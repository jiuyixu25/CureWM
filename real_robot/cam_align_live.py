"""Live external-camera alignment aid: 50/50 blend of the 08-31 reference and the live view, the
reference's edges drawn in green, and a running ORB/RANSAC estimate of centre shift + scale.
Move the tripod until the green edges sit on the live scene and the numbers turn green.

    conda run --no-capture-output -n robot python cam_align_live.py [--ref <init_scene.jpg>]

Keys: q quit   s save a snapshot (blend + live) to --out   e toggle edge overlay   b toggle blend
Falls back to writing --out/live_blend.jpg twice a second when no GUI is available (open it in
an image viewer that reloads on change, e.g. eog).  Robot should be at HOME (as in the reference).
Read-only: no robot motion, cameras released on exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam_check import homography  # noqa: E402
from common import DATA_ROOT  # noqa: E402

TOL_PX, TOL_SCALE = 8.0, 0.02


class TkViewer:
    """Minimal window for headless OpenCV builds (cv2 here has no highgui)."""

    def __init__(self, title, scale=0.75):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.Image, self.ImageTk, self.scale = Image, ImageTk, scale
        self.root = tk.Tk(); self.root.title(title)
        self.label = tk.Label(self.root); self.label.pack()
        self.key, self.closed = None, False
        self.root.bind("<Key>", lambda e: setattr(self, "key", e.char))
        self.root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "closed", True))

    def show(self, bgr):
        if self.closed:
            return "q"
        small = cv2.resize(bgr, None, fx=self.scale, fy=self.scale)
        im = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=im); self.label.image = im
        self.root.update_idletasks(); self.root.update()
        k, self.key = self.key, None
        return k



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=None)
    p.add_argument("--out", default="/tmp/cam_align")
    p.add_argument("--seconds", type=float, default=0, help="auto-exit after this many seconds (0 = run until q)")
    args = p.parse_args()
    t_end = time.time() + args.seconds if args.seconds > 0 else None
    ref_path = Path(args.ref) if args.ref else sorted((DATA_ROOT / "val" / "T1").glob("demo_*"))[0] / "init_scene.jpg"
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise SystemExit(f"cannot read {ref_path}")
    h, w = ref.shape[:2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(ref_gray, 80, 160)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import pyrealsense2 as rs
    from droid.misc.parameters import hand_camera_id
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
    ext = [s for s in serials if s != hand_camera_id]
    if len(ext) != 1:
        raise SystemExit(f"need one external camera, found {ext}")
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(ext[0]); cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    pipe.start(cfg)

    viewer = None
    try:
        viewer = TkViewer("cam_align  (q quit, s snapshot, e edges, b blend)")
    except Exception as e:  # noqa: BLE001
        print(f"[align] no window ({type(e).__name__}) — writing {out}/live_blend.jpg twice a second; open it in eog")
    show_edges, show_blend, k, txt, ok, last_write = True, True, 0, "estimating...", False, 0.0
    try:
        while t_end is None or time.time() < t_end:
            live = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
            k += 1
            if k % 5 == 0:
                H, inl = homography(ref_gray, cv2.cvtColor(live, cv2.COLOR_BGR2GRAY))
                if H is not None:
                    c = np.float32([[[w / 2, h / 2]]])
                    d = cv2.perspectiveTransform(c, H)[0, 0] - c[0, 0]
                    sc = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
                    ok = np.hypot(*d) < TOL_PX and abs(sc - 1) < TOL_SCALE
                    hint = ("closer than ref -> move BACK" if sc > 1 + TOL_SCALE else
                            "farther than ref -> move CLOSER" if sc < 1 - TOL_SCALE else "distance ok")
                    pan = ("scene left -> pan camera LEFT" if d[0] < -TOL_PX else
                           "scene right -> pan camera RIGHT" if d[0] > TOL_PX else "pan ok")
                    tilt = ("scene up -> tilt UP / lower cam" if d[1] < -TOL_PX else
                            "scene down -> tilt DOWN / raise cam" if d[1] > TOL_PX else "tilt ok")
                    txt = f"dx {d[0]:+.0f}px  dy {d[1]:+.0f}px  scale {sc:.3f}  inl {inl} | {hint}; {pan}; {tilt}"
                else:
                    ok, txt = False, f"no homography ({inl} matches)"
            vis = cv2.addWeighted(ref, 0.5, live, 0.5, 0) if show_blend else live.copy()
            if show_edges:
                vis[edges > 0] = (0, 255, 0)
            cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "ALIGNED" if ok else "MOVE CAMERA", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
            if viewer is not None:
                key = viewer.show(vis)
                if key == "q":
                    break
                if key == "e":
                    show_edges = not show_edges
                if key == "b":
                    show_blend = not show_blend
                if key == "s":
                    cv2.imwrite(str(out / f"snap_{int(time.time())}_blend.jpg"), vis)
                    cv2.imwrite(str(out / f"snap_{int(time.time())}_live.jpg"), live)
                    print(f"[align] saved snapshot -> {out}   {txt}")
            elif time.time() - last_write > 0.5:
                cv2.imwrite(str(out / "live_blend.jpg"), vis); last_write = time.time()
                print(f"\r[align] {txt}     ", end="", flush=True)
    finally:
        pipe.stop()
        if viewer is not None:
            try:
                viewer.root.destroy()
            except Exception:  # noqa: BLE001
                pass
        cv2.imwrite(str(out / "final_live.jpg"), live)
        print(f"\n[align] final: {txt}  ({'ALIGNED' if ok else 'NOT aligned'}); final frame -> {out}/final_live.jpg")


if __name__ == "__main__":
    main()
