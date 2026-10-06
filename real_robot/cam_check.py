"""External-camera alignment check against a T1 reference frame. No robot, read-only.

The external D415 was re-positioned for the cube sessions (cubes_camB, 09-11). Every
cup episode recorded after that must come from the 08-31 pose, or the world model sees
a viewpoint it was never adapted to and the probe measures the camera, not the model.

    conda run -n robot python cam_check.py            # default reference = first val/T1 demo
    conda run -n robot python cam_check.py --ref <init_scene.jpg> --out /tmp/cam_check

Writes live.jpg, blend.jpg (50/50 overlay) and side.jpg, then estimates an ORB+RANSAC
homography reference->live and reports the shift of the image centre and the scale.
Aligned when |shift| < 8 px and |scale-1| < 2 %.  Repeat after every nudge of the tripod.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT  # noqa: E402


def grab_external(warmup_frames: int = 30):
    import pyrealsense2 as rs
    from droid.misc.parameters import hand_camera_id
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
    ext = [s for s in serials if s != hand_camera_id]
    if len(ext) != 1:
        raise SystemExit(f"[cam] need exactly one external camera, found {ext} (all: {serials})")
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(ext[0])
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(warmup_frames):                 # let auto-exposure settle
            pipe.wait_for_frames()
        frame = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
    finally:
        pipe.stop()
    return ext[0], frame


def homography(ref_gray, live_gray):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(live_gray, None)
    if d1 is None or d2 is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 12:
        return None, len(matches)
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(mask.sum()) if mask is not None else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=None, help="reference external frame (default: first val/T1 demo)")
    p.add_argument("--out", default="/tmp/cam_check")
    args = p.parse_args()
    ref_path = Path(args.ref) if args.ref else sorted((DATA_ROOT / "val" / "T1").glob("demo_*"))[0] / "init_scene.jpg"
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise SystemExit(f"[cam] cannot read reference {ref_path}")
    serial, live = grab_external()
    if live.shape != ref.shape:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "live.jpg"), live)
    cv2.imwrite(str(out / "blend.jpg"), cv2.addWeighted(ref, 0.5, live, 0.5, 0))
    cv2.imwrite(str(out / "side.jpg"), np.hstack([ref, live]))
    H, inl = homography(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(live, cv2.COLOR_BGR2GRAY))
    print(f"[cam] external {serial}; reference {ref_path}")
    if H is None:
        print(f"[cam] homography failed ({inl} matches), scene too different. Compare {out}/side.jpg by eye")
        return
    h, w = ref.shape[:2]
    c = np.float32([[[w / 2, h / 2]]])
    shift = cv2.perspectiveTransform(c, H)[0, 0] - c[0, 0]
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    ok = np.hypot(*shift) < 8 and abs(scale - 1) < 0.02
    print(f"[cam] inliers={inl}  centre shift dx={shift[0]:+.1f}px dy={shift[1]:+.1f}px  scale={scale:.3f}  "
          f"-> {'ALIGNED' if ok else 'MOVE CAMERA'}  (images in {out})")
    if not ok:
        print("[cam] hint: positive dx = scene appears shifted RIGHT in live -> pan camera right/left to compensate; "
              "scale>1 = camera closer than reference")


if __name__ == "__main__":
    main()
