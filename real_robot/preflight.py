"""CureWM-Real session preflight — read-only, never moves the robot.

Checks: robot ping, NUC zerorpc port, both D415 serials, disk space, camera
lock file, data-root writability. Run before every collection session:
    conda run -n robot python preflight.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, LOCK_FILE, SESSION_LOG, check_port  # noqa: E402

OK, BAD = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
failures = 0


def report(ok: bool, msg: str):
    global failures
    print(f"  {OK if ok else BAD} {msg}")
    if not ok:
        failures += 1


def main():
    from droid.misc.parameters import hand_camera_id, nuc_ip, robot_ip

    print("== CureWM-Real preflight ==")

    r = subprocess.run(["ping", "-c", "1", "-W", "2", robot_ip], capture_output=True)
    report(r.returncode == 0, f"robot {robot_ip} ping")

    report(check_port(nuc_ip, 4242),
           f"NUC {nuc_ip}:4242 zerorpc (launch_server.sh on the NUC if ✗)")

    try:
        import pyrealsense2 as rs
        serials = {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}
    except Exception as e:  # noqa: BLE001
        serials = set()
        print(f"  {BAD} pyrealsense2: {e}")
    report(hand_camera_id in serials, f"wrist camera {hand_camera_id}")
    ext = serials - {hand_camera_id}
    report(len(ext) == 1, f"external camera ({', '.join(ext) if ext else 'MISSING'})")

    # Dual-camera 720p stress read (5 s): catches marginal USB links that pass
    # single-frame tests but starve under sustained recording load (08-31 lesson).
    if hand_camera_id in serials and len(serials) >= 2:
        try:
            import time as _t
            pipes = []
            for s_ in serials:
                p_, c_ = rs.pipeline(), rs.config()
                c_.enable_device(s_); c_.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                p_.start(c_); pipes.append((s_, p_))
            t0 = _t.time(); miss = {s_: 0 for s_, _ in pipes}; got = {s_: 0 for s_, _ in pipes}
            while _t.time() - t0 < 5:
                for s_, p_ in pipes:
                    try: p_.wait_for_frames(1000); got[s_] += 1
                    except Exception: miss[s_] += 1
            for s_, p_ in pipes:
                try: p_.stop()
                except Exception: pass
            for s_ in got:
                report(miss[s_] == 0 and got[s_] > 40,
                       f"stress {s_}: {got[s_]} frames / {miss[s_]} timeouts in 5s")
        except Exception as e:  # noqa: BLE001
            report(False, f"camera stress test errored: {e}")

    free_gb = shutil.disk_usage(str(DATA_ROOT.parent)).free / 1e9
    report(free_gb > 50, f"disk free {free_gb:.0f} GB")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    probe = DATA_ROOT / ".write_probe"
    try:
        probe.write_text("ok"); probe.unlink()
        report(True, f"data root writable ({DATA_ROOT})")
    except OSError as e:
        report(False, f"data root {DATA_ROOT}: {e}")

    if LOCK_FILE.exists():
        with open(LOCK_FILE) as f:
            lock = json.load(f)
        for s, v in lock.items():
            print(f"  · camlock {s}: exp={v['exposure']:.0f} gain={v['gain']:.0f} "
                  f"wb={v['white_balance']:.0f}")
    else:
        print("  · no camera lock yet — first recorder run will create it "
              "(set final lighting BEFORE that run)")

    if SESSION_LOG.exists():
        tail = SESSION_LOG.read_text().strip().split("\n")[-3:]
        print("  · session log tail:")
        for line in tail:
            print(f"      {line[:110]}")

    n_demo = len(list((DATA_ROOT / "demos").glob("*/demo_*")))
    n_val = len(list((DATA_ROOT / "val").glob("*/demo_*")))
    n_rep = len(list((DATA_ROOT / "replays").glob("*/rep_*"))) \
        + len(list((DATA_ROOT / "replays_val").glob("*/rep_*")))
    print(f"  · inventory: {n_demo} demos, {n_val} val demos, {n_rep} replays")

    print(f"== {'ALL GREEN' if failures == 0 else f'{failures} CHECK(S) FAILED'} ==")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
