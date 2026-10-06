"""CureWM-Real shared utilities: episode I/O, camera AE/AWB locking, phase
annotation, and the safety envelope for replay.

Conventions (mirror scripts/tests/collect_lerobot_dataset.py):
  action  float32 (7,) = absolute target [x, y, z, roll, pitch, yaw, gripper]
  state   float32 (7,) = measured        [x, y, z, roll, pitch, yaw, gripper]
  gripper: 0 = open, 1 = closed (recorded as tap-toggle *intent*, bimodal)
  15 Hz control loop; wrist cam = droid.misc.parameters.hand_camera_id.

Episode directory layout (one dir per episode):
  meta.json     kind/task/family/severity/... (see save_episode)
  traj.npz      actions, states, joints, timestamps, movement
  wrist/%05d.jpg  ext/%05d.jpg
  init_scene.jpg  frame-0 external view (object-placement reference for replays)
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONTROL_HZ = 15
JPEG_QUALITY = 92
GRASP_SETTLE_FRAMES = 15          # ~1 s @15 Hz: "close, hold still ~1s" per demo SOP
GRIP_CLOSE_THRESH = 0.5           # intent >= 0.5 counts as "closed"

DATA_ROOT = Path(os.environ.get("CUREWM_DATA_ROOT", Path.home() / "curewm_data"))
LOCK_FILE = DATA_ROOT / "camera_lock.json"
SESSION_LOG = DATA_ROOT / "session_log.jsonl"

FAMILIES = ("insufficient_grip", "premature_release", "carry_slip", "wrist_tilt")
SEMANTIC_FAMILIES = ("wrong_target",)      # target selection; needs a multi-object scene
ALL_FAMILIES = FAMILIES + SEMANTIC_FAMILIES
SEVERITIES = (0.6, 0.8, 1.0)
MT_SEVERITIES = (0.6, 1.0)                 # wrong_target: near / far distractor


# --------------------------------------------------------------------------
# Episode I/O
# --------------------------------------------------------------------------

@dataclass
class EpisodeBuffer:
    """Accumulates one episode in memory, then flushes to disk atomically."""
    out_dir: Path
    meta: dict
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    movement: list = field(default_factory=list)
    wrist_imgs: list = field(default_factory=list)   # BGR uint8 (imwrite expects BGR)
    ext_imgs: list = field(default_factory=list)

    def add(self, action7, state7, joints7, ts, moving, wrist_bgr, ext_bgr):
        self.actions.append(np.asarray(action7, dtype=np.float32))
        self.states.append(np.asarray(state7, dtype=np.float32))
        self.joints.append(np.asarray(joints7, dtype=np.float32))
        self.timestamps.append(float(ts))
        self.movement.append(bool(moving))
        self.wrist_imgs.append(wrist_bgr)
        self.ext_imgs.append(ext_bgr)

    def __len__(self):
        return len(self.actions)

    def save(self) -> Path:
        assert len(self) > 0, "empty episode"
        tmp = self.out_dir.with_name(self.out_dir.name + ".tmp")
        (tmp / "wrist").mkdir(parents=True, exist_ok=True)
        (tmp / "ext").mkdir(parents=True, exist_ok=True)
        for i, (w, e) in enumerate(zip(self.wrist_imgs, self.ext_imgs)):
            cv2.imwrite(str(tmp / "wrist" / f"{i:05d}.jpg"), w,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(str(tmp / "ext" / f"{i:05d}.jpg"), e,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        cv2.imwrite(str(tmp / "init_scene.jpg"), self.ext_imgs[0])
        np.savez_compressed(
            tmp / "traj.npz",
            actions=np.stack(self.actions),
            states=np.stack(self.states),
            joints=np.stack(self.joints),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            movement=np.asarray(self.movement, dtype=bool),
        )
        self.meta["frames"] = len(self)
        self.meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(tmp / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        tmp.rename(self.out_dir)          # atomic-ish publish
        return self.out_dir


def load_episode(ep_dir: str | Path) -> tuple[dict, dict]:
    ep_dir = Path(ep_dir)
    with open(ep_dir / "meta.json") as f:
        meta = json.load(f)
    npz = np.load(ep_dir / "traj.npz")
    traj = {k: npz[k] for k in npz.files}
    return meta, traj


def log_session(entry: dict):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with open(SESSION_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False,
                           default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


# --------------------------------------------------------------------------
# Camera AE / AWB locking
# --------------------------------------------------------------------------
# One lock file for the whole collection campaign: created once on day 1 by
# letting auto-exposure settle, then every later session re-applies the SAME
# values, so demos and their perturbed replays (possibly days apart) share
# identical imaging parameters.

def _color_sensor(cam):
    """RGB sensor of an open droid RealSenseCamera (needs cam._profile)."""
    import pyrealsense2 as rs
    dev = cam._profile.get_device()
    for s in dev.query_sensors():
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            return s
    raise RuntimeError(f"no RGB sensor on {cam.serial_number}")


def lock_cameras(env, warmup_s: float = 3.0, lock_file: Path = LOCK_FILE) -> dict:
    """Freeze exposure/gain/white-balance on every connected camera.

    First run: let AE/AWB converge on the current scene, read the values,
    switch to manual, persist to `lock_file`. Later runs: re-apply stored
    values verbatim (and complain if a serial is missing from the file).
    Call AFTER the camera pipelines are streaming (i.e. after one successful
    env.read_cameras()).
    """
    import pyrealsense2 as rs

    stored = {}
    if lock_file.exists():
        with open(lock_file) as f:
            stored = json.load(f)

    applied = {}
    for serial, cam in env.camera_reader.camera_dict.items():
        sensor = _color_sensor(cam)
        if serial in stored:
            v = stored[serial]
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, v["exposure"])
            sensor.set_option(rs.option.gain, v["gain"])
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            sensor.set_option(rs.option.white_balance, v["white_balance"])
            print(f"[camlock] {serial}: re-applied stored "
                  f"exp={v['exposure']:.0f} gain={v['gain']:.0f} wb={v['white_balance']:.0f}")
            applied[serial] = v
        else:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            sensor.set_option(rs.option.enable_auto_white_balance, 1)
            t_end = time.time() + warmup_s
            while time.time() < t_end:      # keep frames flowing so AE converges
                env.read_cameras()
            v = {
                "exposure": float(sensor.get_option(rs.option.exposure)),
                "gain": float(sensor.get_option(rs.option.gain)),
                "white_balance": float(sensor.get_option(rs.option.white_balance)),
            }
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, v["exposure"])
            sensor.set_option(rs.option.gain, v["gain"])
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            sensor.set_option(rs.option.white_balance, v["white_balance"])
            print(f"[camlock] {serial}: NEW lock exp={v['exposure']:.0f} "
                  f"gain={v['gain']:.0f} wb={v['white_balance']:.0f}")
            stored[serial] = v
            applied[serial] = v

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as f:
        json.dump(stored, f, indent=2)
    return applied


def configure_cameras(env):
    """Image-only trajectory mode on both D415s; returns {'wrist': serial, 'external': serial}."""
    from droid.misc.parameters import hand_camera_id

    cam_dict = env.camera_reader.camera_dict
    if len(cam_dict) < 2:
        raise RuntimeError(f"need 2 cameras, found {list(cam_dict)}")
    if hand_camera_id not in cam_dict:
        raise RuntimeError(f"wrist camera {hand_camera_id} not connected; found {list(cam_dict)}")
    external = [s for s in cam_dict if s != hand_camera_id]
    if len(external) != 1:
        raise RuntimeError(f"expected exactly one external camera, found {external}")
    for cam in cam_dict.values():
        cam.set_reading_parameters(image=True, depth=False, pointcloud=False,
                                   concatenate_images=False)
    env.camera_reader.set_trajectory_mode()
    obs = env.read_cameras()[0]           # opens pipelines + sanity frame
    for s in cam_dict:
        if f"{s}_left" not in obs.get("image", {}):
            raise RuntimeError(f"camera {s} not streaming")
    return {"wrist": hand_camera_id, "external": external[0]}


def grab_views(camera_obs, serials) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(wrist_bgr, ext_bgr) from env.read_cameras()[0] / get_observation(); None on drop."""
    imgs = camera_obs.get("image", {})
    return imgs.get(serials["wrist"] + "_left"), imgs.get(serials["external"] + "_left")


def recover_camera(env, serial, locks: dict | None = None):
    """Hot-restart one camera pipeline in place and re-apply its lock.

    Used by the recorder when a camera stops delivering frames (persistent
    wait_for_frames timeouts) — root cause unclear (single-process dual-stream
    stall not reproducible standalone), so we heal instead of dying.
    """
    import pyrealsense2 as rs
    cam = env.camera_reader.camera_dict[serial]
    print(f"[camrecover] restarting pipeline for {serial} ...", flush=True)
    try:
        cam._stop_pipeline_only()
    except Exception:
        pass
    time.sleep(0.5)
    cam.set_trajectory_mode()          # re-opens the pipeline
    if locks and serial in locks:
        v = locks[serial]
        sensor = _color_sensor(cam)
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, v["exposure"])
        sensor.set_option(rs.option.gain, v["gain"])
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.white_balance, v["white_balance"])
    print(f"[camrecover] {serial} pipeline restarted", flush=True)


# --------------------------------------------------------------------------
# Phase annotation (real-robot mirror of failure_perturbations.annotate_phases)
# --------------------------------------------------------------------------
# Sim used simulator contacts for grasp settle; on hardware we use the demo
# SOP ("close, hold ~1 s") as a fixed settle window instead. Deterministic,
# recorded in meta, and validated by the monotonicity gate like everything else.

def annotate_phases(actions: np.ndarray) -> dict:
    """Segment approach / grasp / carry / place from the gripper intent channel.

    Returns dict of frame indices:
      first_close: first frame commanded closed  (T if never closes)
      carry_start: first_close + GRASP_SETTLE_FRAMES (clipped to T)
      t_release:   first frame commanded open after carry_start (T if none)
    """
    T = len(actions)
    g = actions[:, 6]
    closed = g >= GRIP_CLOSE_THRESH
    if not closed.any():
        return {"T": T, "first_close": T, "carry_start": T, "t_release": T}
    first_close = int(np.argmax(closed))
    carry_start = min(first_close + GRASP_SETTLE_FRAMES, T)
    open_after = ~closed
    open_after[:carry_start] = False
    t_release = int(np.argmax(open_after)) if open_after.any() else T
    return {"T": T, "first_close": first_close,
            "carry_start": carry_start, "t_release": t_release}


# --------------------------------------------------------------------------
# Safety envelope for replay
# --------------------------------------------------------------------------

class SafetyEnvelope:
    """Workspace box auto-derived from the source demo, plus per-step jump guard.

    The mechanical families never move xyz off the demo path (wrist_tilt is
    orientation-only), so violations indicate a bug — clamp, count, and report.
    `wrong_target` re-aims the reach at another object by design; pass the
    perturbed action stream as `extra_xyz` so the box is the union of the two
    paths (the per-step jump guard still catches runtime blow-ups).
    """
    def __init__(self, demo_actions: np.ndarray, margin=0.10, z_margin=0.005,
                 max_step_m=0.06, max_step_rad=0.35, extra_xyz=None):
        xyz = demo_actions[:, :3]
        if extra_xyz is not None:
            xyz = np.concatenate([xyz, np.asarray(extra_xyz, float)[:, :3]], axis=0)
        self.lo = xyz.min(0) - np.array([margin, margin, z_margin])
        self.hi = xyz.max(0) + np.array([margin, margin, margin])
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.violations = 0
        self._prev = None

    def filter(self, action7: np.ndarray) -> np.ndarray:
        a = action7.copy()
        clipped = np.clip(a[:3], self.lo, self.hi)
        if not np.allclose(clipped, a[:3], atol=1e-9):
            self.violations += 1
            a[:3] = clipped
        if self._prev is not None:
            step = a[:3] - self._prev[:3]
            n = np.linalg.norm(step)
            if n > self.max_step_m:
                self.violations += 1
                a[:3] = self._prev[:3] + step * (self.max_step_m / n)
        self._prev = a.copy()
        return a


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def check_port(host: str, port: int, timeout=2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def episode_name(prefix: str, task_id: str) -> str:
    return f"{prefix}_{task_id}_{time.strftime('%m%d_%H%M%S')}"
