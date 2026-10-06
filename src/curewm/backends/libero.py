"""SimBackend for LIBERO (robosuite/MuJoCo), plus the official demonstration loader.

Generates failure-inducing counterfactual pairs in the same domain as
Cosmos-Policy-LIBERO, for measuring optimism before treatment and for evaluating
after post-training.

Conventions:
- A LIBERO action is 7-dimensional OSC_POSE in [-1, 1]. Dimension 7 is the gripper and
  uses **+1 = close, -1 = open**, the opposite sign from ManiSkill. The engine's own
  convention is grip in [0 closed, 1 open]; the conversion happens only at this file's
  boundary.
- Contact is approximated by the gripper close command plus a fixed three-step settling
  delay. A general robosuite geom-contact query would need per-task object names, so this
  conservative heuristic is used instead; revisit it if a family's severity curve looks
  wrong.
- Environments are created lazily per task and cached, so sort demonstrations by task
  before feeding them to the engine.
- Export before running: MUJOCO_GL=egl, __EGL_VENDOR_LIBRARY_FILENAMES=
  /usr/share/glvnd/egl_vendor.d/10_nvidia.json, PYTHONPATH=<LIBERO checkout>, and
  LIBERO_CONFIG_PATH=<an isolated config root>, which keeps ~/.libero untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from curewm.perturbations import Trajectory

GRASP_ESTABLISH_STEPS = 3  # steps after the close command before the grasp counts as settled


class LiberoBackend:
    """One environment, rendered off-screen: 128x128 agentview, flipped vertically
    because robosuite's off-screen buffer is upside down."""

    def __init__(self, suite: str = "libero_goal", image_size: int = 224, render: bool = True):
        self.suite_name = suite
        self.bench = benchmark.get_benchmark_dict()[suite]()
        self.image_size = image_size
        self.render = render  # False skips rendering (an order of magnitude faster) and
                              # records only states, actions and rewards
        self._env = None
        self._task_id: int | None = None

    def _ensure_env(self, task_id: int):
        if self._task_id == task_id and self._env is not None:
            return
        if self._env is not None:
            self._env.close()
        task = self.bench.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if self.render:
            self._env = OffScreenRenderEnv(bddl_file_name=bddl,
                                           camera_heights=self.image_size,
                                           camera_widths=self.image_size)
        else:
            from libero.libero.envs.env_wrapper import ControlEnv
            self._env = ControlEnv(bddl_file_name=bddl,
                                   has_offscreen_renderer=False, use_camera_obs=False)
        self._task_id = task_id

    @property
    def _sim(self):
        s = getattr(self._env, "sim", None)
        return s if s is not None else self._env.env.sim

    # -------------------------------------------------- SimBackend protocol
    def reset_to(self, init_state: dict) -> None:
        self._ensure_env(int(init_state["task_id"]))
        self._env.reset()
        self._env.set_init_state(init_state["mj_state"])

    def rollout(self, actions: np.ndarray) -> dict:
        a = actions.astype(np.float32).copy()
        a[:, 6] = 1.0 - 2.0 * a[:, 6]  # grip [0 closed, 1 open] -> LIBERO [+1 closed, -1 open]
        frames, wrist_frames, proprio, contacts, sim_states = [], [], [], [], []
        success = False
        closed_since = None
        for t in range(len(a)):
            obs, _r, done, _info = self._env.step(a[t])
            if self.render:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1], dtype=np.uint8))
                wrist_frames.append(np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1], dtype=np.uint8))
            proprio.append(np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                           obs["robot0_gripper_qpos"]]).astype(np.float32))
            sim_states.append(np.asarray(self._sim.get_state().flatten(), dtype=np.float64))
            if a[t, 6] > 0:  # close command
                closed_since = t if closed_since is None else closed_since
            else:
                closed_since = None
            contacts.append(closed_since is not None and (t - closed_since) >= GRASP_ESTABLISH_STEPS)
            if self._env.check_success():
                success = True
        T = len(a)
        rewards = np.zeros(T, dtype=np.uint8); rewards[-1] = int(success)  # official semantics: last step only
        dones = np.zeros(T, dtype=np.uint8); dones[-1] = 1
        out = {"proprio": np.stack(proprio), "contacts": np.array(contacts, dtype=bool),
               "sim_states": np.stack(sim_states), "rewards": rewards, "dones": dones,
               "libero_actions": a.astype(np.float64), "success": bool(success)}
        if self.render:
            out["frames"] = np.stack(frames)
            out["wrist_frames"] = np.stack(wrist_frames)
        return out

    def close(self):
        if self._env is not None:
            self._env.close()


# -------------------------------------------------- official demonstration loading

def load_libero_demos(suite: str = "libero_goal", demos_per_task: int = 5,
                      max_tasks: int | None = None,
                      task_ids: list[int] | None = None) -> list[Trajectory]:
    """Read the official hdf5 at datasets/<suite>/<task>_demo.hdf5, taking the first N
    demonstrations per task.  Each demonstration carries the full MuJoCo state sequence;
    states[0] is the initial state the simulator can be reset to exactly.  An explicit
    task_ids takes precedence, which is how work is sharded across processes."""
    bench = benchmark.get_benchmark_dict()[suite]()
    root = Path(get_libero_path("datasets")) / suite
    out: list[Trajectory] = []
    if task_ids is None:
        n_tasks = bench.get_num_tasks() if max_tasks is None else min(max_tasks, bench.get_num_tasks())
        task_ids = list(range(n_tasks))
    for ti in task_ids:
        task = bench.get_task(ti)
        f = root / f"{task.name}_demo.hdf5"
        if not f.exists():
            print(f"[loader] {f.name} is missing, skipping")
            continue
        with h5py.File(f, "r") as h:
            keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos_per_task]
            for k in keys:
                g = h["data"][k]
                a = np.asarray(g["actions"], dtype=np.float32)
                a[:, 6] = (1.0 - a[:, 6]) / 2.0  # LIBERO [+1 closed, -1 open] -> grip [0 closed, 1 open]
                out.append(Trajectory(
                    actions=a,
                    init_state={"task_id": ti, "mj_state": np.asarray(g["states"][0])},
                    task_id=f"{suite}/t{ti:02d}",
                    meta={"task_name": task.name, "demo": k, "language": task.language}))
    print(f"[loader] {suite}: {len(out)} demos from {len(task_ids)} tasks")
    return out
