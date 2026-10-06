"""SimBackend for ManiSkill3, plus the ManiSkill trajectory loader.

Conventions (see the header of perturbations.py):
- A ManiSkill action is pd_ee_delta_pose, shape (7,), with gripper in [-1 closed, +1 open].
- The engine's own convention is grip in [0 closed, 1 open]; the conversion happens only
  at this file's boundary.
- Contact is approximated by agent.is_grasping(target), which is all the phase split
  needs: whether the object is held or not.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  registers the environments

from curewm.perturbations import Trajectory

# Attribute name of each task's grasp target on the env; register new tasks here
TASK_TARGET_ATTR = {
    "PickCube-v1": "cube",
    "StackCube-v1": "cubeA",
    "PegInsertionSide-v1": "peg",
    "PullCubeTool-v1": "cube",
    "PlaceSphere-v1": "obj",
    "LiftPegUpright-v1": "peg",
}


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class ManiSkill3Backend:
    """One environment, CPU physics with GPU rendering, using roughly 1-2 GB of VRAM."""

    def __init__(self, env_id: str, max_episode_steps: int = 400):
        self.env = gym.make(
            env_id, num_envs=1, obs_mode="rgb",
            control_mode="pd_ee_delta_pose", sim_backend="auto",
            max_episode_steps=max_episode_steps,
        )
        self.env_id = env_id
        self._target_attr = TASK_TARGET_ATTR.get(env_id)

    # -------------------------------------------------- SimBackend protocol
    def reset_to(self, init_state: dict) -> None:
        seed = init_state.get("seed")
        self.env.reset(seed=int(seed) if seed is not None else None)
        st = init_state.get("state_dict")
        if st is not None:
            self.env.unwrapped.set_state_dict(_to_torch(st))

    def rollout(self, actions: np.ndarray) -> dict:
        a = actions.astype(np.float32).copy()
        a[:, 6] = a[:, 6] * 2.0 - 1.0  # grip [0,1] -> [-1,1]
        frames, contacts = [], []
        success = False
        for t in range(len(a)):
            obs, _rew, _term, trunc, info = self.env.step(a[t])
            frames.append(self._rgb(obs))
            contacts.append(self._grasping())
            s = info.get("success")
            if s is not None and bool(_np(s).reshape(-1)[0]):
                success = True  # latch on first success; for most tasks it is irreversible
            if bool(_np(trunc).reshape(-1)[0]):
                break
        return {
            "frames": np.stack(frames).astype(np.uint8),
            "contacts": np.array(contacts, dtype=bool),
            "success": success,
            "obj_states": None,
            "ee_states": None,
        }

    # -------------------------------------------------- internals
    def _rgb(self, obs) -> np.ndarray:
        cams = obs["sensor_data"]
        cam = cams.get("base_camera") or next(iter(cams.values()))
        return _np(cam["rgb"])[0]

    def _grasping(self) -> bool:
        if self._target_attr is None:
            return False
        obj = getattr(self.env.unwrapped, self._target_attr, None)
        if obj is None:
            return False
        try:
            return bool(_np(self.env.unwrapped.agent.is_grasping(obj)).reshape(-1)[0])
        except Exception:
            return False


def _to_torch(x):
    if isinstance(x, dict):
        return {k: _to_torch(v) for k, v in x.items()}
    return torch.as_tensor(x)


# -------------------------------------------------- trajectory loading

def _state0(node):
    """Walk the h5 env_states group recursively and return the state dict at t=0."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[0])
    return {k: _state0(v) for k, v in node.items()}


def load_ms_demos(h5_path: str | Path, env_id: str, max_demos: int | None = None) -> list[Trajectory]:
    """Load a ManiSkill trajectory already converted to pd_ee_delta_pose (.h5 plus the
    matching .json)."""
    h5_path = Path(h5_path)
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    episodes = meta["episodes"][: max_demos or None]
    out: list[Trajectory] = []
    with h5py.File(h5_path, "r") as f:
        for ep in episodes:
            g = f[f"traj_{ep['episode_id']}"]
            a = np.asarray(g["actions"], dtype=np.float32)
            a[:, 6] = (a[:, 6] + 1.0) / 2.0  # grip [-1,1] -> [0,1]
            init: dict = {"seed": (ep.get("reset_kwargs") or {}).get("seed")}
            if "env_states" in g:
                try:
                    init["state_dict"] = _state0(g["env_states"])
                except Exception:
                    pass
            out.append(Trajectory(actions=a, init_state=init, task_id=env_id,
                                  meta={"episode_id": ep["episode_id"]}))
    print(f"[loader] {h5_path.name}: {len(out)} demos, T~{int(np.mean([d.T for d in out]))}")
    return out
