"""In-process demonstration collection, calling ManiSkill3's panda motion-planning solver directly.

The official run.py is not used because its multiprocess fork plus rendering
segfaults on this machine.  This script is single-process and renders nothing
(obs_mode="none").  Failed demonstrations are still written: generate_pairs filters them
out downstream with a nominal replay, so there is no need to screen them here.
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.record import RecordEpisode


def get_solver(env_id: str):
    from mani_skill.examples.motionplanning.panda import solutions as S
    table = {
        "PickCube-v1": S.solvePickCube,
        "StackCube-v1": S.solveStackCube,
        "PegInsertionSide-v1": S.solvePegInsertionSide,
        "PlaceSphere-v1": S.solvePlaceSphere,
        "LiftPegUpright-v1": S.solveLiftPegUpright,
    }
    return table[env_id]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("-n", type=int, default=30, help="number of successful demonstrations to collect")
    p.add_argument("--out", default="demos/mp")
    args = p.parse_args()

    solve = get_solver(args.env_id)
    env = gym.make(args.env_id, num_envs=1, obs_mode="none",
                   control_mode="pd_joint_pos", sim_backend="physx_cpu")
    env = RecordEpisode(env, output_dir=f"{args.out}/{args.env_id}",
                        trajectory_name="mp", save_video=False)

    ok = tried = seed = 0
    while ok < args.n and tried < args.n * 4:
        try:
            res = solve(env, seed=seed, debug=False, vis=False)
            info = res[-1] if isinstance(res, tuple) else {}
            success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        except Exception as e:  # one bad demonstration must not take down the batch
            print(f"seed {seed}: solver error: {e}", flush=True)
            success = False
        tried += 1
        ok += int(success)
        print(f"seed {seed}: success={success} ({ok}/{args.n})", flush=True)
        seed += 1
    env.close()
    print(f"[collect] {ok} successes / {tried} tried -> {args.out}/{args.env_id}")


if __name__ == "__main__":
    main()
