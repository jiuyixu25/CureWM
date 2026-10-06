# Hardware pipeline

The same protocol as in simulation, on a Franka arm driven through
[DROID](https://github.com/droid-dataset/droid): collect scripted demonstrations, replay
them action by action, perturb one replay at a chosen severity, and label the outcome from
gripper telemetry with an operator veto.

## Requirements

- A DROID installation on this workstation, and a control box (the NUC, in DROID's
  terminology) running DROID's zerorpc server, reachable over SSH with **key-based
  authentication**.
- Passwordless sudo on the control box for the two `pkill` lines in `restart_stack.sh`,
  e.g. in `/etc/sudoers.d/curewm`:
  ```
  <user> ALL=(root) NOPASSWD: /usr/bin/pkill
  ```
- `pip install -r ../requirements/real_robot.txt`

## Configuration

Nothing is hardcoded; everything comes from the environment.

| Variable | Meaning | Default |
|---|---|---|
| `CUREWM_ROBOT_HOST` | `user@host` of the control box | required |
| `CUREWM_ROBOT_PORT` | zerorpc port | `4242` |
| `CUREWM_DROID_ROOT` | DROID checkout on the control box | `$HOME/droid` |
| `CUREWM_DATA_ROOT`  | where recordings are written | `~/curewm_data` |
| `GRIPPER_CLOSE_FORCE` | gripper closing force, newtons | `50` |

The gripper force matters: it is part of the experimental condition, and sessions run at
different forces are not interchangeable. Record which one a session used.

## Order of operations

The DROID zerorpc shell crashes on a second `launch_controller`, so **a fresh server is
required before every recorder or replayer run**:

```bash
export CUREWM_ROBOT_HOST=user@control-box
bash restart_stack.sh
```

Then:

```bash
python3 scripted_demo.py --task-id T1 --num 20 --per-spot 3   # collect demonstrations
python3 make_plan.py     --task-id T1 --append                # plan the perturbations
python3 replay_perturbed.py --plan plan_T1.json               # replay and label
```

`scripted_demo.py --dry` prints a generated trajectory without touching the robot, which is
the fastest way to check a change. `perturb.py --episode <dir> --family X` does the same
for a perturbation, on a recorded episode.

`auto_collect.py` runs the multi-object (cube) variant, where the robot re-arranges the
scene itself between episodes; `--confirm` asks before each episode and lets the operator
discard one before it is written.

## Calibration

`scripted_demo.py --calibrate` teaches the object spots and the plate position.
`cam_check.py` compares the current external camera view against a stored reference frame
and reports translation and scale residuals; `cam_align_live.py` does it interactively.
`point_plate.py` locates the plate by closing the gripper and probing downward for contact.

Between sessions the camera and the plate drift. Re-align before collecting, and record the
residuals: a session whose scene has shifted enough can fall below the rendering-quality
screen the paper applies, in which case its readings are not interpretable.
