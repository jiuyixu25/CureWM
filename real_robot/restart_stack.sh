#!/usr/bin/env bash
# Restart the robot control stack: clear the control box, start a fresh zerorpc server,
# wait for its port.
#
# The DROID zerorpc shell crashes on a second launch_controller, so a fresh server has to
# be started before every recorder or replayer run.  A failed cleanup must abort: an old
# server leaves the port falsely open and everything after it fails.
#
# Configure by environment, with no credentials in this file:
#   CUREWM_ROBOT_HOST     user@host of the control box (NUC)  (required)
#   CUREWM_ROBOT_PORT     zerorpc port                        (default 4242)
#   CUREWM_DROID_ROOT     DROID checkout on the control box   (default ~/droid)
#   GRIPPER_CLOSE_FORCE   gripper closing force in N          (default 50)
#
# Prerequisites on the control box:
#   - key-based SSH from this machine (ssh-copy-id), so no password is ever passed
#   - passwordless sudo for the two pkill lines below, e.g. in /etc/sudoers.d/curewm:
#       <user> ALL=(root) NOPASSWD: /usr/bin/pkill
set -u
: "${CUREWM_ROBOT_HOST:?set CUREWM_ROBOT_HOST=user@host for the control box}"
PORT=${CUREWM_ROBOT_PORT:-4242}
DROID_ROOT=${CUREWM_DROID_ROOT:-\$HOME/droid}
FORCE=${GRIPPER_CLOSE_FORCE:-50}

CLEAN_OK=0
for try in 1 2 3; do
  echo "[stack] clearing stale processes on the control box (attempt $try)..."
  OUT=$(timeout 25 ssh -o ConnectTimeout=8 -o BatchMode=yes "$CUREWM_ROBOT_HOST" "
    pkill -9 -f 'python run_serve[r].py' 2>/dev/null
    sudo -n pkill -9 -f 'franka_panda_clien[t]|franka_hand_clien[t]|launch_robo[t].py|launch_grippe[r].py' 2>/dev/null
    sudo -n pkill -9 -x run_server 2>/dev/null
    sleep 1
    rm -f $DROID_ROOT/server_nohup.log
    cd $DROID_ROOT/scripts/server && setsid nohup env GRIPPER_CLOSE_FORCE='$FORCE' \
      bash launch_server.sh > $DROID_ROOT/server_nohup.log 2>&1 < /dev/null &
    echo CLEAN_AND_SPAWNED" 2>&1)
  if echo "$OUT" | grep -q CLEAN_AND_SPAWNED; then CLEAN_OK=1; break; fi
  sleep 5
done
[ "$CLEAN_OK" = 1 ] || { echo "[stack] cleanup or spawn failed; is the control box reachable?"; exit 1; }

echo "[stack] waiting for port $PORT ..."
for i in $(seq 1 30); do
  if timeout 10 ssh -o ConnectTimeout=6 -o BatchMode=yes "$CUREWM_ROBOT_HOST" \
      "ss -tln | grep -q ':$PORT'" 2>/dev/null; then
    echo "[stack] port $PORT up (gripper force ${FORCE} N) -- ready."
    echo "[stack] note: this server accepts exactly one launch_controller."
    exit 0
  fi
  sleep 3
done
echo "[stack] timed out: port $PORT is not listening"; exit 1
