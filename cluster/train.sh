#!/usr/bin/env bash
# Post-training launcher for the Cosmos-Policy LIBERO arm.
#
# Every arm in the paper is this same launcher with a different rollout mixture in
# CUREWM_ROLLOUT_DIR. That covers CureWM, the step-matched nominal control, and the
# own-failure and other-demonstration ablations. Nothing else differs: they share the
# initialization, schedule, seed, batch size and sampling ratios.
#
# Required:
#   CUREWM_ROOT           working root holding third_party/ and train_output/
#   CUREWM_ROLLOUT_DIR    rollout mixture for this arm
#   CUREWM_INIT_PT        released weights to fine-tune from
# Optional:
#   CUREWM_SEED (0)  CUREWM_BATCH_SIZE (4)  CUREWM_MAX_ITER (20000)
#   CUREWM_SAVE_ITER (2500)  CUREWM_NUM_WORKERS (2)  CUREWM_OUTPUT_ROOT
#   CUREWM_EXTRA_OVERRIDES  extra key=value overrides passed through to the trainer
#
# Credentials come from the environment, never from this file.  If the tokenizer you use
# is gated, export HF_TOKEN in your shell or log in with `huggingface-cli login` first.
set -eu
: "${CUREWM_ROOT:?set CUREWM_ROOT to the working root}"
: "${CUREWM_ROLLOUT_DIR:?set CUREWM_ROLLOUT_DIR to the rollout mixture for this arm}"
: "${CUREWM_INIT_PT:?set CUREWM_INIT_PT to the released weights}"

CP=$CUREWM_ROOT/third_party/cosmos-policy
export HF_HOME=${HF_HOME:-$CUREWM_ROOT/hf_cache}
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=${WANDB_MODE:-offline}

# Inside the container the CUDA stub has to be visible to triton, and the inherited
# compiler variables confuse the build.
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:/sbin:/usr/sbin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/.singularity.d/libs
export CUDA_HOME=/usr/local/cuda
unset GCC_EXEC_PREFIX CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH PYTHONPATH CC CXX
TLIB=$CP/.venv/lib/python3.10/site-packages/triton/backends/nvidia/lib
[ -e /usr/local/cuda/lib64/stubs/libcuda.so ] && \
  ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$TLIB/libcuda.so" 2>/dev/null || true

export IMAGINAIRE_OUTPUT_ROOT=${CUREWM_OUTPUT_ROOT:-$CUREWM_ROOT/train_output}
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
export BASE_DATASETS_DIR=$CUREWM_ROOT/third_party

# Read by the injected experiment config (see experiment_config_additions.py).
export FAILSAFE_ROLLOUT_DIR=$CUREWM_ROLLOUT_DIR
export FAILSAFE_INIT_PT=$CUREWM_INIT_PT
export FAILSAFE_BATCH_SIZE=${CUREWM_BATCH_SIZE:-4}
export FAILSAFE_MAX_ITER=${CUREWM_MAX_ITER:-20000}
export FAILSAFE_SAVE_ITER=${CUREWM_SAVE_ITER:-2500}
export FAILSAFE_NUM_WORKERS=${CUREWM_NUM_WORKERS:-2}
SEED=${CUREWM_SEED:-0}

echo "[train] seed=$SEED out=$IMAGINAIRE_OUTPUT_ROOT rollout=$FAILSAFE_ROLLOUT_DIR"
cd "$CP"
exec .venv/bin/torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero__failsafe_ft_v1 \
  trainer.seed=$SEED ${CUREWM_EXTRA_OVERRIDES:-}
