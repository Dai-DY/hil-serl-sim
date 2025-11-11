#bash run_actor.sh --eval_checkpoint_step=30000 and --eval_n_trajs=100 --checkpoint_path=../../../../../gpufree-data/hilserl_ouput/five_seed_no_hil
#!/bin/bash
clear
set -e  # Exit immediately on error

if [ $# -lt 2 ]; then
    echo "   Usage: $0 <exp_name> <step> [n_trajs]"
    echo "   exp_name: Experiment directory name (e.g., 'five_seed_no_hil')"
    echo "   step:     Checkpoint step number (e.g., 30000)"
    echo "   n_trajs:  (Optional) Number of evaluation trajectories (default: 100)"
    exit 1
fi

EXP_NAME="$1"
STEP="$2"
N_TRAJS="${3:-100}"  # Default: 100

# ✅ Construct checkpoint path (relative to current directory)
# Assuming current dir is: /root/Project/hil-serl-sim/examples/experiments/pick_cube_sim/
# Checkpoint location: /root/gpufree-data/hilserl_ouput/$EXP_NAME/checkpoint_$STEP
CHECKPOINT_PATH="../../../../../gpufree-data/hilserl_ouput/$EXP_NAME"

# 🔍 Safety check: verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    ABS_PATH="$(realpath "$CHECKPOINT_PATH" 2>/dev/null || echo 'Could not resolve absolute path')"
    echo "   Checkpoint directory not found!"
    echo "   Relative path: $CHECKPOINT_PATH"
    echo "   Resolves to:   $ABS_PATH"
    echo "   Please verify <exp_name> and <step>."
    exit 1
fi

echo "   Checkpoint directory found: $CHECKPOINT_PATH"
echo "   → Evaluating step=$STEP with $N_TRAJS trajectories"

# 🚀 Launch evaluation
bash run_actor.sh \
    --eval_checkpoint_step="$STEP" \
    --eval_n_trajs="$N_TRAJS" \
    --checkpoint_path="$CHECKPOINT_PATH"