export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=five_seed_no_hil \
    --demo_path=/root/Project/hil-serl-sim/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_5_demos_2025-11-07_11-21-47.pkl \
    --learner \