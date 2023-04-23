CUDA_VISIBLE_DEVICES=3 xvfb-run -a python eval_sampleSucc_main.py \
    --categories Box,Bucket,Dishwasher,Display,Microwave,Bench,Bowl,Keyboard2  \
    --primact_type pushing  \
    --aff1_version model_aff_fir   \
    --actor1_version model_actor_fir   \
    --critic1_version model_critic_fir   \
    --aff2_version model_aff_sec   \
    --actor2_version model_actor_sec   \
    --critic2_version model_critic_sec   \
    --CA_path  ../logs/finetune/exp-FINETUNE  \
    --CA_eval_epoch 8-0 \
    --out_folder sampleSucc_CA_results \
    --val_data_dir ../data/PUSH_TEST_DATA \
    --val_buffer_max_num 500  \
    --coordinate_system cambase \
    --target_part_state closed  \
    --start_dist 0.45   \
    --final_dist 0.10   \
    --move_steps 3500   \
    --wait_steps 2000   \
    --num_processes 10 \
    --z_dim 32        \
    --repeat_num 3    \
    --use_CA        \
    --euler_threshold 10 \
    --task_threshold 30 \
    --num_ctpt1 10       \
    --num_ctpt2 10       \
    --rv1 100            \
    --rv2 100            \
    --num_pair1 10       \
    --aff_topk 0.1   \
    --critic_topk1 0.01 \
    --critic_topk 0.001 \
    --no_gui






