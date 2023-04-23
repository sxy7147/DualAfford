CUDA_VISIBLE_DEVICES=1 xvfb-run -a python eval_sampleSucc_main.py \
    --categories Box,Bucket,Dishwasher,Display,Microwave,Bench,Bowl,Keyboard2  \
    --primact_type pushing  \
    --aff1_version model_aff_fir   \
    --aff1_path  ../2gripper_logs/affordance/exp-AFF1   \
    --aff1_eval_epoch 40 \
    --actor1_version model_actor_fir   \
    --actor1_path  ../2gripper_logs/actor/exp-ACTOR1  \
    --actor1_eval_epoch 80 \
    --critic1_version model_critic_fir   \
    --critic1_path  ../2gripper_logs/critic/exp-CRITIC1  \
    --critic1_eval_epoch 40 \
    --aff2_version model_aff_sec   \
    --aff2_path  ../2gripper_logs/affordance/exp-AFF2  \
    --aff2_eval_epoch 40 \
    --actor2_version model_actor_sec   \
    --actor2_path  ../2gripper_logs/actor/exp-ACTOR2  \
    --actor2_eval_epoch 80 \
    --critic2_version model_critic_sec   \
    --critic2_path  ../2gripper_logs/critic/exp-CRITIC2  \
    --critic2_eval_epoch 20 \
    --out_folder sampleSucc_results \
    --val_data_dir ../data/2gripper_data/PUSH_TEST_DATA \
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






