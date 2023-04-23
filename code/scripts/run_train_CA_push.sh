CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train_collaborative_adaptation.py \
    --exp_suffix xxx \
    --category_types Box,Bucket,Dishwasher,Display,Microwave,Bench,Bowl,Keyboard2  \
    --primact_type pushing \
    --aff1_version model_aff_fir   \
    --aff1_path  ../logs/affordance/exp-AFF1   \
    --aff1_eval_epoch 40 \
    --actor1_version model_actor_fir   \
    --actor1_path  ../logs/actor/exp-ACTOR1  \
    --actor1_eval_epoch 80 \
    --critic1_version model_critic_fir   \
    --critic1_path  ../logs/critic/exp-CRITIC1  \
    --critic1_eval_epoch 40 \
    --aff2_version model_aff_sec   \
    --aff2_path  ../logs/affordance/exp-AFF2  \
    --aff2_eval_epoch 40 \
    --actor2_version model_actor_sec   \
    --actor2_path  ../logs/actor/exp-ACTOR2  \
    --actor2_eval_epoch 80 \
    --critic2_version model_critic_sec   \
    --critic2_path  ../2logs/critic/exp-CRITIC2  \
    --critic2_eval_epoch 20 \
    --out_folder collaborative_adaptation \
    --offline_data_dir ../data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/PUSH_TEST_DATA \
    --train_buffer_max_num 12000  \
    --val_buffer_max_num 500  \
    --feat_dim 128   \
    --batch_size 32 \
    --lr 0.001      \
    --lr_decay_every 50 \
    --critic1_lr 0.0001 \
    --critic1_lr_decay_every 300 \
    --aff_lr 0.0001     \
    --aff_lr_decay_every 300 \
    --loss1_weight 1.0  \
    --loss2_weight 0.0 \
    --num_ctpt1 10  \
    --num_ctpt2 10  \
    --num_pair1 10  \
    --rv1 100        \
    --rv2 100        \
    --z_dim 32      \
    --target_part_state closed  \
    --start_dist 0.45   \
    --final_dist 0.10   \
    --move_steps 3500   \
    --wait_steps 2000   \
    --coordinate_system cambase \
    --euler_threshold 3 \
    --task_threshold 30 \
    --aff_topk 0.1   \
    --critic_topk 0.001 \
    --critic_topk1 0.01 \





