CUDA_VISIBLE_DEVICES=0 python train_affordance.py \
    --exp_suffix xxx \
    --category_types Box,Dishwasher,Display,Microwave,Printer,Bench,Keyboard2  \
    --primact_type pushing \
    --model_version model_aff_fir \
    --actor_version model_actor_fir   \
    --actor_path  ../2gripper_logs/actor/exp-ACTOR2_MODEL  \
    --actor_eval_epoch 80 \
    --critic_version model_critic_fir   \
    --critic_path  ../2gripper_logs/critic/exp-CRITIC2_MODEL   \
    --critic_eval_epoch 40 \
    --offline_data_dir ../data/2gripper_data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/2gripper_data/PUSH_TEST_DATA \
    --train_buffer_max_num 24000  \
    --val_buffer_max_num 1000  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 500 \
    --z_dim 32      \
    --topk  100       \
    --coordinate_system cambase \
    --exchange_ctpts  \


