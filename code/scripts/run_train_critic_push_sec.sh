CUDA_VISIBLE_DEVICES=0 python train_critic.py \
    --exp_suffix xxx \
    --category_types Box,Dishwasher,Display,Microwave,Printer,Bench,Keyboard2  \
    --model_version model_critic_sec  \
    --primact_type pushing \
    --offline_data_dir ../data/2gripper_data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/2gripper_data/PUSH_TEST_DATA \
    --train_buffer_max_num 24000  \
    --val_buffer_max_num 1000  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 500 \
    --succ_proportion 0.4 \
    --fail_proportion 0.8 \
    --coordinate_system cambase \
    --exchange_ctpts  \















