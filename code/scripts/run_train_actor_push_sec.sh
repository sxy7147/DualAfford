CUDA_VISIBLE_DEVICES=1 python train_actor.py \
    --exp_suffix xxx \
    --category_types Box,Dishwasher,Display,Microwave,Printer,Bench,Keyboard2  \
    --model_version model_actor_sec \
    --primact_type pushing \
    --offline_data_dir ../data/2gripper_data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/2gripper_data/PUSH_TEST_DATA \
    --train_buffer_max_num 24000  \
    --val_buffer_max_num 1000  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 500 \
    --lbd_kl 0.02     \
    --lbd_dir 5.0     \
    --z_dim 32   \
    --coordinate_system cambase \
    --exchange_ctpts  \


