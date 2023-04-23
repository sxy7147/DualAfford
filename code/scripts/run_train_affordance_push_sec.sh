CUDA_VISIBLE_DEVICES=0 python train_affordance.py \
    --exp_suffix xxx \
    --category_types Box,Dishwasher,Display,Microwave,Printer,Bench,Keyboard2  \
    --primact_type pushing \
    --model_version model_aff_sec \
    --actor_version model_actor_sec   \
    --actor_path  ../logs/actor/exp-ACTOR1_MODEL  \
    --actor_eval_epoch 80 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-CRITIC1_MODEL  \
    --critic_eval_epoch 20 \
    --offline_data_dir ../data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/PUSH_TEST_DATA \
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



