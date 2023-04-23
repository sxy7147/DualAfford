CUDA_VISIBLE_DEVICES=0 python train_critic.py \
    --exp_suffix xxx \
    --category_types Box,Dishwasher,Display,Microwave,Printer,Bench,Keyboard2  \
    --primact_type pushing \
    --model_version model_critic_fir \
    --aff_version model_aff_sec   \
    --aff_path  ../logs/affordance/exp-AFFORDANCE2_MODEL  \
    --aff_eval_epoch 40 \
    --actor_version model_actor_sec   \
    --actor_path  ../logs/actor/exp-ACTOR2_MODEL  \
    --actor_eval_epoch 80 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-CRITIC2_MODEL  \
    --critic_eval_epoch 20 \
    --offline_data_dir ../data/PUSH_TRAIN_DATA  \
    --val_data_dir ../data/PUSH_TEST_DATA \
    --train_buffer_max_num 24000  \
    --val_buffer_max_num 1000  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 500 \
    --topk2 1000      \
    --succ_proportion 0.4 \
    --fail_proportion 0.8 \
    --coordinate_system cambase \
    --loss_type L1Loss    \
    --exchange_ctpts  \













