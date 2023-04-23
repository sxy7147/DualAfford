xvfb-run -a python collect_data_SAC.py \
    --category Display \
    --primact_type pickup \
    --date "default" \
    --out_dir ../data/2gripper_data/collect_data_RL_partialPC \
    --batch_size 512  \
    --num_workers 12 \
    --soft_q_lr 2e-4  \
    --policy_lr 2e-4  \
    --alpha_lr  2e-4  \
    --update_itr 1 \
    --state_add_pose \
    --vis_gif  \
    --use_pam  \
    --use_RL_pred \
    --fix_joint \
    --pam_exp_name finalexp-model_all_final-pulling-None-train_all_v1 \
    --pam_model_epoch 81 \
    --pam_model_version model_3d_legacy

# This is RL training script. Feel free to modify:
#   category: to other existing categories in the dataset
#   primact_type: to other primact types (normally only used in pickup)
#   collect_mode: to other train/test/val splits, notice that each category are only for either train/val or test
#   date: to identify your experiment (just leave it as default)
#   out_dir: to other saving paths
