xvfb-run -a python collect_data_SAC_new.py \
    --category Display \
    --primact_type pickup  \
    --date "XXXX" \
    --out_dir ../data/2gripper_data/collect_data_RL_partialPC \
    --RL_mode test \
    --RL_load_date "default" \
    --RL_exp_name SAC_XXXXX \
    --num_workers 5 \
    --state_add_pose \
    --use_RL_pred \
    --vis_gif  \
    --fix_joint \
    --use_pam  \
    --pam_rate 0.75 \
    --pam_exp_name finalexp-model_all_final-pulling-None-train_all_v1 \
    --pam_model_epoch 81 \
    --pam_model_version model_3d_legacy \


# This is RL testing (used for collecting data) script. Feel free to modify:
#   category, primact_type, collect_mode, date, out_dir: as introduced in the training script
#   RL_load_date: to the identification (date) of previous trained RL model, for loading your trained RL model
#   RL_exp_name: to the suitable epoch you think
# The default model date and epoch are demonstrated in gen_cate_setting.py
# Or you can just assign the argument RL_ckpt to specify the ckpt files, this will overlap RL_load_date and RL_exp_name

