import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from subprocess import call
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from tensorboardX import SummaryWriter
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
ctx = torch.multiprocessing.get_context("spawn")


def run_jobs(idx_process, conf, transition_Q, data):
    f = open(os.devnull, 'w')

    position1, position2, dir1, dir2, mat44, cam2cambase, pc_center, file_dir, file_id = data
    up1, forward1 = dir1[0: 3], dir1[3: 6]
    up2, forward2 = dir2[0: 3], dir2[3: 6]

    if conf.coordinate_system == 'world':
        position_world1, up_world1, forward_world1, position_world2, up_world2, forward_world2 = \
            position1.reshape(3), up1.reshape(3), forward1.reshape(3), position2.reshape(3), up2.reshape(3), forward2.reshape(3)
    elif conf.coordinate_system == 'cambase':
        cambase_batch = [position1.reshape(3), up1.reshape(3), forward1.reshape(3),
                         position2.reshape(3), up2.reshape(3), forward2.reshape(3)]
        is_pc = [True, False, False, True, False, False]
        camera_batch = utils.batch_coordinate_transform(cambase_batch, is_pc, transform_type='cambase2cam', cam2cambase=cam2cambase, pc_center=pc_center)
        world_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2world', mat44=mat44)
        position_world1, up_world1, forward_world1, position_world2, up_world2, forward_world2 = world_batch

    repeat_i = 0

    str_position_world1 = 'str' + ','.join([str(i) for i in position_world1.tolist()])
    str_up_world1 = 'str' + ','.join([str(i) for i in up_world1.tolist()])
    str_forward_world1 = 'str' + ','.join([str(i) for i in forward_world1.tolist()])
    str_position_world2 = 'str' + ','.join([str(i) for i in position_world2.tolist()])
    str_up_world2 = 'str' + ','.join([str(i) for i in up_world2.tolist()])
    str_forward_world2 = 'str' + ','.join([str(i) for i in forward_world2.tolist()])
    cmd = 'python eval_sampleSucc.py --file_dir %s --file_id %d --repeat_id %d ' \
          '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f --move_steps %d --wait_steps %d ' \
          '--position_world1 %s --up_world1 %s --forward_world1 %s --position_world2 %s --up_world2 %s --forward_world2 %s ' \
          '--task_threshold %d --euler_threshold %d --save_results ' \
          '--out_dir %s --no_gui ' \
          % (file_dir, file_id, repeat_i,
             conf.density, conf.damping, conf.target_part_state, conf.start_dist, conf.final_dist, conf.move_steps, conf.wait_steps,
             str_position_world1, str_up_world1, str_forward_world1, str_position_world2, str_up_world2, str_forward_world2,
             conf.task_threshold, conf.euler_threshold,
             os.path.join(conf.exp_dir, conf.out_folder))
    if conf.not_check_contactError:
        pass
    else:
        cmd += '--check_contactError '
    cmd += '> /dev/null 2>&1'
    # cmd += '> /dev/null'

    ret = call(cmd, shell=True)
    gt_result = 1 if ret == 0 else 0
    transition_Q.put([idx_process, gt_result])

    f.close()



def train(conf, train_data_list, val_data_list, flog):
    # create training and validation datasets and data loaders

    data_features = ['part_pc', 'task', 'shape_id', 'mat44', 'cam2cambase', 'pc_centers', 'cur_dir', 'result_idx']

    if conf.primact_type in ['pushing', 'topple', 'pickup']:
        task_input_dim = 3
    elif conf.primact_type in ['rotating']:
        task_input_dim = 1

    # load network model
    aff1_def = utils.get_model_module(conf.aff1_version)
    affordance1 = aff1_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    affordance1.load_state_dict(torch.load(os.path.join(conf.aff1_path, 'ckpts', '%s.pth' % conf.aff1_eval_epoch)))
    affordance1.to(device).eval()

    actor1_def = utils.get_model_module(conf.actor1_version)
    actor1 = actor1_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim, task_input_dim=task_input_dim)
    actor1.load_state_dict(torch.load(os.path.join(conf.actor1_path, 'ckpts', '%s.pth' % conf.actor1_eval_epoch)))
    actor1.to(device).eval()

    critic1_def = utils.get_model_module(conf.critic1_version)
    critic1 = critic1_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    critic1.load_state_dict(torch.load(os.path.join(conf.critic1_path, 'ckpts', '%s.pth' % conf.critic1_eval_epoch)))
    critic1.to(device).eval()

    aff2_def = utils.get_model_module(conf.aff2_version)
    affordance2 = aff2_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    affordance2.load_state_dict(torch.load(os.path.join(conf.aff2_path, 'ckpts', '%s.pth' % conf.aff2_eval_epoch)))
    affordance2.to(device).eval()

    actor2_def = utils.get_model_module(conf.actor2_version)
    actor2 = actor2_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim, task_input_dim=task_input_dim)
    actor2.load_state_dict(torch.load(os.path.join(conf.actor2_path, 'ckpts', '%s.pth' % conf.actor2_eval_epoch)))
    actor2.to(device).eval()

    critic2_def = utils.get_model_module(conf.critic2_version)
    critic2 = critic2_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    critic2.load_state_dict(torch.load(os.path.join(conf.critic2_path, 'ckpts', '%s.pth' % conf.critic2_eval_epoch)))
    critic2.to(device).eval()

    affordance1_opt = torch.optim.Adam(affordance1.parameters(), lr=conf.aff_lr, weight_decay=conf.weight_decay)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=conf.critic1_lr, weight_decay=conf.weight_decay)
    affordance_opt = torch.optim.Adam(affordance2.parameters(), lr=conf.aff_lr, weight_decay=conf.weight_decay)
    critic_opt = torch.optim.Adam(critic2.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    utils.printout(flog, '\n' + str(affordance1) + '\n' + str(actor1) + '\n' + str(critic1) + '\n' + str(affordance2) + '\n' + str(actor2) + '\n' + str(critic2) + '\n')

    affordance1_lr_scheduler = torch.optim.lr_scheduler.StepLR(affordance1_opt, step_size=conf.aff_lr_decay_every, gamma=conf.lr_decay_by)
    critic1_lr_scheduler = torch.optim.lr_scheduler.StepLR(critic1_opt, step_size=conf.critic1_lr_decay_every, gamma=conf.lr_decay_by)
    affordance2_lr_scheduler = torch.optim.lr_scheduler.StepLR(affordance_opt, step_size=conf.aff_lr_decay_every, gamma=conf.lr_decay_by)
    critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(critic_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    if conf.continue_to_play:
        affordance1.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance1.pth' % conf.saved_epoch)))
        affordance1_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance1_optimizer.pth' % conf.saved_epoch)))
        affordance1_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance1_lr_scheduler.pth' % conf.saved_epoch)))
        affordance2.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance2.pth' % conf.saved_epoch)))
        affordance_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance2_optimizer.pth' % conf.saved_epoch)))
        affordance2_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-affordance2_lr_scheduler.pth' % conf.saved_epoch)))

        actor1.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-actor1.pth' % conf.saved_epoch)))
        actor2.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-actor2.pth' % conf.saved_epoch)))

        critic1.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic1.pth' % conf.saved_epoch)))
        critic1_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic1_optimizer.pth' % conf.saved_epoch)))
        critic1_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic1_lr_scheduler.pth' % conf.saved_epoch)))
        critic2.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic2.pth' % conf.saved_epoch)))
        critic_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic2_optimizer.pth' % conf.saved_epoch)))
        critic_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%s-critic2_lr_scheduler.pth' % conf.saved_epoch)))

    utils.optimizer_to_device(affordance1_opt, conf.device)
    utils.optimizer_to_device(critic1_opt, conf.device)
    utils.optimizer_to_device(affordance_opt, conf.device)
    utils.optimizer_to_device(critic_opt, conf.device)




    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR   SampleSucc  loss1   loss2   aff_loss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'val'))


    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.train_buffer_max_num, \
                                        img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, only_true_data=True,
                                        coordinate_system=conf.coordinate_system,
                                        cat2freq=conf.cat2freq)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.val_buffer_max_num, \
                                      img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, only_true_data=True,
                                      coordinate_system=conf.coordinate_system,
                                      cat2freq=conf.val_cat2freq)
    ### load data for the current epoch
    print("len of train data list", len(train_data_list))
    train_dataset.load_data(train_data_list)
    utils.printout(flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=0, drop_last=True,
                                                   collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)

    val_dataset.load_data(val_data_list)
    utils.printout(flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True,
                                                 pin_memory=True, num_workers=0, drop_last=True,
                                                 collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)
    print('train_num_batch: %d, val_num_batch: %d' % (train_num_batch, val_num_batch))

    last_train_console_log_step, last_val_console_log_step = None, None

    # start training
    start_time = time.time()
    start_epoch = 0
    if conf.continue_to_play:
        start_epoch = int(conf.saved_epoch.split('-')[0])

    accumulation_steps, accumulation_data = 0, 0


    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        ### print log
        if not conf.no_console_log:
            utils.printout(flog, f'training run {conf.exp_name}')
            utils.printout(flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        val_fraction_done = 0.0
        val_batch_ind = -1


        ep_loss, ep_cnt = 0, 0
        train_ep_loss, train_cnt = 0, 0
        num_succ, num_fail, num_select = 0, 0, 0
        val_succ, val_fail = 0, 0
        num_update = 0


        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                                                       train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # save checkpoint
            if epoch % 1 == 0 and train_batch_ind % 50 == 0:

                if conf.continue_to_play and int(conf.saved_epoch.split('-')[0]) == epoch and int(conf.saved_epoch.split('-')[1]) == train_batch_ind:
                    pass
                elif os.path.exists(os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance1.pth' % (epoch, train_batch_ind))):
                    pass
                else:
                    with torch.no_grad():
                        utils.printout(flog, 'Saving checkpoint ...... ')
                        torch.save(affordance1.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance1.pth' % (epoch, train_batch_ind)))
                        torch.save(actor1.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-actor1.pth' % (epoch, train_batch_ind)))
                        torch.save(critic1.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic1.pth' % (epoch, train_batch_ind)))
                        torch.save(affordance2.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance2.pth' % (epoch, train_batch_ind)))
                        torch.save(actor2.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-actor2.pth' % (epoch, train_batch_ind)))
                        torch.save(critic2.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic2.pth' % (epoch, train_batch_ind)))

                        torch.save(affordance1_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance1_optimizer.pth' % (epoch, train_batch_ind)))
                        torch.save(critic1_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic1_optimizer.pth' % (epoch, train_batch_ind)))
                        torch.save(affordance_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance2_optimizer.pth' % (epoch, train_batch_ind)))
                        torch.save(critic_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic2_optimizer.pth' % (epoch, train_batch_ind)))
                        torch.save(affordance1_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance1_lr_scheduler.pth' % (epoch, train_batch_ind)))
                        torch.save(critic1_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic1_lr_scheduler.pth' % (epoch, train_batch_ind)))
                        torch.save(affordance2_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-affordance2_lr_scheduler.pth' % (epoch, train_batch_ind)))
                        torch.save(critic_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-%d-critic2_lr_scheduler.pth' % (epoch, train_batch_ind)))

                        utils.printout(flog, 'DONE')

            # set models to training mode
            affordance1.train()
            actor1.train()
            critic1.train()
            affordance2.train()
            actor2.train()
            critic2.train()
            network = [affordance1, critic1, actor1, affordance2, critic2, actor2]


            total_loss, losses, num_succ, num_fail, num_select, actor_losses\
                = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False,
                          step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch,
                          start_time=start_time, log_console=log_console, log_tb=not conf.no_tb_log,
                          tb_writer=train_writer, lr=critic_opt.param_groups[0]['lr'], aff_lr=affordance_opt.param_groups[0]['lr'], flog=flog,
                          num_succ=num_succ, num_fail=num_fail, num_select=num_select, num_update=num_update)

            train_ep_loss += total_loss
            train_cnt += 1


            aff1_loss, aff2_loss, critic1_loss = losses
            affordance1_opt.zero_grad()
            aff1_loss.backward()
            affordance1_opt.step()
            affordance1_lr_scheduler.step()

            affordance_opt.zero_grad()
            aff2_loss.backward()
            affordance_opt.step()
            affordance2_lr_scheduler.step()

            critic1_opt.zero_grad()
            critic1_loss.backward()
            critic1_opt.step()
            critic1_lr_scheduler.step()


            ######## need to revise ############
            cur_loss = total_loss / conf.batch_size
            cur_loss.backward()
            accumulation_steps += 1
            if num_select >= (num_update + 1) * conf.batch_size:
                critic_opt.step()
                critic_lr_scheduler.step()
                critic_opt.zero_grad()
                accumulation_steps = 0
                num_update += 1
                torch.cuda.empty_cache()

            print('num_succ: %d, num_fail: %d, num_select: %d, accumulation_data: %d, accumulation_steps: %d, num_update: %d' % (num_succ, num_fail, num_select, num_select % conf.batch_size, accumulation_steps, num_update))


            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind + 1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                                                           val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                affordance1.eval()
                actor1.eval()
                critic1.eval()
                affordance2.eval()
                actor2.eval()
                critic2.eval()

                with torch.no_grad():
                    val_loss, losses, val_succ, val_fail, _, actor_losses\
                        = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True,
                                  step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch,
                                  start_time=start_time, log_console=log_console, log_tb=not conf.no_tb_log,
                                  tb_writer=val_writer, lr=critic_opt.param_groups[0]['lr'], aff_lr=affordance_opt.param_groups[0]['lr'], flog=flog,
                                  num_succ=val_succ, num_fail=val_fail, num_select=None, num_update=None)
                ep_loss += val_loss
                ep_cnt += 1

        utils.printout(flog, "epoch: %d, total_train_loss: %f, total_val_loss: %f" %
                       (epoch, train_ep_loss / num_select, ep_loss / ep_cnt))



def forward(batch, data_features, network, conf, \
            is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
            log_console=False, log_tb=False, tb_writer=None, lr=None, aff_lr=None, flog=None,
            num_succ=None, num_fail=None, num_select=None, num_update=None):

    batch_size = conf.batch_size
    pcs = torch.cat(batch[data_features.index('part_pc')], dim=0).float().to(device)
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

    task = torch.tensor(np.array(batch[data_features.index('task')])).float().view(batch_size, -1).to(device)
    mat44 = batch[data_features.index('mat44')]
    cam2cambase = batch[data_features.index('cam2cambase')]
    pc_centers = batch[data_features.index('pc_centers')]

    cur_dir = batch[data_features.index('cur_dir')]
    result_id = batch[data_features.index('result_idx')]


    affordance1, critic1, actor1, affordance2, critic, actor2 = network
    num_ctpt1, num_ctpt2, rv1, rv2 = conf.num_ctpt1, conf.num_ctpt2, conf.rv1, conf.rv2
    num_pair1 = conf.num_pair1

    with torch.no_grad():
        # module1
        aff_scores = affordance1.inference_whole_pc(pcs, task).view(batch_size, conf.num_point_per_shape)      # B * N
        aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, conf.num_point_per_shape)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
        selected_idx_idx = torch.randint(0, int(conf.num_point_per_shape * conf.aff_topk), size=(batch_size, num_ctpt1))
        selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
        position1s = pcs.clone()[batch_idx, selected_idx].view(batch_size * num_ctpt1, 3)

        dir1s = actor1.actor_sample_n_finetune(pcs, task, position1s, rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)

        critic_scores = critic1.forward_n_finetune(pcs, task, position1s, dir1s, rvs_ctpt=num_ctpt1, rvs=rv1).view(batch_size, num_ctpt1 * rv1)
        critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
        selected_idx_idx = torch.randint(0, int(num_ctpt1 * rv1 * conf.critic_topk1), size=(batch_size, num_pair1))
        selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
        position1 = position1s.view(batch_size, num_ctpt1, 3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)
        dir1 = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx, selected_idx].view(batch_size * num_pair1, 6)

        # module2
        aff_scores = affordance2.inference_whole_pc_finetune(pcs, task, position1, dir1).view(batch_size * num_pair1, conf.num_point_per_shape)
        aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size * num_pair1, conf.num_point_per_shape)
        batch_idx = torch.tensor(range(batch_size * num_pair1)).view(batch_size * num_pair1, 1)
        selected_idx_idx = torch.randint(0, int(conf.num_point_per_shape * conf.aff_topk), size=(batch_size * num_pair1, num_ctpt2))
        selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
        expanded_pcs = pcs.repeat(1, num_pair1, 1).reshape(batch_size * num_pair1, conf.num_point_per_shape, 3)
        position2 = expanded_pcs.clone()[batch_idx, selected_idx].view(batch_size * num_pair1 * num_ctpt2, 3)

        expanded_position1 = position1.unsqueeze(dim=1).repeat(1, num_ctpt2, 1).reshape(batch_size * num_pair1 * num_ctpt2, 3)
        expanded_dir1s = dir1.unsqueeze(dim=1).repeat(1, num_ctpt2, 1).reshape(batch_size * num_pair1 * num_ctpt2, 6)
        dir2s = actor2.actor_sample_n_finetune(pcs, task, expanded_position1, position2, expanded_dir1s, rvs_ctpt=num_pair1 * num_ctpt2, rvs=rv2).contiguous().view(batch_size * num_pair1 * num_ctpt2 * rv2, 6)

        expanded_expanded_dir1s = expanded_dir1s.unsqueeze(dim=1).repeat(1, rv2, 1).reshape(batch_size * num_pair1 * num_ctpt2 * rv2, 6)

        critic_scores = critic.forward_n_finetune(pcs, task, expanded_position1, position2, expanded_expanded_dir1s, dir2s, rvs_ctpt=num_pair1 * num_ctpt2, rvs=rv2).view(batch_size, num_pair1 * num_ctpt2 * rv2)
        critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_pair1 * num_ctpt2 * rv2)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
        selected_idx_idx = torch.randint(0, int(num_pair1 * num_ctpt2 * rv2 * conf.critic_topk), size=(batch_size, 1))
        selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
        # pred_scores = critic_scores[batch_idx, selected_idx]

    with torch.no_grad():
        position1 = expanded_position1.reshape(batch_size, num_pair1 * num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
        position2 = position2.reshape(batch_size, num_pair1 * num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
        dir1 = expanded_expanded_dir1s.view(batch_size, num_pair1 * num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)
        dir2 = dir2s.view(batch_size, num_pair1 * num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)

    # update affordance1
    aff_pred_score1 = affordance1.forward(pcs, task, position1)
    with torch.no_grad():
        recon_dir1 = actor1.actor_sample_n(pcs, task, position1, rvs=conf.rv_cnt).contiguous().view(batch_size * conf.rv_cnt, 6)
        aff_gt_scores1 = critic1.forward_n(pcs, task, position1, recon_dir1, rvs=conf.rv_cnt)  # dir1: B*6; dir2: (B*rvs) * 6
        aff_gt_score1 = aff_gt_scores1.view(batch_size, conf.rv_cnt, 1).topk(k=conf.topk, dim=1)[0].mean(dim=1)
    aff1_loss = affordance1.get_loss(aff_pred_score1.view(batch_size, -1), aff_gt_score1.view(batch_size, -1))
    if conf.exchange_ctpts:
        aff_pred_score1_exchange = affordance1.forward(pcs, task, position2)
        with torch.no_grad():
            recon_dir2_exchange = actor1.actor_sample_n(pcs, task, position2, rvs=conf.rv_cnt).contiguous().view(batch_size * conf.rv_cnt, 6)
            aff_gt_scores1_exchange = critic1.forward_n(pcs, task, position2, recon_dir2_exchange, rvs=conf.rv_cnt)  # dir1: B*6; dir2: (B*rvs) * 6
            aff_gt_score1_exchange = aff_gt_scores1_exchange .view(batch_size, conf.rv_cnt, 1).topk(k=conf.topk, dim=1)[0].mean(dim=1)
        aff1_loss_exchange = affordance1.get_loss(aff_pred_score1_exchange.view(batch_size, -1), aff_gt_score1_exchange.view(batch_size, -1))
        aff1_loss = aff1_loss + aff1_loss_exchange

    # update affordance2
    aff_pred_score2 = affordance2.forward(pcs, task, position1, position2, dir1)
    with torch.no_grad():
        recon_dir2 = actor2.actor_sample_n(pcs, task, position1, position2, dir1, rvs=conf.rv_cnt).contiguous().view(batch_size * conf.rv_cnt, 6)
        aff_gt_scores2 = critic.forward_n(pcs, task, position1, position2, dir1, recon_dir2, rvs=conf.rv_cnt)  # dir1: B*6; dir2: (B*rvs) * 6
        aff_gt_score2 = aff_gt_scores2.view(batch_size, conf.rv_cnt, 1).topk(k=conf.topk, dim=1)[0].mean(dim=1)
    aff2_loss = affordance2.get_loss(aff_pred_score2.view(batch_size, -1), aff_gt_score2.view(batch_size, -1))
    if conf.exchange_ctpts:
        aff_pred_score2_exchange = affordance2.forward(pcs, task, position2, position1, dir2)
        with torch.no_grad():
            recon_dir2_exchange = actor2.actor_sample_n(pcs, task, position2, position1, dir2, rvs=conf.rv_cnt).contiguous().view(batch_size * conf.rv_cnt, 6)
            aff_gt_scores2_exchange = critic.forward_n(pcs, task, position2, position1, dir2, recon_dir2_exchange, rvs=conf.rv_cnt)  # dir1: B*6; dir2: (B*rvs) * 6
            aff_gt_score2_exchange = aff_gt_scores2_exchange.view(batch_size, conf.rv_cnt, 1).topk(k=conf.topk, dim=1)[0].mean(dim=1)
        aff2_loss_exchange = affordance2.get_loss(aff_pred_score2_exchange.view(batch_size, -1), aff_gt_score2_exchange.view(batch_size, -1))
        aff2_loss = aff2_loss + aff2_loss_exchange

    # update critic1
    critic_pred_score1 = critic1.forward(pcs, task, position1, dir1)
    with torch.no_grad():
        aff_scores = affordance2.inference_whole_pc(pcs, task, position1, dir1)  # B * N * 1
        selected_ctpts_idx = aff_scores.view(batch_size, conf.num_point_per_shape, 1).topk(k=conf.rvs_ctpt, dim=1)[1]  # B * rvs_ctpt * 1  (idx)
        selected_ctpts_idx = selected_ctpts_idx.view(batch_size * conf.rvs_ctpt, 1)
        pcs_idx = torch.tensor(range(batch_size)).reshape(batch_size, 1).unsqueeze(dim=1).repeat(1, conf.rvs_ctpt, 1).reshape(batch_size * conf.rvs_ctpt, 1)
        selected_ckpts = pcs[pcs_idx, selected_ctpts_idx].reshape(batch_size * conf.rvs_ctpt, 3)

        recon_dir2 = actor2.actor_sample_n_diffCtpts(pcs, task, position1, selected_ckpts, dir1, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs).contiguous().reshape(batch_size * conf.rvs_ctpt * conf.rvs, 6)
        expanded_dir1 = dir1.unsqueeze(dim=1).repeat(1, conf.rvs_ctpt * conf.rvs, 1).reshape(batch_size * conf.rvs_ctpt * conf.rvs, -1)
        gt_scores = critic.forward_n_diffCtpts(pcs, task, position1, selected_ckpts, expanded_dir1, recon_dir2, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs)  # dir1: B*6; dir2: (B*rvs) * 6
        gt_score = gt_scores.view(batch_size, conf.rvs_ctpt * conf.rvs, 1).topk(k=conf.topk2, dim=1)[0].mean(dim=1)
    critic1_loss = critic1.get_L1_loss(critic_pred_score1.view(batch_size, -1), gt_score.view(batch_size, -1)).mean()
    if conf.exchange_ctpts:
        critic_pred_score1_exchange = critic1.forward(pcs, task, position2, dir2)
        with torch.no_grad():
            aff_scores_exchange = affordance2.inference_whole_pc(pcs, task, position2, dir2)  # B * N * 1
            selected_ctpts_idx = aff_scores_exchange.view(batch_size, conf.num_point_per_shape, 1).topk(k=conf.rvs_ctpt, dim=1)[1]  # B * rvs_ctpt * 1  (idx)
            selected_ctpts_idx = selected_ctpts_idx.view(batch_size * conf.rvs_ctpt, 1)
            pcs_idx = torch.tensor(range(batch_size)).reshape(batch_size, 1).unsqueeze(dim=1).repeat(1, conf.rvs_ctpt, 1).reshape(batch_size * conf.rvs_ctpt, 1)
            selected_ckpts = pcs[pcs_idx, selected_ctpts_idx].reshape(batch_size * conf.rvs_ctpt, 3)

            recon_dir1_exchange = actor2.actor_sample_n_diffCtpts(pcs, task, position2, selected_ckpts, dir2, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs).contiguous().reshape(batch_size * conf.rvs_ctpt * conf.rvs, 6)
            expanded_dir2_exchange = dir2.unsqueeze(dim=1).repeat(1, conf.rvs_ctpt * conf.rvs, 1).reshape(batch_size * conf.rvs_ctpt * conf.rvs, -1)
            gt_scores_exchange = critic.forward_n_diffCtpts(pcs, task, position2, selected_ckpts, expanded_dir2_exchange, recon_dir1_exchange, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs)  # dir1: B*6; dir2: (B*rvs) * 6
            gt_score_exchange = gt_scores_exchange.view(batch_size, conf.rvs_ctpt * conf.rvs, 1).topk(k=conf.topk2, dim=1)[0].mean(dim=1)
        critic1_loss_exchange = critic1.get_L1_loss(critic_pred_score1_exchange.view(batch_size, -1), gt_score_exchange.view(batch_size, -1)).mean()
        critic1_loss = critic1_loss + critic1_loss_exchange

    losses = [aff1_loss, aff2_loss, critic1_loss]

    # update actors (optional)

    pred_scores = critic.forward(pcs, task, position1, position2, dir1, dir2)  # after sigmoid
    if conf.exchange_ctpts:
        pred_scores_exchange = critic.forward(pcs, task, position2, position1, dir2, dir1)  # after sigmoid


    # online sampling
    position1 = position1.detach().cpu().numpy()
    position2 = position2.detach().cpu().numpy()
    dir1 = dir1.detach().cpu().numpy()
    dir2 = dir2.detach().cpu().numpy()

    processes = []
    trans_q = ctx.Queue()
    for idx in range(conf.batch_size):
        data = [position1[idx], position2[idx], dir1[idx], dir2[idx],
                mat44[idx], cam2cambase[idx], pc_centers[idx], cur_dir[idx], result_id[idx]]
        p = ctx.Process(target=run_jobs, args=(idx, conf, trans_q, data))
        p.start()
        processes.append(p)

    gt_results = np.zeros([batch_size])
    select_tensor = np.zeros([batch_size])
    for p in processes:
        results = trans_q.get()
        idx, result = results
        print('idx: %d\t result: %d\t pred:' % (idx, result), pred_scores[idx].detach().cpu().numpy())
        gt_results[idx] = result

        if not is_val:
            if result == 1 and num_select < (num_update + 1) * batch_size:
                num_succ += 1
                select_tensor[idx] = 1
                num_select += 1
            elif result == 0 and num_fail < num_succ * 0.5 and num_select < (num_update + 1) * batch_size:
                num_fail += 1
                select_tensor[idx] = 1
                num_select += 1

    for p in processes:
        p.join()


    gt_results = torch.tensor(gt_results).view(batch_size, 1).to(device)
    select_tensor = torch.tensor(select_tensor).view(batch_size, 1).to(device)

    # update critic2
    if conf.loss_type == 'crossEntropy':
        loss1 = critic.get_ce_loss_total(pred_scores, gt_results)
    loss2 = torch.min(torch.abs(pred_scores - torch.zeros(batch_size, 1).to(device)), torch.abs(pred_scores - torch.ones(batch_size, 1).to(device)))
    loss = (loss1 * select_tensor).sum() * conf.loss1_weight + (loss2 * select_tensor).sum() * conf.loss2_weight
    print('select_tensor: ', select_tensor.view(-1))

    if conf.exchange_ctpts:
        if conf.loss_type == 'crossEntropy':
            loss1_exchange = critic.get_ce_loss_total(pred_scores_exchange, gt_results)
        loss2_exchange = torch.min(torch.abs(pred_scores_exchange - torch.zeros(batch_size, 1).to(device)), torch.abs(pred_scores_exchange - torch.ones(batch_size, 1).to(device)))
        loss_exchange = (loss1_exchange * select_tensor).sum() * conf.loss1_weight + (loss2_exchange * select_tensor).sum() * conf.loss2_weight
        loss = loss + loss_exchange


    # update actors (optional)
    actor_losses = []

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{aff_lr:>5.2E} '''
                           f'''{gt_results.sum().item() / batch_size:>10.5f}'''
                           f'''{loss1.mean().item():>10.5f}'''
                           f'''{loss2.mean().item():>10.5f}'''
                           )
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('ce_loss', loss1.mean().item(), step)
            tb_writer.add_scalar('loss2', loss2.mean().item(), step)
            tb_writer.add_scalar('lr', lr, step)
            tb_writer.add_scalar('aff_lr', aff_lr, step)
            tb_writer.add_scalar('accu', gt_results.sum().item() / batch_size, step)
            tb_writer.add_scalar('num_succ', num_succ, step)
            tb_writer.add_scalar('num_fail', num_fail, step)
            tb_writer.add_scalar('aff1_loss', aff1_loss.item() / batch_size, step)
            tb_writer.add_scalar('aff2_loss', aff2_loss.item() / batch_size, step)
            tb_writer.add_scalar('critic1_loss', critic1_loss.item() / batch_size, step)

    return loss, losses, num_succ, num_fail, num_select, actor_losses


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--cat2freq', type=str, default=None)
    parser.add_argument('--val_cat2freq', type=str, default=None)
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--offline_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir6', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
    parser.add_argument('--out_folder', type=str, default='xxx')

    # train networks
    parser.add_argument('--aff1_version', type=str, default=None)
    parser.add_argument('--aff1_path', type=str, default=None)
    parser.add_argument('--aff1_eval_epoch', type=str, default=None)
    parser.add_argument('--actor1_version', type=str, default=None, help='model def file')
    parser.add_argument('--actor1_path', type=str)
    parser.add_argument('--actor1_eval_epoch', type=str)
    parser.add_argument('--critic1_version', type=str, default=None, help='model def file')
    parser.add_argument('--critic1_path', type=str)
    parser.add_argument('--critic1_eval_epoch', type=str)

    parser.add_argument('--aff2_version', type=str, default=None)
    parser.add_argument('--aff2_path', type=str, default=None)
    parser.add_argument('--aff2_eval_epoch', type=str, default=None)
    parser.add_argument('--actor2_version', type=str, default=None, help='model def file')
    parser.add_argument('--actor2_path', type=str)
    parser.add_argument('--actor2_eval_epoch', type=str)
    parser.add_argument('--critic2_version', type=str, default=None, help='model def file')
    parser.add_argument('--critic2_path', type=str)
    parser.add_argument('--critic2_eval_epoch', type=str)

    parser.add_argument('--aff_topk', type=float, default=0.1)
    parser.add_argument('--critic_topk', type=float, default=0.01)
    parser.add_argument('--critic_topk1', type=float, default=0.01)

    parser.add_argument('--density', type=float, default=2.0)
    parser.add_argument('--damping', type=int, default=10)
    parser.add_argument('--target_part_state', type=str, default='random-middle')
    parser.add_argument('--start_dist', type=float, default=0.30)
    parser.add_argument('--final_dist', type=float, default=0.10)
    parser.add_argument('--move_steps', type=int, default=2000)
    parser.add_argument('--wait_steps', type=int, default=2000)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../2gripper_logs/finetune', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')
    parser.add_argument('--continue_to_play', action='store_true', default=False)
    parser.add_argument('--saved_epoch', type=str, default=0)
    parser.add_argument('--not_check_contactError', action='store_true', default=False)

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=8192)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--task_feat_dim', type=int, default=32)
    parser.add_argument('--cp_feat_dim', type=int, default=32)
    parser.add_argument('--dir_feat_dim', type=int, default=32)
    parser.add_argument('--rv_cnt', type=int, default=100)
    parser.add_argument('--rvs_ctpt', type=int, default=10)
    parser.add_argument('--rvs', type=int, default=100)
    parser.add_argument('--num_ctpt1', type=int, default=10)
    parser.add_argument('--num_ctpt2', type=int, default=10)
    parser.add_argument('--num_pair1', type=int, default=10)
    parser.add_argument('--rv1', type=int, default=10)
    parser.add_argument('--rv2', type=int, default=10)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--topk2', type=int, default=1000)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--coordinate_system', type=str, default='world')
    parser.add_argument('--exchange_ctpts', action='store_true', default=False)
    parser.add_argument('--task_threshold', type=int, default=15)
    parser.add_argument('--euler_threshold', type=int, default=5)

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_buffer_max_num', type=int, default=20000)
    parser.add_argument('--val_buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--aff_lr', type=float, default=0.001)
    parser.add_argument('--critic1_lr', type=float, default=0.001)
    parser.add_argument('--actor_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--aff_lr_decay_every', type=float, default=5000)
    parser.add_argument('--critic1_lr_decay_every', type=float, default=5000)
    parser.add_argument('--actor_lr_decay_every', type=float, default=5000)
    parser.add_argument('--critic_score_threshold', type=float, default=0.5)

    # loss weights
    parser.add_argument('--loss_type', type=str, default='crossEntropy')
    parser.add_argument('--loss1_weight', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss2_weight', type=float, default=0.0, help='loss weight')

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=2, help='number of optimization steps beween console log prints')


    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-finetune-{conf.setting}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))

    train_data_list = []
    offline_data_dir_list = [conf.offline_data_dir, conf.offline_data_dir2, conf.offline_data_dir3,
                             conf.offline_data_dir4, conf.offline_data_dir5, conf.offline_data_dir6]
    for data_dir in offline_data_dir_list:
        train_data_list.extend(utils.append_data_list(data_dir, only_true_data=True, primact_type=conf.primact_type))
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))
    print(train_data_list)

    val_data_list = []
    val_data_dir_list = [conf.val_data_dir, conf.val_data_dir2, conf.val_data_dir3,
                         conf.val_data_dir4, conf.val_data_dir5, conf.val_data_dir6]
    for data_dir in val_data_dir_list:
        val_data_list.extend(utils.append_data_list(data_dir, only_true_data=True, primact_type=conf.primact_type))
    utils.printout(flog, 'len(val_data_list): %d' % len(val_data_list))


    ### start training
    train(conf, train_data_list, val_data_list, flog)

    ### before quit
    flog.close()
