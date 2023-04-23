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
import torch.nn.functional as F
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
from tensorboardX import SummaryWriter


def train(conf, train_data_list, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['part_pc', 'task', 'ctpt1', 'ctpt2', 'dir1', 'dir2', 'success', 'shape_id', 'result_idx']

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    if conf.primact_type in ['pushing', 'topple', 'pickup']:
        task_input_dim = 3
    elif conf.primact_type in ['rotating']:
        task_input_dim = 1

    # create models
    network = model_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    if conf.continue_to_play:
        network.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % conf.saved_epoch)))
        network_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % conf.saved_epoch)))
        network_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % conf.saved_epoch)))

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    affordance, actor, critic = None, None, None
    # load aff2 + actor2 + critic2
    if conf.model_version == 'model_critic_fir':
        aff_def = utils.get_model_module(conf.aff_version)
        affordance = aff_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
        affordance.load_state_dict(torch.load(os.path.join(conf.aff_path, 'ckpts', '%s-network.pth' % conf.aff_eval_epoch)))
        affordance.to(conf.device)

        actor_def = utils.get_model_module(conf.actor_version)
        actor = actor_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim, task_input_dim=task_input_dim)
        actor.load_state_dict(torch.load(os.path.join(conf.actor_path, 'ckpts', '%s.pth' % conf.actor_eval_epoch)))
        actor.to(conf.device)

        critic_def = utils.get_model_module(conf.critic_version)
        critic = critic_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
        critic.load_state_dict(torch.load(os.path.join(conf.critic_path, 'ckpts', '%s-network.pth' % conf.critic_eval_epoch)))
        critic.to(conf.device)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        train_writer = SummaryWriter(os.path.join(conf.tb_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.tb_dir, 'val'))


    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.train_buffer_max_num, \
                                        img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal,
                                        succ_proportion=conf.succ_proportion, fail_proportion=conf.fail_proportion,
                                        coordinate_system=conf.coordinate_system, exchange_ctpts=conf.exchange_ctpts,
                                        cat2freq=conf.cat2freq)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.val_buffer_max_num, \
                                      img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal,
                                      succ_proportion=conf.succ_proportion, fail_proportion=conf.fail_proportion,
                                      coordinate_system=conf.coordinate_system, exchange_ctpts=conf.exchange_ctpts,
                                      cat2freq=conf.val_cat2freq)

    ### load data for the current epoch
    print("len of train data list", len(train_data_list))
    train_dataset.load_data(train_data_list)  # 每个实验路径的list
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=0, drop_last=True,
                                                   collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)

    val_dataset.load_data(val_data_list)
    utils.printout(conf.flog, str(val_dataset))
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
        start_epoch = conf.saved_epoch


    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):   # 每个epoch重新获得一次train dataset
        utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Waiting epoch-{epoch} data ]')

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        val_fraction_done = 0.0
        val_batch_ind = -1

        ep_loss, ep_cnt = 0, 0
        train_ep_loss, train_cnt = 0, 0

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # save checkpoint
            if epoch % 2 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            total_loss = critic_forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                                        step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                                        log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'],
                                        affordance=affordance, actor=actor, critic=critic)

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            train_ep_loss += total_loss
            train_cnt += 1

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
                network.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    loss = critic_forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                                           step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                                           log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'],
                                           affordance=affordance, actor=actor, critic=critic)
                    ep_loss += loss
                    ep_cnt += 1

        utils.printout(flog, "epoch: %d, total_train_loss: %f, total_val_loss: %f" % (epoch, train_ep_loss / train_cnt, ep_loss / ep_cnt))


def critic_forward(batch, data_features, network, conf,
                   is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
                   log_console=False, log_tb=False, tb_writer=None, lr=None,
                   affordance=None, actor=None, critic=None):

    batch_size = conf.batch_size
    pcs = torch.cat(batch[data_features.index('part_pc')], dim=0).float().to(device)
    # pcs = torch.tensor(batch[data_features.index('part_pc')]).float().view(batch_size, -1, 3).to(device)
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

    task = torch.tensor(np.array(batch[data_features.index('task')])).float().view(batch_size, -1).to(device)
    ctpt1 = torch.tensor(np.array(batch[data_features.index('ctpt1')])).float().view(batch_size, -1).to(device)
    ctpt2 = torch.tensor(np.array(batch[data_features.index('ctpt2')])).float().view(batch_size, -1).to(device)
    dir1 = torch.tensor(np.array(batch[data_features.index('dir1')])).float().view(batch_size, -1).to(device)
    dir2 = torch.tensor(np.array(batch[data_features.index('dir2')])).float().view(batch_size, -1).to(device)
    gt_result = torch.tensor(np.array(batch[data_features.index('success')])).float().view(batch_size, -1).to(device)

    # input to the pipeline
    if conf.model_version == 'model_critic_sec':
        pred_score = network.forward(pcs, task, ctpt1, ctpt2, dir1, dir2)   # after sigmoid
        loss = network.get_ce_loss_total(pred_score, gt_result)
        total_loss = loss.mean()

    elif conf.model_version == 'model_critic_fir':
        pred_score = network.forward(pcs, task, ctpt1, dir1)    # B * 1
        with torch.no_grad():
            aff_scores = affordance.inference_whole_pc(pcs, task, ctpt1, dir1)  # B * N * 1
            selected_ckpts_idx = aff_scores.view(batch_size, conf.num_point_per_shape, 1).topk(k=conf.rvs_ctpt, dim=1)[1]   # B * rvs_ctpt * 1  (idx)
            selected_ckpts_idx = selected_ckpts_idx.view(batch_size * conf.rvs_ctpt, 1)                                     # (B * rvs_ctpt) * 1
            pcs_idx = torch.tensor(range(batch_size)).reshape(batch_size, 1).unsqueeze(dim=1).repeat(1, conf.rvs_ctpt, 1).reshape(batch_size * conf.rvs_ctpt, 1)
            selected_ckpts = pcs[pcs_idx, selected_ckpts_idx].reshape(batch_size * conf.rvs_ctpt, 3)                                         # (B * rvs_ctpt) * 3

            recon_dir2 = actor.actor_sample_n_diffCtpts(pcs, task, ctpt1, selected_ckpts, dir1, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs).contiguous().reshape(batch_size * conf.rvs_ctpt * conf.rvs, 6)    # (B * rvs_ctpt * rvs) * 6
            expanded_dir1 = dir1.unsqueeze(dim=1).repeat(1, conf.rvs_ctpt * conf.rvs, 1).reshape(batch_size * conf.rvs_ctpt * conf.rvs, -1)
            gt_scores = critic.forward_n_diffCtpts(pcs, task, ctpt1, selected_ckpts, expanded_dir1, recon_dir2, rvs_ctpt=conf.rvs_ctpt, rvs=conf.rvs)   # dir1: B*6; dir2: (B*rvs) * 6
            gt_score = gt_scores.view(batch_size, conf.rvs_ctpt * conf.rvs, 1).topk(k=conf.topk2, dim=1)[0].mean(dim=1)

        if conf.loss_type == 'crossEntropy':
            loss = network.get_ce_loss_total(pred_score.view(batch_size, -1), gt_score.view(batch_size, -1))
        elif conf.loss_type == 'L1Loss':
            loss = network.get_L1_loss(pred_score.view(batch_size, -1), gt_score.view(batch_size, -1))
        total_loss = loss.mean()



    if is_val:
        pred = pred_score.detach().cpu().numpy() > conf.critic_score_threshold
        Fscore, precision, recall, accu = utils.cal_Fscore(np.array(pred, dtype=np.int32), gt_result.detach().cpu().numpy())

    # display information
    data_split = 'val' if is_val else 'train'
    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('critic_loss', total_loss.item(), step)
            tb_writer.add_scalar('critic_lr', lr, step)
        if is_val and log_tb and tb_writer is not None:
            tb_writer.add_scalar('Fscore', Fscore, step)
            tb_writer.add_scalar('precision', precision, step)
            tb_writer.add_scalar('recall', recall, step)
            tb_writer.add_scalar('accu', accu, step)


    return total_loss


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

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../2gripper_logs/critic', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=8192)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--task_feat_dim', type=int, default=32)
    parser.add_argument('--cp_feat_dim', type=int, default=32)
    parser.add_argument('--dir_feat_dim', type=int, default=32)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--coordinate_system', type=str, default='world')
    parser.add_argument('--loss_type', type=str, default='crossEntropy')
    parser.add_argument('--succ_proportion', type=float, default=0.50)
    parser.add_argument('--fail_proportion', type=float, default=0.75)
    parser.add_argument('--exchange_ctpts', action='store_true', default=False)

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_buffer_max_num', type=int, default=20000)
    parser.add_argument('--val_buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--critic_score_threshold', type=float, default=0.5)
    # loss weights

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # continue to play
    parser.add_argument('--continue_to_play', action='store_true', default=False, help='continue to play')
    parser.add_argument('--saved_critic_dir', type=str, default=None)
    parser.add_argument('--saved_epoch', type=int, default=None)

    # train critic1
    parser.add_argument('--aff_version', type=str, default=None)
    parser.add_argument('--aff_path', type=str, default=None)
    parser.add_argument('--aff_eval_epoch', type=str, default=None)
    parser.add_argument('--actor_version', type=str, default=None)
    parser.add_argument('--actor_path', type=str, default=None)
    parser.add_argument('--actor_eval_epoch', type=str, default=None)
    parser.add_argument('--critic_version', type=str, default=None)
    parser.add_argument('--critic_path', type=str, default=None)
    parser.add_argument('--critic_eval_epoch', type=str, default=None)
    parser.add_argument('--topk2', type=int, default=1000)
    parser.add_argument('--rvs', type=int, default=100)
    parser.add_argument('--rvs_ctpt', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=32)


    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.makedirs(conf.exp_dir)
        os.mkdir(conf.tb_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))


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
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))


    train_data_list = []
    offline_data_dir_list = [conf.offline_data_dir, conf.offline_data_dir2, conf.offline_data_dir3,
                             conf.offline_data_dir4, conf.offline_data_dir5, conf.offline_data_dir6]
    for data_dir in offline_data_dir_list:
        train_data_list.extend(utils.append_data_list(data_dir, primact_type=conf.primact_type))
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))
    print(train_data_list)

    val_data_list = []
    val_data_dir_list = [conf.val_data_dir, conf.val_data_dir2, conf.val_data_dir3,
                         conf.val_data_dir4, conf.val_data_dir5, conf.val_data_dir6]
    for data_dir in val_data_dir_list:
        val_data_list.extend(utils.append_data_list(data_dir, primact_type=conf.primact_type))
    utils.printout(flog, 'len(val_data_list): %d' % len(val_data_list))
    print('val_data_list: ')
    for idx in range(len(val_data_list)):
        print(val_data_list[idx])


    ### start training
    print('train_data_list: ', train_data_list[0])
    train(conf, train_data_list, val_data_list)


    ### before quit
    # close file log
    flog.close()

