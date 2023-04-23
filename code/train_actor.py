"""
    Train the full model
"""

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
from PIL import Image
from subprocess import call
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def train(conf, train_data_list, val_data_list):
    # create training and validation datasets and data loaders

    data_features = ['part_pc', 'task', 'ctpt1', 'ctpt2', 'dir1', 'dir2', 'success', 'shape_id']

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    if conf.primact_type in ['pushing', 'topple', 'pickup']:
        task_input_dim = 3
    elif conf.primact_type in ['rotating']:
        task_input_dim = 1

    # create models
    network = model_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim,
                                z_dim=conf.z_dim, lbd_kl=conf.lbd_kl, lbd_dir=conf.lbd_dir, task_input_dim=task_input_dim)
    utils.printout(conf.flog, '\n' + str(network) + '\n')


    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    # send parameters to device

    if conf.continue_to_play:
        network.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % conf.saved_epoch)))
        network_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % conf.saved_epoch)))
        network_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % conf.saved_epoch)))

    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration     LR    TotalLoss  KLLoss   DirLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'val'))


    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.train_buffer_max_num, \
                                        img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, only_true_data=True,
                                        coordinate_system=conf.coordinate_system, exchange_ctpts=conf.exchange_ctpts,
                                        cat2freq=conf.cat2freq)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.val_buffer_max_num, \
                                      img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, only_true_data=True,
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
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
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
    for epoch in range(start_epoch, conf.epochs):
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
            if epoch % 5 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            total_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                                 step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                                 log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

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
                    val_loss = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                                 step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                                 log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer,
                                 lr=network_opt.param_groups[0]['lr'])
                ep_loss += val_loss
                ep_cnt += 1

        utils.printout(flog, "epoch: %d, total_train_loss: %f, total_val_loss: %f" % (epoch, train_ep_loss / train_cnt, ep_loss / ep_cnt))



def forward(batch, data_features, network, conf, \
            is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
            log_console=False, log_tb=False, tb_writer=None, lr=None):
    torch.cuda.empty_cache()

    batch_size = conf.batch_size
    pcs = torch.cat(batch[data_features.index('part_pc')], dim=0).float().to(device)
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

    task = torch.tensor(np.array(batch[data_features.index('task')])).float().view(batch_size, -1).to(device)
    ctpt1 = torch.tensor(np.array(batch[data_features.index('ctpt1')])).float().view(batch_size, -1).to(device)
    ctpt2 = torch.tensor(np.array(batch[data_features.index('ctpt2')])).float().view(batch_size, -1).to(device)
    dir1 = torch.tensor(np.array(batch[data_features.index('dir1')])).float().view(batch_size, -1).to(device)
    dir2 = torch.tensor(np.array(batch[data_features.index('dir2')])).float().view(batch_size, -1).to(device)

    kl_loss, dir_loss = np.array(0), np.array(0)

    if conf.model_version == 'model_actor_sec':
        losses = network.get_loss(pcs, task, ctpt1, ctpt2, dir1, dir2)
        kl_loss = losses['kl']
        dir_loss = losses['dir']
        total_loss = losses['tot']

    elif conf.model_version == 'model_actor_fir':
        losses = network.get_loss(pcs, task, ctpt1, dir1)
        kl_loss = losses['kl']
        dir_loss = losses['dir']
        total_loss = losses['tot']


    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{lr:>5.2E} '''
                           # f'''{actor_loss.item():>10.5f}'''
                           f'''{total_loss.item():>10.5f}'''
                           f'''{kl_loss.item():>10.5f}'''
                           f'''{dir_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('kl_loss', kl_loss.item(), step)
            tb_writer.add_scalar('dir_loss', dir_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)


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
    parser.add_argument('--offline_data_dir7', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs/actor', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')
    parser.add_argument('--continue_to_play', action='store_true', default=False)
    parser.add_argument('--saved_epoch', type=int, default=0)

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=8192)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--task_feat_dim', type=int, default=32)
    parser.add_argument('--cp_feat_dim', type=int, default=32)
    parser.add_argument('--dir_feat_dim', type=int, default=32)
    parser.add_argument('--rv_dim', type=int, default=10)
    parser.add_argument('--rv_cnt', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--coordinate_system', type=str, default='world')
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
    parser.add_argument('--loss_weight_critic', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_actor_coverage', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_action_score', type=float, default=100.0, help='loss weight')

    # CAVE
    parser.add_argument('--lbd_kl', type=float, default=1.0)
    parser.add_argument('--lbd_dir', type=float, default=1.0)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')


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
    train(conf, train_data_list, val_data_list)

    ### before quit
    # close file log
    flog.close()
