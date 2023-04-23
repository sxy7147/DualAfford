import os
import sys
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
import torch
import time
import random
import imageio
from subprocess import call
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp_suffix', type=str, help='exp suffix')
parser.add_argument('--categories', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
parser.add_argument('--cat2freq', type=str, default=None)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_folder', type=str, default='xxx')

parser.add_argument('--coordinate_system', type=str, default='world')
parser.add_argument('--repeat_num', type=int, default=1)

parser.add_argument('--val_data_dir', type=str, help='data directory')
parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
parser.add_argument('--val_buffer_max_num', type=int, default=20000)

parser.add_argument('--rv_cnt', type=int, default=100)
parser.add_argument('--rvs_ctpt', type=int, default=10)
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--task_feat_dim', type=int, default=32)
parser.add_argument('--cp_feat_dim', type=int, default=32)
parser.add_argument('--dir_feat_dim', type=int, default=32)
parser.add_argument('--num_point_per_shape', type=int, default=8192)

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

parser.add_argument('--CA_path', type=str)
parser.add_argument('--CA_eval_epoch', type=str)

parser.add_argument('--aff1_topk', type=float, default=0.1)
parser.add_argument('--aff2_topk', type=float, default=0.1)
parser.add_argument('--aff_topk', type=float, default=0.1)
parser.add_argument('--critic_topk1', type=float, default=0.01)
parser.add_argument('--critic_topk', type=float, default=0.01)
parser.add_argument('--num_ctpt1', type=int, default=10)
parser.add_argument('--num_ctpt2', type=int, default=10)
parser.add_argument('--rv1', type=int, default=100)
parser.add_argument('--rv2', type=int, default=100)
parser.add_argument('--num_pair1', type=int, default=10)
parser.add_argument('--num_ctpts', type=int, default=10)
parser.add_argument('--rvs', type=int, default=100)

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--static_friction', type=float, default=4.0)
parser.add_argument('--dynamic_friction', type=float, default=4.0)

parser.add_argument('--draw_aff_map', action='store_true', default=False)
parser.add_argument('--num_draw', type=int, default=1)

parser.add_argument('--task_threshold', type=int, default=15)
parser.add_argument('--euler_threshold', type=int, default=5)
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--use_CA', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_jobs(idx_process, args, transition_Q, cur_file_list, out_dir):
    device = args.device if torch.cuda.is_available() else "cpu"

    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)

    primact_type = args.primact_type
    if primact_type in ['pushing', 'topple', 'pickup']:
        task_input_dim = 3
    elif primact_type in ['rotating']:
        task_input_dim = 1


    # load models
    aff1_def = utils.get_model_module(args.aff1_version)
    affordance1 = aff1_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor1_def = utils.get_model_module(args.actor1_version)
    actor1 = actor1_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim, task_input_dim=task_input_dim)
    critic1_def = utils.get_model_module(args.critic1_version)
    critic1 = critic1_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    aff2_def = utils.get_model_module(args.aff2_version)
    affordance2 = aff2_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor2_def = utils.get_model_module(args.actor2_version)
    actor2 = actor2_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim, task_input_dim=task_input_dim)
    critic2_def = utils.get_model_module(args.critic2_version)
    critic2 = critic2_def.Network(args.feat_dim, args.task_feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    if not args.use_CA:
        affordance1.load_state_dict(torch.load(os.path.join(args.aff1_path, 'ckpts', '%s.pth' % args.aff1_eval_epoch)))
        actor1.load_state_dict(torch.load(os.path.join(args.actor1_path, 'ckpts', '%s.pth' % args.actor1_eval_epoch)))
        critic1.load_state_dict(torch.load(os.path.join(args.critic1_path, 'ckpts', '%s.pth' % args.critic1_eval_epoch)))
        affordance2.load_state_dict(torch.load(os.path.join(args.aff2_path, 'ckpts', '%s.pth' % args.aff2_eval_epoch)))
        actor2.load_state_dict(torch.load(os.path.join(args.actor2_path, 'ckpts', '%s.pth' % args.actor2_eval_epoch)))
        critic2.load_state_dict(torch.load(os.path.join(args.critic2_path, 'ckpts', '%s.pth' % args.critic2_eval_epoch)))
    else:
        affordance1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance1.pth' % args.CA_eval_epoch)))
        actor1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor1.pth' % args.CA_eval_epoch)))
        critic1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic1.pth' % args.CA_eval_epoch)))
        affordance2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance2.pth' % args.CA_eval_epoch)))
        actor2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor2.pth' % args.CA_eval_epoch)))
        critic2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic2.pth' % args.CA_eval_epoch)))

    affordance1.to(device).eval()
    actor1.to(device).eval()
    critic1.to(device).eval()
    affordance2.to(device).eval()
    actor2.to(device).eval()
    critic2.to(device).eval()



    for file in cur_file_list:
        for repeat_i in range(args.repeat_num):
            torch.cuda.empty_cache()
            batch_size = 1

            file_dir, file_id = file
            with open(os.path.join(file_dir, 'result_%d.json' % file_id), 'r') as fin:
                result_data = json.load(fin)

            with h5py.File(os.path.join(file_dir, 'cam_XYZA_%d.h5' % file_id), 'r') as fin:
                cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                cam_XYZA_pts = fin['pc'][:].astype(np.float32)

            shape_id = int(result_data['shape_id'])
            category = result_data['category']
            camera_metadata = result_data['camera_metadata']
            mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
            cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)
            save_aff_dir = os.path.join(out_dir, file_dir.split('/')[-2], 'affordance_maps')

            pcs, pc_center = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system=args.coordinate_system, mat44=mat44, cam2cambase=cam2cambase)
            pcs = pcs.float().contiguous().to(device)
            input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
            input_pcid2 = furthest_point_sample(pcs, args.num_point_per_shape).long().reshape(-1)  # BN
            pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)

            if primact_type in ['pushing', 'topple', 'pickup']:
                if primact_type == 'topple':
                    traj = np.array(result_data['trajectory'], dtype=np.float32)
                    task_world = traj / np.linalg.norm(traj)
                else:
                    task_world = np.array(result_data['traj_dir'], dtype=np.float32)

                if args.coordinate_system == 'world':
                    task = torch.from_numpy(task_world).float().view(batch_size, -1).to(device)
                elif args.coordinate_system == 'cambase':
                    task_cam = utils.coordinate_transform(task_world, False, transform_type='world2cam', mat44=mat44)
                    task_cambase = utils.coordinate_transform(task_cam, False, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                    task = torch.from_numpy(task_cambase).float().view(batch_size, -1).to(device)

            elif primact_type in ['rotating']:
                task = torch.from_numpy(np.array(result_data['beta'], dtype=np.float32)).view(batch_size, -1).to(device)


            ''' inference '''
            num_ctpt1, num_ctpt2, rv1, rv2 = args.num_ctpt1, args.num_ctpt2, args.rv1, args.rv2
            num_pair1 = args.num_pair1

            with torch.no_grad():
                # print('aff1')
                aff_scores = affordance1.inference_whole_pc(pcs, task).view(batch_size, args.num_point_per_shape)  # B * N
                aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, args.num_point_per_shape)
                batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
                selected_idx_idx = torch.randint(0, int(args.num_point_per_shape * args.aff_topk), size=(batch_size, num_ctpt1))
                selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
                position1s = pcs.clone()[batch_idx, selected_idx].view(batch_size * num_ctpt1, 3)

                # print('actor1')
                dir1s = actor1.actor_sample_n_finetune(pcs, task, position1s, rvs_ctpt=num_ctpt1, rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)

                # print('critic1')
                critic_scores = critic1.forward_n_finetune(pcs, task, position1s, dir1s, rvs_ctpt=num_ctpt1, rvs=rv1).view(batch_size, num_ctpt1 * rv1)
                critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
                batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
                selected_idx_idx = torch.randint(0, int(num_ctpt1 * rv1 * args.critic_topk1), size=(batch_size, num_pair1))
                selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
                position1 = position1s.view(batch_size, num_ctpt1, 3)[batch_idx, selected_idx // rv1].view(batch_size * num_pair1, 3)
                dir1 = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx, selected_idx].view(batch_size * num_pair1, 6)

                # print('aff2')
                aff_scores = affordance2.inference_whole_pc_finetune(pcs, task, position1, dir1, rvs_ctpt=num_pair1).view(batch_size * num_pair1, args.num_point_per_shape)
                aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size * num_pair1, args.num_point_per_shape)
                batch_idx = torch.tensor(range(batch_size * num_pair1)).view(batch_size * num_pair1, 1)
                selected_idx_idx = torch.randint(0, int(args.num_point_per_shape * args.aff_topk), size=(batch_size * num_pair1, num_ctpt2))
                selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]
                expanded_pcs = pcs.repeat(1, num_pair1, 1).reshape(batch_size * num_pair1, args.num_point_per_shape, 3)
                position2 = expanded_pcs.clone()[batch_idx, selected_idx].view(batch_size * num_pair1 * num_ctpt2, 3)

                # print('actor2')
                expanded_position1 = position1.unsqueeze(dim=1).repeat(1, num_ctpt2, 1).reshape(batch_size * num_pair1 * num_ctpt2, 3)
                expanded_dir1s = dir1.unsqueeze(dim=1).repeat(1, num_ctpt2, 1).reshape(batch_size * num_pair1 * num_ctpt2, 6)
                dir2s = actor2.actor_sample_n_finetune(pcs, task, expanded_position1, position2, expanded_dir1s, rvs_ctpt=num_pair1 * num_ctpt2, rvs=rv2).contiguous().view(batch_size * num_pair1 * num_ctpt2 * rv2, 6)

                # print('critic2')
                expanded_expanded_dir1s = expanded_dir1s.unsqueeze(dim=1).repeat(1, rv2, 1).reshape(batch_size * num_pair1 * num_ctpt2 * rv2, 6)
                critic_scores = critic2.forward_n_finetune(pcs, task, expanded_position1, position2, expanded_expanded_dir1s, dir2s, rvs_ctpt=num_pair1 * num_ctpt2, rvs=rv2).view(batch_size, num_pair1 * num_ctpt2 * rv2)
                critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_pair1 * num_ctpt2 * rv2)
                batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
                selected_idx_idx = torch.randint(0, int(num_pair1 * num_ctpt2 * rv2 * args.critic_topk), size=(batch_size, 1))
                selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
                pred_scores = critic_scores[batch_idx, selected_idx]

                position1 = expanded_position1.reshape(batch_size, num_pair1 * num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
                position2 = position2.reshape(batch_size, num_pair1 * num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
                dir1 = expanded_expanded_dir1s.view(batch_size, num_pair1 * num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)
                dir2 = dir2s.view(batch_size, num_pair1 * num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)

                with torch.no_grad():
                    if args.draw_aff_map and repeat_i < args.num_draw:
                        aff_scores = affordance1.inference_whole_pc(pcs, task).detach().cpu().numpy().reshape(-1)
                        aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
                        fn = os.path.join(save_aff_dir, '%d_%s_%s_%s_%s' % (file_id, str(repeat_i), category, shape_id, 'map1'))
                        utils.draw_affordance_map(fn, cam2cambase, mat44,
                                                  pcs[0].detach().cpu().numpy(), aff_scores,
                                                  coordinate_system=args.coordinate_system,
                                                  ctpt1=position1[0].detach().cpu().numpy(),
                                                  type='0')
                        aff_scores = affordance2.inference_whole_pc(pcs, task, position1, dir1).detach().cpu().numpy().reshape(-1)  # B * N * 1
                        aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
                        fn = os.path.join(save_aff_dir, '%d_%s_%s_%s_%s' % (file_id, str(repeat_i), category, shape_id, 'map2'))
                        utils.draw_affordance_map(fn, cam2cambase, mat44,
                                                  pcs[0].detach().cpu().numpy(), aff_scores,
                                                  coordinate_system=args.coordinate_system,
                                                  ctpt1=position1[0].detach().cpu().numpy(),
                                                  ctpt2=position2[0].detach().cpu().numpy(),
                                                  type='2')

                dir1 = dir1.view(6).detach().cpu().numpy()
                dir2 = dir2.view(6).detach().cpu().numpy()


            position1 = position1.view(3).detach().cpu().numpy()
            position2 = position2.view(3).detach().cpu().numpy()
            up1, forward1 = dir1[0: 3], dir1[3: 6]
            up2, forward2 = dir2[0: 3], dir2[3: 6]

            if args.coordinate_system == 'cambase':
                cambase_batch = [position1.reshape(3), up1.reshape(3), forward1.reshape(3),
                                 position2.reshape(3), up2.reshape(3), forward2.reshape(3)]
                is_pc = [True, False, False, True, False, False]
                camera_batch = utils.batch_coordinate_transform(cambase_batch, is_pc, transform_type='cambase2cam', cam2cambase=cam2cambase, pc_center=pc_center)
                world_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2world', mat44=mat44)
                position_world1, up_world1, forward_world1, position_world2, up_world2, forward_world2 = world_batch
            elif args.coordinate_system == 'world':
                position_world1, up_world1, forward_world1, position_world2, up_world2, forward_world2 = position1, up1, forward1, position2, up2, forward2

            str_position_world1 = 'str' + ','.join([str(i) for i in position_world1.tolist()])
            str_up_world1 = 'str' + ','.join([str(i) for i in up_world1.tolist()])
            str_forward_world1 = 'str' + ','.join([str(i) for i in forward_world1.tolist()])
            str_position_world2 = 'str' + ','.join([str(i) for i in position_world2.tolist()])
            str_up_world2 = 'str' + ','.join([str(i) for i in up_world2.tolist()])
            str_forward_world2 = 'str' + ','.join([str(i) for i in forward_world2.tolist()])
            cmd = 'python eval_sampleSucc.py --file_dir %s --file_id %s --repeat_id %d ' \
                  '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f --move_steps %d --wait_steps %d ' \
                  '--position_world1 %s --up_world1 %s --forward_world1 %s --position_world2 %s --up_world2 %s --forward_world2 %s ' \
                  '--static_friction %f --dynamic_friction %f --task_threshold %d --euler_threshold %d '    \
                  '--out_dir %s --save_results --no_gui ' \
                  % (file_dir, str(file_id), repeat_i,
                     args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist, args.move_steps, args.wait_steps,
                     str_position_world1, str_up_world1, str_forward_world1, str_position_world2, str_up_world2, str_forward_world2,
                     args.static_friction, args.dynamic_friction, args.task_threshold, args.euler_threshold,
                     out_dir)
            cmd += '> /dev/null 2>&1'

            ret = call(cmd, shell=True)
            if ret == 0:
                transition_Q.put(['succ', file_id, repeat_i])
            elif ret == 1:
                transition_Q.put(['fail', file_id, repeat_i])
            else:
                transition_Q.put(['invalid', file_id, repeat_i])



def main():
    if args.use_CA:
        out_dir = os.path.join(args.CA_path, args.out_folder)
    else:
        out_dir = os.path.join(args.actor2_path, args.out_folder)
    print('out_dir: ', out_dir)

    dir_list = [args.val_data_dir, args.val_data_dir2, args.val_data_dir3, args.val_data_dir4, args.val_data_dir5, args.val_data_dir6]
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            continue
        cur_child_dir = cur_dir.split('/')[-1]
        if not os.path.exists(os.path.join(out_dir, cur_child_dir)):
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'succ_gif'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'fail_gif'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'invalid_gif'))

            os.makedirs(os.path.join(out_dir, cur_child_dir, 'succ_files'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'fail_files'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'invalid_files'))

            os.makedirs(os.path.join(out_dir, cur_child_dir, 'affordance_maps'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'critic_maps'))
            os.makedirs(os.path.join(out_dir, cur_child_dir, 'BEGIN'))


    # load files
    categories = args.categories.split(',')
    category_cnts = dict()
    for category in categories:
        category_cnts[category] = 0
    cat_list, shape_list, _, _ = utils.get_shape_list(all_categories=args.categories, mode='all', primact_type=args.primact_type)

    freq_dict = dict()
    if args.cat2freq:
        freqs = [int(x) for x in args.cat2freq.split(',')]
        for idx in range(len(freqs)):
            freq_dict[categories[idx]] = freqs[idx]
    else:
        for idx in range(len(categories)):
            freq_dict[categories[idx]] = 1e5

    file_list = []
    total_file = 0


    if args.primact_type in ['pushing', 'rotating', 'topple']:
        dir_name = 'succ_files'
    elif args.primact_type in ['pickup']:
        dir_name = 'dual_succ_files'

    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            continue
        for file in sorted(os.listdir(os.path.join(cur_dir, dir_name))):
            if file[-4:] != 'json':
                continue
            file_id = int(file.split('.')[0].split('_')[1])
            with open(os.path.join(cur_dir, dir_name, file), 'r') as fin:
                result_data = json.load(fin)
            cur_cat, shape_id = result_data['category'], result_data['shape_id']

            if category_cnts[cur_cat] < freq_dict[cur_cat]:
                category_cnts[cur_cat] += 1
                file_list.append([os.path.join(cur_dir, dir_name), file_id])
                total_file += 1
            if total_file >= args.val_buffer_max_num:
                break
    num_file_per_process = total_file // args.num_processes + 1

    print(category_cnts)
    succ_dict = dict()
    for item in file_list:
        _, file_id = item
        succ_dict[file_id] = 0


    trans_q = ctx.Queue()
    for idx_process in range(args.num_processes):
        cur_file_list = file_list[idx_process * num_file_per_process: min((idx_process + 1) * num_file_per_process, total_file)]
        p = ctx.Process(target=run_jobs, args=(idx_process, args, trans_q, cur_file_list, out_dir))
        p.start()


    total, max_trial = 0, 0
    cnt_dict = {'succ': 0, 'fail': 0, 'invalid': 0}

    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            results = trans_q.get()
            result, file_id, repeat_id = results

            cnt_dict[result] += 1
            total += 1
            if result == 'succ':
                succ_dict[file_id] += 1

            print(
                'Episode: {} | Valid_portion: {:.4f} | Succ_portion: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, (cnt_dict['succ'] + cnt_dict['fail']) / total, cnt_dict['succ'] / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()

            if total >= total_file * args.repeat_num:
                exit(0)


if __name__ == '__main__':
    sys.exit(main())
