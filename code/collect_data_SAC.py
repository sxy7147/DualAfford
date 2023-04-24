import math
import random
import numpy as np
import torch
import argparse
import time
import datetime
import gc
import os
import sys
import shutil
from PIL import Image
import json
import imageio
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

torch.multiprocessing.set_start_method('forkserver', force=True)  # critical for make multiprocessing work
from models.SAC import ReplayBuffer, SAC_Trainer
from sapien.core import Pose
from env_rl import Env
from camera_rl import Camera
from robots.panda_robot_rl import Robot
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample


class DivError(Exception):
    pass


class Fail(Exception):
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--category', type=str)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--date', type=str)
parser.add_argument('--unique_suf', type=str, default="")
parser.add_argument('--out_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--begin_epoch', type=int, default=0)
parser.add_argument('--model_save_freq', type=int, default=50)

# env setting
parser.add_argument('--collect_mode', type=str, default='train')
parser.add_argument('--fix_cam', action='store_true', default=False)
parser.add_argument('--fix_joint', action='store_true', default=False)
parser.add_argument('--joint_range', nargs=2, type=float, default=[0.0, 1.0])
parser.add_argument('--show_gui', action='store_true', default=False, help='show_gui [default: False]')
parser.add_argument('--use_collision', action='store_true', default=False)
parser.add_argument("--use_HD", action="store_true", default=False)
parser.add_argument("--use_pam_act", action="store_true", default=False)
parser.add_argument("--use_pam", action="store_true", default=False)
parser.add_argument("--act_type", type=int, default=0)
parser.add_argument("--browse_mode", action="store_true", default=False)
parser.add_argument("--pam_rate", type=float, default=0.95)
parser.add_argument("--RL_act_rate", type=float, default=1.0)
parser.add_argument("--use_waiting", action="store_true", default=False)
parser.add_argument('--vis_gif', action='store_true', default=False)

# RL
parser.add_argument('--action_range', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--explore_steps', type=int, default=0)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--use_RL_pred', action='store_true', default=False)
parser.add_argument('--RL_continue', action='store_true', default=False)
parser.add_argument("--RL_mode", type=str, default="train")
parser.add_argument("--RL_load_suf", type=str, default="")
parser.add_argument("--RL_exp_name", type=str, default="")
parser.add_argument("--RL_ckpt", type=str, default=None)
parser.add_argument('--shape_id', type=int, default=0)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--replay_buffer_size', type=int, default=16384)
parser.add_argument("--reward_shape_num", type=int, default=0)
parser.add_argument('--reward_scale', type=float, default=1.0)
parser.add_argument('--alpha_lr', type=float, default=3e-4)
parser.add_argument('--soft_q_lr', type=float, default=3e-4)
parser.add_argument('--policy_lr', type=float, default=3e-4)
parser.add_argument('--update_itr', type=int, default=1)
parser.add_argument('--state_add_pose', action='store_true', default=False)
parser.add_argument('--penalty_obj_movement', action='store_true', default=False)
parser.add_argument('--AUTO_ENTROPY', action='store_true', default=True)
parser.add_argument('--DETERMINISTIC', action='store_true', default=False)

# HEU
parser.add_argument('--use_HEU', action='store_true', default=False)
parser.add_argument('--add_HEU_intoBuffer', action='store_true', default=False)

# contact point predict
parser.add_argument('--pam_exp_name', type=str, default=None, help='name of the training run')
parser.add_argument('--pam_model_version', type=str, default=None)
parser.add_argument('--pam_model_epoch', type=int, default=None, help='epoch')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_predict_aff_map(cam_XYZA, pam_net, device):
    mask = (cam_XYZA[:, :, 3] > 0.5)                        # 448 x 448 (True or False)
    pc = cam_XYZA[mask, :3]

    grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
    grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)   # 2 x 448 x 448
    pcids = grid_xy[:, mask].T

    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000]

    pc = pc[idx, :]                                         # M x 3
    pcids = pcids[idx, :]                                   # M x 2
    pc[:, 0] -= 5
    pc = torch.from_numpy(pc).unsqueeze(0).to(device)

    input_pcid = furthest_point_sample(pc, 10000).long().reshape(-1)
    pc = pc[:, input_pcid, :3]                      # 1 x N x 3
    pcids = pcids[input_pcid.cpu().numpy()]         # N x 2

    # action_score_map
    with torch.no_grad():
        pred_action_score_map = pam_net.inference_action_score(pc)[0]       # N
        pred_action_score_map = pred_action_score_map.cpu().numpy()

    return pc[0].cpu().numpy(), pcids, pred_action_score_map


def replay_push(replay_buffer, RL_pred, RL_mode, state=None, action=None, reward=None, next_state=None, done=None):
    if RL_pred and RL_mode == "train":
        replay_buffer.push(state, action, reward, next_state, done)


def get_pam_target_seg_mask(pam_pcids, pred_aff_score):
    score_rank = np.argsort(pred_aff_score)[::-1]
    num_point = score_rank.shape[0]
    score_rank = score_rank[:num_point // 10]
    select_map = np.zeros([448, 448], dtype=np.int32)
    select_map[pam_pcids[score_rank][:, 0], pam_pcids[score_rank][:, 1]] = 1
    return select_map


def get_first_point_pc(pc, pcids, x, y, device):
    idx = np.where(np.all((pcids == [x, y]), axis=1))[0][0]
    fpc = np.vstack([pc[idx: idx + 1], pc[:idx], pc[idx + 1:]])
    return torch.from_numpy(fpc).to(device)[None]


def thread_info(process_id, s):
    print(f"<Process {process_id}>: {s}")


def unknown_value(pose):
    return np.isnan(pose.p).any() or np.isinf(pose.p).any() or np.isnan(pose.q).any() or np.isinf(pose.q).any()


def worker(process_id, sac_trainer, trans_queue, replay_buffer, args, pam_net=None):
    '''
    the function for sampling with multi-processing
    '''
    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + process_id)

    # setup env
    thread_info(process_id, "Creating env")
    env = Env(show_gui=args.show_gui, set_ground=True, fix_joint=args.fix_joint)
    thread_info(process_id, "Env created!")

    # setup camera
    cam_theta, cam_phi = 2.8847928720705545, 1.0333818132766017  #####
    thread_info(process_id, "Creating camera...")
    cam = Camera(env, phi=cam_phi, theta=cam_theta, raise_camera=args.raise_camera, base_theta=args.base_theta,
                 random_initialize=False, fixed_position=args.fix_cam, restrict_dir=args.restrict_dir,
                 low_view=args.low_view)  # [0.5π, 1.5π] , exact_dir=True
    cam.change_pose()
    thread_info(process_id, "Camera created!")
    if args.show_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

    # RL setting
    state_dim = 13 if args.state_add_pose else 6
    action_dim = 12
    invalid_state = np.zeros(state_dim)

    # category-shape setting
    cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list_full(
        all_categories=args.category, primact=args.primact_type, mode=args.collect_mode
    )
    stable_dict = dict()

    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(args.material_static_friction, args.material_dynamic_friction, 0.01)
    if 'pushing' in args.primact_type or 'rotating' in args.primact_type or "topple" in args.primact_type:
        robot_scale = args.gripper_scale
        open_gripper = False
    elif 'pickup' in args.primact_type or "pulling" in args.primact_type:
        robot_scale = args.gripper_scale
        open_gripper = True
    else:
        robot_scale = args.gripper_scale
        open_gripper = False
    robot1 = Robot(env, robot_urdf_fn, robot_material, name="robot1", open_gripper=open_gripper, scale=robot_scale)
    robot2 = Robot(env, robot_urdf_fn, robot_material, name="robot2", open_gripper=open_gripper, scale=robot_scale)

    # training loop
    for eps in range(args.max_episodes):
        torch.cuda.empty_cache()
        out_info = dict()

        # set camera pose
        if not args.fix_cam:
            cam.change_pose()
            if args.show_gui:
                env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

        # load object
        if args.shape_id != 0:
            shape_id = args.shape_id
            selected_cat = shape2cat_dict[str(shape_id)]
        else:
            selected_cat = cat_list[random.randint(0, len(cat_list) - 1)]
            shape_id = cat2shape_dict[selected_cat][random.randint(0, len(cat2shape_dict[selected_cat]) - 1)]
        if shape_id not in stable_dict:
            stable_dict[shape_id] = [2, 2]
        thread_info(process_id, f"{stable_dict}")
        if np.random.rand() >= stable_dict[shape_id][0] / stable_dict[shape_id][1]:
            continue
        if args.urdf_type == "removed":
            object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd_rm.urdf" % str(shape_id)
            thread_info(process_id, f">>> Removed parts! Shape_id: {shape_id}")
        elif args.urdf_type == "fixbase":
            object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd_fixbase.urdf" % str(shape_id)
            thread_info(process_id, f">>> Fixbase! Shape_id: {shape_id}")
        elif args.urdf_type == "processed":
            object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd.urdf" % str(shape_id)
            thread_info(process_id, f">>> Processed! Shape_id: {shape_id}")
        elif args.urdf_type == "shapenet":
            object_urdf_fn = "../../Sapien_dataset/dataset2/%s/mobility_vhacd.urdf" % str(shape_id)
            thread_info(process_id, f">>> ShapeNet! Shape_id: {shape_id}")
        else:
            object_urdf_fn = '../../Sapien_dataset/dataset/%s/mobility.urdf' % str(shape_id)
            thread_info(process_id, f">>> Origin! Shape_id: {shape_id}")
        object_material = env.get_material(args.material_static_friction, args.material_dynamic_friction, 0.01)

        joint_angles = env.load_object(
            object_urdf_fn, object_material, joint_range=args.joint_range, target_part_id=-1, stiffness=args.stiffness, damping=args.damping,
            scale=args.obj_scale, density=args.density, lie_down=args.lie_down, use_collision=args.use_collision
        )
        stable_dict[shape_id][1] += 1

        # wait for the object's still
        thread_info(process_id, "Begin Waiting Still...")
        still_timesteps = utils.wait_for_object_still(env)
        thread_info(process_id, "Still waiting OK!")
        if still_timesteps < 5000:
            thread_info(process_id, 'Error: Object Not Still!\n')
            env.remove_all_objects()
            continue

        ### use the GT vision
        thread_info(process_id, "Get observation...")
        rgb, depth = cam.get_observation()      # can not delete
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
        cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]
        gt_all_link_mask = cam.get_id_link_mask(env.all_link_ids)       # (448, 448), 0 ~ len(all_link_ids)
        xs, ys = np.where(gt_all_link_mask > 0)
        if len(xs) == 0:
            thread_info(process_id, 'Error: No Link Pixel! Quit!\n')
            env.remove_all_objects()
            continue

        # select a target part for stable testing
        thread_info(process_id, "Finding target part...")
        gt_target_link_mask, prev_origin_world, target_pose,\
            target_link_mat44, target_x, target_y = utils.select_target_part(env, cam)

        if gt_target_link_mask is None:
            thread_info(process_id, "Error: No fixed target part!\n")
            env.remove_all_objects()
            continue
        stable_dict[shape_id][0] += 1

        # select point
        if args.use_pam and np.random.rand() < args.pam_rate:
            thread_info(process_id, "Predicting aff.....")
            pam_pc, pam_pcids, pred_aff_score = get_predict_aff_map(cam_XYZA, pam_net, args.device)
            select_seg_mask = get_pam_target_seg_mask(pam_pcids, pred_aff_score)
            out_info["seg_select"] = "pam"
        else:
            thread_info(process_id, "No Predicting aff.....")
            pam_pc, pam_pcids, pred_aff_score = None, None, None
            select_seg_mask = gt_all_link_mask
            out_info["seg_select"] = "random"

        # select 2 contact points
        xs, ys = np.where(select_seg_mask > 0)
        if len(xs) == 0:
            thread_info(process_id, 'Error: No Movable Link Pixel! Quit!\n')
            env.remove_all_objects()
            continue

        idx1, idx2, x1, y1, x2, y2 = None, None, None, None, None, None
        position_world1, position_cam1, position_world2, position_cam2 = None, None, None, None
        position_world1_xyz1, position_world2_xyz1 = None, None
        found = False
        for _ in range(100):
            idx1, idx2 = np.random.randint(len(xs)), np.random.randint(len(xs))
            x1, y1, x2, y2 = xs[idx1], ys[idx1], xs[idx2], ys[idx2]
            # given contact points, the agent predicts the directions
            position_world1, position_cam1, position_world1_xyz1 = utils.get_contact_point(cam, cam_XYZA, x1, y1)
            position_world2, position_cam2, position_world2_xyz1 = utils.get_contact_point(cam, cam_XYZA, x2, y2)
            if idx1 != idx2 and np.linalg.norm(position_world1 - position_world2) > 0.25:
                found = True
                break
        if not found:
            thread_info(process_id, "Error: Select point not found!\n")
            env.remove_all_objects()
            break
        thread_info(process_id, f"Select points: p1={position_world1}, p2={position_world2}")

        # begin to predict
        env.render()
        cur_qpos = env.get_object_qpos()
        cur_root_pos = env.get_object_root_pose()
        if args.show_gui:
            env.wait_to_start()

        out_info['random_seed'] = args.random_seed
        out_info['pixel_locs'] = [int(target_x), int(target_y)]
        out_info['target_link_mat44'] = target_link_mat44.tolist()
        out_info['prev_origin_world'] = prev_origin_world.tolist()
        out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
        out_info['camera_metadata'] = cam.get_metadata_json()
        out_info['joint_angles'] = joint_angles
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper
        out_info['shape_id'] = shape_id
        out_info['category'] = selected_cat
        out_info['primact_type'] = args.primact_type
        out_info['pixel1_idx'] = int(idx1)
        out_info['pixel2_idx'] = int(idx2)
        out_info['obj_pose_p'] = target_pose.p.tolist()
        out_info['obj_pose_q'] = target_pose.q.tolist()
        out_info['position_cam1'] = position_cam1.tolist()  # contact point at camera c
        out_info['position_world1'] = position_world1.tolist()  # world
        out_info['position_cam2'] = position_cam2.tolist()  # contact point at camera c
        out_info['position_world2'] = position_world2.tolist()  # world

        # take state and the RL policy and predict an action
        if args.use_RL_pred and np.random.rand() < args.RL_act_rate:
            thread_info(process_id, "predicting RL action!")
            out_info["act_select"] = "RL"
            state = np.concatenate([position_world1, position_world2])
            if args.state_add_pose:
                state = np.concatenate([position_world1, position_world2, target_pose.p, target_pose.q])
            if args.RL_mode == "test" or eps > args.explore_steps:
                action = sac_trainer.policy_net.get_action(state, deterministic=args.DETERMINISTIC)
            else:
                action = sac_trainer.policy_net.sample_action()

            # calculate the concrete direction of grippers
            dirm1 = utils.bgs(torch.from_numpy(action[0: 6 ]).to(args.device).contiguous().reshape(-1, 2, 3).permute(0, 2, 1))
            dirm2 = utils.bgs(torch.from_numpy(action[6: 12]).to(args.device).contiguous().reshape(-1, 2, 3).permute(0, 2, 1))
            if args.use_pam and eps < args.reward_shape_num // args.num_workers:
                world2cam = torch.from_numpy(np.linalg.inv(cam.mat44[:3, :3])).to(args.device).type(torch.float32)
                ppc1 = get_first_point_pc(pam_pc, pam_pcids, x1, y1, args.device)
                ppc2 = get_first_point_pc(pam_pc, pam_pcids, x2, y2, args.device)
                cam_up1, cam_forward1 = (world2cam @ dirm1[0, :, 0])[None], (world2cam @ dirm1[0, :, 1])[None]
                cam_up2, cam_forward2 = (world2cam @ dirm2[0, :, 0])[None], (world2cam @ dirm2[0, :, 1])[None]
                shape_reward1 = float((pam_net.inference_critic(ppc1, cam_up1, cam_forward1, True).item() - 1) * 0.5)
                shape_reward2 = float((pam_net.inference_critic(ppc2, cam_up2, cam_forward2, True).item() - 1) * 0.5)
                shape_reward = shape_reward1 + shape_reward2
            else:
                shape_reward = 0.0
                shape_reward1, shape_reward2 = 0, 0
            dirm1 = dirm1.detach().cpu().numpy()
            dirm2 = dirm2.detach().cpu().numpy()
            up1, forward1 = dirm1[0, :, 0], dirm1[0, :, 1]
            up2, forward2 = dirm2[0, :, 0], dirm2[0, :, 1]

        elif args.use_pam and args.use_pam_act:
            thread_info(process_id, "Where2act action!")
            out_info["act_select"] = "pam"
            state, action = None, None
            shape_reward1, shape_reward2 = 0, 0
            shape_reward = 0.0
            cam2world = cam.mat44[:3, :3]
            ppc1 = get_first_point_pc(pam_pc, pam_pcids, x1, y1, args.device)
            ppc2 = get_first_point_pc(pam_pc, pam_pcids, x2, y2, args.device)
            pred1_6d = pam_net.inference_actor(ppc1)[0]                                         # RV_CNT x 6
            pred1_Rs = pam_net.actor.bgs(pred1_6d.reshape(-1, 3, 2)).detach().cpu().numpy()     # RV_CNT x 3 x 3
            pred2_6d = pam_net.inference_actor(ppc2)[0]                                         # RV_CNT x 6
            pred2_Rs = pam_net.actor.bgs(pred2_6d.reshape(-1, 3, 2)).detach().cpu().numpy()     # RV_CNT x 3 x 3
            up1, forward1 = cam2world @ pred1_Rs[0, :, 0], cam2world @ pred1_Rs[0, :, 1]
            up2, forward2 = cam2world @ pred2_Rs[0, :, 0], cam2world @ pred2_Rs[0, :, 1]

        else:
            state, action = None, None
            up1, forward1 = np.random.randn(3).astype(np.float32), np.random.randn(3).astype(np.float32)
            up2, forward2 = np.random.randn(3).astype(np.float32), np.random.randn(3).astype(np.float32)
            if args.use_HD:
                thread_info(process_id, "Hard code action!")
                out_info["act_select"] = "HD"
                up1[2] = -10.0
                up2[2] = -10.0
            else:
                thread_info(process_id, "Random action!")
                out_info["act_select"] = "random"
            dir1 = torch.from_numpy(np.concatenate([up1, forward1])).contiguous().reshape(-1, 2, 3).permute(0, 2, 1)
            dir1 = utils.bgs(dir1).permute(0, 2, 1).contiguous().view(-1).detach().cpu().numpy()
            up1, forward1 = dir1[0: 3], dir1[3: 6]
            dir2 = torch.from_numpy(np.concatenate([up2, forward2])).contiguous().reshape(-1, 2, 3).permute(0, 2, 1)
            dir2 = utils.bgs(dir2).permute(0, 2, 1).contiguous().view(-1).detach().cpu().numpy()
            up2, forward2 = dir2[0: 3], dir2[3: 6]
            shape_reward = 0.0
            shape_reward1, shape_reward2 = 0, 0

        out_info["shape_reward1"] = shape_reward1
        out_info["shape_reward2"] = shape_reward2

        gif_imgs = []
        if "pulling" in args.primact_type:
            g1r, g2r = 5 + np.random.rand() * 5, 5 + np.random.rand() * 5
            thread_info(process_id, f"g1r: {g1r}, g2r: {g2r}")
            up1[2] = -math.tan(g1r * math.pi / 180) * np.linalg.norm(up1[:2])
            up2[2] = -math.tan(g2r * math.pi / 180) * np.linalg.norm(up2[:2])
        try:
            up1 /= np.linalg.norm(up1)
            forward1 /= np.linalg.norm(forward1)
            up2 /= np.linalg.norm(up2)
            forward2 /= np.linalg.norm(forward2)
            if (abs(up1 @ forward1) > 0.99) or (abs(up2 @ forward2) > 0.99):
                thread_info(process_id, "Up/Forward collision!")
                raise DivError()
            thread_info(process_id, f"\tup1: {up1}, forward1: {forward1}, up2: {up2}, forward2: {forward2}")

            pos1_pose1, pos1_rotmat1, pos2_pose1, pos2_rotmat1, pos3_rotmat1 = utils.get_rotmat_full(
                cam, position_world1, up1, forward1, act_type=args.act_type,
                number='1', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist
            )
            pos1_pose2, pos1_rotmat2, pos2_pose2, pos2_rotmat2, pos3_rotmat2 = utils.get_rotmat_full(
                cam, position_world2, up2, forward2, act_type=args.act_type,
                number='2', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist
            )
            thread_info(process_id, f"\tpos1_pose1: {pos1_pose1}, pos2_pose1: {pos2_pose1}, pos1_pose2: {pos1_pose2}, pos2_pose2: {pos2_pose2}")
            if unknown_value(pos1_pose1) or unknown_value(pos1_pose2) or unknown_value(pos2_pose1) or unknown_value(pos2_pose2):
                thread_info(process_id, "Inf/NaN value!")
                raise DivError()

            # setup robots
            robot1.load_gripper()
            robot2.load_gripper()

            # simulation, the robots move
            if args.browse_mode:
                result, grasp1, grasp2, next_grasp1, next_grasp2, fimg = "INVALID", False, False, False, False, None
                utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps * 2, cam=cam)
            else:
                result, fimg, sim_gif_imgs, grasp1, grasp2, next_grasp1, next_grasp2 = utils.simulate(
                    env, cam, args.primact_type, robot1, robot2, args.wait_steps, args.move_steps,
                    pos1_pose1, pos2_rotmat1, pos3_rotmat1, pos1_pose2, pos2_rotmat2, pos3_rotmat2, args.use_waiting
                )
                gif_imgs.extend(sim_gif_imgs)
            next_target_pose = env.get_target_part_pose()

            env.remove_all_objects()
            env.scene.remove_articulation(robot1.robot)
            env.scene.remove_articulation(robot2.robot)

        except:
            result = "INVALID"
            grasp1, grasp2, next_grasp1, next_grasp2 = False, False, False, False
            pos1_pose1, pos1_rotmat1, pos2_pose1, pos2_rotmat1, pos3_rotmat1 = None, None, None, None, None
            pos1_pose2, pos1_rotmat2, pos2_pose2, pos2_rotmat2, pos3_rotmat2 = None, None, None, None, None
            fimg = None
            next_target_pose = None
            env.remove_all_objects()

        if result == "INVALID":
            reward, success = 0, False
            next_state = invalid_state
            robot1_gif, robot2_gif = [], []
            robot1_fimg, robot2_fimg = None, None
        else:
            # VALID
            target_part_trans = next_target_pose.to_transformation_matrix()     # world -> target part, transformation matrix 4*4 SE3
            transition = np.linalg.inv(target_part_trans) @ target_link_mat44
            alpha, beta, gamma = utils.rotationMatrixToEulerAngles(transition)

            # calculate displacement
            next_origin_world_xyz1 = target_part_trans @ np.array([0, 0, 0, 1])
            next_origin_world = next_origin_world_xyz1[:3]
            trajectory = next_origin_world - prev_origin_world
            task_success, div_error, traj_len, traj_dir = utils.check_task_success(
                trajectory, alpha, beta, gamma, args.primact_type, grip_dir1=up1, grip_dir2=up2, out_info=out_info
            )
            thread_info(process_id, f"trajectory: {trajectory}, traj_len: {traj_len}, traj_dir: {traj_dir}")
            thread_info(process_id, f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")
            thread_info(process_id, f"task_success: {task_success} !")
            
            if "pulling" in args.primact_type:
                robot1_success, robot2_success = False, False
                robot1_gif, robot2_gif, robot1_fimg, robot2_fimg = [], [], None, None
            else:
                robot1_success, robot2_success = True, True
                robot1_gif, robot2_gif, robot1_fimg, robot2_fimg = [], [], None, None
                # only using one robot to succeed is considered a failure
                try:
                    if args.RL_mode != "vis" and not task_success:
                        raise Fail()
                    robot1_success, robot1_gif, robot1_fimg = utils.check_single_success(
                        env, cam, robot1, out_info, args.primact_type, object_urdf_fn, object_material, args.category,
                        cur_root_pos, cur_qpos, args.obj_scale, args.density, pos1_pose1, pos2_rotmat1, pos3_rotmat1,
                        args.wait_steps, args.move_steps, use_collision=args.use_collision
                    )
                    thread_info(process_id, f"robot1_success: {robot1_success} !")
                    if robot1_success:
                        raise Fail()
                    robot2_success, robot2_gif, robot2_fimg = utils.check_single_success(
                        env, cam, robot2, out_info, args.primact_type, object_urdf_fn, object_material, args.category,
                        cur_root_pos, cur_qpos, args.obj_scale, args.density, pos1_pose2, pos2_rotmat2, pos3_rotmat2,
                        args.wait_steps, args.move_steps, use_collision=args.use_collision
                    )
                    thread_info(process_id, f"robot2_success: {robot2_success} !")
                except Fail:
                    pass

            # calculate success
            success = task_success and not (robot1_success or robot2_success)
            if "pushing" in args.primact_type or "rotating" in args.primact_type or "topple" in args.primact_type:
                thread_info(process_id, "No need grasp")
            elif "pickup" in args.primact_type or "pulling" in args.primact_type or "fetch" in args.primact_type:
                thread_info(process_id, "Need grasp")
                success = success and grasp1 and grasp2 and next_grasp1 and next_grasp2

            reward = 0 if div_error else utils.cal_reward(
                args.primact_type, task_success, alpha, beta, gamma, traj_len, up1, up2, trajectory,
                grasp1, grasp2, next_grasp1, next_grasp2
            )

            # get next state
            if out_info["act_select"] == "RL":
                next_position_world1 = target_part_trans @ (np.linalg.inv(target_link_mat44) @ position_world1_xyz1)
                next_position_world1 = next_position_world1[:3]
                next_position_world2 = target_part_trans @ (np.linalg.inv(target_link_mat44) @ position_world2_xyz1)
                next_position_world2 = next_position_world2[:3]
                if args.state_add_pose:
                    next_state = np.concatenate([next_position_world1, next_position_world2,
                                                 next_target_pose.p, next_target_pose.q]).reshape(-1)
                else:
                    next_state = np.concatenate([next_position_world1, next_position_world2]).reshape(-1)

                out_info['next_position_world1'] = next_position_world1.tolist()
                out_info['next_position_world2'] = next_position_world2.tolist()
                out_info['state'] = state.tolist()
                out_info['action'] = action.tolist()
                out_info['next_state'] = next_state.tolist()
            else:
                next_state = None

            out_info['task_success'] = 'True' if task_success else 'False'
            out_info['success'] = 'True' if success else 'False'
            out_info['target_part_trans'] = target_part_trans.tolist()
            out_info['transition'] = transition.tolist()
            out_info['alpha'] = alpha.tolist()
            out_info['beta'] = beta.tolist()
            out_info['gamma'] = gamma.tolist()
            out_info['next_obj_pose_p'] = next_target_pose.p.tolist()
            out_info['next_obj_pose_q'] = next_target_pose.q.tolist()
            out_info['reward'] = reward

        out_info['result'] = result
        img_pack = {
            "gt_target_link_mask": gt_target_link_mask, "select_seg_mask": select_seg_mask,
            "gif_imgs": gif_imgs, "fimg": fimg,
            "robot1_gif": robot1_gif, "robot1_fimg": robot1_fimg, "robot2_gif": robot2_gif, "robot2_fimg": robot2_fimg,
        }
        if args.use_pam:
            img_pack["pam_pc"] = pam_pc
            img_pack["pred_aff_map"] = pred_aff_score

        replay_push(replay_buffer, args.use_RL_pred, args.RL_mode, state, action, reward + shape_reward, next_state, True)
        trans_queue.put([out_info, cam_XYZA_list, img_pack, success, reward])
        gc.collect()
        print("")


def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()


def get_cate_setting_args(args):
    setting_file = open("./cate_setting.json", "r")
    all_dict = json.load(setting_file)
    setting_file.close()
    args.cate_id = all_dict["shape_id"][args.category]
    cs_dict = all_dict[args.primact_type]
    args.obj_scale = cs_dict["cate_setting"][args.category]["obj_scale"]
    args.urdf_type = cs_dict["cate_setting"][args.category]["urdf"]
    args.joint_range = cs_dict["cate_setting"][args.category]["joint_range"]
    args.base_theta = cs_dict["cate_setting"][args.category]["base_theta"]
    args.restrict_dir = cs_dict["cate_setting"][args.category]["restrict_dir"]
    args.low_view = cs_dict["cate_setting"][args.category]["low_view"]
    args.lie_down = cs_dict["cate_setting"][args.category]["lie_down"]
    args.density = cs_dict["density"]
    if args.RL_mode == "test":
        args.RL_load_date = cs_dict["RL_load_date"]
        args.RL_exp_name = "SAC_" + cs_dict["cate_setting"][args.category]["RL_exp_num"]
    elif args.RL_continue:
        args.RL_load_date = args.date
        args.RL_load_suf = "old"
    args.material_dynamic_friction = cs_dict["material_dynamic_friction"]
    args.material_static_friction = cs_dict["material_static_friction"]
    args.scene_dynamic_friction = cs_dict["scene_dynamic_friction"]
    args.scene_static_friction = cs_dict["scene_static_friction"]
    args.stiffness = cs_dict["stiffness"]
    args.damping = cs_dict["damping"]
    args.gripper_scale = cs_dict["gripper_scale"]
    args.start_dist = cs_dict["start_dist"]
    args.final_dist = cs_dict["final_dist"]
    args.move_steps = cs_dict["move_steps"]
    args.wait_steps = cs_dict["wait_steps"]
    args.raise_camera = 0.0
    args.manage_address = f"{args.date}{args.cate_id}{args.unique_suf}.com"
    if args.browse_mode:
        args.collect_mode = "all"
    if args.use_HD:
        dir_pre = "HD"
    elif args.use_RL_pred:
        dir_pre = "SAC" if args.RL_mode == "train" else "SACVAL"
    elif args.use_pam_act:
        dir_pre = "PAMACT"
    else:
        dir_pre = "Random"
    args.out_folder = f"{dir_pre}_{args.category}_{args.primact_type}_" + \
                      f"{args.date}_id{args.cate_id}{'' if args.unique_suf == '' else '_'}{args.unique_suf}"
    if args.show_gui:
        args.num_workers = 1
    args.device = args.device if torch.cuda.is_available() else "cpu"
    
    return args


if __name__ == '__main__':
    print("OK!!!!")
    args = parser.parse_args()
    args = get_cate_setting_args(args)

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    out_dir = os.path.join(args.out_dir, args.out_folder)
    print('out_dir: ', out_dir)
    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, overwrite? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()
        shutil.rmtree(out_dir)
    else:
        os.makedirs(out_dir)

    # load the paths
    result_dual_succ_dir, saved_dual_succ_dir = os.path.join(out_dir, 'dual_succ_gif'), os.path.join(out_dir, 'dual_succ_files')
    result_fail_dir, saved_fail_dir = os.path.join(out_dir, 'fail_gif'), os.path.join(out_dir, 'fail_files')
    result_invalid_dir, saved_invalid_dir = os.path.join(out_dir, 'invalid_gif'), os.path.join(out_dir, 'invalid_files')
    os.makedirs(result_dual_succ_dir), os.makedirs(saved_dual_succ_dir)
    os.makedirs(result_fail_dir), os.makedirs(saved_fail_dir)
    os.makedirs(result_invalid_dir), os.makedirs(saved_invalid_dir)
    result_dirs = [result_dual_succ_dir, result_fail_dir, result_invalid_dir]
    dir_items = ["select_seg", "target_link", "gif", "fimg", "single_gif", "single_fimg"]
    if args.use_pam:
        dir_items.append("pred_aff_map")
    for d in result_dirs:
        for im in dir_items:
            os.makedirs(os.path.join(d, im))

    tb_writer = SummaryWriter(os.path.join(out_dir, 'tb_logs'))

    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager(address=args.manage_address)
    manager.start()
    if args.use_RL_pred:
        state_dim = 13 if args.state_add_pose else 6
        action_dim = 12
        if args.RL_mode == "train":
            replay_buffer = manager.ReplayBuffer(args.replay_buffer_size)  # share the replay buffer through manager
        else:
            replay_buffer = None
        sac_trainer = SAC_Trainer(
            replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim,
            action_range=args.action_range, device=args.device
        )

        # share the global parameters in multiprocessing
        sac_trainer.soft_q_net1.share_memory()
        sac_trainer.soft_q_net2.share_memory()
        sac_trainer.target_soft_q_net1.share_memory()
        sac_trainer.target_soft_q_net2.share_memory()
        sac_trainer.policy_net.share_memory()   # model
        sac_trainer.log_alpha.share_memory_()   # variable
        ShareParameters(sac_trainer.soft_q_optimizer1)
        ShareParameters(sac_trainer.soft_q_optimizer2)
        ShareParameters(sac_trainer.policy_optimizer)
        ShareParameters(sac_trainer.alpha_optimizer)

        if args.RL_mode == "test" or args.RL_continue:
            print("Loading checkpoint...")
            if args.RL_ckpt is None:
                args.RL_load_folder = f"SAC_{args.category}_{args.primact_type}_{args.RL_load_date}_" + \
                                  f"id{args.cate_id}{'' if args.RL_load_suf == '' else '_'}{args.RL_load_suf}"
                args.RL_ckpt = os.path.join(args.out_dir, args.RL_load_folder, args.RL_exp_name)
            sac_trainer.load_model(args.RL_ckpt)
            print("Model loaded!")
    else:
        replay_buffer = None
        sac_trainer = None
        state_dim, action_dim = None, None

    # load where2act model
    if args.use_pam:
        pam_train_conf = torch.load(os.path.join('logs', "w2a", args.pam_exp_name, 'conf.pth'))
        pam_model_def = utils.get_model_module(args.pam_model_version)
        pam_net = pam_model_def.Network(pam_train_conf.feat_dim, pam_train_conf.rv_dim, pam_train_conf.rv_cnt)
        print('Loading ckpt from ', os.path.join('logs', "w2a", args.pam_exp_name, 'ckpts'), args.pam_model_epoch)
        data_to_restore = torch.load(os.path.join('logs', "w2a", args.pam_exp_name, 'ckpts', f'{args.pam_model_epoch}-network.pth'))
        pam_net.load_state_dict(data_to_restore, strict=False)
        print('DONE\n')
        pam_net.to(args.device)     # send to device
        pam_net.eval()
    else:
        pam_net = None

    trans_queue = mp.Queue()        # used for get rewards from all processes and plot the curve
    processes = []
    for i in range(args.num_workers):
        process = Process(target=worker, args=(i, sac_trainer, trans_queue, replay_buffer, args, pam_net))
        process.daemon = True       # all processes closed when the main stops
        processes.append(process)
    [p.start() for p in processes]

    # add heu
    if args.add_HEU_intoBuffer:
        json_dir = os.path.join(args.out_dir, "HEU_succ_json")
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                with open(os.path.join(json_dir, file)) as fin:
                    result_data = json.load(fin)
                    state, action = np.array(result_data['state'], dtype=np.int32), np.array(result_data['action'], dtype=np.int32)
                    reward, next_state = result_data['reward'], np.array(result_data['next_state'], dtype=np.int32)
                    replay_buffer.push(state, action, reward, next_state, True)
            break

    epoch, saved_fail_epoch, saved_invalid_epoch = 0, 0, 0
    saved_epoch = args.begin_epoch
    num_valid, num_single_succ, num_fail, num_dual_succ, num_invalid = 0, 0, 0, 0, 0
    dual_succ_record = []
    pam_dual_succ_record = [0] if args.use_pam else None
    no_pam_dual_succ, no_pam_num = 0, 1
    shape_success = dict()
    t_begin = datetime.datetime.now()
    while True:             # keep getting the episode reward from the queue
        # dynamically adding process, if you want to speed up the collecting process
        if os.path.exists(os.path.join(out_dir, "Add")):
            os.rmdir(os.path.join(out_dir, "Add"))
            p = Process(target=worker, args=(len(processes), sac_trainer, trans_queue, replay_buffer, args, pam_net))
            p.daemon = True       # all processes closed when the main stops
            p.start()
            processes.append(p)
        t0 = time.time()
        epoch += 1
        out_info, cam_XYZA_list, img_pack, success, reward = trans_queue.get()
        out_info['epoch'] = epoch
        shape_id = out_info["shape_id"]
        shape_cate = out_info["category"]
        if shape_id not in shape_success:
            shape_success[shape_id] = [0, 0, 0, 0, 0]   # num, num_invalid, num_fail, num_single_succ, num_dual_succ
        shape_success[shape_id][0] += 1

        valid, dual_succ, fail = False, False, False
        if out_info['result'] == 'INVALID':
            num_invalid += 1
            shape_success[shape_id][1] += 1
            actual_movement, robot1_actual_movement, robot2_actual_movement = -1, -1, -1
        else:
            valid = True
            num_valid += 1
            actual_movement = out_info['traj_len']
            robot1_actual_movement = out_info["robot1_traj_len"] if "robot1_traj_len" in out_info else -1
            robot2_actual_movement = out_info["robot2_traj_len"] if "robot2_traj_len" in out_info else -1
            if success:
                dual_succ = True
                num_dual_succ += 1
                shape_success[shape_id][4] += 1
            else:
                fail = True
                num_fail += 1
                shape_success[shape_id][2] += 1
                tc = out_info["task_success"] == "True"
                r1c = "robot1_success" in out_info and out_info["robot1_success"] == "True"
                r2c = "robot2_success" in out_info and out_info["robot2_success"] == "True"
                num_single_succ += tc and (r1c or r2c)
                shape_success[shape_id][3] += tc and (r1c or r2c)
        shape_fout = open(os.path.join(out_dir, "shape_success.json"), "w")
        json.dump(shape_success, shape_fout)
        shape_fout.close()

        # calculate nearby success rate
        dual_succ_record.append(int(dual_succ))
        if out_info["seg_select"] == "pam" and out_info["act_select"] == "RL":
            pam_dual_succ_record.append(int(dual_succ))
        else:
            no_pam_dual_succ += int(dual_succ)
            no_pam_num += 1
        dual_succ_nearby = sum(dual_succ_record[-1000:]) / len(dual_succ_record[-1000:])
        pam_dual_succ_nearby = sum(pam_dual_succ_record[-1000:]) / len(pam_dual_succ_record[-1000:]) if args.use_pam else 0.0

        # update
        if args.use_RL_pred and args.RL_mode == "train":
            print('Length of ReplayBuffer: ', replay_buffer.get_length())
            if replay_buffer.get_length() > args.batch_size:
                for i in range(args.update_itr):
                    p_loss, v_loss = sac_trainer.update(
                        args.batch_size, reward_scale=args.reward_scale, auto_entropy=args.AUTO_ENTROPY,
                        target_entropy=-1. * action_dim
                    )
                    tb_writer.add_scalar('policy_loss', p_loss, epoch * args.update_itr + i)
                    tb_writer.add_scalar('value_loss', v_loss, epoch * args.update_itr + i)
            if epoch % args.model_save_freq == 0 and epoch > 0:
                sac_trainer.save_model(path=os.path.join(out_dir, 'SAC_%d' % epoch))

        tb_writer.add_scalar('reward', reward, epoch)
        tb_writer.add_scalar('shape_reward', out_info["shape_reward1"] + out_info["shape_reward2"], epoch)
        tb_writer.add_scalar('dual_succ_nearby', dual_succ_nearby, epoch)
        if args.use_pam:
            tb_writer.add_scalar('pam_dual_succ_nearby', pam_dual_succ_nearby, epoch)
        tb_writer.add_scalar('num_dual_succ', num_dual_succ, epoch)
        tb_writer.add_scalar('num_fail', num_fail, epoch)
        tb_writer.add_scalar('num_invalid', num_invalid, epoch)
        tb_writer.add_scalar('num_single_succ', num_single_succ, epoch)

        # save json and save gif
        gt_target_link_mask = img_pack["gt_target_link_mask"]
        select_seg_mask = img_pack["select_seg_mask"]
        if args.RL_mode == "train":
            fail_limit, invalid_limit = 5, 5
            fail_bias = 1000 if epoch % 3 == 0 else 0
            invalid_bias = 1000 if epoch % 25 == 0 else 0
        elif args.RL_mode == "test":
            fail_limit, invalid_limit = 2, 2
            fail_bias, invalid_bias = 0, 0
        else:
            fail_limit, invalid_limit = 50, 50
            fail_bias, invalid_bias = 1000, 0

        if dual_succ:
            saved_epoch += 1
            utils.save_data_full(saved_dual_succ_dir, saved_epoch, out_info, cam_XYZA_list, gt_target_link_mask, whole_pc=None)
            result_save_dir = result_dual_succ_dir
        elif fail and num_dual_succ * fail_limit + fail_bias >= saved_fail_epoch:
            saved_epoch += 1
            saved_fail_epoch += 1
            utils.save_data_full(saved_fail_dir, saved_epoch, out_info, cam_XYZA_list, gt_target_link_mask, whole_pc=None)
            result_save_dir = result_fail_dir
        elif (not valid) and num_dual_succ * invalid_limit + invalid_bias >= saved_invalid_epoch:
            saved_epoch += 1
            saved_invalid_epoch += 1
            utils.save_data_full(saved_invalid_dir, saved_epoch, out_info, cam_XYZA_list, gt_target_link_mask, whole_pc=None)
            result_save_dir = result_invalid_dir
        else:
            result_save_dir = None

        if result_save_dir is not None:
            file_str = f"{shape_id}_{saved_epoch}_{shape_cate}"

            gt_target_link_mask[gt_target_link_mask > 0] = 1
            select_seg_mask[select_seg_mask > 0] = 1
            slc_path = os.path.join(result_save_dir, "select_seg", f'slc_{file_str}.png')
            link_path = os.path.join(result_save_dir, "target_link", f'link_{file_str}.png')
            img_link = Image.fromarray((gt_target_link_mask * 255).astype(np.uint8))
            img_seg = Image.fromarray((select_seg_mask * 255).astype(np.uint8))
            img_link.save(link_path)
            img_seg.save(slc_path)
            if len(img_pack["gif_imgs"]) > 0:
                gif_path = os.path.join(result_save_dir, "gif", f'{file_str}_%.3f_%.2f.gif' % (actual_movement, reward))
                imageio.mimsave(gif_path, img_pack["gif_imgs"])
            if len(img_pack["robot1_gif"]) > 0:
                robot1_gif_path = os.path.join(result_save_dir, "single_gif", f'robot1_{file_str}_%.3f.gif' % robot1_actual_movement)
                imageio.mimsave(robot1_gif_path, img_pack["robot1_gif"])
            if len(img_pack["robot2_gif"]) > 0:
                robot2_gif_path = os.path.join(result_save_dir, "single_gif", f'robot2_{file_str}_%.3f.gif' % robot2_actual_movement)
                imageio.mimsave(robot2_gif_path, img_pack["robot2_gif"])
            if img_pack["fimg"] is not None:
                fimg_path = os.path.join(result_save_dir, "fimg", f'fimg_{file_str}.png')
                img_pack["fimg"].save(fimg_path)                    # first frame
            if img_pack["robot1_fimg"] is not None:
                robot1_fimg_path = os.path.join(result_save_dir, "single_fimg", f'robot1_fimg_{file_str}.png')
                img_pack["robot1_fimg"].save(robot1_fimg_path)      # first frame
            if img_pack["robot2_fimg"] is not None:
                robot2_fimg_path = os.path.join(result_save_dir, "single_fimg", f'robot2_fimg_{file_str}.png')
                img_pack["robot2_fimg"].save(robot2_fimg_path)      # first frame
            if out_info["seg_select"] == "pam":
                pam_path = os.path.join(result_save_dir, "pred_aff_map", f"action_score_map_full_{saved_epoch}_{out_info['shape_id']}")
                utils.render_pts_label_png(pam_path, img_pack["pam_pc"], img_pack["pred_aff_map"])

        print(
            f'Episode: {epoch}  | Valid_portion: %.4f | Succ_portion: %.4f | Succ_nearby: %.4f | Pam_nearby: %.4f | No_pam: %.4f' %
            (num_valid / epoch, num_dual_succ / epoch, dual_succ_nearby, pam_dual_succ_nearby, no_pam_dual_succ / no_pam_num)
        )
        print(f"Running Time: %.4f | Total Time:" % (time.time() - t0), datetime.datetime.now() - t_begin)
        print('actual_movement:', actual_movement, '\tvalid:', valid, '\tsuccess:', success, '\treward: ', reward)
        print('num_dual_succ: ', num_dual_succ, "\tnum_fail: ", num_fail, "\tnum_single_succ: ", num_single_succ, "\tnum_invalid:", num_invalid)
        del out_info, img_pack
        gc.collect()

