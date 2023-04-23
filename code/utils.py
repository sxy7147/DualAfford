import os
import sys
import h5py
import torch
import numpy as np
import importlib
import random
import shutil
import math
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from subprocess import call
from sapien.core import Pose
import json
import torch.nn.functional as F
from camera import Camera


def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)


def collate_feats(b):
    return list(zip(*b))


def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def export_label(out, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f\n' % (l[i]))


def export_pts_label(out, v, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], l[i]))


def render_pts_label_png(out, v, l):
    export_pts(out + '.pts', v)
    export_label(out + '.label', l)
    export_pts_label(out + '.feats', v, l)
    cmd = 'xvfb-run -a ~/thea/TheaDepsUnix/Source/TheaPrefix/bin/Thea/RenderShape %s.pts -f %s.feats %s.png 448 448 -v 1,0,0,-5,0,0,0,0,1 >> /dev/null' % (out, out, out)
    call(cmd, shell=True)
    # print("calling", cmd)


def render_png_given_pts(out):
    cmd = 'xvfb-run -a ~/thea/TheaDepsUnix/Source/TheaPrefix/bin/Thea/RenderShape %s.pts -f %s.feats %s.png 448 448 -v 1,0,0,-5,0,0,0,0,1 >> /dev/null' % (out, out, out)
    call(cmd, shell=True)


def export_pts_color_obj(out, v, c):
    with open(out + '.obj', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))


def export_pts_color_pts(out, v, c):
    with open(out + '.pts', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))


def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta


def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint


def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector

    Args:
        pose: (4, 4) transformation matrix

    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """

    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (
            1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta


def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x


def get_random_number(l, r):
    return np.random.rand() * (r - l) + l


def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()


def calc_part_motion_degree(part_motion):
    return part_motion * 180.0 / 3.1415926535


def radian2degree(radian):
    return radian * 180.0 / np.pi


def degree2radian(degree):
    return degree / 180.0 * np.pi


def cal_Fscore(pred, labels):
    TP, TN, FN, FP = 0, 0, 0, 0
    TP += ((pred == 1) & (labels == 1)).sum()  
    TN += ((pred == 0) & (labels == 0)).sum()  
    FN += ((pred == 0) & (labels == 1)).sum()  
    FP += ((pred == 1) & (labels == 0)).sum()  
    try:
        p = TP / (TP + FP)
    except:
        p = 0
    try:
        r = TP / (TP + FN)
    except:
        r = 0
    try:
        F1 = 2 * r * p / (r + p)
    except:
        F1 = 0

    acc = (pred == labels).sum() / len(pred)
    return F1, p, r, acc


def cal_included_angle(x, y):
    len_x = np.linalg.norm(x)
    len_y = np.linalg.norm(y)
    cos_ = (x @ y) / (len_x * len_y)
    angle_radian = np.arccos(np.clip(cos_, -1 + 1e-6, 1 - 1e-6))
    angle_degree = angle_radian * 180 / np.pi
    len_projection = len_x * cos_   # the projection of x on y
    return angle_degree, len_projection


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])    # pitch (Y-axis)
        y = math.atan2(-R[2, 0], sy)        # yaw (Z-axis)
        z = math.atan2(R[1, 0], R[0, 0])    # roll (X-axis)
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi])


def get_rotmat(cam, position_world, up, forward, number, out_info, start_dist=0.20, final_dist=0.08):
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
    out_info['position_world' + number] = position_world.tolist()  # world
    out_info['gripper_direction_world' + number] = up.tolist()
    out_info['gripper_direction_camera' + number] = up_cam.tolist()
    out_info['gripper_forward_direction_world' + number] = forward.tolist()
    out_info['gripper_forward_direction_camera' + number] = forward_cam.tolist()
    rotmat = np.eye(4).astype(np.float32)  # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - up * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)
    out_info['target_rotmat_world' + number] = final_rotmat.tolist()

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - up * start_dist
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    out_info['start_rotmat_world' + number] = start_rotmat.tolist()

    return start_pose, start_rotmat, final_pose, final_rotmat


def cal_final_pose(cam, cam_XYZA, x, y, number, out_info, start_dist=0.20, final_dist=0.08, given_up=None, given_forward=None, given=False):
    # get pixel 3D position (cam/world)
    position_cam = cam_XYZA[x, y, :3]   # contact point
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]
    out_info['position_cam' + number] = position_cam.tolist()   # contact point at camera c
    out_info['position_world' + number] = position_world.tolist()   # world

    # get pixel 3D pulling direction (cam/world)
    gt_nor = cam.get_normal_map()
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
    out_info['norm_direction_camera' + number] = direction_cam.tolist()
    out_info['norm_direction_world' + number] = direction_world.tolist()

    # The initial direction obeys Gaussian distribution
    degree = np.abs(np.random.normal(loc=0, scale=25, size=[1]))
    radian = degree * np.pi / 180
    threshold = 1 * np.pi / 180
    # sample a random direction in the hemisphere (cam/world)
    action_direction_cam = np.random.randn(3).astype(np.float32)
    action_direction_cam /= np.linalg.norm(action_direction_cam)
    # while action_direction_cam @ direction_cam > -np.cos(np.pi / 6):  # up_norm_thresh: 30
    num_trial = 0
    while (action_direction_cam @ direction_cam > -np.cos(radian + threshold) or action_direction_cam @ direction_cam < -np.cos(radian - threshold))\
            and num_trial < 2000:  # up_norm_thresh: 30
        action_direction_cam = np.random.randn(3).astype(np.float32)
        action_direction_cam /= np.linalg.norm(action_direction_cam)
        num_trial += 1
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
    if given == True:
        action_direction_world = given_up
    out_info['gripper_direction_world' + number] = action_direction_world.tolist()

    # compute final pose
    up = np.array(action_direction_world, dtype=np.float32)
    forward = np.random.randn(3).astype(np.float32)
    while abs(up @ forward) > 0.99:
        forward = np.random.randn(3).astype(np.float32)
    if given == True:
        forward = given_forward
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
    out_info['gripper_forward_direction_world' + number] = forward.tolist()
    rotmat = np.eye(4).astype(np.float32)   # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)
    out_info['target_rotmat_world' + number] = final_rotmat.tolist()

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - action_direction_world * start_dist
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    out_info['start_rotmat_world' + number] = start_rotmat.tolist()

    return start_pose, start_rotmat, final_pose, final_rotmat, up, forward



def dual_gripper_move_to_target_pose(robot1, robot2, target_ee_pose1, target_ee_pose2, num_steps, vis_gif=False, vis_gif_interval=200, cam=None):
    imgs = []

    executed_time1 = num_steps * robot1.timestep
    executed_time2 = num_steps * robot2.timestep
    spatial_twist1 = robot1.calculate_twist(executed_time1, target_ee_pose1)
    spatial_twist2 = robot2.calculate_twist(executed_time2, target_ee_pose2)

    rdis1, rdis2, ddis = 1, 1, 0
    for i in range(num_steps):
        if i % 100 == 0:
            spatial_twist1 = robot1.calculate_twist((num_steps - i) * robot1.timestep, target_ee_pose1)
            spatial_twist2 = robot2.calculate_twist((num_steps - i) * robot2.timestep, target_ee_pose2)

        qvel1 = robot1.compute_joint_velocity_from_twist(spatial_twist1)
        # robot1.internal_controller(qvel1)
        robot1.internal_controller(qvel1 / math.exp(max(0, -ddis)))
        qvel2 = robot2.compute_joint_velocity_from_twist(spatial_twist2)
        # robot2.internal_controller(qvel2)
        robot2.internal_controller(qvel2 / math.exp(max(0, ddis)))
        robot2.env.step()
        robot2.env.render()
        if vis_gif and ((i + 1) % vis_gif_interval == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
        if vis_gif and (i == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)

    if vis_gif:
        return imgs
    

def dual_gripper_wait_n_steps(robot1, robot2, n, vis_gif=False, vis_gif_interval=200, cam=None):
    imgs = []

    robot1.clear_velocity_command()
    robot2.clear_velocity_command()
    for i in range(n):
        passive_force1 = robot1.robot.compute_passive_force()
        passive_force2 = robot2.robot.compute_passive_force()
        robot1.robot.set_qf(passive_force1)
        robot2.robot.set_qf(passive_force2)
        robot2.env.step()
        robot2.env.render()
        if vis_gif and ((i + 1) % vis_gif_interval == 0):
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            imgs.append(fimg)
    robot1.robot.set_qf([0] * robot1.robot.dof)
    robot2.robot.set_qf([0] * robot2.robot.dof)

    if vis_gif:
        return imgs



def append_data_list(file_dir, only_true_data=False, append_root_dir=False, primact_type='pushing'):
    data_list = []
    if file_dir != 'xxx':
        if 'RL' in file_dir:
            data_list.append(os.path.join(file_dir, 'dual_succ_files'))
        else:
            data_list.append(os.path.join(file_dir, 'succ_files'))

        if not only_true_data:
            data_list.append(os.path.join(file_dir, 'fail_files'))
            data_list.append(os.path.join(file_dir, 'invalid_files'))

        if append_root_dir:
            data_list.append(file_dir)
        print('data_list: ', data_list)
    return data_list


def save_data(saved_dir, epoch, out_info, cam_XYZA_list, gt_target_link_mask, whole_pc=None, repeat_id=None, init_cam_XYZA_list=None, final_cam_XYZA_list=None):
    if repeat_id == None:
        symbol = str(epoch)
    else:
        symbol = str(epoch) + '_' + str(repeat_id)

    with open(os.path.join(saved_dir, 'result_%s.json' % symbol), 'w') as fout:
        json.dump(out_info, fout)

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA = cam_XYZA_list
    save_h5(os.path.join(saved_dir, 'cam_XYZA_%s.h5' % symbol), [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
    if init_cam_XYZA_list != None:
        init_cam_XYZA_id1, init_cam_XYZA_id2, init_cam_XYZA_pts, init_cam_XYZA = init_cam_XYZA_list
        save_h5(os.path.join(saved_dir, 'init_cam_XYZA_%s.h5' % symbol), [(init_cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                          (init_cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                          (init_cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
    if final_cam_XYZA_list != None:
        final_cam_XYZA_id1, final_cam_XYZA_id2, final_cam_XYZA_pts, final_cam_XYZA = final_cam_XYZA_list
        save_h5(os.path.join(saved_dir, 'final_cam_XYZA_%s.h5' % symbol), [(final_cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                                                                          (final_cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                                                                          (final_cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])

    if whole_pc != None:
        np.savez(os.path.join(saved_dir, 'collision_visual_shape_%s' % symbol), pts=whole_pc)

    Image.fromarray((gt_target_link_mask > 0).astype(np.uint8) * 255).save(
        os.path.join(saved_dir, 'interaction_mask_%s.png' % symbol))



# input sz bszx3x2
# input: bs * 3 * 2
# tensor(forward, up).reshape(-1, 2, 3).permute(0, 2, 1)
def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


# batch geodesic loss for rotation matrices
def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)


# 6D-Rot loss
# input sz bszx6
def get_6d_rot_loss(pred_6d, gt_6d):
    # [bug fixed]
    # pred_Rs = self.bgs(pred_6d.reshape(-1, 3, 2))
    # gt_Rs = self.bgs(gt_6d.reshape(-1, 3, 2))
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    return theta


def check_success(trajectory, alpha, beta, gamma, primact_type, out_info=None, lieDown=False, threshold=3, grip_dir1=None, grip_dir2=None):
    div_error, success = False, False

    if not out_info:    # useless
        out_info = {}

    if 'pushing' in primact_type:
        # the projection on x-y plane
        traj_projection = trajectory
        traj_projection[-1] = 0.0
        traj_len = np.linalg.norm(traj_projection)
        if traj_len > 1e-6:
            traj_dir = traj_projection / np.linalg.norm(traj_projection)
        else:
            traj_dir = np.array([0, 0, 0])
            div_error = True

        if np.abs(alpha) < threshold and np.abs(beta) < threshold and np.abs(gamma) < threshold and traj_len >= 0.05:
            success = True

    if 'rotating' in primact_type:
        traj_len = np.linalg.norm(trajectory)
        traj_dir = None
        task = beta
        if not lieDown:
            if np.abs(alpha) < threshold and np.abs(beta) > 10 and np.abs(gamma) < threshold:  # rorate and can translation ######
                success = True
        else:
            if np.abs(alpha) > 10 and np.abs(beta) < threshold and np.abs(gamma) < threshold:  # rorate and can translation ######
                success = True

    if 'topple' in primact_type:
        # the projection on x-y plane
        traj_projection = trajectory
        traj_projection[-1] = 0.0
        traj_len = np.linalg.norm(traj_projection)
        if traj_len > 1e-6:
            traj_dir = traj_projection / np.linalg.norm(traj_projection)
        else:
            traj_dir = np.array([0, 0, 0])
            div_error = True
        if ((np.abs(alpha) + np.abs(gamma)) > 10) and (np.abs(beta) < threshold):
            success = True

    if 'pickup' in primact_type:
        # the projection on z-axis
        traj_len = trajectory[2]
        try:
            traj_dir = trajectory / np.linalg.norm(trajectory)
        except:
            traj_dir = np.array([0, 0, 0], dtype=np.float32)
            div_error = True
        task = traj_dir
        if np.abs(alpha) < threshold and np.abs(beta) < threshold and np.abs(gamma) < threshold and traj_len >= 0.05:  # translation but not rotate
            success = True

    return success, div_error, traj_len, traj_dir


def get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system, mat44=None, cam2cambase=None, gt_target_link_mask=None):
    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    mask = (out[:, :, 3] > 0.5)
    pc = out[mask, :3]
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000 - 1]
    pc = pc[idx, :]
    pc[:, 0] -= 5
    pc_centers = None
    if coordinate_system == 'world':  # cam2world
        pc = (mat44[:3, :3] @ pc.T).T
    elif coordinate_system == 'cambase':  # cam2cambase
        pc = pc @ np.transpose(cam2cambase, (1, 0))
        pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
        pc_centers = pc_centers[0]
        pc -= pc_centers
    out = torch.from_numpy(pc).unsqueeze(0)
    return out, pc_centers


def wait_for_object_still(env, cam=None, visu=False):
    print('start wait for still')
    still_timesteps, wait_timesteps = 0, -1
    imgs = []
    las_qpos = env.get_object_qpos()            
    las_root_pose = env.get_object_root_pose()  
    no_qpos = (las_qpos.shape == (0,))
    while still_timesteps < 5000 and wait_timesteps < 20000:
        env.step()
        env.render()
        cur_qpos = env.get_object_qpos()
        cur_root_pose = env.get_object_root_pose()

        invalid_contact = False
        for c in env.scene.get_contacts():
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    invalid_contact = True
                    break
            if invalid_contact:
                break
        if (no_qpos or np.max(np.abs(cur_qpos - las_qpos)) < 1e-5) and \
           (np.max(np.abs(cur_root_pose.p - las_root_pose.p)) < 1e-5) and \
           (np.max(np.abs(cur_root_pose.q - las_root_pose.q)) < 1e-5) and (not invalid_contact):
            still_timesteps += 1
        else:
            still_timesteps = 0
        las_qpos = cur_qpos
        las_root_pose = cur_root_pose
        wait_timesteps += 1
        if visu and wait_timesteps % 200 == 0:
            rgb_pose, _ = cam.get_observation()
            fimg = (rgb_pose * 255).astype(np.uint8)
            fimg = Image.fromarray(fimg)
            for idx in range(5):
                imgs.append(fimg)
    if visu:
        return still_timesteps, imgs
    else:
        return still_timesteps


def get_shape_list(all_categories, mode='train', primact_type='push'):
    train_file_dir = "../stats/train_where2actPP_train_data_list.txt"
    val_file_dir = "../stats/train_where2actPP_test_data_list.txt"
    if primact_type == 'pickup':
        train_file_dir = "../stats/train_where2actPP_train_data_list_pickup.txt"
        val_file_dir = "../stats/train_where2actPP_val_data_list_pickup.txt"
    cat_list = all_categories.split(',')

    train_shape_list, val_shape_list = [], []
    val_cat_shape_id_dict, train_cat_shape_id_dict = {}, {}
    shape_cat_dict = {}

    for cat in cat_list:
        train_cat_shape_id_dict[cat] = []
        val_cat_shape_id_dict[cat] = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in cat_list:
                continue
            train_shape_list.append(shape_id)
            train_cat_shape_id_dict[cat].append(shape_id)
            shape_cat_dict[shape_id] = cat

    with open(val_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in cat_list:
                continue
            val_shape_list.append(shape_id)
            val_cat_shape_id_dict[cat].append(shape_id)
            shape_cat_dict[shape_id] = cat

    if mode == 'train':
        return cat_list, train_shape_list, shape_cat_dict, train_cat_shape_id_dict
    elif mode == 'val':
        return cat_list, val_shape_list, shape_cat_dict, val_cat_shape_id_dict
    elif mode == 'all':
        all_shape_list = train_shape_list + val_shape_list
        all_cat_shape_id_list = {}
        for cat in cat_list:
            all_cat_shape_id_list[cat] = train_cat_shape_id_dict[cat] + val_cat_shape_id_dict[cat]
        return cat_list, all_shape_list, shape_cat_dict, all_cat_shape_id_list



def draw_affordance_map(fn, cam2cambase, mat44, pcs, pred_aff_map, coordinate_system='cambase', ctpt1=None, ctpt2=None, type='0'):
    print('fn: ', fn)

    if coordinate_system == 'cambase':
        pc_camera = pcs @ np.linalg.inv(np.transpose(cam2cambase, (1, 0)))
        ctpt1 = ctpt1 @ np.linalg.inv(np.transpose(cam2cambase, (1, 0)))
        if type == '2':
            ctpt2 = ctpt2 @ np.linalg.inv(np.transpose(cam2cambase, (1, 0)))
    elif coordinate_system == 'world':
        pc_camera = (np.linalg.inv(mat44[:3, :3]) @ pcs.T).T
        ctpt1 = (np.linalg.inv(mat44[:3, :3]) @ ctpt1.T).T
        if type == '2':
            ctpt2 = (np.linalg.inv(mat44[:3, :3]) @ ctpt2.T).T
    else:
        pc_camera = pcs

    if type== '1' or type == '2':
        ctpt1s = []
        for k in range(300):  # jitter
            cur_pt = np.zeros(3)
            cur_pt[0] = ctpt1[0] + np.random.random() * 0.02 - 0.01
            cur_pt[1] = ctpt1[1] + np.random.random() * 0.02 - 0.01
            cur_pt[2] = ctpt1[2] + np.random.random() * 0.02 - 0.01
            ctpt1s.append(cur_pt)
        ctpt1s = np.array(ctpt1s)
        ctpt1s_color = np.ones(300)

    if type == '2':
        ctpt2s = []
        for k in range(300):  # jitter
            cur_pt = np.zeros(3)
            cur_pt[0] = ctpt2[0] + np.random.random() * 0.02 - 0.01
            cur_pt[1] = ctpt2[1] + np.random.random() * 0.02 - 0.01
            cur_pt[2] = ctpt2[2] + np.random.random() * 0.02 - 0.01
            ctpt2s.append(cur_pt)
        ctpt2s = np.array(ctpt2s)
        ctpt2s_color = np.zeros(300)

    if type == '0':
        render_pts_label_png(fn, pc_camera, pred_aff_map)
    elif type == '1':
        render_pts_label_png(fn, np.concatenate([pc_camera, ctpt1s]), np.concatenate([pred_aff_map, ctpt1s_color]))
    elif type == '2':
        render_pts_label_png(fn, np.concatenate([pc_camera, ctpt1s, ctpt2s]), np.concatenate([pred_aff_map, ctpt1s_color, ctpt2s_color]))
    else:
        pass


def coordinate_transform(item, is_pc, transform_type='cambase2world', mat44=None, cam2cambase=None, pc_center=None):
    if transform_type == 'cam2world':
        transformed_item = (mat44[:3, :3] @ item.T).T
    elif transform_type == 'world2cam':
        transformed_item = (np.linalg.inv(mat44[:3, :3]) @ item.T).T
    elif transform_type == 'cam2cambase':
        transformed_item = item @ np.transpose(cam2cambase, (1, 0))
        if is_pc:
            transformed_item -= pc_center
    elif transform_type == 'cambase2cam':
        if is_pc:
            transformed_item = item + pc_center
        else:
            transformed_item = item
        transformed_item = transformed_item @ np.linalg.inv(np.transpose(cam2cambase, (1, 0)))

    return transformed_item


def batch_coordinate_transform(batch, is_pc, transform_type='cambase2world', mat44=None, cam2cambase=None, pc_center=None):
    transformed_batch = []
    for idx in range(len(batch)):
        transformed_item = coordinate_transform(batch[idx], is_pc[idx], transform_type, mat44, cam2cambase, pc_center)
        transformed_batch.append(transformed_item)
    return transformed_batch


def get_data_info(cur_data, cur_type='type0', exchange_ctpts=False, given_task=None):
    cur_dir, shape_id, category, \
    pixel1_idx1, contact_point1, gripper_up1, gripper_forward1, \
    pixel1_idx2, contact_point2, gripper_up2, gripper_forward2, \
    task, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc_center,  \
    pixel_ids, target_link_mat44, target_part_trans, transition, \
    contact_point_world1, gripper_up_world1, gripper_forward_world1, \
    contact_point_world2, gripper_up_world2, gripper_forward_world2 = cur_data

    if cur_type == 'type0':     # succ
        pass
    elif cur_type == 'type1':   # succ2fail
        task = given_task
        success = False
    elif cur_type == 'type2':   # fail
        pass
    elif cur_type == 'type3':   # invalid
        task = given_task
    elif cur_type == 'type4':
        pass

    if exchange_ctpts:
        return (cur_dir, shape_id, category,
                pixel1_idx2, contact_point2, gripper_up2, gripper_forward2,
                pixel1_idx1, contact_point1, gripper_up1, gripper_forward1,
                task, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc_center,
                pixel_ids, target_link_mat44, target_part_trans, transition,
                contact_point_world2, gripper_up_world2, gripper_forward_world2,
                contact_point_world1, gripper_up_world1, gripper_forward_world1,)
    else:
        return (cur_dir, shape_id, category,
                pixel1_idx1, contact_point1, gripper_up1, gripper_forward1,
                pixel1_idx2, contact_point2, gripper_up2, gripper_forward2,
                task, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc_center,
                pixel_ids, target_link_mat44, target_part_trans, transition,
                contact_point_world1, gripper_up_world1, gripper_forward_world1,
                contact_point_world2, gripper_up_world2, gripper_forward_world2)


