import os
import numpy as np
from PIL import Image
import utils
from argparse import ArgumentParser
from sapien.core import Pose, ArticulationJointType
from env import Env, ContactError, SVDError
from camera import Camera
from robots.panda_robot import Robot
import random
import imageio
import json


parser = ArgumentParser()
parser.add_argument('--trial_id', type=int)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--gripper_id', type=int, default=0)

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--threshold', type=int, default=3)

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()



trial_id = args.trial_id
gripper_id = args.gripper_id
out_dir = args.out_dir

with open(os.path.join(args.out_dir, 'tmp_succ_files', 'result_%d.json' % args.trial_id), 'r') as fin:
    result_data = json.load(fin)

shape_id = result_data['shape_id']
category = result_data['category']
primact_type = result_data['primact_type']


if args.random_seed is not None:
    np.random.seed(args.random_seed)


def save_singe_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    # utils.save_data(os.path.join(out_dir, 'tmp_succ_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
    imageio.mimsave(os.path.join(out_dir, 'tmp_succ_gif', '%d_%s_%s_succ_single%d.gif' % (trial_id, category, shape_id, gripper_id)), gif_imgs)

def save_single_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    # utils.save_data(os.path.join(out_dir, 'fail_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
    imageio.mimsave(os.path.join(out_dir, 'tmp_succ_gif', '%d_%s_%s_fail_single%d.gif' % (trial_id, category, shape_id, gripper_id)), gif_imgs)

def save_single_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg):
    # utils.save_data(os.path.join(out_dir, 'invalid_files'), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
    fimg.save(os.path.join(out_dir, 'tmp_succ_gif', '%d_%s_%s_invalid_single%d.png' % (trial_id, category, shape_id, gripper_id)))







out_info = dict()
gif_imgs = []
fimg = None
success = False


# setup env
print("creating env")
env = Env(show_gui=(not args.no_gui), set_ground=True)

# setup camera
cam_theta, cam_phi = result_data['camera_metadata']['theta'], result_data['camera_metadata']['phi']
cam = Camera(env, phi=cam_phi, theta=cam_theta, random_position=False, restrict_dir=True)  # [0.5π, 1.5π]
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
print("camera created")


smaller_categories = ['Pliers', 'Kettle', 'Pen', 'Remote', 'Bowl']
medium_categories = ['KitchenPot', 'Toaster', 'Basket', 'Bucket', 'Dishwasher']
ShapeNet_categories = ['Bench', 'Sofa', 'Bowl', 'Basket', 'Keyboard2', 'Jar', 'Chair2']


if category not in ShapeNet_categories:
    object_urdf_fn = '../data/dataset/%s/mobility.urdf' % str(shape_id)
else:
    object_urdf_fn = '../data/dataset2/%s/mobility_vhacd.urdf' % str(shape_id)
object_material = env.get_material(4, 4, 0.01)

joint_angles = result_data['joint_angles']
if category in smaller_categories:
    joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, target_part_id=-1, scale=0.5, density=args.density, damping=args.damping, given_joint_angles=joint_angles)
elif category in medium_categories:
    joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, target_part_id=-1, scale=0.75, density=args.density, damping=args.damping, given_joint_angles=joint_angles)
else:
    joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, target_part_id=-1, scale=1.0, density=args.density, damping=args.damping, given_joint_angles=joint_angles)

# wait for the object's still
still_timesteps = utils.wait_for_object_still(env)


### use the GT vision
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

# pc, pc_centers = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 'world', mat44=np.array(cam.get_metadata_json()['mat44'], dtype=np.float32))
# pc = pc.detach().cpu().numpy().reshape(-1, 3)
object_movable_link_ids = env.movable_link_ids
object_all_link_ids = env.all_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_movable_link_mask(object_all_link_ids)  # (448, 448), 0(unmovable) - id(all)

# sample a pixel on target part
x, y = result_data['pixel_locs'][0], result_data['pixel_locs'][1]
xs, ys = np.where(gt_all_link_mask > 0)
target_part_id = object_all_link_ids[gt_all_link_mask[x, y] - 1]
env.set_target_object_part_actor_id2(target_part_id)  # for get_target_part_pose
gt_target_link_mask = cam.get_movable_link_mask([target_part_id])

# calculate position    world = trans @ local
target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()  # local2world
prev_origin_world_xyz1 = target_link_mat44 @ np.array([0, 0, 0, 1])
prev_origin_world = prev_origin_world_xyz1[:3]
obj_pose = env.get_target_part_pose()


env.render()
idx1, idx2 = result_data['pixel1_idx'], result_data['pixel2_idx']
x1, y1 = xs[idx1], ys[idx1]
x2, y2 = xs[idx2], ys[idx2]

# move back
env.render()
if not args.no_gui:
    env.wait_to_start()
    pass

out_info['random_seed'] = args.random_seed
out_info['pixel_locs'] = [int(x), int(y)]
out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
out_info['camera_metadata'] = cam.get_metadata_json()
out_info['object_state'] = args.target_part_state
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
out_info['shape_id'] = shape_id
out_info['category'] = category
out_info['primact_type'] = primact_type
out_info['pixel1_idx'] = int(idx1)
out_info['pixel2_idx'] = int(idx2)
out_info['target_link_mat44'] = target_link_mat44.tolist()
out_info['prev_origin_world'] = prev_origin_world.tolist()
out_info['obj_pose_p'] = obj_pose.p.tolist()
out_info['obj_pose_q'] = obj_pose.q.tolist()
out_info['success'] = 'False'
out_info['result'] = 'VALID'


if args.gripper_id == 0:
    given_up = np.array(result_data['gripper_direction_world1'], dtype=np.float32)
    given_forward = np.array(result_data['gripper_forward_direction_world1'], dtype=np.float32)
    start_pose, start_rotmat, final_pose, final_rotmat, _, _ = \
        utils.cal_final_pose(cam, cam_XYZA, x1, y1, number='1', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist, given=True, given_up=given_up, given_forward=given_forward)
elif args.gripper_id == 1:
    given_up = np.array(result_data['gripper_direction_world2'], dtype=np.float32)
    given_forward = np.array(result_data['gripper_forward_direction_world2'], dtype=np.float32)
    start_pose, start_rotmat, final_pose, final_rotmat, _, _ = \
        utils.cal_final_pose(cam, cam_XYZA, x2, y2, number='2', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist, given=True, given_up=given_up, given_forward=given_forward)

# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot_scale = 2 if 'pickup' in primact_type else 2

robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=False, scale=robot_scale)

try:
    robot.robot.set_root_pose(start_pose)
    env.render()
    rgb_pose, _ = cam.get_observation()
    fimg = (rgb_pose * 255).astype(np.uint8)
    fimg = Image.fromarray(fimg)

    env.step()

    robot.close_gripper()
    env.step()
    env.render()

    imgs = robot.move_to_target_pose(final_rotmat, num_steps=args.move_steps, vis_gif=True, cam=cam)
    gif_imgs.extend(imgs)
    imgs = robot.wait_n_steps(n=args.wait_steps, vis_gif=True, cam=cam)
    gif_imgs.extend(imgs)

except Exception:
    print("Contact Error!")
    out_info['result'] = 'INVALID'
    save_single_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg)
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot.robot)
    env.close()
    exit(2)

''' check success '''

next_obj_pose = env.get_target_part_pose()
target_part_trans = env.get_target_part_pose().to_transformation_matrix()  # 得到世界坐标系 -> target part坐标系的 transformation matrix 4*4 SE3
transition = np.linalg.inv(target_part_trans) @ target_link_mat44
alpha, beta, gamma = utils.rotationMatrixToEulerAngles(transition)  # eulerAngles(trans) = eulerAngles(prev_mat44) - eulerAngle(then_mat44)
out_info['target_part_trans'] = target_part_trans.tolist()
out_info['transition'] = transition.tolist()
out_info['alpha'] = alpha.tolist()
out_info['beta'] = beta.tolist()
out_info['gamma'] = gamma.tolist()
out_info['next_obj_pose_p'] = next_obj_pose.p.tolist()
out_info['next_obj_pose_q'] = next_obj_pose.q.tolist()

# calculate displacement
next_origin_world_xyz1 = target_part_trans @ np.array([0, 0, 0, 1])
next_origin_world = next_origin_world_xyz1[:3]
trajectory = next_origin_world - prev_origin_world
success, div_error, traj_len, traj_dir = utils.check_success(trajectory, alpha, beta, gamma, primact_type, out_info, threshold=args.threshold)
out_info['success'] = 'True' if success else 'False'
out_info['trajectory'] = trajectory.tolist()
out_info['traj_len'] = traj_len.tolist()
if primact_type == 'pushing' or primact_type == 'topple':
    out_info['traj_dir'] = traj_dir.tolist()



if div_error:
    print('NAN!')
    save_single_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg)
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot.robot)
    env.close()
    exit(2)


env.scene.remove_articulation(robot.robot)
env.scene.remove_articulation(env.object)
env.close()


if success:
    if 'pushing' in primact_type:
        task = np.array(result_data['traj_dir'], dtype=np.float32)
        included_angle, _ = utils.cal_included_angle(task, traj_dir)
    elif 'rotating' in primact_type:
        task = np.array(result_data['beta'], dtype=np.float32)
        included_angle = np.abs(task - beta)
    elif 'topple' in primact_type:
        task = np.array(result_data['traj_dir'], dtype=np.float32)
        included_angle, _ = utils.cal_included_angle(task, traj_dir)

    if included_angle < 5:
        save_singe_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
        exit(0)
    else:
        save_single_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
        exit(1)
else:
    save_single_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
    exit(1)


