import os
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
from sapien.core import Pose, ArticulationJointType
from env import Env, ContactError, SVDError
from camera import Camera
from robots.panda_robot import Robot
import random
import imageio
import math


parser = ArgumentParser()
parser.add_argument('--file_dir', type=str)
parser.add_argument('--file_id', type=str)
parser.add_argument('--repeat_id', type=int)

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--stiffness', type=int, default=10)
parser.add_argument('--damping', type=int, default=100)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--static_friction', type=float, default=4.0)
parser.add_argument('--dynamic_friction', type=float, default=4.0)
parser.add_argument('--scene_static_friction', type=float, default=0.30)
parser.add_argument('--scene_dynamic_friction', type=float, default=0.30)
parser.add_argument('--gripper_scale', type=float, default=2.0)

parser.add_argument('--position_world1', type=str)
parser.add_argument('--up_world1', type=str)
parser.add_argument('--forward_world1', type=str)
parser.add_argument('--position_world2', type=str)
parser.add_argument('--up_world2', type=str)
parser.add_argument('--forward_world2', type=str)

parser.add_argument('--task_threshold', type=int, default=15)
parser.add_argument('--euler_threshold', type=int, default=5)
parser.add_argument('--check_contactError', action='store_true', default=False)
parser.add_argument('--out_dir', type=str, default='/media/wuruihai/sixt/2gripper_logs/visu/BEGIN')
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()



file_id = args.file_id
repeat_id = args.repeat_id
out_dir = os.path.join(args.out_dir, args.file_dir.split('/')[-2])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if args.save_results and (not os.path.exists(os.path.join(out_dir, 'succ_files'))):
    os.makedirs(os.path.join(out_dir, 'succ_files')), os.makedirs(os.path.join(out_dir, 'succ_gif'))
    os.makedirs(os.path.join(out_dir, 'fail_files')), os.makedirs(os.path.join(out_dir, 'fail_gif'))
    os.makedirs(os.path.join(out_dir, 'invalid_files')), os.makedirs(os.path.join(out_dir, 'invalid_gif'))


with open(os.path.join(args.file_dir, 'result_%s.json' % args.file_id), 'r') as fin:
    result_data = json.load(fin)

shape_id = result_data['shape_id']
category = result_data['category']
primact_type = result_data['primact_type']
random_seed = result_data['random_seed']
np.random.seed(random_seed)

smaller_categories = ['Pliers', 'Kettle', 'Pen', 'Remote', 'Bowl']
medium_categories = ['KitchenPot', 'Toaster', 'Basket', 'Bucket', 'Dishwasher']
ShapeNet_categories = ['Bench', 'Sofa', 'Bowl', 'Basket', 'Keyboard2', 'Jar', 'Chair2']

if primact_type in ['pushing', 'rotating', 'topple']:
    if category in smaller_categories:
        args.object_scale = 0.5
    elif category in medium_categories:
        args.object_scale = 0.75
    else:
        args.object_scale = 1.0

elif primact_type in ['pickup']:
    cs_dict = json.load(open("./cate_setting.json", "r"))[primact_type]
    args.object_scale = cs_dict["cate_setting"][category]["obj_scale"]
    args.urdf_type = cs_dict["cate_setting"][category]["urdf"]
    args.density = cs_dict["density"]
    args.stiffness = cs_dict["stiffness"]
    args.damping = cs_dict["damping"]
    args.start_dist = cs_dict["start_dist"]
    args.final_dist = cs_dict["final_dist"]
    args.move_steps = cs_dict["move_steps"]
    args.wait_steps = cs_dict["wait_steps"]
    args.dynamic_friction = cs_dict["material_dynamic_friction"]
    args.static_friction = cs_dict["material_static_friction"]
    args.scene_dynamic_friction = cs_dict["scene_dynamic_friction"]
    args.scene_static_friction = cs_dict["scene_static_friction"]
    args.gripper_scale = cs_dict["gripper_scale"]



def save_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    utils.save_data(os.path.join(out_dir, 'succ_files'), file_id, out_info, cam_XYZA_list, gt_target_link_mask, repeat_id=repeat_id)
    imageio.mimsave(os.path.join(out_dir, 'succ_gif', '%s_%d_%s_%s.gif' % (file_id, repeat_id, category, shape_id)), gif_imgs)

def save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs):
    utils.save_data(os.path.join(out_dir, 'fail_files'), file_id, out_info, cam_XYZA_list, gt_target_link_mask, repeat_id=repeat_id)
    imageio.mimsave(os.path.join(out_dir, 'fail_gif', '%s_%d_%s_%s.gif' % (file_id, repeat_id, category, shape_id)), gif_imgs)

def save_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg):
    utils.save_data(os.path.join(out_dir, 'invalid_files'), file_id, out_info, cam_XYZA_list, gt_target_link_mask, repeat_id=repeat_id)
    fimg.save(os.path.join(out_dir, 'invalid_gif', '%s_%d_%s_%s.png' % (file_id, repeat_id, category, shape_id)))




out_info = dict()
gif_imgs = []
fimg = None
success = False


# setup env
print("creating env")
env = Env(show_gui=(not args.no_gui), set_ground=True,
          static_friction=args.scene_static_friction, dynamic_friction=args.scene_dynamic_friction)

# setup camera
cam_theta, cam_phi = result_data['camera_metadata']['theta'], result_data['camera_metadata']['phi']
cam = Camera(env, phi=cam_phi, theta=cam_theta, random_position=False, restrict_dir=True)  # [0.5π, 1.5π]
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
print("camera created")



if primact_type in ['pushing', 'rotating', 'topple']:
    if category not in ShapeNet_categories:
        object_urdf_fn = '../../Sapien_dataset/dataset/%s/mobility.urdf' % str(shape_id)
    else:
        object_urdf_fn = '../../Sapien_dataset/dataset2/%s/mobility_vhacd.urdf' % str(shape_id)
    object_material = env.get_material(args.static_friction, args.dynamic_friction, 0.01)

elif primact_type in ['pickup']:
    if args.urdf_type == "removed":
        object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd_rm.urdf" % str(shape_id)
        print(">>> Removed parts! Shape_id: ", shape_id)
    elif args.urdf_type == "fixbase":
        object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd_fixbase.urdf" % str(shape_id)
        print(">>> Fixbase! Shape_id: ", shape_id)
    elif args.urdf_type == "processed":
        object_urdf_fn = "../../Sapien_dataset/dataset/%s/mobility_vhacd.urdf" % str(shape_id)
        print(">>> Processed! Shape_id: ", shape_id)
    elif args.urdf_type == "shapenet":
        object_urdf_fn = "../../Sapien_dataset/dataset2/%s/mobility_vhacd.urdf" % str(shape_id)
        print(">>> ShapeNet! Shape_id: ", shape_id)
    else:
        object_urdf_fn = '../../Sapien_dataset/dataset/%s/mobility.urdf' % str(shape_id)
        print(">>> Origin! Shape_id: ", shape_id)
    object_material = env.get_material(args.static_friction, args.dynamic_friction, 0.01)



joint_angles = result_data['joint_angles']
joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, target_part_id=-1, scale=args.object_scale,
                               density=args.density, stiffness=args.stiffness, damping=args.damping, given_joint_angles=joint_angles)

# wait for the object's still
still_timesteps = utils.wait_for_object_still(env)


### use the GT vision
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

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
if not args.no_gui:
    env.wait_to_start()
    pass

out_info['random_seed'] = random_seed
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
out_info['target_link_mat44'] = target_link_mat44.tolist()
out_info['prev_origin_world'] = prev_origin_world.tolist()
out_info['obj_pose_p'] = obj_pose.p.tolist()
out_info['obj_pose_q'] = obj_pose.q.tolist()
out_info['success'] = 'False'
out_info['result'] = 'VALID'

if primact_type in ['pushing', 'topple', 'pickup']:
    if primact_type == 'topple':
        traj = np.array(result_data['trajectory'], dtype=np.float32)
        task = traj / np.linalg.norm(traj)
    else:
        task = np.array(result_data['traj_dir'], dtype=np.float32)

elif primact_type in ['rotating']:
    task = np.array(result_data['beta'], dtype=np.float32)
out_info['task'] = task.tolist()


position_world1 = np.array([float(i) for i in args.position_world1[3:].split(',')], dtype=np.float32)
up_world1 = np.array([float(i) for i in args.up_world1[3:].split(',')], dtype=np.float32)
forward_world1 = np.array([float(i) for i in args.forward_world1[3:].split(',')], dtype=np.float32)
position_world2 = np.array([float(i) for i in args.position_world2[3:].split(',')], dtype=np.float32)
up_world2 = np.array([float(i) for i in args.up_world2[3:].split(',')], dtype=np.float32)
forward_world2 = np.array([float(i) for i in args.forward_world2[3:].split(',')], dtype=np.float32)

start_pose1, start_rotmat1, final_pose1, final_rotmat1 = utils.get_rotmat(cam, position_world1, up_world1, forward_world1,
                                                                          number='1', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist)
start_pose2, start_rotmat2, final_pose2, final_rotmat2 = utils.get_rotmat(cam, position_world2, up_world2, forward_world2,
                                                                          number='2', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist)



# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(args.static_friction, args.dynamic_friction, 0.01)

open_gripper = True if primact_type in ['pickup'] else False
robot1 = Robot(env, robot_urdf_fn, robot_material, open_gripper=open_gripper, scale=args.gripper_scale)
robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=open_gripper, scale=args.gripper_scale)


try:
    try:
        if args.check_contactError:
            env.dual_start_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id, robot2.gripper_actor_ids, True)

        robot1.robot.set_root_pose(start_pose1)
        robot2.robot.set_root_pose(start_pose2)

        # save img
        env.render()
        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose * 255).astype(np.uint8)
        fimg = Image.fromarray(fimg)

        env.step()

    except Exception:
        print('contact error')
        out_info['result'] = 'INVALID'
        if args.save_results:
            save_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg)
        env.scene.remove_articulation(env.object)
        env.scene.remove_articulation(robot1.robot)
        env.scene.remove_articulation(robot2.robot)
        env.close()
        exit(2)

    if args.check_contactError:
        env.dual_end_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id, robot2.gripper_actor_ids, False)

    if primact_type in ['pushing', 'rotating', 'topple']:
        robot1.close_gripper()
        robot2.close_gripper()
    elif primact_type in ['pickup']:
        robot1.open_gripper()
        robot2.open_gripper()
    env.step()
    env.render()

    imgs = utils.dual_gripper_move_to_target_pose(robot1, robot2, final_rotmat1, final_rotmat2, num_steps=args.move_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    imgs = utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)

    if primact_type in ['pickup']:
        robot1.close_gripper()
        robot2.close_gripper()
        utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
        imgs = utils.dual_gripper_move_to_target_pose(robot1, robot2, start_rotmat1, start_rotmat2, num_steps=args.move_steps, cam=cam, vis_gif=True)
        gif_imgs.extend(imgs)
        imgs = utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
        gif_imgs.extend(imgs)


except Exception:
    print("Contact Error!")
    out_info['result'] = 'INVALID'
    if args.save_results:
        save_invalid_data(out_info, cam_XYZA_list, gt_target_link_mask, fimg)
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot1.robot)
    env.scene.remove_articulation(robot2.robot)
    env.close()
    exit(2)


''' check success '''

next_obj_pose = env.get_target_part_pose()
target_part_trans = env.get_target_part_pose().to_transformation_matrix()
transition = np.linalg.inv(target_part_trans) @ target_link_mat44
alpha, beta, gamma = utils.rotationMatrixToEulerAngles(transition)  # eulerAngles(trans) = eulerAngles(prev_mat44) - eulerAngle(then_mat44)
out_info['target_part_trans'] = target_part_trans.tolist()
out_info['transition'] = transition.tolist()
out_info['alpha'] = alpha.tolist()
out_info['beta'] = beta.tolist()
out_info['gamma'] = gamma.tolist()
out_info['next_obj_pose_p'] = next_obj_pose.p.tolist()
out_info['next_obj_pose_q'] = next_obj_pose.q.tolist()
print('alpha, beta, gamma: ', alpha, beta, gamma)

# calculate displacement
next_origin_world_xyz1 = target_part_trans @ np.array([0, 0, 0, 1])
next_origin_world = next_origin_world_xyz1[:3]
trajectory = next_origin_world - prev_origin_world
success, div_error, traj_len, traj_dir = utils.check_success(trajectory, alpha, beta, gamma, primact_type, out_info, threshold=args.euler_threshold, grip_dir1=up_world1, grip_dir2=up_world2)
out_info['success'] = 'True' if success else 'False'
out_info['trajectory'] = trajectory.tolist()
out_info['traj_len'] = traj_len.tolist()
if primact_type in ['pushing', 'topple', 'pickup', 'pulling']:
    out_info['traj_dir'] = traj_dir.tolist()


if div_error:
    print('NAN!')
    if args.save_results:
        save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot1.robot)
    env.scene.remove_articulation(robot2.robot)
    env.close()
    exit(1)


env.scene.remove_articulation(robot1.robot)
env.scene.remove_articulation(robot2.robot)
env.scene.remove_articulation(env.object)
env.close()


if success:
    if primact_type in ['pushing', 'topple', 'pickup']:
        included_angle, _ = utils.cal_included_angle(task, traj_dir)
        out_info['included_angle'] = included_angle.tolist()
    elif 'rotating' in primact_type:
        included_angle = np.abs(task - beta)
        out_info['included_angle'] = included_angle.tolist()

    if primact_type in ['pushing', 'topple', 'pickup']:
        if included_angle < args.task_threshold:
            if args.save_results:
                save_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
            exit(0)
        else:
            if args.save_results:
                save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
            exit(1)
    elif 'rotating' in primact_type:
        if (task * beta > 0) and included_angle < args.task_threshold:
            if args.save_results:
                save_succ_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
            exit(0)
        else:
            if args.save_results:
                save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
            exit(1)

else:
    if args.save_results:
        save_fail_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs)
    exit(1)
