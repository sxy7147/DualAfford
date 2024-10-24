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


parser = ArgumentParser()
parser.add_argument('--trial_id', type=int)
parser.add_argument('--shape_id', type=str)
parser.add_argument('--category', type=str)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)

parser.add_argument('--density', type=float, default=1.0)
parser.add_argument('--damping', type=int, default=1.0)
parser.add_argument('--target_part_state', type=str, default='closed')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)
parser.add_argument('--threshold', type=int, default=3)

parser.add_argument('--save_data', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')

args = parser.parse_args()



shape_id = args.shape_id
category = args.category
trial_id = args.trial_id
primact_type = args.primact_type
out_dir = args.out_dir
if args.random_seed is not None:
    np.random.seed(args.random_seed)


        
def save_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs, result):
    if args.save_data:
        utils.save_data(os.path.join(out_dir, '%s_files' % result), trial_id, out_info, cam_XYZA_list, gt_target_link_mask)
        imageio.mimsave(os.path.join(out_dir, '%s_gif' % result, '%d_%s_%s.gif' % (trial_id, category, shape_id)), gif_imgs)



out_info = dict()
gif_imgs = []
fimg = None
success = False


# setup env
print("creating env")
env = Env(show_gui=(not args.no_gui), set_ground=True, static_friction=4.0, dynamic_friction=4.0)

# setup camera
cam = Camera(env, random_position=True, restrict_dir=True)  # [0.5π, 1.5π]
print("camera created")
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)


smaller_categories = ['Pliers', 'Kettle', 'Pen', 'Remote', 'Bowl', 'USB']
medium_categories = ['KitchenPot', 'Toaster', 'Basket', 'Bucket', 'Dishwasher']
ShapeNet_categories = ['Bench', 'Sofa', 'Bowl', 'Basket', 'Keyboard2', 'Jar', 'Chair2']


if category not in ShapeNet_categories:
    object_urdf_fn = '../../dataset/%s/mobility.urdf' % str(shape_id)
else:
    # object_urdf_fn = '../../dataset2/%s/mobility_vhacd.urdf' % str(shape_id)
    object_urdf_fn = '../../dataset2/%s/mobility.urdf' % str(shape_id)
object_material = env.get_material(4, 4, 0.01)
try:
    if category in smaller_categories:
        scale = 0.2
    elif category in medium_categories:
        scale = 0.75
    else:
        scale = 1.0
    joint_angles = env.load_object(object_urdf_fn, object_material, state=args.target_part_state, target_part_id=-1, scale=scale, density=args.density, damping=args.damping)
    print('joint_angles', joint_angles)
except Exception:
    print('error while load object')
    env.close()
    exit(3)


# wait for the object's still
still_timesteps = utils.wait_for_object_still(env)

# save the gif
# still_imgs = []
# still_timesteps, imgs = utils.wait_for_object_still(env, cam=cam, visu=True)
# still_imgs.extend(imgs)
# # if still_timesteps < 5000:
# imageio.mimsave(os.path.join(args.out_dir, 'wait_gif', '%d_%s_%s.gif' % (trial_id, category, shape_id)), still_imgs)

if still_timesteps < 5000:
    print('Object Not Still!')
    env.scene.remove_articulation(env.object)
    env.close()
    exit(3)


### use the GT vision
rgb, depth = cam.get_observation()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
cam_XYZA_list = [cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, cam_XYZA]

# pc, pc_centers = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 'world', mat44=np.array(cam.get_metadata_json()['mat44'], dtype=np.float32))
# pc = pc.detach().cpu().numpy().reshape(-1, 3)
gt_movable_link_mask = cam.get_link_mask(env.movable_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_fixed_link_mask = cam.get_link_mask(env.fixed_link_ids)  # (448, 448), 0(unmovable) - id(movable)
gt_all_link_mask = cam.get_link_mask(env.all_link_ids)  # (448, 448), 0(unmovable) - id(all)

# sample a pixel on target part
xs, ys = np.where(gt_all_link_mask > 0)
if len(xs) == 0:
    env.scene.remove_articulation(env.object)
    env.close()
    print('can not find any points in the scene')
    exit(3)


target_joint_type = ArticulationJointType.FIX
tot_trial = 0
while tot_trial < 50:
    idx = np.random.randint(len(xs))
    x, y = xs[idx], ys[idx]
    target_part_id = env.all_link_ids[gt_all_link_mask[x, y] - 1]
    env.set_target_object_part_actor_id2(target_part_id)
    if env.target_object_part_joint_type == target_joint_type:
        break 
    else:
        tot_trial += 1
if env.target_object_part_joint_type != target_joint_type:
    env.scene.remove_articulation(env.object)
    env.close()
    print('can not find proper points for the first gripper')
    exit(3)
gt_target_link_mask = cam.get_link_mask([target_part_id])

# calculate position    world = trans @ local
target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()  # local2world
prev_origin_world_xyz1 = target_link_mat44 @ np.array([0, 0, 0, 1])
prev_origin_world = prev_origin_world_xyz1[:3]
obj_pose = env.get_target_part_pose()


# env.render()
# idx1, idx2 = 0, 0
# while idx1 == idx2:
#     idx1, idx2 = np.random.randint(len(xs)), np.random.randint(len(xs))
# x1, y1 = xs[idx1], ys[idx1]
# x2, y2 = xs[idx2], ys[idx2]
xs1, ys1 = np.where(gt_fixed_link_mask > 0)
idx1 = np.random.randint(len(xs1))
x1, y1 = xs[idx1], ys[idx1]


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
# out_info['pixel1_idx'] = int(idx1)
# out_info['pixel2_idx'] = int(idx2)
out_info['target_link_mat44'] = target_link_mat44.tolist()
out_info['prev_origin_world'] = prev_origin_world.tolist()
out_info['obj_pose_p'] = obj_pose.p.tolist()
out_info['obj_pose_q'] = obj_pose.q.tolist()
out_info['success'] = 'False'
# out_info['success_bd'] = 'False'
out_info['result'] = 'VALID'

start_pose1, start_rotmat1, final_pose1, final_rotmat1, _, _ = utils.cal_final_pose(cam, cam_XYZA, x1, y1, number='1', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist)
# start_pose2, start_rotmat2, final_pose2, final_rotmat2, _, _ = utils.cal_final_pose(cam, cam_XYZA, x2, y2, number='2', out_info=out_info, start_dist=args.start_dist, final_dist=args.final_dist)

# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot_scale = 3

robot1 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)
# robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)

try:

    try:
        # activate contact checking
        # env.dual_start_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id, robot2.gripper_actor_ids, True)
        env.start_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, True)

        robot1.robot.set_root_pose(start_pose1)
        # robot2.robot.set_root_pose(start_pose2)
        robot1.open_gripper()

        # save img
        env.render()
        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose * 255).astype(np.uint8)
        fimg = Image.fromarray(fimg)

        env.step()

    except Exception:
        print('contact error')
        out_info['result'] = 'INVALID'
        save_data(out_info, cam_XYZA_list, gt_target_link_mask, [fimg], 'invalid')
        env.scene.remove_articulation(env.object)
        env.scene.remove_articulation(robot1.robot)
        # env.scene.remove_articulation(robot2.robot)
        env.close()
        exit(2)

    # env.dual_end_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, robot2.hand_actor_id, robot2.gripper_actor_ids, False)
    env.end_checking_contact(robot1.hand_actor_id, robot1.gripper_actor_ids, False)

    env.step()
    env.render()

    # imgs = utils.dual_gripper_move_to_target_pose(robot1, robot2, final_rotmat1, final_rotmat2, num_steps=args.move_steps, cam=cam, vis_gif=True)
    imgs = robot1.move_to_target_pose(final_rotmat1, num_steps=args.move_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    # imgs = utils.dual_gripper_wait_n_steps(robot1, robot2, n=args.wait_steps, cam=cam, vis_gif=True)
    imgs = robot1.wait_n_steps(n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)

    robot1.close_gripper()
    imgs = robot1.wait_n_steps(n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)

    imgs = robot1.move_to_target_pose(start_rotmat1, num_steps=args.move_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)
    imgs = robot1.wait_n_steps(n=args.wait_steps, cam=cam, vis_gif=True)
    gif_imgs.extend(imgs)


except Exception:
    print("Contact Error!")
    out_info['result'] = 'INVALID'
    save_data(out_info, cam_XYZA_list, gt_target_link_mask, [fimg], 'invalid')
    env.scene.remove_articulation(env.object)
    env.scene.remove_articulation(robot1.robot)
    # env.scene.remove_articulation(robot2.robot)
    env.close()
    exit(2)
   
# TODO 
robot2 = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_scale)


''' check success '''

next_obj_pose = env.get_target_part_pose()
target_part_trans = env.get_target_part_pose().to_transformation_matrix()  # world coordinate -> target part transformation matrix 4*4 SE3
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
# print('before check success')
# success, div_error, traj_len, traj_dir = utils.check_success(trajectory, alpha, beta, gamma, primact_type, out_info, threshold=args.threshold)
success = False     # TODO: implement the succ metrics
pickup_pose = env.object.get_root_pose()
pickup_position = pickup_pose.p.flatten()
if pickup_position[2] > 0.1:
    success = True
out_info['success'] = 'True' if success else 'False'
out_info['trajectory'] = trajectory.tolist()
# out_info['traj_len'] = traj_len.tolist()
# if args.primact_type in ['pushing', 'topple']:
#     out_info['traj_dir'] = traj_dir.tolist()
# print('after check success')


# print('before div error', div_error)
# if div_error:
#     print('NAN!')
#     env.scene.remove_articulation(env.object)
#     env.scene.remove_articulation(robot1.robot)
#     env.scene.remove_articulation(robot2.robot)
#     env.close()
#     exit(2)
# print('after div error')


env.scene.remove_articulation(robot1.robot)
env.scene.remove_articulation(robot2.robot)
env.scene.remove_articulation(env.object)
env.close()


if success:
    save_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs, 'succ')
    exit(0)
else:
    save_data(out_info, cam_XYZA_list, gt_target_link_mask, gif_imgs, 'fail')
    exit(1)











