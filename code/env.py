"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig, ArticulationJointType
from sapien.core.pysapien import ActorBase, VulkanRenderer
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number
# import trimesh
import ipdb
import math


class ContactError(Exception):
    pass


class SVDError(Exception):
    pass


class Env(object):
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500,
                 object_position_offset=0.0, succ_ratio=0.1, set_ground=False,
                 static_friction=0.30, dynamic_friction=0.30):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True

        self.renderer = sapien.OptifuserRenderer(config=render_config)
        # self.renderer = sapien.VulkanRenderer()
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0
        scene_config.default_static_friction = static_friction
        scene_config.default_dynamic_friction = dynamic_friction

        self.scene = self.engine.create_scene(config=scene_config)
        if set_ground:
            self.scene.add_ground(altitude=0.0, render=False)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.contact_error = False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, state='closed', target_part_id=-1, scale=1.0, density=1.0, stiffness=100, damping=10, lieDown=False, given_joint_angles=None, given_pose=None):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = scale
        self.object = loader.load(urdf, {"material": material, "density": density})
        if given_pose:
            pose = given_pose
        else:
            if not lieDown:
                pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
            else:
                # pose = Pose([self.object_position_offset, 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0])
                pose = Pose([self.object_position_offset, 0, 0], [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.all_joint_types = [j.type for j in self.object.get_joints()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        target_part_joint_idx = -1
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    target_part_joint_idx = len(joint_angles)
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                # print("l, r:", l, r)
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'middle':
                    joint_angles.append(float((l + r) / 2))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)

        if given_joint_angles:
            joint_angles = given_joint_angles

        self.object.set_qpos(joint_angles)
        # print('joint_angles: ', joint_angles)
        # self.object.set_qpos([-0.6, -0.6])
        if target_part_id >= 0:
            return joint_angles, target_part_joint_idx
        return joint_angles


    def get_target_part_axes(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[1, 0]), float(mat[2, 0]), float(-mat[0, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')
        return joint_axes

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes


    def set_target_object_part_actor_id2(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id     # not movable
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_child_link().get_id() == actor_id:
                self.target_object_part_actor_link = j.get_child_link()

        # monitor the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_child_link().get_id() == actor_id:
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
            idx += 1


    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_object_root_pose(self):
        return self.object.get_root_pose()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        # ipdb.set_trace()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_robot_hand_actor_idhand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False

    def dual_start_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id, robot2_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot1_hand_actor_id = robot1_hand_actor_id
        self.robot2_hand_actor_id = robot2_hand_actor_id
        self.robot1_gripper_actor_ids = robot1_gripper_actor_ids
        self.robot2_gripper_actor_ids = robot2_gripper_actor_ids
        self.contact_error = False

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def dual_end_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id, robot2_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot1_hand_actor_id = robot1_hand_actor_id
        self.robot2_hand_actor_id = robot2_hand_actor_id
        self.robot1_gripper_actor_ids = robot1_gripper_actor_ids
        self.robot2_gripper_actor_ids = robot2_gripper_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                raise ContactError()


    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                       contact, valid = True, True
                if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
                        self.contact_error = True
                        return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False
        return True


    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None
