"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig, ArticulationJointType
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number, ContactError
import trimesh
import ipdb
import math


class SVDError(Exception):
    pass


class Env(object):
    def __init__(self, flog=None, render_rate=20, timestep=1 / 500, static_friction=0.3, dynamic_friction=0.3,
                 object_position_offset=0.0, succ_ratio=0.1, show_gui=True, set_ground=True, fix_joint=True):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.fix_joint = fix_joint
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
        self.renderer.enable_global_axes(False)

        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0 + object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0
        scene_config.default_static_friction = static_friction      # default: 0.30
        scene_config.default_dynamic_friction = dynamic_friction    # default: 0.30

        self.scene = self.engine.create_scene(config=scene_config)
        if set_ground:
            self.scene.add_ground(altitude=0.0, render=False)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1 + object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1 + object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1 + object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.objects = list()
        self.object = None
        self.object_target_joint = None
        self.all_link_ids = list()
        self.all_joint_types = list()
        self.joint_angles_lower = None
        self.joint_angles_upper = None

        # check contact
        self.check_contact = False
        self.contact_error = False

        # target part
        self.target_object_part_actor_link = None
        self.target_object_part_joint_id = None
        self.target_object_part_joint_type = None
        self.non_target_object_part_actor_id = None
        self.target_object_part_actor_id = None

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, joint_range=(0.0, 1.0), target_part_id=-1, scale=1.0, density=1.0, stiffness=100, damping=10,
                    lie_down=False, given_pose=None, given_joint_angles=None, use_collision=False):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = scale
        print("Loading object...")
        self.object = loader.load(urdf, {"material": material, "density": density})
        self.objects.append(self.object)
        if given_pose:
            pose = given_pose
        else:
            if lie_down:
                pose = Pose([self.object_position_offset, 0, 0], [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0])
            else:
                pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        if use_collision:
            self.set_collision_bodies()
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids.extend([l.get_id() for l in self.object.get_links()])
        self.all_joint_types.extend([j.type for j in self.object.get_joints()])
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)

        # set initial qpos
        if given_joint_angles:
            joint_angles = given_joint_angles
            idx_j = 0
            for j in self.object.get_joints():
                if j.get_dof() == 1:
                    if j.get_child_link().get_id() == target_part_id:
                        target_part_joint_idx = idx_j
                    if self.fix_joint:
                        j.set_limits(np.array([[joint_angles[idx_j], joint_angles[idx_j]]]))
                    idx_j += 1
        else:
            joint_angles = []
            self.joint_angles_lower = []
            self.joint_angles_upper = []
            target_part_joint_idx = -1
            llim, rlim = joint_range[0], joint_range[1]
            if llim < rlim:
                print("Relative joints!")
            else:
                print("Absolute joints!")
            for j in self.object.get_joints():
                if j.get_dof() == 1:
                    if j.get_child_link().get_id() == target_part_id:
                        target_part_joint_idx = len(joint_angles)
                    l = float(process_angle_limit(j.get_limits()[0, 0]))
                    r = float(process_angle_limit(j.get_limits()[0, 1]))
                    self.joint_angles_lower.append(l)
                    self.joint_angles_upper.append(r)
                    if llim < rlim:
                        jr = float(get_random_number(llim, rlim))
                        jrt = (1 - jr) * l + jr * r
                    else:
                        jrt = max(min(r, llim), l)
                    print("l, r, jrt:", l, r, jrt)
                    joint_angles.append(jrt)
                    if self.fix_joint:
                        j.set_limits(np.array([[jrt, jrt]]))

        print("\tjoint angles:", joint_angles)
        self.object.set_qpos(joint_angles)
        
        if target_part_id >= 0:
            return joint_angles, target_part_joint_idx
        return joint_angles

    def remove_all_objects(self):
        for obj in self.objects:
            self.scene.remove_articulation(obj)
        self.objects = list()
        self.all_link_ids = list()
        self.all_joint_types = list()
        self.target_object_part_actor_link = None
        self.target_object_part_joint_id = None
        self.target_object_part_joint_type = None

    def load_real_object(self, urdf, material, joint_angles=None):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        self.object = loader.load(urdf, {"material": material})
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=100, damping=10)

        if joint_angles is not None:
            self.object.set_qpos(joint_angles)

        return None

    def set_main_object(self, idx):
        self.object = self.objects[idx]

    def update_and_set_joint_angles_all(self, state='closed'):
        joint_angles = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
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
        self.object.set_qpos(joint_angles)
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

    def get_target_part_axes_dir(self, target_part_id):
        joint_axes = self.get_target_part_axes(target_part_id=target_part_id)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.5:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_axes_dir_new(self, target_part_id):
        joint_axes = self.get_target_part_axes_new(target_part_id=target_part_id)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.1:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_origins_new(self, target_part_id):
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    joint_origins = pos.p.tolist()
                    # ipdb.set_trace()
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def get_target_part_origins(self, target_part_id):
        print("attention!!! origin")
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    # joint_origins = pos.p.tolist()
                    # ipdb.set_trace()
                    mat = pos.to_transformation_matrix()
                    # print("pos:", pos.p)
                    joint_origins = [float(-mat[1, 3]), float(mat[2, 3]), float(-mat[0, 3])]
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def update_joint_angle(self, joint_angles, target_part_joint_idx, state, task_lower,
                           push=True, pull=False, drawer=False):
        if push:
            if drawer:
                l = max(self.joint_angles_lower[target_part_joint_idx],
                        self.joint_angles_lower[target_part_joint_idx] + task_lower)
                r = self.joint_angles_upper[target_part_joint_idx]
            else:
                l = max(self.joint_angles_lower[target_part_joint_idx],
                        self.joint_angles_lower[target_part_joint_idx] + task_lower * np.pi / 180)
                r = self.joint_angles_upper[target_part_joint_idx]
        if pull:
            if drawer:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower
            else:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower * np.pi / 180
        if state == 'closed':
            joint_angles[target_part_joint_idx] = (float(l))
        elif state == 'open':
            joint_angles[target_part_joint_idx] = float(r)
        elif state == 'middle':
            joint_angles[target_part_joint_idx] = float((l + r) / 2)
        elif state == 'random-middle':
            joint_angles[target_part_joint_idx] = float(get_random_number(l, r))
        elif state == 'random-middle-open':
            joint_angles[target_part_joint_idx] = float(get_random_number(r * 0.8, r))
        elif state == 'random-closed-middle':
            if np.random.random() < 0.5:
                joint_angles[target_part_joint_idx] = float(get_random_number(l, r))
            else:
                joint_angles[target_part_joint_idx] = float(l)
        else:
            raise ValueError('ERROR: object init state %s unknown!' % state)
        return joint_angles

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()

        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                    self.target_object_part_joint_type = j.type
                idx += 1

    def set_target_object_part_actor_id2(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id  # not movable
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # monitor the target joint
        idx = 0

        for j in self.object.get_joints():
            # if j.get_dof() == 1:
            if j.get_child_link().get_id() == actor_id:
                print("2.32.32.3", j)

                self.target_object_part_actor_link = j.get_child_link()
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
            idx += 1

    def set_target_joint_actor_id(self, target_joint_type):
        idx = 0
        actor_id = None
        for j in self.object.get_joints():
            # if j.get_dof() == 1:
            if j.type == target_joint_type:
                self.target_object_part_actor_link = j.get_child_link()
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
                actor_id = j.get_child_link().get_id()
                if self.flog is not None:
                    self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
                self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))
            idx += 1
        return actor_id

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
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False

    def dual_start_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id,
                                    robot2_gripper_actor_ids, strict):
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

    def dual_end_checking_contact(self, robot1_hand_actor_id, robot1_gripper_actor_ids, robot2_hand_actor_id,
                                  robot2_gripper_actor_ids, strict):
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
            if not self.dual_check_contact_is_valid():
                raise ContactError()

    def check_main_part(self, part_id):
        main_joint_type = [ArticulationJointType.FIX, ArticulationJointType.PRISMATIC]
        for joint in self.object.get_joints():
            if joint.get_child_link().get_id() == part_id:
                if joint.type in main_joint_type:
                    return True
                else:
                    return False

    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False
        valid = False
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
                        print("## Gripper-Object collision", c.actor1.name, c.actor2.name)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id:
                    if self.check_contact_strict:
                        print("## Hand collision", c.actor1.name, c.actor2.name)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
                    print("## Gripper-Hand collision", c.actor1.name, c.actor2.name)
                    self.contact_error = True
                    return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False
        return True

    def dual_check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False
        valid = False
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot1_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                        (aid2 in self.robot1_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                    contact, valid = True, True
                if (aid1 in self.robot2_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                        (aid2 in self.robot2_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                    contact, valid = True, True

                if (aid1 in self.robot1_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                        (aid2 in self.robot1_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        print("## Gripper1-Object collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if (aid1 in self.robot2_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                        (aid2 in self.robot2_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True

                if aid1 == self.robot1_hand_actor_id or aid2 == self.robot1_hand_actor_id:
                    if self.check_contact_strict:
                        print("## Hand1 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if aid1 == self.robot2_hand_actor_id or aid2 == self.robot2_hand_actor_id:
                    if self.check_contact_strict:
                        print("## Hand2 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True

                # starting pose should have no collision at all
                if (aid1 in self.robot1_gripper_actor_ids or aid1 == self.robot1_hand_actor_id or
                    aid2 in self.robot1_gripper_actor_ids or aid2 == self.robot1_hand_actor_id) and self.first_timestep_check_contact:
                    print("## Start Hand1 collision", c.actor1.name, c.actor2.name, aid1, aid2)
                    self.contact_error = True
                    return False
                if (aid1 in self.robot2_gripper_actor_ids or aid1 == self.robot2_hand_actor_id or
                    aid2 in self.robot2_gripper_actor_ids or aid2 == self.robot2_hand_actor_id) and self.first_timestep_check_contact:
                    print("## Start Hand2 collision", c.actor1.name, c.actor2.name, aid1, aid2)
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

    def get_global_mesh(self, obj):
        final_vs = []
        final_fs = []
        vid = 0
        for l in obj.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0]
                v[:, 1] *= vscale[1]
                v[:, 2] *= vscale[2]
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def get_part_mesh(self, obj, part_id):
        final_vs = [];
        final_fs = [];
        vid = 0;
        for l in obj.get_links():
            l_id = l.get_id()
            if l_id != part_id:
                continue
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0];
                v[:, 1] *= vscale[1];
                v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def set_collision_bodies(self):
        for l in self.object.get_links():
            for s in l.get_visual_bodies():
                s.set_visibility(0)
            for s in l.get_collision_visual_bodies():
                s.set_visibility(1)

    def sample_pc(self, v, f, n_points=4096):
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
        return points

    def check_drawer(self):
        for j in self.object.get_joints():
            if j.get_dof() == 1 and (j.type == ArticulationJointType.PRISMATIC):
                return True
        return False
