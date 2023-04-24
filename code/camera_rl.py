"""
    an RGB-D camera
"""
import numpy as np
from sapien.core import Pose


class Camera(object):

    def __init__(self, env, image_size1=448, image_size2=448, fov=35, near=0.1, far=100.0,
                 dist=5.0, phi=np.pi / 10, theta=np.pi, raise_camera=0.0, base_theta=1.0,
                 random_initialize=False, fixed_position=False, restrict_dir=False, exact_dir=False,
                 low_view=False, real_data=False):
        # set camera intrinsics
        self.env = env
        builder = env.scene.create_actor_builder()
        # camera_mount_actor = builder.build_kinematic()
        camera_mount_actor = builder.build(is_kinematic=True)
        self.camera_mount_actor = camera_mount_actor
        self.camera = env.scene.add_mounted_camera(
            'camera', camera_mount_actor, Pose(), image_size1, image_size2, 0, np.deg2rad(fov), near, far
        )

        # log parameters
        self.restrict_dir = restrict_dir
        self.exact_dir = exact_dir
        self.low_view = low_view
        self.fixed_position = fixed_position
        self.real_data = real_data
        self.near = near
        self.far = far
        self.dist = dist
        # self.pos = pos
        self.raise_camera = raise_camera
        self.base_theta = base_theta

        if random_initialize:
            self.pick_view()
        else:
            self.theta = theta
            self.phi = phi

        self.pos = None
        self.mat44 = None
        self.base_mat44 = None
        self.cam2cambase = None

    def change_pose(self, phi=None, theta=None):
        # set camera extrinsics
        if not self.fixed_position:
            self.pick_view()
        if phi is not None:
            self.phi = phi
            self.theta = theta

        self.pos = np.array([self.dist * np.cos(self.phi) * np.cos(self.theta),
                             self.dist * np.cos(self.phi) * np.sin(self.theta),
                             self.dist * np.sin(self.phi)])

        # world base
        forward = -self.pos / np.linalg.norm(self.pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        self.mat44 = np.eye(4)
        self.mat44[:3, :3] = np.vstack([forward, left, up]).T
        self.mat44[:3, 3] = self.pos                # mat44 is cam2world
        self.mat44[0, 3] += self.env.object_position_offset
        self.mat44[2, 3] += self.raise_camera       # to object's center
        self.camera_mount_actor.set_pose(Pose.from_transformation_matrix(self.mat44))

        # compute camera-base frame (camera-center, world-up-z, camera-front-x)
        cb_up = np.array([0, 0, 1], dtype=np.float32)
        cb_left = np.cross(cb_up, forward)
        cb_left /= np.linalg.norm(cb_left)
        cb_forward = np.cross(cb_left, cb_up)
        cb_forward /= np.linalg.norm(cb_forward)
        self.base_mat44 = np.eye(4)
        self.base_mat44[:3, :3] = np.vstack([cb_forward, cb_left, cb_up]).T
        self.base_mat44[:3, 3] = self.pos               # cambase2world
        self.base_mat44[2, 3] += self.raise_camera      # to object's center
        self.cam2cambase = np.linalg.inv(self.base_mat44) @ self.mat44  # cam2cambase
        self.cam2cambase = self.cam2cambase[:3, :3]

        self.env.step()
        self.env.render()

    def pick_view(self):
        if self.exact_dir:
            self.theta = self.base_theta * np.pi
        elif self.restrict_dir:
            self.theta = (self.base_theta + (np.random.random() - 0.5)) * np.pi
            # self.theta = np.pi / 2
        else:
            self.theta = np.random.random() * np.pi * 2
        print(">>> THETA =", self.theta)
        if self.low_view:
            self.phi = (np.random.random() + 0.4) * np.pi / 6
        else:
            self.phi = (np.random.random() + 1) * np.pi / 6
        if self.real_data:
            self.theta = np.random.random() * np.pi * 2
            self.phi = np.random.random() * np.pi * 2

    def get_observation(self):
        self.camera.take_picture()
        rgba = self.camera.get_color_rgba()
        rgba = (rgba * 255).clip(0, 255).astype(np.float32) / 255
        white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.float32)
        mask = np.tile(rgba[:, :, 3:4], [1, 1, 3])
        rgb = rgba[:, :, :3] * mask + white * (1 - mask)
        depth = self.camera.get_depth().astype(np.float32)
        return rgb, depth

    def compute_camera_XYZA(self, depth):
        camera_matrix = self.camera.get_camera_matrix()[:3, :3]
        y, x = np.where(depth < 1)
        z = self.near * self.far / (self.far + depth * (self.near - self.far))
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points = (permutation @ np.dot(np.linalg.inv(camera_matrix), np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T
        return y, x, points

    @staticmethod
    def compute_XYZA_matrix(id1, id2, pts, size1, size2):
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[id1, id2, :3] = pts
        out[id1, id2, 3] = 1
        return out

    def get_normal_map(self):
        nor = self.camera.get_normal_rgba()
        # convert from PartNet-space (x-right, y-up, z-backward) to SAPIEN-space (x-front, y-left, z-up)
        new_nor = np.array(nor, dtype=np.float32)
        new_nor[:, :, 0] = -nor[:, :, 2]
        new_nor[:, :, 1] = -nor[:, :, 0]
        new_nor[:, :, 2] = nor[:, :, 1]
        return new_nor

    def get_movable_link_mask(self, link_ids):
        link_seg = self.camera.get_segmentation()
        link_mask = np.zeros((link_seg.shape[0], link_seg.shape[1])).astype(np.uint8)
        for idx, lid in enumerate(link_ids):
            cur_link_pixels = int(np.sum(link_seg == lid))
            if cur_link_pixels > 0:
                link_mask[link_seg == lid] = idx + 1
        return link_mask

    def get_id_link_mask(self, link_ids):
        link_seg = self.camera.get_segmentation()

        # TODO: Debug
        print("NUMTOTLINK:", len(np.where(link_seg > 0)[0]))

        link_mask = np.zeros((link_seg.shape[0], link_seg.shape[1])).astype(np.uint8)
        for idx, lid in enumerate(link_ids):
            cur_link_pixels = int(np.sum(link_seg == lid))
            if cur_link_pixels > 0:
                link_mask[link_seg == lid] = idx + 1
        return link_mask

    def get_name_seg_mask(self, keyword=['handle']):
        # read part seg partid2renderids
        partid2renderids = dict()   # {leg1:[id1,id2], leg2:[id_x, id_y]}
        for k in self.env.scene.render_id_to_visual_name:
            # print(k, self.env.scene.render_id_to_visual_name[k])
            if self.env.scene.render_id_to_visual_name[k].split('-')[0] in keyword:   # e.g. leg-1 / base_body-3
                part_id = int(self.env.scene.render_id_to_visual_name[k].split('-')[-1])
                if part_id not in partid2renderids:
                    partid2renderids[part_id] = []
                partid2renderids[part_id].append(k)
        # generate 0/1 target mask
        part_seg = self.camera.get_obj_segmentation()        # (448, 448) render_id
        # print('part_seg: ', part_seg, part_seg.shape)

        dest_mask = np.zeros((part_seg.shape[0], part_seg.shape[1])).astype(np.uint8)
        for partid in partid2renderids:
            cur_part_mask = np.isin(part_seg, partid2renderids[partid])
            cur_part_mask_pixels = int(np.sum(cur_part_mask))
            if cur_part_mask_pixels > 0:
                dest_mask[cur_part_mask] = partid
                print("dest part get!")
        return dest_mask                                      # (448, 448) part_id

    def get_object_mask(self):
        rgba = self.camera.get_albedo_rgba()
        return rgba[:, :, 3] > 0.5

    # return camera parameters
    def get_metadata(self):
        return {
            'pose': self.camera.get_pose(),
            'near': self.camera.get_near(),
            'far': self.camera.get_far(),
            'width': self.camera.get_width(),
            'height': self.camera.get_height(),
            'fov': self.camera.get_fovy(),
            'camera_matrix': self.camera.get_camera_matrix(),
            'projection_matrix': self.camera.get_projection_matrix(),
            'model_matrix': self.camera.get_model_matrix(),
            'mat44': self.mat44,
            'cam2cambase': self.cam2cambase,
            'basemat44': self.base_mat44,
        }

    # return camera parameters
    def get_metadata_json(self):
        return {
            'dist': self.dist,
            'theta': self.theta,
            'phi': self.phi,
            'near': self.camera.get_near(),
            'far': self.camera.get_far(),
            'width': self.camera.get_width(),
            'height': self.camera.get_height(),
            'fov': self.camera.get_fovy(),
            'camera_matrix': self.camera.get_camera_matrix().tolist(),
            'projection_matrix': self.camera.get_projection_matrix().tolist(),
            'model_matrix': self.camera.get_model_matrix().tolist(),
            'mat44': self.mat44.tolist(),
            'cam2cambase': self.cam2cambase.tolist(),
            'basemat44': self.base_mat44.tolist(),
        }
