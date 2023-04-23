import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
# from pyquaternion import Quaternion
from camera import Camera
import random
import utils
from utils import get_6d_rot_loss
import copy
import ipdb



class SAPIENVisionDataset(data.Dataset):

    def __init__(self, primact_types, category_types, data_features, buffer_max_num, img_size=224,
                 only_true_data=False, succ_proportion=0.4, fail_proportion=0.8, coordinate_system='world',
                 exchange_ctpts=False, find_task_for_invalid=True, cat2freq=None,
                 no_true_false_equal=False):
        self.primact_types = primact_types
        self.category_types = category_types
        self.data_features = data_features
        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.only_true_data = only_true_data
        self.succ_proportion = succ_proportion
        self.fail_proportion = fail_proportion
        self.coordinate_system = coordinate_system
        self.exchange_ctpts = exchange_ctpts
        self.find_task_for_invalid = find_task_for_invalid

        self.dataset = dict()
        for primact_type in primact_types:
            self.dataset[primact_type] = []

        cat_list, shape_list, _, _ = utils.get_shape_list(all_categories=category_types, mode='all', primact_type=self.primact_types[0])
        self.shape_list = shape_list
        self.category_dict = {'success': {}, 'fail1': {}, 'fail': {}, 'invalid': {}}
        for cat in cat_list:
            self.category_dict['success'][cat] = 0
            self.category_dict['fail1'][cat] = 0
            self.category_dict['fail'][cat] = 0
            self.category_dict['invalid'][cat] = 0

        self.freq_dict = dict()
        if cat2freq:
            freqs = [int(x) for x in cat2freq.split(',')]
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = freqs[idx]
        else:
            for idx in range(len(cat_list)):
                self.freq_dict[cat_list[idx]] = 1e4


    def load_data(self, data_list):
        bar = ProgressBar()
        num_pos, num_neg1, num_neg2, num_invalid = 0, 0, 0, 0
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            if not os.path.exists(cur_dir):
                continue

            for root, dirs, files in os.walk(cur_dir):
                for file in sorted(files):
                    if 'json' not in file:
                        continue
                    result_idx = int(file.split('.')[0][7:])
                    with open(os.path.join(cur_dir, 'result_%d.json' % result_idx), 'r') as fin:
                        try:
                            result_data = json.load(fin)
                        except Exception:
                            continue

                    epoch = result_idx
                    shape_id = int(result_data['shape_id'])
                    category = result_data['category']
                    cur_primact_type = result_data['primact_type']
                    if cur_primact_type not in self.primact_types:
                        continue
                    if str(shape_id) not in self.shape_list:
                        continue
                    if category not in self.category_types:
                        continue

                    valid = False if 'invalid' in cur_dir else True
                    success = True if 'succ' in cur_dir else False

                    if self.only_true_data and (not success):
                        continue
                    if (num_neg2 >= num_pos * (self.fail_proportion - self.succ_proportion)) and valid and (not success):
                        continue
                    if (num_invalid >= num_pos * (1 - self.fail_proportion)) and (not valid):
                        continue
                    if num_pos >= self.buffer_max_num and success:
                        continue

                    if success and self.category_dict['success'][category] >= self.freq_dict[category]:
                        continue
                    elif valid and (not success) and self.category_dict['fail'][category] >= self.category_dict['success'][category] * (self.fail_proportion - self.succ_proportion):
                        continue
                    elif (not valid) and self.category_dict['invalid'][category] >= self.category_dict['success'][category] * (1 - self.fail_proportion):
                        continue

                    if 'gripper_direction_world1' not in result_data.keys():
                        continue
                    contact_point_world1 = np.array(result_data['position_world1'], dtype=np.float32)
                    gripper_up_world1 = np.array(result_data['gripper_direction_world1'], dtype=np.float32)
                    gripper_forward_world1 = np.array(result_data['gripper_forward_direction_world1'], dtype=np.float32)
                    contact_point_world2 = np.array(result_data['position_world2'], dtype=np.float32)
                    gripper_up_world2 = np.array(result_data['gripper_direction_world2'], dtype=np.float32)
                    gripper_forward_world2 = np.array(result_data['gripper_forward_direction_world2'], dtype=np.float32)

                    camera_metadata = result_data['camera_metadata']
                    mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
                    cam2cambase = np.array(camera_metadata['cam2cambase'], dtype=np.float32)

                    target_link_mat44, target_part_trans, transition = None, None, None     # not used
                    joint_angles = np.array(result_data['joint_angles'], dtype=np.float32)
                    pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)


                    task = None     
                    if valid:
                        if cur_primact_type == 'pushing':
                            task = np.array(result_data['traj_dir'], dtype=np.float32)  # norm, in world coordinate system
                            task_world = np.array(result_data['traj_dir'], dtype=np.float32)
                        elif cur_primact_type == 'rotating':
                            task = np.array(result_data['beta'], dtype=np.float32)
                        elif cur_primact_type == 'topple':  #######################
                            traj = np.array(result_data['trajectory'], dtype=np.float32)
                            task = traj / np.linalg.norm(traj)
                            task_world = traj / np.linalg.norm(traj)
                        elif cur_primact_type == 'pickup':
                            task = np.array(result_data['traj_dir'], dtype=np.float32)
                            task_world = np.array(result_data['traj_dir'], dtype=np.float32)
                        if np.isnan(task).any() or (not np.any(task)):
                            continue

                    with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%d.h5' % result_idx), 'r') as fin:
                        cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                        cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                        cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                    _, pc_center = utils.get_part_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, coordinate_system='cambase', mat44=mat44, cam2cambase=cam2cambase)


                    if self.coordinate_system == 'world':
                        contact_point1, gripper_up1, gripper_forward1, contact_point2, gripper_up2, gripper_forward2 = \
                            contact_point_world1, gripper_up_world1, gripper_forward_world1, contact_point_world2, gripper_up_world2, gripper_forward_world2
                    elif self.coordinate_system == 'cambase':
                        world_batch = [contact_point_world1.copy(), gripper_up_world1.copy(), gripper_forward_world1.copy(), contact_point_world2.copy(), gripper_up_world2.copy(), gripper_forward_world2.copy()]
                        is_pc = [True, False, False, True, False, False]
                        camera_batch = utils.batch_coordinate_transform(world_batch, is_pc, transform_type='world2cam', mat44=mat44)
                        cambase_batch = utils.batch_coordinate_transform(camera_batch, is_pc, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                        contact_point1, gripper_up1, gripper_forward1, contact_point2, gripper_up2, gripper_forward2 = cambase_batch
                        if valid and cur_primact_type in ['pushing', 'topple', 'pickup']:
                            task = utils.coordinate_transform(task, False, transform_type='world2cam', mat44=mat44)
                            task = utils.coordinate_transform(task, False, transform_type='cam2cambase', cam2cambase=cam2cambase)


                    pixel1_idx1, pixel1_idx2 = None, None
                    cur_data = (cur_dir, shape_id, category,
                                pixel1_idx1, contact_point1, gripper_up1, gripper_forward1,
                                pixel1_idx2, contact_point2, gripper_up2, gripper_forward2,
                                task, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc_center,
                                pixel_ids, target_link_mat44, target_part_trans, transition,
                                contact_point_world1, gripper_up_world1, gripper_forward_world1,
                                contact_point_world2, gripper_up_world2, gripper_forward_world2)

                    if success:
                        self.category_dict['success'][category] += 1
                        self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=False, cur_type='type0'))
                        num_pos += 1
                        if self.exchange_ctpts:
                            self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=True, cur_type='type0'))
                            num_pos += 1

                        # get negative data by changing the task the positive data
                        if not self.only_true_data:
                            if (num_neg1 < num_pos * self.succ_proportion) and (self.category_dict['fail1'][category] <= self.category_dict['success'][category] * self.succ_proportion):
                                if cur_primact_type == 'pushing':
                                    task_neg = np.array([random.random(), random.random(), 0], dtype=np.float32)  # world
                                    angle_degree, _ = utils.cal_included_angle(task_neg, task_world)
                                    while angle_degree < 30:
                                        task_neg = np.array([random.random(), random.random(), 0], dtype=np.float32)
                                        angle_degree, _ = utils.cal_included_angle(task_neg, task_world)
                                    if self.coordinate_system == 'cambase':     # world2cambase
                                        task_neg = utils.coordinate_transform(task_neg, False, transform_type='world2cam', mat44=mat44)
                                        task_neg = utils.coordinate_transform(task_neg, False, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                                    task_neg = task_neg / np.linalg.norm(task_neg)
                                if cur_primact_type == 'rotating':
                                    task_neg = np.array(random.random(), dtype=np.float32) * 100 - 50    # （-50， 50)
                                    angle_degree = np.abs(task_neg - task)
                                    while angle_degree < 20:
                                        task_neg = np.array(random.random(), dtype=np.float32) * 100 - 50
                                        angle_degree = np.abs(task_neg - task)
                                if cur_primact_type == 'topple':
                                    task_neg = np.array([random.random(), random.random(), 0], dtype=np.float32)  # world
                                    angle_degree, _ = utils.cal_included_angle(task_neg, task_world)
                                    while angle_degree < 30:
                                        task_neg = np.array([random.random(), random.random(), 0], dtype=np.float32)
                                        angle_degree, _ = utils.cal_included_angle(task_neg, task_world)
                                    if self.coordinate_system == 'cambase':     # world2cambase
                                        task_neg = utils.coordinate_transform(task_neg, False, transform_type='world2cam', mat44=mat44)
                                        task_neg = utils.coordinate_transform(task_neg, False, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                                    task_neg = task_neg / np.linalg.norm(task_neg)
                                if cur_primact_type == 'pickup':
                                    task_neg = np.array([random.random(), random.random(), random.random()], dtype=np.float32)  # cambase
                                    angle_degree, _ = utils.cal_included_angle(task_neg, task)
                                    while angle_degree < 45:
                                        task_neg = np.array([random.random(), random.random(), random.random()], dtype=np.float32)
                                        angle_degree, _ = utils.cal_included_angle(task_neg, task)
                                    task_neg = task_neg / np.linalg.norm(task_neg)
                                self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=False, cur_type='type1', given_task=task_neg))
                                self.category_dict['fail1'][category] += 1
                                num_neg1 += 1
                                if self.exchange_ctpts:
                                    self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=True, cur_type='type1', given_task=task_neg))
                                    num_neg1 += 1

                    elif valid and (not success):
                        # if num_neg2 < self.buffer_max_num * (self.fail_proportion - self.succ_proportion):
                        self.category_dict['fail'][category] += 1
                        self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=False, cur_type='type2'))
                        num_neg2 += 1
                        if self.exchange_ctpts:
                            self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=True, cur_type='type2'))
                            num_neg2 += 1

                    elif not valid:     # contact error, the last file_list
                        if self.find_task_for_invalid:
                            refer_data_idx = None
                            min_dis = 10000
                            for idx_positive_data in range(len(self.dataset[cur_primact_type])):
                                positive_data = self.dataset[cur_primact_type][idx_positive_data]
                                if positive_data[13] == False:  # negative data
                                    continue
                                if positive_data[1] != shape_id:
                                    continue

                                # cambase
                                post_ctpt1_world, post_up1, post_forward1 = positive_data[25], positive_data[26], positive_data[27]
                                post_ctpt2_world, post_up2, post_forward2 = positive_data[28], positive_data[29], positive_data[30]

                                dis1 = (np.linalg.norm(post_ctpt1_world - contact_point_world1) + np.linalg.norm(post_ctpt2_world - contact_point_world2)) * 15
                                dir1 = get_6d_rot_loss(torch.from_numpy(np.concatenate([post_up1, post_forward1])), torch.from_numpy(np.concatenate([gripper_up_world1, gripper_forward_world1])))[0].item() + \
                                       get_6d_rot_loss(torch.from_numpy(np.concatenate([post_up2, post_forward2])), torch.from_numpy(np.concatenate([gripper_up_world2, gripper_forward_world2])))[0].item()
                                cur_dis = dis1 + dir1

                                if cur_dis < min_dis:  # find min(dis_ctpt + dir_rot_loss)
                                    min_dis = cur_dis
                                    refer_data_idx = idx_positive_data

                            if refer_data_idx == None:
                                continue
                            refer_data = copy.deepcopy(self.dataset[cur_primact_type][refer_data_idx])
                            task = refer_data[11].copy()
                            post_camera_metadata = refer_data[18]
                            post_mat44, post_cam2cambase = np.array(post_camera_metadata['mat44'], dtype=np.float32), np.array(post_camera_metadata['cam2cambase'], dtype=np.float32)
                            post_pc_center = refer_data[20]
                            if self.coordinate_system == 'world':
                                pass
                            elif self.coordinate_system == 'cambase':
                                if cur_primact_type in ['pushing', 'topple', 'pickup']:
                                    task = utils.coordinate_transform(task, False, transform_type='cambase2cam', cam2cambase=post_cam2cambase, pc_center=post_pc_center)
                                    task = utils.coordinate_transform(task, False, transform_type='cam2world', mat44=post_mat44)
                                    task = utils.coordinate_transform(task, False, transform_type='world2cam', mat44=mat44)
                                    task = utils.coordinate_transform(task, False, transform_type='cam2cambase', cam2cambase=cam2cambase, pc_center=pc_center)
                            self.category_dict['invalid'][category] += 1
                            self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=False, cur_type='type3', given_task=task))
                            num_invalid += 1
                            if self.exchange_ctpts:
                                self.dataset[cur_primact_type].append(utils.get_data_info(cur_data, exchange_ctpts=True, cur_type='type3', given_task=task))
                                num_invalid += 1


        print('positive data: %d; negative data1: %d; negative data2: %d; contact error: %d' % (num_pos, num_neg1, num_neg2, num_invalid))
        print('category distribution: \nsuccess:', self.category_dict['success'], '\nfail1:', self.category_dict['fail1'], '\nfail:', self.category_dict['fail'], '\ninvalid:', self.category_dict['invalid'])


    def __str__(self):
        strout = '[SAPIENVisionDataset %d] primact_types: %s, img_size: %d\n' % \
                (len(self), ','.join(self.primact_types), self.img_size)
        for primact_type in self.primact_types:
            # strout += '\t<%s> True: %d False: %d\n' % (primact_type, len(self.true_data[primact_type]), len(self.false_data[primact_type]))
            strout += '\t<%s>\n' % primact_type
        return strout

    def __len__(self):
        max_data = 0
        for primact_type in self.primact_types:
            max_data = max(max_data, len(self.dataset[primact_type]))
        return max_data * len(self.primact_types)

    def __getitem__(self, index):
        primact_id = index % len(self.primact_types)
        primact_type = self.primact_types[primact_id]
        index = index // len(self.primact_types)


        cur_dir, shape_id, category, \
        idx1, contact_point1, up1, forward1,\
        idx2, contact_point2, up2, forward2, \
        task, valid, success, epoch, result_idx, mat44, cam2cambase, camera_metadata, joint_angles, pc_center, \
        pixel_ids, target_link_mat44, target_part_trans, transition, \
        contact_point_world1, gripper_up_world1, gripper_forward_world1, \
        contact_point_world2, gripper_up_world2, gripper_forward_world2 = self.dataset[primact_type][index]


        # print(result, is_original)
        data_feats = ()
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'part_pc':
                with h5py.File(os.path.join(cur_dir, 'cam_XYZA_%d.h5' % result_idx), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                mask = (out[:, :, 3] > 0.5)
                pc = out[mask, :3]
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000-1]
                pc = pc[idx, :]
                pc[:, 0] -= 5

                pc_world = (mat44[:3, :3] @ pc.T).T
                if self.coordinate_system == 'world':   # cam2world
                    pc = (mat44[:3, :3] @ pc.T).T
                elif self.coordinate_system == 'cambase':   # cam2cambase
                    pc = pc @ np.transpose(cam2cambase, (1, 0))
                    pc_centers = (pc.max(axis=0, keepdims=True) + pc.min(axis=0, keepdims=True)) / 2
                    pc_centers = pc_centers[0]
                    pc -= pc_centers
                out = torch.from_numpy(pc).unsqueeze(0)
                data_feats = data_feats + (out,)


            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'primact_type':
                data_feats = data_feats + (primact_type,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'task':
                data_feats = data_feats + (task,)

            elif feat == 'ctpt1':
                ctpt1 = contact_point1
                data_feats = data_feats + (ctpt1,)

            elif feat == 'ctpt2':
                ctpt2 = contact_point2
                data_feats = data_feats + (ctpt2,)

            elif feat == 'dir1':
                dir1 = np.concatenate([up1, forward1])
                data_feats = data_feats + (dir1,)

            elif feat == 'dir2':
                dir2 = np.concatenate([up2, forward2])
                data_feats = data_feats + (dir2,)

            elif feat == 'target_link_mat44':
                data_feats = data_feats + (target_link_mat44,)

            elif feat == 'target_part_trans':
                data_feats = data_feats + (target_part_trans,)

            elif feat == 'pc_centers':
                data_feats = data_feats + (pc_center,)

            elif feat == 'valid':
                data_feats = data_feats + (valid,)

            elif feat == 'success':
                data_feats = data_feats + (success,)

            elif feat == 'result_idx':   # epoch = result_idx
                data_feats = data_feats + (result_idx,)

            elif feat == 'camera_metadata':
                data_feats = data_feats + (camera_metadata,)

            elif feat == 'joint_angles':
                data_feats = data_feats + (joint_angles,)

            elif feat == 'pixel_ids':
                data_feats = data_feats + (pixel_ids,)

            elif feat == 'cam2cambase':
                data_feats = data_feats + (cam2cambase,)

            elif feat == 'mat44':
                data_feats = data_feats + (mat44,)

            elif feat == 'pixel1_idx':
                data_feats = data_feats + (idx1,)

            elif feat == 'pixel2_idx':
                data_feats = data_feats + (idx2,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

