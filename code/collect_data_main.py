import os
import sys
import numpy as np
import utils
from argparse import ArgumentParser
import time
import random
import multiprocessing as mp
from subprocess import call
import datetime


parser = ArgumentParser()
parser.add_argument('--category', type=str)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--mode', type=str, default='train/val/all')

parser.add_argument('--density', type=float, default=2.0)
parser.add_argument('--damping', type=int, default=10)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--start_dist', type=float, default=0.30)
parser.add_argument('--final_dist', type=float, default=0.10)
parser.add_argument('--move_steps', type=int, default=2000)
parser.add_argument('--wait_steps', type=int, default=2000)

parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--not_check_dual', action='store_true', default=False)

args = parser.parse_args()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def copy_file(trail_id, category, shape_id, succ, out_dir):
    tmp_file_dir = os.path.join(out_dir, 'tmp_succ_files')
    tmp_gif_dir = os.path.join(out_dir, 'tmp_succ_gif')
    if succ:
        target_file_dir = os.path.join(out_dir, 'succ_files')
        target_gif_dir = os.path.join(out_dir, 'succ_gif')
    else:
        target_file_dir = os.path.join(out_dir, 'fail_files')
        target_gif_dir = os.path.join(out_dir, 'fail_gif')

    cmd = 'cp %s %s/' % (os.path.join(tmp_file_dir, 'result_%d.json' % trail_id), target_file_dir)
    call(cmd, shell=True)
    cmd = 'cp %s %s/' % (os.path.join(tmp_file_dir, 'cam_XYZA_%d.h5' % trail_id), target_file_dir)
    call(cmd, shell=True)
    cmd = 'cp %s %s/' % (os.path.join(tmp_file_dir, 'interaction_mask_%d.png' % trail_id), target_file_dir)
    call(cmd, shell=True)
    cmd = 'cp %s %s/' % (os.path.join(tmp_gif_dir, '%d_%s_%s.gif' % (trail_id, category, shape_id)), target_gif_dir)
    call(cmd, shell=True)



def run_jobs(idx_process, args, transition_Q):
    random.seed(datetime.datetime.now())
    setup_seed(random.randint(1, 1000) + idx_process)
    sum_trial = 100000

    cat_list, shape_list, shape2cat_dict, cat2shape_dict = utils.get_shape_list(all_categories=args.category, mode=args.mode)

    for trial in range(args.start_epoch, sum_trial):
        cur_trial = sum_trial * idx_process + trial
        cur_random_seed = np.random.randint(10000000)

        # load object
        selected_cat = cat_list[random.randint(0, len(cat_list) - 1)]
        shape_id = cat2shape_dict[selected_cat][random.randint(0, len(cat2shape_dict[selected_cat]) - 1)]
        print('shape_id: ', shape_id, selected_cat, cur_trial)

        cmd = 'python collect_data.py --trial_id %d --shape_id %s --category %s --primact_type %s --random_seed %d ' \
              '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f ' \
              '--move_steps %d --wait_steps %d --out_dir %s --no_gui ' \
              % (cur_trial, shape_id, selected_cat, args.primact_type, cur_random_seed,
                 args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist,
                 args.move_steps, args.wait_steps, args.out_dir)
        if trial % args.save_interval == 0:
            cmd += '--save_data '
        cmd += '> /dev/null 2>&1'

        ret = call(cmd, shell=True)
        if ret == 1:
            transition_Q.put(['fail', trial])
        elif ret == 2:
            transition_Q.put(['invalid', trial])


        if ret == 0:
            # check dual
            ret0, ret1 = 1, 1
            if not args.not_check_dual:
                cmd = 'python collect_data_checkDual.py --trial_id %d --random_seed %d --gripper_id %d ' \
                      '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f ' \
                      '--move_steps %d --wait_steps %d --out_dir %s --no_gui ' \
                      % (cur_trial, cur_random_seed, 0,
                         args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist,
                         args.move_steps, args.wait_steps, args.out_dir)
                ret0 = call(cmd, shell=True)

                cmd = 'python collect_data_checkDual.py --trial_id %d --random_seed %d --gripper_id %d ' \
                      '--density %f --damping %d --target_part_state %s --start_dist %f --final_dist %f ' \
                      '--move_steps %d --wait_steps %d --out_dir %s --no_gui ' \
                      % (cur_trial, cur_random_seed, 1,
                         args.density, args.damping, args.target_part_state, args.start_dist, args.final_dist,
                         args.move_steps, args.wait_steps, args.out_dir)
                ret1 = call(cmd, shell=True)

            if ret0 == 0 or ret1 == 0:
                transition_Q.put(['fail', trial])     # single succ
                copy_file(cur_trial, selected_cat, shape_id, succ=False, out_dir=args.out_dir)
            else:
                transition_Q.put(['succ', trial])     # dual succ
                copy_file(cur_trial, selected_cat, shape_id, succ=True, out_dir=args.out_dir)




if __name__ == '__main__':
    out_dir = args.out_dir
    print('out_dir: ', out_dir)
    if os.path.exists(out_dir):
        response = input('Out directory "%s" already exists, continue? (y/n) ' % out_dir)
        if response != 'y' and response != 'Y':
            sys.exit()


    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, 'succ_gif'))
        os.makedirs(os.path.join(out_dir, 'fail_gif'))
        os.makedirs(os.path.join(out_dir, 'invalid_gif'))
        os.makedirs(os.path.join(out_dir, 'tmp_succ_gif'))

        os.makedirs(os.path.join(out_dir, 'succ_files'))
        os.makedirs(os.path.join(out_dir, 'fail_files'))
        os.makedirs(os.path.join(out_dir, 'invalid_files'))
        os.makedirs(os.path.join(out_dir, 'tmp_succ_files'))


    trans_q = mp.Queue()
    for idx_process in range(args.num_processes):
        p = mp.Process(target=run_jobs, args=(idx_process, args, trans_q))
        p.start()


    total, max_trial = 0, 0
    cnt_dict = {'succ': 0, 'fail': 0, 'invalid': 0}

    t0 = time.time()
    t_begin = datetime.datetime.now()
    while True:
        if not trans_q.empty():
            results = trans_q.get()
            result, trial_id = results

            cnt_dict[result] += 1
            total += 1
            max_trial = max(max_trial, trial_id)

            print(
                'Episode: {} | trial_id: {} | Valid_portion: {:.4f} | Succ_portion: {:.4f} | Running Time: {:.4f} | Total Time:'.format(
                    total, max_trial, (cnt_dict['succ'] + cnt_dict['fail']) / total, cnt_dict['succ'] / total, time.time() - t0), datetime.datetime.now() - t_begin
            )
            t0 = time.time()

            if total >= 10000:
                for idx_process in range(args.num_processes):
                    p.terminate()
                    p.join()
                break

