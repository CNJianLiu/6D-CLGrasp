import os
import cv2
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import argparse
from lib.utils import draw_detections


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='/mnt/HDD1/dataset/CAMERA_REAL/', help='camera_val, real_test') # NOCS dataset path
parser.add_argument('--result_dir', type=str, default='results/REAL_at+globalshape/', help='result directory') # REAL or CAMERA predicted results path
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
opt = parser.parse_args()

def visualize():
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    # get test data list
    # REAL
    file_path = 'Real/test_list.txt'
    # CAMERA
    # file_path = 'CAMERA/val_list.txt'
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data, file_path))]
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        img_path_parsing = img_path.split('/')

        # load result
        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(opt.result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'rb') as f:
            result = cPickle.load(f)

        # load NOCS result _REAL
        nocs_pkl_path = os.path.join('/mnt/HDD3/lj/6D-CLGrasp/results/nocs_results/real_test/', 'results_test_{}_{}.pkl'.format(
            img_path_parsing[-2], img_path_parsing[-1]))
        # load NOCS result _CAMERA
        # nocs_pkl_path = os.path.join('/mnt/HDD3/lj/6D-CLGrasp/results/nocs_results/val/', 'results_val_{}_{}.pkl'.format(
        #     img_path_parsing[-2], img_path_parsing[-1]))

        with open(nocs_pkl_path, 'rb') as f:
            nocs_result = cPickle.load(f)

        result['nocs_RTs'] = nocs_result['pred_RTs']
        result['nocs_scales'] = nocs_result['pred_scales']
        result['nocs_class_ids'] = nocs_result['pred_class_ids']
        
        img = cv2.imread(img_path + '_color.png')
        draw_detections(img, '/mnt/HDD3/lj/6D-CLGrasp/results/vis_results/REAL_at+globalshape', img_path_parsing[-2], img_path_parsing[-1], real_intrinsics, result['pred_RTs'], result['pred_scales'], result['pred_class_ids'], result['gt_RTs'], result['gt_scales'], result['gt_class_ids'], result['nocs_RTs'], result['nocs_scales'], result['nocs_class_ids'], draw_nocs=True)

if __name__ == '__main__':
    visualize()