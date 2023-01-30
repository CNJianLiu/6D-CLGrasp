import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from lib.CLGrasp import CLGraspNet
from lib.loss import *
from data.pose_dataset import PoseDataset
from lib.utils import setup_logger, compute_sRT_errors
from lib.align import estimateSimilarityTransform

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CAMERA+Real', help='CAMERA or CAMERA+Real')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--max_epoch', type=int, default=15, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
# if need resume the model
# parser.add_argument('--start_epoch', type=int, default=, help='which epoch to start')
# parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='train_results/real_at+globalshape/', help='directory to save train results')
parser.add_argument('--val_size', type=int, default=1500, help='val size for a epoch')
opt = parser.parse_args()

#original
opt.decay_epoch = [0, 5, 10] # lr=0.0001, batchsize=16, num_workers=10, max_epoch = 15, train_steps = 12000
opt.decay_rate = [1.0, 0.6, 0.3]

opt.corr_wt = 1.0
opt.cd_wt = 5.0
opt.entropy_wt = 0.0001 
opt.deform_wt = 0.01

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))    
    else:
        print("Not enough GPU hardware devices available")    
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.create_file_writer(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    logger.propagate = 0

    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    # model & loss
    estimator = CLGraspNet(opt.n_cat, opt.nv_prior, num_structure_points=256)
    estimator.cuda()
    estimator = nn.DataParallel(estimator)

    criterion = Loss(opt.corr_wt, opt.cd_wt, opt.entropy_wt, opt.deform_wt)
    lowrank_criterion = LowRank_Loss()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))

    # dataset
    train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size)
    val_dataset = PoseDataset(opt.dataset, 'test', opt.data_dir, opt.n_pts, opt.img_size) 

    # start training
    st_time = time.time()
    train_steps = 12000 
    global_step = train_steps * (opt.start_epoch - 1)
    n_decays = len(opt.decay_epoch)

    assert len(opt.decay_rate) == n_decays
    decay_count = 0
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i
    current_lr = opt.lr * opt.decay_rate[decay_count]
    optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr) 
    train_size = train_steps * opt.batch_size 
    indices = []
    page_start = -train_size
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))

        # create optimizer and adjust learning rate if needed
        if decay_count < len(opt.decay_rate):
            if epoch > opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if opt.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len+real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3*n_repeat*real_len) + real_indices*n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start+train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                                                       num_workers=opt.num_workers, pin_memory=True)
        estimator.train()
        for i, data in enumerate(train_dataloader, 1):
            points, rgb, choose, cat_id, model, prior, sRT, nocs = data
            points = points.cuda()
            rgb = rgb.cuda()
            choose = choose.cuda()
            cat_id = cat_id.cuda()
            model = model.cuda()
            prior = prior.cuda()
            sRT = sRT.cuda()
            nocs = nocs.cuda()
            structure_points, assign_mat, deltas = estimator(points, rgb, choose, cat_id, prior)

            loss1, corr_loss, cd_loss, entropy_loss, deform_loss = criterion(assign_mat, deltas, prior, nocs, model)
            loss2 = lowrank_criterion(structure_points, points)
            
            loss = 1.0 * loss1 + 1.0 * loss2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # write results to tensorboard
            with tb_writer.as_default():
                tf.summary.scalar('learning_rate', current_lr, step=global_step)
                tf.summary.scalar('train_loss', loss.item(), step=global_step)
                tf.summary.scalar('corr_loss', corr_loss.item(), step=global_step)
                tf.summary.scalar('cd_loss', cd_loss.item(), step=global_step)
                tf.summary.scalar('entropy_loss', entropy_loss.item(), step=global_step)
                tf.summary.scalar('deform_loss', deform_loss.item(), step=global_step)
                tf.summary.scalar('lowrank_loss', loss2.item(), step=global_step)
                tb_writer.flush()             

            if i % 10 == 0:
                logger.info('Batch {0} Loss:{1:f}, corr_loss:{2:f}, cd_loss:{3:f}, entropy_loss:{4:f}, deform_loss:{5:f}, lowrank_loss:{6:f}'.format(
                    i, loss.item(), corr_loss.item(), cd_loss.item(), entropy_loss.item(), deform_loss.item(), loss2.item()))
        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))


        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) +
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))
        val_loss = 0.0
        total_count = np.zeros((opt.n_cat,), dtype=int)
        strict_success = np.zeros((opt.n_cat,), dtype=int)    
        easy_success = np.zeros((opt.n_cat,), dtype=int)      
        iou_success = np.zeros((opt.n_cat,), dtype=int)       

        # sample validation subset
        val_size = opt.val_size
        val_idx = random.sample(list(range(val_dataset.length)), val_size)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                                                 num_workers=opt.num_workers, pin_memory=True)
        estimator.eval()
        for i, data in enumerate(val_dataloader, 1):
            points, rgb, choose, cat_id, model, prior, sRT, nocs = data
            points = points.cuda()
            rgb = rgb.cuda()
            choose = choose.cuda()
            cat_id = cat_id.cuda()
            model = model.cuda()
            prior = prior.cuda()
            sRT = sRT.cuda()
            nocs = nocs.cuda()
            with torch.no_grad():
                structure_points, assign_mat, deltas = estimator(points, rgb, choose, cat_id, prior)
            loss1, _, _, _, _ = criterion(assign_mat, deltas, prior, nocs, model)
            loss2 = lowrank_criterion(structure_points, points)
            loss = 1.0 * loss1 + 1.0 * loss2        
    
            # estimate pose and scale
            inst_shape = prior + deltas
            assign_mat = F.softmax(assign_mat, dim=2)
            nocs_coords = torch.bmm(assign_mat, inst_shape)
            nocs_coords = nocs_coords.detach().cpu().numpy()[0]
            points = points.cpu().numpy()[0]
            
            choose = choose.cpu().numpy()[0]
            _, choose = np.unique(choose, return_index=True)
            nocs_coords = nocs_coords[choose, :]
            points = points[choose, :]
            _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, points)

            # evaluate pose
            cat_id = cat_id.item()
            if pred_sRT is not None:
                sRT = sRT.detach().cpu().numpy()[0]
                R_error, T_error, IoU = compute_sRT_errors(pred_sRT, sRT)
                if R_error < 5 and T_error < 0.05:
                    strict_success[cat_id] += 1
                if R_error < 10 and T_error < 0.05:
                    easy_success[cat_id] += 1
                if IoU < 0.1:
                    iou_success[cat_id] += 1
            total_count[cat_id] += 1
            val_loss += loss.item()
            if i % 100 == 0:
                logger.info('Batch {0} Loss:{1:f}'.format(i, loss.item()))
        # compute accuracy
        strict_acc = 100 * (strict_success / total_count)
        easy_acc = 100 * (easy_success / total_count)
        iou_acc = 100 * (iou_success / total_count)
        for i in range(opt.n_cat):
            logger.info('{} accuracies:'.format(val_dataset.cat_names[i]))
            logger.info('5^o 5cm: {:4f}'.format(strict_acc[i]))
            logger.info('10^o 5cm: {:4f}'.format(easy_acc[i]))
            logger.info('IoU < 0.1: {:4f}'.format(iou_acc[i]))
        strict_acc = np.mean(strict_acc)
        easy_acc = np.mean(easy_acc)
        iou_acc = np.mean(iou_acc)
        val_loss = val_loss / val_size
        with tb_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tf.summary.scalar('5^o5cm_acc', strict_acc, step=global_step)
            tf.summary.scalar('10^o5cm_acc', easy_acc, step=global_step)
            tf.summary.scalar('iou_acc', iou_acc, step=global_step)
            tb_writer.flush()        

        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('Overall accuracies:')
        logger.info('5^o 5cm: {:4f} 10^o 5cm: {:4f} IoU: {:4f}'.format(strict_acc, easy_acc, iou_acc))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))

if __name__ == '__main__':
    train()