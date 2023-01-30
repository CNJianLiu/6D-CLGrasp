import os
import time
import argparse
import torch
import tensorflow as tf
from lib.auto_encoder import GSENet
from lib.loss import ChamferLoss
from data.shape_dataset import ShapeDataset
from lib.utils import setup_logger


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='number of points, needed if use points')
parser.add_argument('--emb_dim', type=int, default=512, help='dimension of latent embedding [default: 512]')
parser.add_argument('--h5_file', type=str, default='data/obj_models/ShapeNetCore_4096.h5', help='h5 file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='train_results/autoencoder', help='directory to save train results')
opt = parser.parse_args()

opt.repeat_epoch = 10
opt.decay_step = 5000
opt.decay_epoch = [0, 150, 300, 450]
opt.decay_rate = [1.0, 0.6, 0.3, 0.1]

def train_net():
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.create_file_writer(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    estimator = GSENet(opt.emb_dim, opt.num_point)
    estimator.cuda()
    criterion = ChamferLoss()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))
    # dataset
    train_dataset = ShapeDataset(opt.h5_file, mode='train', augment=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.num_workers)
    val_dataset = ShapeDataset(opt.h5_file, mode='val', augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)
    # train
    st_time = time.time()
    global_step = ((train_dataset.length + opt.batch_size - 1) // opt.batch_size) * opt.repeat_epoch * (opt.start_epoch - 1)
    
    n_decays = len(opt.decay_epoch)
    assert len(opt.decay_rate) == n_decays
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i
    current_lr = opt.lr * opt.decay_rate[decay_count]
    optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)

    for epoch in range(opt.start_epoch, opt.max_epoch+1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        batch_idx = 0
        estimator.train()
        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(train_dataloader):
                # label must be zero_indexed
                batch_xyz, batch_label = data
                batch_xyz = batch_xyz[:, :, :3].cuda()
                optimizer.zero_grad()
                embedding, point_cloud = estimator(batch_xyz)
                loss, _, _ = criterion(point_cloud, batch_xyz)
            with tb_writer.as_default():
                tf.summary.scalar('learning_rate', current_lr, step=global_step)
                tf.summary.scalar('train_loss', loss.item(), step=global_step)
                tb_writer.flush()  

                # backward
                loss.backward()
                optimizer.step()
                global_step += 1
                batch_idx += 1

                if batch_idx % 10 == 0:
                    logger.info('Batch {0} Loss:{1:f}'.format(batch_idx, loss))
        
        # adjust learning rate if needed
        if decay_count < len(opt.decay_rate):
            if epoch >= opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1
        
        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))
        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))
        estimator.eval()
        val_loss = 0.0
        for i, data in enumerate(val_dataloader, 1):
            batch_xyz, batch_label = data
            batch_xyz = batch_xyz[:, :, :3].cuda()
            embedding, point_cloud = estimator(batch_xyz)
            loss, _, _ = criterion(point_cloud, batch_xyz)
            val_loss += loss.item()
            logger.info('Batch {0} Loss:{1:f}'.format(i, loss))
        val_loss = val_loss / i
        with tb_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tb_writer.flush()

        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))


if __name__ == '__main__':
    train_net()
