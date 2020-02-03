import argparse
import tensorflow as tf
import os
import sys
import time
import yaml

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from voc_data import create_batch_generator
from anchor import generate_default_boxes
from network import create_ssd
from losses import create_losses

# Paper와 같이 Hyperparameter를 Default로서 설정하였다.
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/train')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num-batches', default=-1, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

# 사용가능한 GPU Device를 설정한다.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# 20개의 Class + 1개의 Background
NUM_CLASSES = 21


# LossFunction과 Backpropagation을 징행한다.
@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(imgs)
        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss
        # l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        # l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        # loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))
    
    return loss, conf_loss, loc_loss

if __name__ == '__main__':
    # Model의 Checkpoints를 저장할 Directory가 없을 경우 생성한다.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 실제 SSD 300에 미리 저장되어 있는 Setting값을 가져와서 적용한다.(Anchor, FeatureMapSize 등)
    with open('./config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = cfg['SSD300']
    default_boxes = generate_default_boxes(config)
    
    # voc_data.py에서 설정한 Dataset을 Batch형태로서 가져온다.
    batch_generator, val_generator, info = create_batch_generator(args.data_dir,default_boxes,
        args.batch_size, args.num_batches,
        mode='train')
    
    # 실제 SSD Model을 설정한다. 만약, Training중이던 Model이 있으면 그대로 가져가서 사용할 수 있다.
    try:
        ssd = create_ssd(NUM_CLASSES,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()
    
    # Hard negative mining을 적용하여 Loss를 구한다.
    criterion = create_losses(args.neg_ratio, NUM_CLASSES)
    steps_per_epoch = info['length'] // args.batch_size

    # 해당 논문에서는 The learning rate decay policy is slightly different for each dataset
    # 로서 설명하였다. 정확한 방법은 나와있지 않아서 아마 원본 Code를 참고하여 만든 것 같다.
    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])
    
    # Optimizer 선언
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_fn,
        momentum=args.momentum)

    # Training의 과정을 저장할 tf.summary를 선언한다.
    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # 지정한 Epoch 만큼 Model을 Training한다.
    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            # print(i)
            
            # Batch 도중에 Loss를 확인한다.
            if (i + 1) % 50 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        
        # Training Data가 아닌 Validation으로서 확인한다.
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        # Training Loss에 관하여 tf.summary를 이용하여 저장
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        # Validation Loss에 관하여 tf.summary를 이용하여 저장
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

        # 일정 Epoch마다 Model을 Keras의 .h5형태로서 저장
        if (epoch + 1) % 10 == 0:
            ssd.save_weights(
                os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
