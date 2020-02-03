import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random

from box_utils import compute_target
from image_utils import horizontal_flip
from functools import partial

# 실질적인 Dataset을 Training Data와 Validation Data로서 나눈 뒤, 
# Batch 처리까지하여 Model에 넣을 수 있게 하는 Preprocessing 단계이다. 

# Data초기에 필요한 Argument들을 정의하는 부분이다. 
class VOCDataset():
    def __init__(self, data_dir, default_boxes,num_examples=-1):
        super(VOCDataset, self).__init__()
        # 미리 정해져있는 20개의 Label을 정의한 것
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        
        # {'aeroplane': 0, 'bicycle': 1,...} 형식으로 Label의 이름과 Index를 Dict Type으로 선언
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])
        # Image 경로
        self.image_dir = data_dir+'/JPEGImages'
        # Annotations(Bounding Box의 Label 및 (xmin,ymin,xmax,ymax)) 경로
        self.anno_dir = data_dir+'/Annotations'
        
        
        # Image와 해당되는 Annotation이 맞는지 확인하기 위한 것.
        self.ids = list(map(lambda x: x[:-4], os.listdir(self.image_dir)))
        # 입력 받는 Default Boxes
        self.default_boxes = default_boxes
        # Model Input으로 들어가는 Image의 Size
        self.new_size = 300

        if num_examples != -1:
            self.ids = self.ids[:num_examples]
        # Trainning Dataset, 전체 Dataset의 75%
        self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        # Validation Dataset, 전체 Dataset의 25%
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]
        # 위의 image_utils.py를 활용하여 Dataset을 원래대로 사용할지 Flip한 Dataset을 사용할지 결정하기 위해서
        self.augmentation = ['original','flip']
    
    # 전체 데이터의 개수 파악  
    def __len__(self):
        return len(self.ids)
    
    # 해당되는 Index의 Image를 반환
    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_path)

        return img
    
    # 해당되는 Index의 Annotation.xml을 통하여 
    # Label,(xmin, ymin, xmax, ymax)을 반환-> 0~1사이의 값으로서 정규화 
    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index]
        anno_path = os.path.join(self.anno_dir, filename + '.xml')
        objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            xmin = (float(bndbox.find('xmin').text) - 1) / w
            ymin = (float(bndbox.find('ymin').text) - 1) / h
            xmax = (float(bndbox.find('xmax').text) - 1) / w
            ymax = (float(bndbox.find('ymax').text) - 1) / h
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    # 실질적인 Dataset 생성을 위하여 필요
    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        
        # 만약 Train인 경우 File적용 Test라면 Filp 적용 X
        if subset == 'train':
            indices = self.train_ids
            # 3. Random하게 Original Image를 사용할지 Flip을 실행할 Image를 사용할지 결정한다.
            augmentation_method = np.random.choice(self.augmentation)
            if augmentation_method == 'flip':
                img, boxes, labels = horizontal_flip(img, boxes, labels)
                
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        for index in range(len(indices)):
            # img, orig_shape = self._get_image(index)
            filename = indices[index]
            img = self._get_image(index)
            
            # 1. Input Image의 Size를 받는다.
            w, h = img.size
            
            # 2. get_annotation()을 통하여 Label과 Bounding Box의 Location을 입력받는다.
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)
            
            # 4. Image의 Size를 Model Input에 맞게 (300,300)으로 바꾼뒤 0 ~ 1 사이의 값으로서 정규화를 한다.
            img = np.array(img.resize(
                (self.new_size, self.new_size)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)
            
            # 5. Utils -> box_utils -> compute_target를 통하여 실제 Label을 Model에 맞는 Label로서 변경한다.
            gt_confs, gt_locs = compute_target(
                self.default_boxes, boxes, labels)

            # 6. Filename, Image, Ground Truth Label, Ground Truth Location을 반환한다
            # Generator로서 특정 Index후 다음 Index로 반환하기 위하여 Return 값을 yield로서 선언
            yield filename, img, gt_confs, gt_locs

# create_batch_generator(): Batch_Size를 입력받아 Dataset을 생성한다.
def create_batch_generator(data_dir,default_boxes,batch_size, num_batches,
                           mode):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    voc = VOCDataset(data_dir,default_boxes,num_examples)

    info = {
        'idx_to_name': voc.idx_to_name,
        'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        'image_dir': voc.image_dir,
        'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen = partial(voc.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, subset='val')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info
