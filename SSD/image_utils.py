import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import tensorflow as tf

from box_utils import compute_iou

# Model이 Predicted한 Object Class와 Box Localization, 
# Image를 입력받아 해당되는 Image에 Box를 표시한 뒤, Class를 표시하고 저장한다.
class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )

        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')

# Image를 Flipped한 뒤 Box의 위치를 조정한다.
def horizontal_flip(img, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels
