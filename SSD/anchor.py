import itertools
import math
import tensorflow as tf


def generate_default_boxes(config):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
        scales: boxes' size relative to image's size
        fm_sizes: sizes of feature maps
        ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4) with format (cx, cy, w, h)
    """

    # Config를 Argument로 받아 미리 지정되어있는 config.yaml File의 Parameter값을 가져오게 된다.
    default_boxes = []
    scales = config['scales']
    fm_sizes = config['fm_sizes']
    ratios = config['ratios']

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            # cx, cy 정의
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            # Aspect ratio가 1인경우
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])
            
            # Aspect ratio가 1이 아닌경우 (2,3)
            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    # Defult Boxes는 N*(cx,cy,w,h)로서 정의되게 된다.
    default_boxes = tf.constant(default_boxes)
    # 0~1 사이의 값으로서 정규화
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes
