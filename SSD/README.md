# SSD (Single Shot MultiBox Detector) - Tensorflow 2.0

## More Information
There's a lot more detail about that code on my blog.  


## Preparation
1. Download <a href="http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html#devkit">PASCAL VOC dataset (2012)</a>, <a href="http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/">PASCAL VOC test (2012)</a> and extract at `./data`  
Directory Structure  
- data/train/JPEGImages
- data/train/Annotations
- data/test/JPEGImages
- data/test/Annotations

2. preprocessing testset  
```
python preprocess_tesy.py
```
**data direcory**  
- data/train/JPEGImages
- data/train/Annotations
- data/test/JPEGImages
- data/test/Annotations
- data/preprocessing_test/JPEGImages
- data/preprocessing_test/Annotations

3. Install necessary dependencies:
```
pip install -r requirements.txt
```

## Training
Arguments for the training script:

```
>> python train.py --help
usage: train.py [-h] [--data-dir DATA_DIR] [--batch-size BATCH_SIZE]
                [--num-batches NUM_BATCHES] [--neg-ratio NEG_RATIO]
                [--initial-lr INITIAL_LR] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--num-epochs NUM_EPOCHS]
                [--checkpoint-dir CHECKPOINT_DIR]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--batch-size` training batch size
-  `--num-batches` number of batches to train (`-1`: train all)
-  `--neg-ratio` ratio used in hard negative mining when computing loss
-  `--initial-lr` initial learning rate
-  `--momentum` momentum value for SGD
-  `--weight-decay` weight decay value for SGD
-  `--num-epochs` number of epochs to train
-  `--checkpoint-dir` checkpoint directory
-  `--gpu-id` GPU ID

example)  
```
python train.py --batch-size 4 --gpu-id 0
```

## Testing
Arguments for the testing script:
```
>> python test.py --help
usage: test.py [-h] [--data-dir DATA_DIR] [--num-examples NUM_EXAMPLES]
               [--pretrained-type PRETRAINED_TYPE]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--checkpoint-path CHECKPOINT_PATH] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--num-examples` number of examples to test (`-1`: test all)
-  `--checkpoint-dir` checkpoint directory
-  `--checkpoint-path` path to a specific checkpoint
-  `--pretrained-type` pretrained weight type (`latest`: automatically look for newest checkpoint in `checkpoint_dir`, `specified`: use the checkpoint specified in `checkpoint_path`)
-  `--gpu-id` GPU ID

example)
```
python test.py --checkpoint-path ./checkpoints/ssd_epoch_110.h5 --num-examples 40
```

## Reference
- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)
- Caffe original implementation: [code](https://github.com/weiliu89/caffe/tree/ssd)
- Pytorch implementation: [code](https://github.com/ChunML/ssd-pytorch)
- Original Code: [code](https://github.com/ChunML/ssd-tf2)
