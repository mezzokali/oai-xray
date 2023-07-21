# python mmdetection/tools/train.py mmdetection/configs/hip/hip.py

_base_ = '../mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py'

# model
model = dict(
    bbox_head=dict(num_classes=2))

# dataset
dataset_type = 'CocoDataset'
classes = ('left', 'right', )
data_root = 'D:/Code/Hip_v2/coco/'


# preprocess
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # remove RandomFlip
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # remove RandomFlip
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        img_prefix=data_root + 'train/',
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix=data_root + 'val/',
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix=data_root + 'test/',
        classes=classes,
        ann_file=data_root + 'annotations/instances_test.json',
        pipeline=test_pipeline))

# learning schedule
runner = dict(type='EpochBasedRunner', max_epochs=100)    # 100 epochs

# Checkpoint hook config
# refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=10)

# log config
log_config = dict(
    interval=50,
    hooks=[dict(type='TensorboardLoggerHook')])    # use tensorboard log

# pre-trained model
load_from = 'D:/Code/Hip_v2/mmdetection/checkpoints/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
