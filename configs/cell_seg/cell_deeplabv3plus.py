from mmseg.datasets.custom_dataset import MyCustomDataset
import sys
sys.path.append('./mmsegmentation')

# from mmengine.config import read_base

# with read_base():
_base_ = [
        '../_base_/models/deeplabv3plus_r50-d8.py',
        '../_base_/datasets/ade20k.py',
        '../_base_/default_runtime.py',
        '../_base_/schedules/schedule_160k.py'
    ]


data_root = 'data/cell_dataset/'
dataset_type = 'MyCustomDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size_divisor=32),  # 使用 size_divisor，而不是固定尺寸
    dict(type='Pad', size_divisor=32, pad_val=0),
    dict(type='PackSegInputs')  # 替代 `Collect`
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline,
        reduce_zero_label=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        reduce_zero_label=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        reduce_zero_label=False,
    )
)
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),  # 控制台文本日志输出
        dict(type='CustomTensorboardLoggerHook', log_dir='./work_dirs/tf_logs')  # 使用自定义的 TensorBoard Hook
    ]
)

default_hooks = dict(
    logger=dict(type='LoggerHook'),
    timer=dict(type='IterTimerHook'),
    # 添加 TensorboardLoggerHook
    tensorboard=dict(type='TensorboardLoggerHook'),
)



optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)