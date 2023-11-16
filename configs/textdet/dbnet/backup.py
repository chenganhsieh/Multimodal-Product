_base_ = [
    '_base_clip_db_r50_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]

# Save checkpoints every 10 epochs, and only keep the latest checkpoint
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=1,
    ))
# Set the maximum number of epochs to 400, and validate the model every 10 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=10)
# Fix learning rate as a constant


# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test = _base_.icdar2015_textdet_test
icdar2015_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)

_base_.optim_wrapper.optimizer.lr = 0.002
