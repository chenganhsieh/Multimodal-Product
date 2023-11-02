_base_ = [
    'dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50_1200e_icdar2015/dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth"

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

_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.002

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
param_scheduler = [
    dict(type='ConstantLR', factor=1.0),
]