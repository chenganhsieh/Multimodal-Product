auto_scale_lr = dict(base_batch_size=8)
default_hooks = dict(
    checkpoint=dict(interval=20, type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2015_textdet_data_root = 'data/icdar2015'
icdar2015_textdet_test = dict(
    ann_file='textdet_test.json',
    data_root='data/icdar2015',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            2260,
            2260,
        ), type='Resize'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    test_mode=True,
    type='OCRDataset')
icdar2015_textdet_train = dict(
    ann_file='textdet_train.json',
    data_root='data/icdar2015',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            keep_ratio=True,
            ratio_range=(
                0.75,
                2.5,
            ),
            scale=(
                800,
                800,
            ),
            type='RandomResize'),
        dict(
            crop_ratio=0.5,
            iter_num=1,
            min_area_ratio=0.2,
            type='TextDetRandomCropFlip'),
        dict(
            prob=0.8,
            transforms=[
                dict(min_side_ratio=0.3, type='RandomCrop'),
            ],
            type='RandomApply'),
        dict(
            prob=0.5,
            transforms=[
                dict(
                    max_angle=30,
                    pad_with_fixed_color=False,
                    type='RandomRotate',
                    use_canvas=True),
            ],
            type='RandomApply'),
        dict(
            prob=[
                0.6,
                0.4,
            ],
            transforms=[
                [
                    dict(keep_ratio=True, scale=800, type='Resize'),
                    dict(target_scale=800, type='SourceImagePad'),
                ],
                dict(keep_ratio=False, scale=800, type='Resize'),
            ],
            type='RandomChoice'),
        dict(direction='horizontal', prob=0.5, type='RandomFlip'),
        dict(
            brightness=0.12549019607843137,
            contrast=0.5,
            op='ColorJitter',
            saturation=0.5,
            type='TorchVisionWrapper'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    type='OCRDataset')
launcher = 'none'
load_from = 'checkpoints/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth',
            type='Pretrained'),
        out_indices=(
            1,
            2,
            3,
        ),
        type='CLIPResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        fourier_degree=5,
        in_channels=256,
        module_loss=dict(num_sample=50, type='FCEModuleLoss'),
        postprocessor=dict(
            alpha=1.2,
            beta=1.0,
            num_reconstr_points=50,
            scales=(
                8,
                16,
                32,
            ),
            score_thr=0.3,
            text_repr_type='quad',
            type='FCEPostprocessor'),
        type='FCEHead'),
    neck=dict(
        act_cfg=None,
        add_extra_convs='on_output',
        in_channels=[
            512,
            1024,
            2048,
        ],
        num_outs=3,
        out_channels=256,
        relu_before_extra_convs=True,
        type='mmdet.FPN'),
    type='FCENet')
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(end=1500, eta_min=1e-07, power=0.9, type='PolyLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2260,
                2260,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2260,
        2260,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
train_cfg = dict(max_epochs=1500, type='EpochBasedTrainLoop', val_interval=20)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='data/icdar2015',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.75,
                    2.5,
                ),
                scale=(
                    800,
                    800,
                ),
                type='RandomResize'),
            dict(
                crop_ratio=0.5,
                iter_num=1,
                min_area_ratio=0.2,
                type='TextDetRandomCropFlip'),
            dict(
                prob=0.8,
                transforms=[
                    dict(min_side_ratio=0.3, type='RandomCrop'),
                ],
                type='RandomApply'),
            dict(
                prob=0.5,
                transforms=[
                    dict(
                        max_angle=30,
                        pad_with_fixed_color=False,
                        type='RandomRotate',
                        use_canvas=True),
                ],
                type='RandomApply'),
            dict(
                prob=[
                    0.6,
                    0.4,
                ],
                transforms=[
                    [
                        dict(keep_ratio=True, scale=800, type='Resize'),
                        dict(target_scale=800, type='SourceImagePad'),
                    ],
                    dict(keep_ratio=False, scale=800, type='Resize'),
                ],
                type='RandomChoice'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                brightness=0.12549019607843137,
                contrast=0.5,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        type='OCRDataset'),
    num_workers=24,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.75,
            2.5,
        ),
        scale=(
            800,
            800,
        ),
        type='RandomResize'),
    dict(
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2,
        type='TextDetRandomCropFlip'),
    dict(
        prob=0.8,
        transforms=[
            dict(min_side_ratio=0.3, type='RandomCrop'),
        ],
        type='RandomApply'),
    dict(
        prob=0.5,
        transforms=[
            dict(
                max_angle=30,
                pad_with_fixed_color=False,
                type='RandomRotate',
                use_canvas=True),
        ],
        type='RandomApply'),
    dict(
        prob=[
            0.6,
            0.4,
        ],
        transforms=[
            [
                dict(keep_ratio=True, scale=800, type='Resize'),
                dict(target_scale=800, type='SourceImagePad'),
            ],
            dict(keep_ratio=False, scale=800, type='Resize'),
        ],
        type='RandomChoice'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        brightness=0.12549019607843137,
        contrast=0.5,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2260,
                2260,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/fcenet_resnet50-oclip_fpn_1500e_icdar2015'
