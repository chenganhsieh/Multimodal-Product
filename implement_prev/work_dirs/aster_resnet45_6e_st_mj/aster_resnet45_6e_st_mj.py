auto_scale_lr = dict(base_batch_size=1024)
cute80_textrecog_data_root = 'data/cute80'
cute80_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/cute80',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
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
dictionary = dict(
    dict_file=
    '/home/biometrics/reserve/Multimodal-Product/implement_prev/configs/textrecog/aster/../../../dicts/english_digits_symbols.txt',
    same_start_end=True,
    type='Dictionary',
    with_end=True,
    with_padding=True,
    with_start=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2013_857_textrecog_test = dict(
    ann_file='textrecog_test_857.json',
    data_root='data/icdar2013',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2013_textrecog_data_root = 'data/icdar2013'
icdar2013_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/icdar2013',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2013_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/icdar2013',
    pipeline=None,
    type='OCRDataset')
icdar2015_1811_textrecog_test = dict(
    ann_file='textrecog_test_1811.json',
    data_root='data/icdar2015',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_data_root = 'data/icdar2015'
icdar2015_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/icdar2015',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/icdar2015',
    pipeline=None,
    type='OCRDataset')
iiit5k_textrecog_data_root = 'data/iiit5k'
iiit5k_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/iiit5k',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
iiit5k_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/iiit5k',
    pipeline=None,
    type='OCRDataset')
launcher = 'none'
load_from = 'checkpoints/textrecog/aster_resnet45_6e_st_mj-cc56eca4.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
mjsynth_sub_textrecog_train = dict(
    ann_file='subset_textrecog_train.json',
    data_root='data/mjsynth',
    pipeline=None,
    type='OCRDataset')
mjsynth_textrecog_data_root = 'data/mjsynth'
mjsynth_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/mjsynth',
    pipeline=None,
    type='OCRDataset')
model = dict(
    backbone=dict(
        arch_channels=[
            32,
            64,
            128,
            256,
            512,
        ],
        arch_layers=[
            3,
            4,
            6,
            6,
            3,
        ],
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        in_channels=3,
        init_cfg=[
            dict(layer='Conv2d', type='Kaiming'),
            dict(layer='BatchNorm2d', type='Constant', val=1),
        ],
        stem_channels=[
            32,
        ],
        strides=[
            (
                2,
                2,
            ),
            (
                2,
                2,
            ),
            (
                2,
                1,
            ),
            (
                2,
                1,
            ),
            (
                2,
                1,
            ),
        ],
        type='ResNet'),
    data_preprocessor=dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        attn_dims=512,
        dictionary=dict(
            dict_file=
            '/home/biometrics/reserve/Multimodal-Product/implement_prev/configs/textrecog/aster/../../../dicts/english_digits_symbols.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        emb_dims=512,
        hidden_size=512,
        in_channels=512,
        max_seq_len=25,
        module_loss=dict(
            flatten=True, ignore_first_char=True, type='CEModuleLoss'),
        postprocessor=dict(type='AttentionPostprocessor'),
        type='ASTERDecoder'),
    encoder=dict(in_channels=512, type='ASTEREncoder'),
    preprocessor=dict(
        in_channels=3,
        num_control_points=20,
        output_image_size=(
            32,
            100,
        ),
        resized_image_size=(
            32,
            64,
        ),
        type='STN'),
    type='ASTER')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0004,
        type='AdamW',
        weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=6,
        convert_to_iter_based=True,
        eta_min=4e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=None)
resume = False
svt_textrecog_data_root = 'data/svt'
svt_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/svt',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
svt_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/svt',
    pipeline=None,
    type='OCRDataset')
svtp_textrecog_data_root = 'data/svtp'
svtp_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='data/svtp',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
svtp_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/svtp',
    pipeline=None,
    type='OCRDataset')
synthtext_an_textrecog_train = dict(
    ann_file='alphanumeric_textrecog_train.json',
    data_root='data/synthtext',
    pipeline=None,
    type='OCRDataset')
synthtext_sub_textrecog_train = dict(
    ann_file='subset_textrecog_train.json',
    data_root='data/synthtext',
    pipeline=None,
    type='OCRDataset')
synthtext_textrecog_data_root = 'data/synthtext'
synthtext_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/synthtext',
    pipeline=None,
    type='OCRDataset')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='data/icdar2015',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                    'instances',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_test.json',
            data_root='data/icdar2015',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            256,
            64,
        ), type='Resize'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
                'instances',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
test_evaluator = dict(
    dataset_prefixes=[
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='data/icdar2015',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
            'instances',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=6, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=1024,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='data/mjsynth',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_train.json',
                data_root='data/synthtext',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_train.json',
            data_root='data/mjsynth',
            pipeline=None,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_train.json',
            data_root='data/synthtext',
            pipeline=None,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(scale=(
            256,
            64,
        ), type='Resize'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='data/mjsynth',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='data/synthtext',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    256,
                    64,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                        'instances',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='data/icdar2015',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                    'instances',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/aster_resnet45_6e_st_mj'
