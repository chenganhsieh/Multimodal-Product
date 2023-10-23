log_config = dict(interval=10, hooks=[dict(type='DetailTextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=1200)
checkpoint_config = dict(by_epoch=True, interval=300)
evaluation = dict(
    interval=5,
    metric='hmean-iou',
    save_best='0_icdar2015_test_hmean-iou:hmean',
    rule='greater')
custom_imports = dict(
    imports=['ocrclip', 'datasets', 'hooks', 'optimizer'],
    allow_failed_imports=False)
opencv_num_threads = 0
mp_start_method = 'fork'
prompt_class_names = ['the pixels of many arbitrary-shape text instances.']
model = dict(
    type='OCRCLIP',
    pretrained='./pretrained/RN50.pt',
    context_length=14,
    class_names=['the pixels of many arbitrary-shape text instances.'],
    use_learnable_prompt=True,
    use_learnable_prompt_only=False,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=640,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=18,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    prompt_generator=dict(
        type='PromptGenerator',
        visual_dim=1024,
        token_embed_dim=512,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    visual_prompt_generator=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2049], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(
            type='DBParamPostprocessor',
            text_repr_type='quad',
            mask_thr=0.28,
            min_text_score=0.2,
            min_text_width=2,
            use_approxPolyDP=True,
            unclip_ratio=1.6,
            arcLength_ratio=0.001,
            max_candidates=3000)),
    identity_head=dict(
        type='IdentityHead',
        downsample_ratio=32.0,
        loss_weight=1.0,
        reduction='mean',
        negative_ratio=3.0,
        bbce_loss=True),
    train_cfg=None,
    test_cfg=None,
    scale_matching_score_map=False)
dataset_type = 'OCRCLIPDataset'
data_root = './data/icdar2015'
train = dict(
    type='OCRCLIPDataset',
    ann_file='./data/icdar2015/instances_training.json',
    img_prefix='./data/icdar2015/imgs',
    data_name='icdar2015',
    pipeline=None)
test = dict(
    type='OCRCLIPDataset',
    ann_file='./data/icdar2015/instances_test.json',
    img_prefix='./data/icdar2015/imgs',
    data_name='icdar2015_test',
    pipeline=None)
train_list = [
    dict(
        type='OCRCLIPDataset',
        ann_file='./data/icdar2015/instances_training.json',
        img_prefix='./data/icdar2015/imgs',
        data_name='icdar2015',
        pipeline=None)
]
test_list = [
    dict(
        type='OCRCLIPDataset',
        ann_file='./data/icdar2015/instances_test.json',
        img_prefix='./data/icdar2015/imgs',
        data_name='icdar2015_test',
        pipeline=None)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline_r18 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
train_pipeline_r18_shrink = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.9),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline_1333_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline_r50dcnv2 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[122.67891434, 116.66876762, 104.00698793],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
train_pipeline_r50dcnv2_shrink = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[122.67891434, 116.66876762, 104.00698793],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.9),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_4068_1152 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1152),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_4068_1152 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1152),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_736_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_736_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_4068_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_4068_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 736),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_800_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_800_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_2944_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_4068_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_2944_800 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_1600_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_vis_1600_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                call_super=False,
                keys=['ann_info'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'ann_info'])
        ])
]
test_pipeline_1024_768 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_2944_768 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2944, 768),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline_list = [[{
    'type': 'LoadImageFromFile',
    'color_type': 'color_ignore_orientation'
}, {
    'type':
    'MultiScaleFlipAug',
    'img_scale': (4068, 1024),
    'flip':
    False,
    'transforms': [{
        'type': 'Resize',
        'img_scale': (2944, 736),
        'keep_ratio': True
    }, {
        'type': 'Normalize',
        'mean': [122.67891434, 116.66876762, 104.00698793],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True
    }, {
        'type': 'Pad',
        'size_divisor': 32
    }, {
        'type': 'ImageToTensor',
        'keys': ['img']
    }, {
        'type': 'Collect',
        'keys': ['img']
    }]
}],
                      [{
                          'type': 'LoadImageFromFile',
                          'color_type': 'color_ignore_orientation'
                      }, {
                          'type':
                          'MultiScaleFlipAug',
                          'img_scale': (4068, 1024),
                          'flip':
                          False,
                          'transforms': [{
                              'type': 'Resize',
                              'img_scale': (2944, 736),
                              'keep_ratio': True
                          }, {
                              'type':
                              'Normalize',
                              'mean':
                              [122.67891434, 116.66876762, 104.00698793],
                              'std': [58.395, 57.12, 57.375],
                              'to_rgb':
                              True
                          }, {
                              'type': 'Pad',
                              'size_divisor': 32
                          }, {
                              'type': 'ImageToTensor',
                              'keys': ['img']
                          }, {
                              'type': 'Collect',
                              'keys': ['img']
                          }]
                      }],
                      [{
                          'type': 'LoadImageFromFile',
                          'color_type': 'color_ignore_orientation'
                      }, {
                          'type':
                          'MultiScaleFlipAug',
                          'img_scale': (800, 800),
                          'flip':
                          False,
                          'transforms': [{
                              'type': 'Resize',
                              'img_scale': (800, 800),
                              'keep_ratio': True
                          }, {
                              'type':
                              'Normalize',
                              'mean':
                              [122.67891434, 116.66876762, 104.00698793],
                              'std': [58.395, 57.12, 57.375],
                              'to_rgb':
                              True
                          }, {
                              'type': 'Pad',
                              'size_divisor': 32
                          }, {
                              'type': 'ImageToTensor',
                              'keys': ['img']
                          }, {
                              'type': 'Collect',
                              'keys': ['img']
                          }]
                      }],
                      [{
                          'type': 'LoadImageFromFile',
                          'color_type': 'color_ignore_orientation'
                      }, {
                          'type':
                          'MultiScaleFlipAug',
                          'img_scale': (4068, 736),
                          'flip':
                          False,
                          'transforms': [{
                              'type': 'Resize',
                              'img_scale': (2944, 736),
                              'keep_ratio': True
                          }, {
                              'type':
                              'Normalize',
                              'mean':
                              [122.67891434, 116.66876762, 104.00698793],
                              'std': [58.395, 57.12, 57.375],
                              'to_rgb':
                              True
                          }, {
                              'type': 'Pad',
                              'size_divisor': 32
                          }, {
                              'type': 'ImageToTensor',
                              'keys': ['img']
                          }, {
                              'type': 'Collect',
                              'keys': ['img']
                          }]
                      }]]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRCLIPDataset',
                ann_file='./data/icdar2015/instances_training.json',
                img_prefix='./data/icdar2015/imgs',
                data_name='icdar2015',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ImgAug',
                args=[['Fliplr', 0.5], {
                    'cls': 'Affine',
                    'rotate': [-10, 10]
                }, ['Resize', [0.5, 3.0]]]),
            dict(type='EastRandomCrop', target_size=(640, 640)),
            dict(type='DBNetTargets', shrink_ratio=0.4),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr',
                    'gt_thr_mask'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRCLIPDataset',
                ann_file='./data/icdar2015/instances_test.json',
                img_prefix='./data/icdar2015/imgs',
                data_name='icdar2015_test',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(4068, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(2944, 736), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.67891434, 116.66876762, 104.00698793],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRCLIPDataset',
                ann_file='./data/icdar2015/instances_test.json',
                img_prefix='./data/icdar2015/imgs',
                data_name='icdar2015_test',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(4068, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(2944, 736), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.67891434, 116.66876762, 104.00698793],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(
    type='Adam',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-07, by_epoch=True)
total_epochs = 1200
work_dir = './config_1023_105111'
gpu_ids = range(0, 3)
