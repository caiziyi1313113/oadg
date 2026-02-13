# Cityscapes-COCO baseline for JDet FasterRCNN (HBB)
# NOTE: Use JDet's COCO dataset loader with Cityscapes COCO json.
'''
它的作用是：把实验名、路径、模型结构、训练/测试超参、数据集、优化器、学习率策略、日志/保存频率
都用一个 Python dict 组织好，供 init_cfg() 读入，
然后 JDetRunner 在内部按这些配置去构建模型/数据集/optimizer/scheduler 并运行。
'''
import os
# Normalize 用的均值/方差，以及是否把 RGB 转 BGR
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
CITYSCAPES_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
num_classes = len(CITYSCAPES_CLASSES) + 1

num_views = 2
lw_jsd_rpn = 0.1
lw_jsd_roi = 10.0
lw_cont = 0.01
temperature = 0.06
random_proposal_cfg = dict(
    bbox_from='oagrb',
    num_bboxes=10,
    scales=(0.01, 0.3),
    ratios=(0.3, 1 / 0.3),
    iou_max=0.7,
    iou_min=0.0,
)

# experiment naming / logging
# 训练输出目录和数据集目录
name = "faster_rcnn_r50_fpn_cityscapes_coco_jdet_oadg"

if os.name == "nt":
    work_dir = r"D:\sim2real\OA-DG\jdet_cityscapes\work_dirs\faster_rcnn_r50_fpn_cityscapes_coco_jdet_oadg"
    cityscapes_root = r"D:\sim2real\OA-DG\ws\data\cityscapes"
else:
    work_dir = "/mnt/d/sim2real/OA-DG/jdet_cityscapes/work_dirs/faster_rcnn_r50_fpn_cityscapes_coco_jdet_oadg"
    cityscapes_root = "/mnt/d/sim2real/OA-DG/ws/data/cityscapes"

# data paths
# 将 D:\ 替换为 /mnt/d/，并将所有 \ 替换为 /
train_img_root = os.path.join(cityscapes_root, 'leftImg8bit', 'train')
val_img_root = os.path.join(cityscapes_root, 'leftImg8bit', 'val')

train_ann = os.path.join(cityscapes_root, 'annotations', 'instancesonly_filtered_gtFine_train.json')
val_ann = os.path.join(cityscapes_root, 'annotations', 'instancesonly_filtered_gtFine_val.json')

# 从清华大学服务器下载模型
# 选择2：可以用 JDet 模型 zoo 的 FasterRCNN-R50-FPN detector 预训练作为 resume_path / load_from
# model 字典：定义整个检测模型（这是配置的核心）
model = dict(
    # 自定义/扩展的 OBB FasterRCNN 版本
    type='FasterRCNNHBBMulti',
    # 告诉 backbone 用 resnet50 的预训练（来自 JDet 模型 zoo）
    pretrained='modelzoo://resnet50',
    # backbone 的配置
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1", "layer2", "layer3", "layer4"],
        pretrained=True,
        norm_eval=True,
    ),
    # FPN 颈部的配置
    neck=dict(
        type='FPN',
        # FPN 的输入通道
        in_channels=[256, 512, 1024, 2048],
        # FPN 的输出通道
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=5,
    ),
    # RPN 头的配置，（提 proposal）
    rpn_head=dict(
        type='FasterrcnnHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLossPlus',
            use_sigmoid=True,
            loss_weight=1.0,
            num_views=num_views,
            additional_loss='jsdv1_3_2aug',
            lambda_weight=lw_jsd_rpn,
        ),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views),
    ),
    # RoI 特征提取
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=0, version=1),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    # bbox_head：二阶段分类+回归
    bbox_head=dict(
        type='Shared2FCContrastiveHeadHBB',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=num_classes,
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        with_module=False,
        with_cont=True,
        cont_predictor_cfg=dict(num_linear=2, feat_channels=256, return_relu=True),
        loss_cls=dict(
            type='CrossEntropyLossPlus',
            use_sigmoid=False,
            loss_weight=1.0,
            num_views=num_views,
            additional_loss='jsdv1_3_2aug',
            lambda_weight=lw_jsd_roi,
        ),
        loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0, num_views=num_views),
        loss_cont=dict(type='ContrastiveLossPlus', loss_weight=lw_cont, num_views=num_views, temperature=temperature),
    ),
    # train_cfg：训练阶段的 assign / sample / nms 等策略
    # 正负样本 IoU 阈值
    # 每张图采多少 anchors（256）以及正样本比例（0.5）
    # RPN proposal 的 nms_pre/nms_post 等
    # 二阶段的 assigner/sampler
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                match_low_quality=True,
                iou_calculator=dict(type='BboxOverlaps2D_v1'),
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000, # 在执行 NMS（非极大值抑制）之前，保留得分最高的框的数量
            nms_post=1000,
            max_num=1000,#在执行 NMS 之后，每张图片最终保留的候选框数量。和nms_post等价
            nms_thr=0.7,
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D_v1'),
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
        random_proposal_cfg=random_proposal_cfg,
    ),
    # 推理阶段阈值和 NMS
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5), 
            max_per_img=100, # align with MMDetection default
        ),
    ),
)

# 在配置文件的 末尾（或者 model 定义之后），添加 load_from 字段。指向你下载或保存的 DOTA 预训练模型路径。
# 指定 DOTA 预训练权重的路径
# 注意这里的加载权重是整个detector级别的，而之前的pretrain是仅有后端初始化
# NOTE: The DOTA FasterRCNN premodel has mismatched class heads (16 classes).
# Use backbone-only pretrain (modelzoo://resnet50) and skip detector-level load.
# set to None when you want to resume from checkpoints
pretrained_weights = "/mnt/d/sim2real/OA-DG/jdet_cityscapes/minimal_jdet.pkl"
#load_from = './premodel/FasterRCNN-R50-FPN.pkl'  # noqa F401
# Note: do not print here; it runs on import and can be misleading in eval scripts.
# dataset settings
# resume_path can be set to a specific ckpt; None -> auto search latest in work_dir
resume_path = None
# 定义 train/val/test 的数据集构建方式
# dataset settings (MMDet-style layout, mapped to JDet COCODataset)
dataset_type = 'CityscapesDataset'
data_root = cityscapes_root

# JDet uses transforms (not mmdet pipelines). Keep equivalent ops here.
train_pipeline = [
    dict(type='ResizeByImgScale', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='OAMix',
        version='augmix',
        num_views=num_views,
        keep_orig=True,
        severity=10,
        random_box_ratio=(3, 1 / 3),
        random_box_scale=(0.01, 0.1),
        oa_random_box_scale=(0.005, 0.1),
        oa_random_box_ratio=(3, 1 / 3),
        spatial_ratio=4,
        sigma_ratio=0.3,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
]


test_pipeline = [
    dict(type='ResizeByImgScale', img_scale=(2048, 1024), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
]
# 这里不采用repeat dataset，直接用iter控制进行充分学习
# 暂时不支持多尺度的测试
dataset = dict(
    train=dict(
        type=dataset_type,
        root=train_img_root,
        anno_file=train_ann,
            use_anno_cats=False,
        transforms=train_pipeline,
        batch_size=1,
        num_workers=1,
        shuffle=True,
    ),
    val=dict(
        type=dataset_type,
        root=val_img_root,
        anno_file=val_ann,
        use_anno_cats=False,
        transforms=test_pipeline,
        batch_size=1,
        num_workers=2,
        shuffle=False,
    ),
    test=dict(
        type=dataset_type,
        root=val_img_root,
        anno_file=val_ann,
        use_anno_cats=False,
        transforms=test_pipeline,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    ),
)

optimizer = dict(
    type='SGD',
    lr=0.00125,
    momentum=0.9,
    weight_decay=0.0001,
    grad_clip=dict(max_norm=35, norm_type=2),
)

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    milestones=[35000],
)

logger = dict(type='RunLogger')

# runtime settings
# 使iter级别
max_iter = 50000
max_epoch = None

# run a val epoch once after training finishes (epoch increments when loop ends)
eval_interval = 1
eval_interval_iter = 1000
# checkpoint_interval = 1
checkpoint_interval_iter = 100
log_interval = 10

val_vis_num = 20
val_vis_score_thr = 0.0
val_vis_dir = None  # 可选，不写默认 work_dir/val_vis


'''
run_net.py 调 init_cfg(config_file)：会执行并加载这个配置文件里的变量（model/dataset/optimizer/...）。

faster_rcnn_r50_fpn_1x_cityscap…

runner = Runner()：你的 Runner 继承自 JDetRunner，基类会读 cfg 并构建：

self.model（按 model dict）

self.train_dataset/val_dataset（按 dataset dict）

self.optimizer（按 optimizer dict）

self.scheduler（按 scheduler dict）

self.logger（按 logger dict）

self.total_iter（来自 max_iter）

runner.run()：虽然你没定义 run()，但它在 基类 JDetRunner 里；它内部会调用 self.train() —— 于是就会用到你覆盖后的训练循环（含 iter-based ckpt）。
'''

