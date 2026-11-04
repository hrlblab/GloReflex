# Mask2Former R50 instance segmentation (official mmdet LSJ recipe, total batch 16, AdamW).
# Changes: one "thing" class and no "stuff" classes, the dataset, and the schedule.
# The official recipe is 50 epochs on COCO (368,750 iterations); it is kept at 50 epochs on
# our data: ceil(n_annotated_train_tiles / 16) * 50 = 156,950 iterations for our training set,
# with the LR milestones rescaled by the same factor. Adjust MAX_ITERS for another dataset.
_base_ = 'mmdet::mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'

# Dataset: single class "glomerulus", tiled instance-segmentation COCO written by
# tools/data/build_instseg_tiles.py; `file_name` is relative to `data_root`.
data_root = 'data/'
metainfo = dict(classes=('glomerulus',))
train_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='instseg_coco/train.json', data_prefix=dict(_delete_=True, img='')))
val_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=metainfo,
                 ann_file='instseg_coco/val.json', data_prefix=dict(_delete_=True, img='')))
test_dataloader = val_dataloader

MAX_ITERS = 156950
MILESTONES = [round(m * MAX_ITERS / 368750) for m in (327778, 355092)]

model = dict(
    panoptic_head=dict(num_things_classes=1, num_stuff_classes=0,
                       loss_cls=dict(class_weight=[1.0, 0.1])),
    panoptic_fusion_head=dict(num_things_classes=1, num_stuff_classes=0))

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=10000)
param_scheduler = dict(_delete_=True, type='MultiStepLR', begin=0, end=MAX_ITERS, by_epoch=False,
                       milestones=MILESTONES, gamma=0.1)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000, save_best='coco/segm_mAP',
                                     max_keep_ckpts=3))

val_evaluator = dict(_delete_=True, type='CocoMetric', ann_file='data/instseg_coco/val.json',
                     metric=['segm'], format_only=False)
test_evaluator = val_evaluator
