# Mask R-CNN R50-FPN 1x (official mmdet recipe: 12 epochs, SGD lr 0.02, total batch 16).
# Only the number of classes and the dataset are changed.
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

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

model = dict(roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

val_evaluator = dict(ann_file='data/instseg_coco/val.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator
