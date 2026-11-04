#!/bin/bash
# Train YOLO11-seg (ultralytics recipe: yolo11l-seg COCO-pretrained, 100 epochs, imgsz 640;
# total batch 64, which is ultralytics' nominal batch, so the LR is unchanged).
# The YOLO dataset is written by tools/data/coco_to_yolo_seg.py. Run from the repository root.
#   bash tools/detectors/train_yolo_seg.sh data/instseg_yolo/dataset.yaml 0,1
# Multi-GPU DDP training needs find_unused_parameters=True (the frozen DFL layer of the
# segmentation head is unused); yolo_ddp_patch/ sets it without changing the training math.
set -euo pipefail
DATA=$1; GPUS=${2:-0}
export PYTHONPATH="$(pwd)/tools/detectors/yolo_ddp_patch:${PYTHONPATH:-}"
python -c "
from ultralytics import YOLO
YOLO('yolo11l-seg.pt').train(data='$DATA', epochs=100, batch=64, imgsz=640, device='$GPUS',
                             project='work_dirs/yolo11seg', name='train', exist_ok=True)
"
