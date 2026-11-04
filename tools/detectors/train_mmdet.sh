#!/bin/bash
# Train an mmdet instance-segmentation detector (Mask R-CNN, SOLOv2, Mask2Former) with its
# official recipe (total batch 16). Run from the repository root.
#   bash tools/detectors/train_mmdet.sh mask_rcnn   0,1     8   # 2 GPUs x 8
#   bash tools/detectors/train_mmdet.sh solov2      0,1     8
#   bash tools/detectors/train_mmdet.sh mask2former 0,1,2,3 4   # 4 GPUs x 4
set -euo pipefail
NAME=$1; GPUS=$2; BPG=$3; PORT=${4:-29500}
CFG=tools/detectors/configs/${NAME}_glo.py
WD=work_dirs/${NAME}
NG=$(echo "$GPUS" | awk -F, '{print NF}')
TRAIN=$(python -c "import mmdet, os; print(os.path.join(os.path.dirname(mmdet.__file__), '.mim/tools/train.py'))")
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --nproc_per_node="$NG" --master_port="$PORT" \
  "$TRAIN" "$CFG" --launcher pytorch --work-dir "$WD" --cfg-options train_dataloader.batch_size="$BPG"
