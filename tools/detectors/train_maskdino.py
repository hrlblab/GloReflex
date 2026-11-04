#!/usr/bin/env python
"""Train MaskDINO with its official driver on the glomerulus dataset.

MaskDINO's LSJ data mapper needs polygon annotations; convert the RLE COCO first
with ``tools/data/coco_rle_to_polygon.py``.

    MASKDINO_ROOT=third_party/MaskDINO python tools/detectors/train_maskdino.py \
        --config-file tools/detectors/configs/maskdino_glo.yaml --num-gpus 4 \
        --data-root data --train-json data/instseg_coco_poly/train.json --val-json data/instseg_coco_poly/val.json
"""
import os
import random
import sys

sys.path.insert(0, os.environ.get("MASKDINO_ROOT", "third_party/MaskDINO"))

from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.data.datasets import register_coco_instances  # noqa: E402
from detectron2.engine import default_argument_parser, launch  # noqa: E402

import train_net  # noqa: E402  (MaskDINO's official training script)


def register(args):
    existing = set(DatasetCatalog.list())
    for name, js in (("glo_instseg_train", args.train_json), ("glo_instseg_val", args.val_json)):
        if name not in existing:
            register_coco_instances(name, {}, js, args.data_root)


def main(args):
    register(args)
    return train_net.main(args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    parser.add_argument("--data-root", required=True, help="directory that COCO file_name is relative to")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    args = parser.parse_args()
    args.dist_url = "tcp://127.0.0.1:" + str(random.randint(1000, 20000))
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
