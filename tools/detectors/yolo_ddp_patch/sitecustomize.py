"""Force ``find_unused_parameters=True`` for DistributedDataParallel.

ultralytics YOLO11-seg multi-GPU training fails with "Expected to have finished
reduction ..." because the frozen DFL convolution receives no gradient. This only
changes a DDP flag (loss, optimiser and batch are unchanged). Loaded through
PYTHONPATH by ``train_yolo_seg.sh`` so it also applies to the spawned DDP workers.
"""
try:
    import torch
    _DDP = torch.nn.parallel.DistributedDataParallel
    _orig = _DDP.__init__

    def _init(self, *args, **kwargs):
        kwargs["find_unused_parameters"] = True
        _orig(self, *args, **kwargs)

    _DDP.__init__ = _init
except Exception:  # noqa: BLE001 -- never break interpreter start-up
    pass
