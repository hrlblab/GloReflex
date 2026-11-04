#!/usr/bin/env python
"""Fine-tune the SAM2.1 refiner (LoRA on Hiera + full mask decoder).

* Prompt = current mask (+ clicks during training), never a box: in the loop the
  mask is updated every round, a box would stay frozen at round 0.
* Iterative click curriculum: for each batch a number of rounds T in [0, 8] is
  drawn. Rounds 1..T-1 evolve the mask without gradients (each adds a click
  sampled from the current error region and feeds the output back as the mask
  prompt); round T is the one with gradients. T = 0 is the mask-only case used
  at deployment.
* Loss = Dice + BCE on rows with GT, plus L1 between the IoU head and the true
  Dice of the predicted mask (target 0 for the false-positive rows that only
  supervise the IoU head).
* 15 training WSIs are held out to monitor the Dice change of a single step.

    python tools/experience/train_refiner.py --data <data_dir>/train --out <weights>/refiner \
        --epochs 3 --batch-size 4 --amp
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gloreflex import refiner as ft
from gloreflex.geometry import dice_np
from gloreflex.refiner import IMAGE_SIZE, forward, mask_to_logits, to_target
from gloreflex.refiner_data import RefinerDataset, sample_click


def save_atomic(model, out: Path, opt, sch, step: int, epoch: int) -> None:
    """Crash-safe checkpoint: write into the spare slot, then switch ``CURRENT``."""
    bad = [n for n, q in model.named_parameters() if q.requires_grad and not torch.isfinite(q).all()]
    if bad:
        raise SystemExit(f"non-finite weights, refusing to save: {bad[:5]}")
    cur = (out / "CURRENT").read_text().strip() if (out / "CURRENT").exists() else "b"
    slot = "b" if cur == "a" else "a"
    d = out / f"ckpt_{slot}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ft.save_finetuned_state(model, d, {"step": step, "epoch": epoch})
    torch.save({"opt": opt.state_dict(), "sch": sch.state_dict(), "step": step, "epoch": epoch},
               d / "trainer.pt")
    tmp = out / "CURRENT.tmp"
    tmp.write_text(slot)
    os.replace(tmp, out / "CURRENT")


def load_latest(model, out: Path, opt, sch):
    if not (out / "CURRENT").exists():
        return 0, 0
    d = out / f"ckpt_{(out / 'CURRENT').read_text().strip()}"
    if not (d / "trainer.pt").exists():
        return 0, 0
    ft.load_finetuned_state(model, d)
    t = torch.load(d / "trainer.pt", map_location="cpu")
    opt.load_state_dict(t["opt"]); sch.load_state_dict(t["sch"])
    print(f"resuming from {d.name}: step {t['step']} epoch {t['epoch']}", flush=True)
    return int(t["step"]), int(t["epoch"])


def build_clicks(priors, gts, has_gt, click_list, rng, dev):
    """Append one new click per sample to its history; returns padded (coords, labels).
    Samples without a valid click are padded with label -1 (ignored by SAM)."""
    for b in range(len(priors)):
        click_list[b].append(sample_click(priors[b], gts[b], rng) if has_gt[b] else None)
    n = max(len(c) for c in click_list)
    coords = np.zeros((len(priors), n, 2), np.float32)
    labels = np.full((len(priors), n), -1, np.int64)
    for b, cs in enumerate(click_list):
        for j, c in enumerate(cs):
            if c is None:
                continue
            x, y, lab, _ = c
            H, W = priors[b].shape
            coords[b, j] = [x / W * IMAGE_SIZE, y / H * IMAGE_SIZE]
            labels[b, j] = lab
    return torch.from_numpy(coords).to(dev), torch.from_numpy(labels).to(dev)


def load_rows(data, max_rows=None, holdout_wsi=15, seed=2026):
    m = pd.read_csv(Path(data) / "manifest.csv")
    m = m[m.use_refiner | m.quality_head_only].copy()
    m["has_gt"] = m.use_refiner
    rng = np.random.default_rng(seed)
    wsis = sorted(m.wsi_id.unique())
    hold = set(rng.choice(wsis, min(holdout_wsi, len(wsis)), replace=False).tolist())
    tr = m[~m.wsi_id.isin(hold)]; ho = m[m.wsi_id.isin(hold)]
    if max_rows:
        tr = tr.head(max_rows); ho = ho.head(max(4, max_rows // 4))
    return tr.to_dict("records"), ho.to_dict("records"), sorted(hold)


@torch.no_grad()
def evaluate(model, loader, dev, rng, amp_ctx, n_clicks=1):
    """Single-step Dice change on the held-out WSIs."""
    model.eval(); recs = []
    for batch in loader:
        img = batch["image"].to(dev, non_blocking=True)
        pri = batch["prior"].numpy(); gt = batch["gt"].numpy()
        hg = batch["has_gt"].numpy().astype(bool)
        if not hg.any():
            continue
        cl = [[] for _ in range(len(pri))]
        cur = [p.copy() for p in pri]
        for _ in range(n_clicks):
            co, la = build_clicks(cur, gt, hg, cl, rng, dev)
            mi = torch.from_numpy(np.stack([mask_to_logits(c) for c in cur])).to(dev)
            with amp_ctx:
                low, _ = forward(model, img, co, la, mi)
            up = F.interpolate(low.float(), size=pri.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0
            cur = up.cpu().numpy().astype(np.uint8)
        for b in range(len(pri)):
            if hg[b]:
                d0, d1 = dice_np(pri[b], gt[b]), dice_np(cur[b], gt[b])
                recs.append({"sample_id": batch["sample_id"][b], "source": batch["source"][b],
                             "dice_prior": d0, "dice_refined": d1, "delta": d1 - d0})
    model.train()
    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    hi = df[df.dice_prior >= 0.95]
    return {"n": len(df), "dice_prior": round(df.dice_prior.mean(), 4),
            "dice_refined": round(df.dice_refined.mean(), 4), "delta_mean": round(df.delta.mean(), 4),
            "frac_worse": round(float((df.delta < 0).mean()), 4), "hi_n": len(hi),
            "hi_delta_mean": round(hi.delta.mean(), 4) if len(hi) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with manifest.csv (build_refiner_trainset.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-decoder", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--iou-weight", type=float, default=1.0)
    ap.add_argument("--max-clicks", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--holdout-wsi", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from sam2.build_sam import build_sam2
    model = build_sam2(ft.CONFIG_NAME, ckpt_path=str(ft.DEFAULT_CKPT_PATH), device=str(dev))
    cfg = ft.configure_finetune(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                lora_dropout=args.lora_dropout)
    print(f"configure_finetune: {json.dumps(cfg)}", flush=True)
    model.train()

    tf = ft.sam_transforms()
    tr_rows, ho_rows, hold = load_rows(args.data, args.max_rows, args.holdout_wsi, args.seed)
    print(f"train {len(tr_rows)} rows / held out {len(ho_rows)} rows ({len(hold)} WSIs)", flush=True)
    json.dump({"holdout_wsi": hold}, open(out / "holdout_wsi.json", "w"), indent=1)
    dl = DataLoader(RefinerDataset(tr_rows, tf, aug=True, seed=args.seed), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    ho_dl = DataLoader(RefinerDataset(ho_rows, tf, aug=False, seed=args.seed), batch_size=args.batch_size,
                       shuffle=False, num_workers=max(1, args.num_workers // 2))

    groups = [{"params": [p for n, p in model.named_parameters()
                          if p.requires_grad and ("lora_A" in n or "lora_B" in n)], "lr": args.lr_lora},
              {"params": [p for n, p in model.named_parameters()
                          if p.requires_grad and "lora_" not in n], "lr": args.lr_decoder}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = args.epochs * len(dl)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
    rng = np.random.default_rng(args.seed)

    step0, ep0 = load_latest(model, out, opt, sch) if args.resume else (0, 0)
    logs, evals, step, t0 = [], [], step0, time.time()
    for ep in range(ep0, args.epochs):
        for batch in dl:
            img = batch["image"].to(dev, non_blocking=True)
            pri = batch["prior"].numpy(); gtn = batch["gt"].numpy()
            hg = batch["has_gt"].numpy().astype(bool)
            T = int(rng.integers(0, args.max_clicks + 1))
            cl = [[] for _ in range(len(pri))]
            cur = [p.copy() for p in pri]
            co = la = None
            for t in range(T):
                co, la = build_clicks(cur, gtn, hg, cl, rng, dev)
                if t == T - 1:
                    break                                    # the last round carries gradients
                mi = torch.from_numpy(np.stack([mask_to_logits(c) for c in cur])).to(dev)
                with torch.no_grad(), amp_ctx:
                    low, _ = forward(model, img, co, la, mi)
                cur = (F.interpolate(low.float(), size=pri.shape[-2:], mode="bilinear",
                                     align_corners=False)[:, 0] > 0).cpu().numpy().astype(np.uint8)
            mi = torch.from_numpy(np.stack([mask_to_logits(c) for c in cur])).to(dev)
            tgt = torch.from_numpy(np.stack([to_target(g) for g in gtn])).to(dev)
            hgt = torch.from_numpy(hg.astype(np.float32)).to(dev)

            with amp_ctx:
                low, iou_pred = forward(model, img, co, la, mi)
                lg = low.float()
                if lg.shape[-2:] != tgt.shape[-2:]:
                    lg = F.interpolate(lg, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
                if hgt.sum() > 0:
                    sel = hgt > 0
                    l_dice = ft.dice_loss_with_logits(lg[sel], tgt[sel])
                    l_bce = ft.bce_loss_with_logits(lg[sel], tgt[sel])
                else:
                    l_dice = l_bce = lg.sum() * 0.0
                with torch.no_grad():                        # IoU head regresses the true Dice
                    pm = (lg.detach() > 0).float()
                    inter = (pm * tgt).sum((1, 2, 3))
                    d_true = (2 * inter / (pm.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + 1e-6)) * hgt
                l_iou = F.l1_loss(iou_pred.float().squeeze(-1), d_true)
                loss = l_dice + l_bce + args.iou_weight * l_iou

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.grad_clip)
            scaler.step(opt); scaler.update(); sch.step()
            if not np.isfinite(float(loss.item())):
                raise SystemExit(f"non-finite loss at step {step}")
            logs.append({"step": step, "epoch": ep, "T": T, "loss": float(loss.item()),
                         "dice": float(l_dice.item()), "bce": float(l_bce.item()),
                         "iou": float(l_iou.item()), "wall_s": round(time.time() - t0, 1)})
            if step % args.log_every == 0:
                print(f"[{ep}] step {step}/{total} T={T} loss {loss.item():.4f}", flush=True)
            if step > 0 and step % args.eval_every == 0:
                e = summarize(evaluate(model, ho_dl, dev, rng, amp_ctx)); e["step"] = step; evals.append(e)
                print(f"  [held out] {json.dumps(e)}", flush=True)
                json.dump(evals, open(out / "eval_log.json", "w"), indent=1)
            if step > 0 and step % args.ckpt_every == 0:
                save_atomic(model, out, opt, sch, step, ep)
                json.dump(logs, open(out / "train_log.json", "w"))
            step += 1

    save_atomic(model, out, opt, sch, step, args.epochs - 1)
    ft.save_finetuned_state(model, out, {"final": True, "step": step})
    json.dump(logs, open(out / "train_log.json", "w"))
    df = evaluate(model, ho_dl, dev, rng, amp_ctx)
    df.to_csv(out / "holdout_eval.csv", index=False)
    fin = summarize(df); fin["step"] = step; evals.append(fin)
    json.dump(evals, open(out / "eval_log.json", "w"), indent=1)
    print(f"final held-out evaluation: {json.dumps(fin)}\n{step} steps / "
          f"{(time.time() - t0) / 3600:.2f} h -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
