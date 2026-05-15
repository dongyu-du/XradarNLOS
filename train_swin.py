"""
Stage 1 Training: Fine-tune SwinUNet on X-band Range-Azimuth spectrograms.

The model produces 2-channel heatmaps detecting LOS and mNLOS reflections.
Currently supports --mode real (real experiment data).

Run from the public/ directory:
    python train_swin.py --mode real \
        --label_dir   /path/to/data/Experiment/1112label_data \
        --train_list  data/data_splits/exp_train_files.txt \
        --val_list    data/data_splits/exp_test_files.txt  \
        --output      checkpoints/swinunet_stage1_realdata.pth \
        --start_ckpt  checkpoints/swinunet_stage1_realdata.pth

Each line in the train/val list is a relative capture path, e.g. scene1/capture1.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm

from utils.dataset import ExpNlosDataset
from utils.eval import evaluate_swin
from utils.loss import HeatmapFocalLoss
from utils.swin import SwinUnet


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SwinUNet on X-band RA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",        default="real", choices=["real"])
    parser.add_argument("--label_dir",   required=True)
    parser.add_argument("--train_list",  required=True)
    parser.add_argument("--val_list",    required=True)
    parser.add_argument("--output",      required=True, help="Best checkpoint save path (.pth)")
    parser.add_argument("--start_ckpt",  default="")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_head",     type=float, default=5e-5)
    parser.add_argument("--lr_drop",     type=int,   default=40,
                        help="Epoch at which LR is halved")
    parser.add_argument("--loss_weight", type=float, nargs=2, default=[1.5, 1.0],
                        metavar=("W_LOS", "W_MNLOS"))
    parser.add_argument("--sigma",       type=float, nargs=2, default=[1.0, 1.0],
                        metavar=("SIGMA_LOS", "SIGMA_MNLOS"))
    parser.add_argument("--save_every",  type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--device",      default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Mode: {args.mode}  |  Device: {device}")

    train_caps = [l.strip() for l in open(args.train_list) if l.strip()]
    val_caps   = [l.strip() for l in open(args.val_list)   if l.strip()]
    print(f"Train: {len(train_caps)} captures  |  Val: {len(val_caps)} captures")

    ds_train = ExpNlosDataset(train_caps, args.label_dir, sigma=tuple(args.sigma))
    loader   = DataLoader(ds_train,
                          batch_sampler=BatchSampler(
                              RandomSampler(ds_train), args.batch_size, drop_last=True),
                          num_workers=4)
    print(f"Valid train captures: {len(ds_train)}")

    model = SwinUnet(img_size=512, num_classes=2).to(device)
    if args.start_ckpt:
        model.load_state_dict(torch.load(args.start_ckpt, map_location=device), strict=True)
        print(f"Loaded: {args.start_ckpt}")

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "swin_unet" in n and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if "swin_unet" not in n and p.requires_grad],
         "lr": args.lr_head},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    criterion = HeatmapFocalLoss()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for radar, target, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                                     leave=False):
            radar, target = radar.to(device), target.to(device)
            out  = model(radar)[..., 128:128 + 256]
            loss = (args.loss_weight[0] * criterion(out[:, 0:1], target[:, 0:1]) +
                    args.loss_weight[1] * criterion(out[:, 1:2], target[:, 1:2]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running += loss.item() * radar.size(0)
            n += radar.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={running/max(n,1):.4f}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            macro_f1 = evaluate_swin(model, val_caps, args.label_dir, device)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), args.output)
                print(f"  >> Best Macro-F1={best_f1:.4f}  saved → {args.output}")

    print(f"\nDone. Best Macro-F1={best_f1:.4f}  Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
