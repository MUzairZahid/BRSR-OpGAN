"""Train BRSR-OpGAN (or the CNN-GAN baseline with --Q 1) on the BRSR benchmark.

Examples
--------
# BRSR-OpGAN, dual-domain loss, on the BRSR dataset (paper setting)
python train.py --dataset brsr --Q 3 --lambda_freq 2

# Time-domain loss only
python train.py --dataset brsr --Q 3 --lambda_freq 0

# CNN-GAN baseline
python train.py --dataset brsr --Q 1

# Second restoration pass (BRSR-OpGAN-D-2P): train on the outputs of a trained first pass
python train.py --dataset brsr --Q 3 --first_pass pretrained_weights/brsr/BRSR_OpGAN_Q3_Dual_Domain.pth
"""
import argparse
import os

import torch

from data import make_dataloaders
from models import ResidualGenerator
from utils import initialize_models, save_config, train_dual_loss

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["brsr", "awgn_baseline"], default="brsr")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--model", choices=["residual", "simple"], default="residual")
    ap.add_argument("--Q", type=int, default=3, help="Self-ONN order (1 = conventional CNN)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_recon", type=float, default=100.0, help="reconstruction loss weight")
    ap.add_argument("--lambda_freq", type=float, default=2.0, help="frequency loss weight (0 = time domain only)")
    ap.add_argument("--no_normalize", action="store_true", help="disable per-signal min-max scaling")
    ap.add_argument("--first_pass", default=None, help="trained first-pass generator (enables 2nd-pass training)")
    ap.add_argument("--first_pass_q", type=int, default=None, help="Q of the first-pass generator (default: --Q)")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--no_eval_train", action="store_true", help="skip train-split SNR during evaluation (faster)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    tag = f"{'2P_' if args.first_pass else ''}q{args.Q}_F{args.lambda_freq:g}_BS{args.batch_size}_{args.model}_{args.dataset}"
    args.out_dir = args.out_dir or os.path.join(HERE, "runs", tag)
    os.makedirs(args.out_dir, exist_ok=True)
    save_config(args, os.path.join(args.out_dir, "config.json"))
    print(f"Output folder: {args.out_dir}")

    loaders = make_dataloaders(args.data_dir, args.dataset, args.batch_size, splits=("train", "validation"),
                               normalize=not args.no_normalize, num_workers=args.num_workers,
                               pin_memory=args.device.startswith("cuda"))
    G, D = initialize_models(args.model, args.Q, args.device)
    pre = None
    if args.first_pass:
        pre = ResidualGenerator(q=args.first_pass_q or args.Q)
        pre.load_state_dict(torch.load(args.first_pass, map_location="cpu"), strict=True)
        pre = pre.to(args.device)

    best = train_dual_loss(G, D, loaders, args.epochs, args.lambda_recon, args.lambda_freq, args.device,
                           args.out_dir, eval_every=args.eval_every, eval_train=not args.no_eval_train,
                           pre_model=pre, lr=args.lr)
    chain = f"{args.first_pass} {best}" if args.first_pass else best
    print(f"\nBest generator: {best}\nEvaluate it with:\n"
          f"  python evaluate.py --dataset {args.dataset} --checkpoint {chain} --q {args.Q}")


if __name__ == "__main__":
    main()
