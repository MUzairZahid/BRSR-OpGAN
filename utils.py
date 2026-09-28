"""Training utilities for BRSR-OpGAN (dual-domain loss, Algorithm 1 of the paper)."""
import json
import os

import torch
import torch.nn as nn

from data import denormalize_signal
from models import Discriminator, Generator, ResidualDiscriminator, ResidualGenerator


def initialize_models(model_type, q, device):
    """Generator and discriminator. q = order of the Self-ONN layers (q = 1 -> conventional CNN)."""
    if model_type == "simple":
        return Generator(q=q).to(device), Discriminator(q=q).to(device)
    if model_type == "residual":
        return ResidualGenerator(q=q).to(device), ResidualDiscriminator(q=q).to(device)
    raise ValueError(f"unknown model type {model_type}")


def calculate_snr(restored, clean):
    """Per-sample SNR in dB over both I/Q channels: 10 log10(P_clean / P_error)."""
    noise = restored - clean
    signal_power = clean[:, 0].pow(2).mean(-1) + clean[:, 1].pow(2).mean(-1)
    noise_power = noise[:, 0].pow(2).mean(-1) + noise[:, 1].pow(2).mean(-1)
    return 10 * torch.log10(signal_power / (noise_power + 1e-6))


def calculate_spectrogram_torch(signals, n_fft=256, hop_length=128, win_length=256, power=2.0):
    """STFT power spectrogram of complex I/Q signals [B, 2, L], min-max scaled to [-1, 1] per signal."""
    x = torch.view_as_complex(signals.permute(0, 2, 1).contiguous())
    window = torch.hann_window(win_length, device=signals.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      return_complex=True, window=window).abs() ** power
    smin = spec.amin(dim=(-2, -1), keepdim=True)
    smax = spec.amax(dim=(-2, -1), keepdim=True)
    return 2 * (spec - smin) / (smax - smin) - 1


@torch.no_grad()
def mean_snr(G, loader, device, pre_model=None):
    """Mean restored SNR (paper protocol: output mapped back with the clean signal's min/max)."""
    G.eval()
    values = []
    for (clean, cmin, cmax), (noisy, _, _), _, _, _ in loader:
        clean, cmin, cmax, noisy = clean.to(device), cmin.to(device), cmax.to(device), noisy.to(device)
        if pre_model is not None:
            noisy = pre_model(noisy)
        restored = G(noisy)
        values.append(calculate_snr(denormalize_signal(restored, cmin, cmax),
                                    denormalize_signal(clean, cmin, cmax)).cpu())
    return torch.cat(values).mean().item()


def train_dual_loss(G, D, dataloaders, num_epochs, lambda_recon, lambda_freq, device, out_dir,
                    eval_every=10, eval_train=True, pre_model=None, lr=5e-4):
    """Adversarial training with time- and frequency-domain L1 reconstruction losses.

    Generator loss: (D(G(x)) - 1)^2 + lambda_recon * (L_time + lambda_freq * L_freq) / k,
    with k = 2 when the frequency loss is used and k = 1 otherwise (as in the original code;
    the defaults lambda_recon = 100, lambda_freq = 2 give the paper's 1:2 time/frequency ratio).
    Discriminator loss: (D(x) - 1)^2 + D(G(x))^2 (least-squares GAN).

    If ``pre_model`` is given (a frozen, already trained generator), G is trained on its outputs:
    this is the second restoration pass (BRSR-OpGAN-D-2P).

    Every ``eval_every`` epochs the mean SNR on the validation split (and train split if
    ``eval_train``) is logged, and the generator with the best validation SNR is saved to
    ``out_dir/generator_best.pth``.
    """
    os.makedirs(out_dir, exist_ok=True)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    l1, mse = nn.L1Loss(), nn.MSELoss()
    division = 1 if lambda_freq == 0 else 2
    if pre_model is not None:
        pre_model.eval()
        for p in pre_model.parameters():
            p.requires_grad_(False)
    best_val, log_path = -float("inf"), os.path.join(out_dir, "training_log.txt")

    for epoch in range(num_epochs):
        G.train(); D.train()
        sums = dict(g=0.0, d=0.0, time=0.0, freq=0.0)
        for (clean, _, _), (noisy, _, _), _, _, _ in dataloaders["train"]:
            clean, noisy = clean.to(device), noisy.to(device)
            if pre_model is not None:
                with torch.no_grad():
                    noisy = pre_model(noisy)
            real = torch.ones(clean.size(0), 1, device=device)
            fake = torch.zeros(clean.size(0), 1, device=device)

            restored = G(noisy)
            loss_time = l1(restored, clean)
            loss_freq = lambda_freq * l1(calculate_spectrogram_torch(restored), calculate_spectrogram_torch(clean))
            loss_g = mse(D(restored), real) + lambda_recon * (loss_time + loss_freq) / division
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            loss_d = mse(D(clean), real) + mse(D(restored.detach()), fake)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            sums["g"] += loss_g.item(); sums["d"] += loss_d.item()
            sums["time"] += loss_time.item(); sums["freq"] += loss_freq.item()

        n = len(dataloaders["train"])
        print(f"Epoch [{epoch + 1}/{num_epochs}] loss G {sums['g']/n:.4f}  loss D {sums['d']/n:.4f}  "
              f"L_time {sums['time']/n:.4f}  L_freq {sums['freq']/n:.4f}", flush=True)

        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            val = mean_snr(G, dataloaders["validation"], device, pre_model)
            line = f"Epoch {epoch}: validation SNR {val:.2f} dB"
            if eval_train:
                line += f", train SNR {mean_snr(G, dataloaders['train'], device, pre_model):.2f} dB"
            if val > best_val:
                best_val = val
                torch.save(G.state_dict(), os.path.join(out_dir, "generator_best.pth"))
                line += "  <- best, saved"
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()},
                       os.path.join(out_dir, "checkpoint_last.pth"))
    return os.path.join(out_dir, "generator_best.pth")


def save_config(args, path):
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2)
