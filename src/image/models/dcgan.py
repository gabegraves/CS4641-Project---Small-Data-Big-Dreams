from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from src.image.augmentation_pipeline import AugmentationConfig, CIFAR10AugmentationPipeline


def _weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, num_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int, num_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


@dataclass
class DCGANConfig:
    latent_dim: int = 128
    feature_maps: int = 64
    num_channels: int = 3
    batch_size: int = 128
    num_epochs: int = 25
    lr: float = 2e-4
    beta1: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: Path = Path("results/dcgan")
    seed: Optional[int] = 1234
    augmentations: Optional[list[str]] = None
    save_every: int = 5
    num_workers: int = 0


def _build_dataloader(config: DCGANConfig) -> DataLoader:
    pipeline = CIFAR10AugmentationPipeline(
        config=AugmentationConfig(normalize=False)
    )
    augmentations = config.augmentations or ["random_crop", "horizontal_flip", "color_jitter"]
    dataset = pipeline.augmented_dataset(augmentations, include_original=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def train_dcgan(config: Optional[DCGANConfig] = None) -> Dict[str, float]:
    cfg = config or DCGANConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    dataloader = _build_dataloader(cfg)
    device = torch.device(cfg.device)

    netG = Generator(cfg.latent_dim, cfg.feature_maps, cfg.num_channels).to(device)
    netG.apply(_weights_init)

    netD = Discriminator(cfg.feature_maps, cfg.num_channels).to(device)
    netD.apply(_weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, cfg.latent_dim, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    optimizerD = torch.optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    history: Dict[str, float] = {}

    total_steps = cfg.num_epochs * math.ceil(len(dataloader.dataset) / cfg.batch_size)
    print(f"Starting DCGAN training for {cfg.num_epochs} epochs ({total_steps} steps).")

    for epoch in range(cfg.num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            b_size = real_images.size(0)
            real_images = real_images.to(device) * 2 - 1

            # Update D - maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, cfg.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # Update G - maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    f"[{epoch}/{cfg.num_epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD_real.item() + errD_fake.item():.4f} "
                    f"Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.num_epochs - 1:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            grid_path = cfg.results_dir / f"epoch_{epoch+1:03d}.png"
            vutils.save_image(fake, grid_path, padding=2, normalize=True)
            torch.save(netG.state_dict(), cfg.results_dir / "generator.pt")
            torch.save(netD.state_dict(), cfg.results_dir / "discriminator.pt")

        history[f"epoch_{epoch+1}_loss_G"] = errG.item()
        history[f"epoch_{epoch+1}_loss_D"] = errD_real.item() + errD_fake.item()

    return {"final_G_loss": errG.item(), "final_D_loss": errD_real.item() + errD_fake.item(), **history}


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a DCGAN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--results-dir", type=Path, default=Path("results/dcgan"))
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=["random_crop", "horizontal_flip", "color_jitter"],
        help="Augmentations to include in the training dataset.",
    )
    args = parser.parse_args()

    cfg = DCGANConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        results_dir=args.results_dir,
        augmentations=list(args.augmentations),
    )
    metrics = train_dcgan(cfg)
    print("Training complete:", metrics)


if __name__ == "__main__":
    run_cli()
