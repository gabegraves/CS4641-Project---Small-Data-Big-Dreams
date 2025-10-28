from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from src.image.augmentation_pipeline import AugmentationConfig, CIFAR10AugmentationPipeline


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual(x)


class UNet(nn.Module):
    def __init__(self, channels: int = 3, base_dim: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.inc = ResidualBlock(channels, base_dim, time_dim)
        self.down1 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.down2 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        self.down3 = ResidualBlock(base_dim * 4, base_dim * 8, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.bot1 = ResidualBlock(base_dim * 8, base_dim * 8, time_dim)
        self.bot2 = ResidualBlock(base_dim * 8, base_dim * 8, time_dim)

        self.up3 = ResidualBlock(base_dim * 8 + base_dim * 8, base_dim * 4, time_dim)
        self.up2 = ResidualBlock(base_dim * 4 + base_dim * 4, base_dim * 2, time_dim)
        self.up1 = ResidualBlock(base_dim * 2 + base_dim * 2, base_dim, time_dim)
        self.outc = nn.Conv2d(base_dim, channels, 1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timesteps)

        x1 = self.inc(x, t)
        x2 = self.down1(self.pool(x1), t)
        x3 = self.down2(self.pool(x2), t)
        x4 = self.down3(self.pool(x3), t)

        h = self.bot1(self.pool(x4), t)
        h = self.bot2(h, t)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up3(torch.cat([h, x4], dim=1), t)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(torch.cat([h, x3], dim=1), t)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(torch.cat([h, x2], dim=1), t)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.outc(h)
        return h


@dataclass
class DDPMConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 128
    num_epochs: int = 100
    lr: float = 2e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: Path = Path("results/ddpm")
    seed: Optional[int] = 42
    augmentations: Optional[list[str]] = None
    sample_every: int = 10
    num_workers: int = 0


class GaussianDiffusion:
    def __init__(self, config: DDPMConfig) -> None:
        self.cfg = config
        timesteps = config.timesteps

        betas = torch.linspace(config.beta_start, config.beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.clamp(posterior_variance, min=1e-20))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in [
            "betas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ]:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device))
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = noise if noise is not None else torch.randn_like(x_start)
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_prod = torch.clamp(
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None], min=1e-12
        )
        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise

    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha = torch.clamp(
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None], min=1e-12
        )
        alpha_t = 1.0 - betas_t
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

        model_mean = sqrt_recip_alpha * (
            x - betas_t * model(x, t) / sqrt_one_minus_alpha
        )

        if (t == 0).all():
            return model_mean

        posterior_variance = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance) * noise

    def p_sample_loop(self, model: nn.Module, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        model.eval()
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
        model.train()
        return img


def _build_dataloader(config: DDPMConfig) -> DataLoader:
    pipeline = CIFAR10AugmentationPipeline(config=AugmentationConfig(normalize=False))
    augmentations = config.augmentations or ["random_crop", "horizontal_flip", "color_jitter"]
    dataset = pipeline.augmented_dataset(augmentations, include_original=True)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def train_ddpm(config: Optional[DDPMConfig] = None) -> None:
    cfg = config or DDPMConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = UNet(channels=3).to(device)
    diffusion = GaussianDiffusion(cfg).to(device)
    dataloader = _build_dataloader(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device) * 2 - 1  # scale to [-1, 1]
            t = torch.randint(0, cfg.timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise=noise)

            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"[Epoch {epoch+1}/{cfg.num_epochs}] Step {step} Loss: {loss.item():.4f}")

        if (epoch + 1) % cfg.sample_every == 0 or epoch == cfg.num_epochs - 1:
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    model,
                    (16, 3, 32, 32),
                    device=device,
                )
                samples = (samples.clamp(-1, 1) + 1) / 2  # back to [0, 1]
                out_path = cfg.results_dir / f"epoch_{epoch+1:03d}.png"
                vutils.save_image(samples, out_path, nrow=4)
                torch.save(model.state_dict(), cfg.results_dir / "ddpm.pt")


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a lightweight DDPM on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--results-dir", type=Path, default=Path("results/ddpm"))
    parser.add_argument("--augmentations", nargs="*", default=["random_crop", "horizontal_flip", "color_jitter"])
    args = parser.parse_args()

    cfg = DDPMConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        results_dir=args.results_dir,
        augmentations=list(args.augmentations),
    )
    train_ddpm(cfg)


if __name__ == "__main__":
    run_cli()
