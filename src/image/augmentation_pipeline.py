from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class AugmentationConfig:
    """
    Configuration bundle for building torchvision transforms used across training
    and visualization. All parameters are optional so the pipeline can be tailored
    from the CLI or a notebook.
    """

    random_crop_size: Optional[int] = 28
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    perspective_distortion: float = 0.2
    perspective_probability: float = 0.3
    rotation_angles: Sequence[int] = (90, 180, 270)
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.05
    grayscale_prob: float = 0.1
    normalize: bool = True
    output_size: int = 32
    imagenet_stats: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    custom_transforms: List[Callable] = field(default_factory=list)

    def to_base_transform(self) -> transforms.Compose:
        ops: List[Callable] = [transforms.ToTensor()]
        if self.normalize:
            ops.append(transforms.Normalize(*self.imagenet_stats))
        return transforms.Compose(ops)


class CIFAR10AugmentationPipeline:
    """
    Utility wrapper that owns a single CIFAR-10 dataset instance and exposes
    preconfigured augmentation transforms. This eliminates redundant downloads
    while making it easy to plug into a training DataLoader or export images.
    """

    def __init__(
        self,
        data_dir: Path | str = "data/cifar10",
        train: bool = True,
        download: bool = True,
        base_transform: Optional[Callable] = None,
        config: Optional[AugmentationConfig] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or AugmentationConfig()
        transform = base_transform or self.config.to_base_transform()
        self.dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=train,
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:  # pragma: no cover - passthrough for convenience
        return len(self.dataset)

    def _build_augmented_transform(self, name: str) -> transforms.Compose:
        cfg = self.config
        base_ops: List[Callable] = []
        if name == "random_crop" and cfg.random_crop_size:
            base_ops.append(
                transforms.RandomResizedCrop(
                    size=cfg.output_size, scale=(cfg.random_crop_size / cfg.output_size, 1.0)
                )
            )
        elif name.startswith("rotation_"):
            degrees = int(name.split("_", 1)[1])
            base_ops.append(transforms.RandomRotation([degrees, degrees]))
        elif name == "horizontal_flip":
            base_ops.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_prob))
        elif name == "vertical_flip":
            base_ops.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip_prob))
        elif name == "perspective":
            base_ops.append(
                transforms.RandomPerspective(
                    distortion_scale=cfg.perspective_distortion,
                    p=cfg.perspective_probability,
                )
            )
        elif name == "color_jitter":
            base_ops.append(
                transforms.ColorJitter(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                )
            )
        elif name == "grayscale":
            base_ops.append(transforms.RandomGrayscale(p=cfg.grayscale_prob))
        else:
            raise KeyError(f"Unknown augmentation: {name}")

        composed = transforms.Compose(
            cfg.custom_transforms
            + base_ops
            + [transforms.ToTensor()]
            + ([transforms.Normalize(*cfg.imagenet_stats)] if cfg.normalize else [])
        )
        return composed

    def available_augmentations(self) -> Dict[str, transforms.Compose]:
        augments: Dict[str, transforms.Compose] = {
            "random_crop": self._build_augmented_transform("random_crop"),
            "horizontal_flip": self._build_augmented_transform("horizontal_flip"),
            "color_jitter": self._build_augmented_transform("color_jitter"),
            "grayscale": self._build_augmented_transform("grayscale"),
            "perspective": self._build_augmented_transform("perspective"),
        }
        if self.config.vertical_flip_prob > 0:
            augments["vertical_flip"] = self._build_augmented_transform("vertical_flip")
        for angle in self.config.rotation_angles:
            augments[f"rotation_{angle}"] = self._build_augmented_transform(
                f"rotation_{angle}"
            )
        return augments

    def augmented_dataset(
        self,
        augmentation_names: Iterable[str],
        include_original: bool = True,
    ) -> Dataset:
        transforms_map = self.available_augmentations()
        datasets_to_concat: List[Dataset] = []

        if include_original:
            datasets_to_concat.append(self.dataset)

        for name in augmentation_names:
            if name not in transforms_map:
                raise KeyError(
                    f"{name} not in available augmentations: {list(transforms_map)}"
                )
            datasets_to_concat.append(
                datasets.CIFAR10(
                    root=self.data_dir,
                    train=self.dataset.train,
                    download=False,
                    transform=transforms_map[name],
                )
            )
        return ConcatDataset(datasets_to_concat)

    def dataloader(
        self,
        augmentation_names: Iterable[str] = (),
        include_original: bool = True,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> DataLoader:
        dataset = self.augmented_dataset(augmentation_names, include_original)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def sample_augmented_images(
        self,
        augmentation_name: str,
        indices: Optional[Sequence[int]] = None,
        num_samples: int = 9,
    ) -> List[torch.Tensor]:
        transforms_map = self.available_augmentations()
        if augmentation_name not in transforms_map:
            raise KeyError(f"Unknown augmentation: {augmentation_name}")

        aug_ds = datasets.CIFAR10(
            root=self.data_dir,
            train=self.dataset.train,
            download=False,
            transform=transforms_map[augmentation_name],
        )
        if indices is None:
            indices = torch.randint(0, len(aug_ds), (num_samples,)).tolist()
        return [aug_ds[i][0] for i in indices]

    @staticmethod
    def show_image_grid(
        tensors: Sequence[torch.Tensor],
        labels: Optional[Sequence[int]] = None,
        title: Optional[str] = None,
    ) -> None:
        if not tensors:
            raise ValueError("No tensors provided to display.")

        columns = math.ceil(math.sqrt(len(tensors)))
        rows = math.ceil(len(tensors) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5))
        axes_array = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (ax, tensor) in enumerate(zip(axes_array, tensors)):
            img = tensor.detach().cpu()
            # Un-normalise for readability.
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(-1, 1, 1)
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            if labels and idx < len(labels):
                ax.set_title(str(labels[idx]), fontsize=8)
            ax.axis("off")
        for ax in axes_array[len(tensors) :]:
            ax.axis("off")
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def export_augmented_grid(
        self,
        output_dir: Path | str,
        augmentation_name: str,
        num_samples: int = 9,
    ) -> Path:
        """
        Save a grid of augmented samples for reports or qualitative inspection.
        Returns the path of the saved figure.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tensors = self.sample_augmented_images(
            augmentation_name, num_samples=num_samples
        )
        fig_path = output_dir / f"{augmentation_name}_grid.png"

        columns = math.ceil(math.sqrt(len(tensors)))
        rows = math.ceil(len(tensors) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5))

        axes_array = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, tensor in zip(axes_array, tensors):
            img = tensor.detach().cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(-1, 1, 1)
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
        for ax in axes_array[len(tensors) :]:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        return fig_path


def build_train_val_loaders(
    config: Optional[AugmentationConfig] = None,
    augmentations: Iterable[str] = ("random_crop", "color_jitter", "horizontal_flip"),
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: Path | str = "data/cifar10",
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience helper used by unit tests and training scripts. Returns train and
    validation loaders sharing the same augmentation catalogue.
    """
    config = config or AugmentationConfig()
    pipeline = CIFAR10AugmentationPipeline(data_dir=data_dir, config=config)
    train_loader = pipeline.dataloader(
        augmentation_names=augmentations,
        include_original=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=config.to_base_transform(),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 augmentations and optionally export sample grids."
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=["random_crop", "color_jitter", "horizontal_flip"],
        help="Subset of augmentation names to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/augmentations"),
        help="Directory used to save visual samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of images per augmentation grid (must be a square number).",
    )
    args = parser.parse_args()

    config = AugmentationConfig()
    pipeline = CIFAR10AugmentationPipeline(config=config)
    for name in args.augmentations:
        path = pipeline.export_augmented_grid(
            output_dir=args.output_dir,
            augmentation_name=name,
            num_samples=args.num_samples,
        )
        print(f"Saved {name} grid to {path}")


if __name__ == "__main__":
    run_cli()
