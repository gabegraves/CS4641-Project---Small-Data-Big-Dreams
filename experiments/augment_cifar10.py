"""
Example script that runs the CIFAR-10 augmentation pipeline defined in
``src.image.augmentation_pipeline`` and exports qualitative grids for reports.
Run with:

    python -m experiments.augment_cifar10 --augmentations random_crop color_jitter
"""

from src.image.augmentation_pipeline import run_cli

if __name__ == "__main__":
    run_cli()
