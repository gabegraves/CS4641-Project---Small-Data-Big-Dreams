# Small Data, Big Dreams

This repository investigates how far we can push classical models and modern generative techniques when data is scarce. The project now backs up its claims with runnable code for:

- Repeatable image augmentation built on a single CIFAR-10 download.
- Two generative image models (DCGAN + lightweight DDPM) that can be trained end-to-end on the augmented data.
- A tabular pipeline that fits CTGAN on the Atlanta travel time dataset and reports how synthetic samples affect downstream regressors.

The notebooks from the original project remain for exploration, while the new `src/` directory provides scriptable, plain-Python workflows meant for reproducible experiments.

---

## Repository Layout

```
├── src/
│   ├── image/
│   │   ├── augmentation_pipeline.py   # Composable CIFAR-10 augmentations and loaders
│   │   └── models/
│   │       ├── dcgan.py               # DCGAN training loop and CLI
│   │       └── ddpm.py                # Lightweight DDPM trainer and sampler
│   └── tabular/
│       └── ctgan_pipeline.py          # CTGAN training + evaluation for travel-time data
├── experiments/                       # Ready-to-run entry points that wrap the src modules
│   ├── augment_cifar10.py
│   ├── train_dcgan.py
│   ├── train_ddpm.py
│   └── run_ctgan.py
├── results/                           # Generated artifacts (figures, checkpoints, metrics)
├── image-data/                        # Legacy notebook + assets
├── tabular/                           # Legacy tabular notebook + raw ARFF dump
├── requirements.txt
└── README.md
```

All generated files and checkpoints are routed to `results/` by default so that the working tree stays clean.

---

## Environment Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

The repository targets Python 3.10+. GPU acceleration is optional but recommended for the image generators.

---

## Datasets

- **CIFAR-10** is downloaded automatically under `data/cifar10/`. The `CIFAR10AugmentationPipeline` keeps a single cached copy of the dataset and builds augmented views on the fly.
- **Travel_Times.csv** (root directory) serves as the default tabular dataset for CTGAN experiments. Adjust `--data-path` to point to alternative CSV/Parquet files with a compatible schema.

Large original assets (e.g., proposal decks, static figures) remain in the repository for archival purposes but are not required for running the code.

---

## Image Workflows

### 1. Augmentation Baselines

Generate qualitative grids for the chosen augmentations and store them under `results/augmentations/`:

```bash
python -m experiments.augment_cifar10 \
    --augmentations random_crop color_jitter horizontal_flip \
    --num-samples 16
```

The underlying pipeline exposes a `DataLoader` factory so that models can train directly on augmented datasets without repeatedly downloading CIFAR-10.

### 2. DCGAN Training

```bash
python -m experiments.train_dcgan \
    --epochs 25 \
    --batch-size 128 \
    --augmentations random_crop horizontal_flip color_jitter
```

Artifacts:
- `results/dcgan/epoch_*.png` – generated samples throughout training.
- `results/dcgan/generator.pt` & `discriminator.pt` – latest checkpoints.

### 3. Diffusion (DDPM) Training

```bash
python -m experiments.train_ddpm \
    --epochs 100 \
    --timesteps 1000
```

Artifacts:
- `results/ddpm/epoch_*.png` – diffusion samples reclaimed to `[0, 1]`.
- `results/ddpm/ddpm.pt` – UNet weights for resuming or sampling.

Both generators share the augmentation pipeline, making it easy to evaluate synthetic images alongside the real and augmented training data.

---

## Tabular Workflow (CTGAN)

Run the end-to-end CTGAN experiment, including baseline, synthetic-only, and blended training regimes:

```bash
python -m experiments.run_ctgan \
    --data-path Travel_Times.csv \
    --epochs 300 \
    --synthetic-samples 5000 \
    --splits 5
```

Outputs:
- `results/tabular/ctgan_results.csv` – side-by-side metrics (R², RMSE, MAE).
- `results/tabular/summary.json` – includes cross-validation statistics for the baseline regressor.

The pipeline relies on scikit-learn transformers, so swapping in a different estimator or feature subset is straightforward.

---

## Results Snapshot (Oct 28, 2025)

- Augmentation study: qualitative grids for `random_crop`, `color_jitter`, and `horizontal_flip` are stored in `results/augmentations/` (curated copies under `reports/latest/augmentations/`).
- DCGAN: 1-epoch warm start on the augmented CIFAR-10 split yields `results/dcgan/epoch_001.png` with final losses `L_G ~= 2.47` and `L_D ~= 0.75`.
- DDPM: lightweight UNet trained for 1 epoch (200 diffusion steps) saves samples to `results/ddpm/epoch_001.png` and weights at `results/ddpm/ddpm.pt`.
- CTGAN (30 epochs, 5k synthetic rows) on the Atlanta travel-time data reports (artifacts in `reports/latest/tabular/`):
  - Baseline RandomForest RMSE 16.7, R^2 0.999
  - Synthetic-only RMSE 574.1, R^2 0.021
  - Blended RMSE 186.9, R^2 0.896
  - 5-fold CV RMSE mean 25.8 +/- 8.2
  Full tables are persisted in `results/tabular/summary.json`.

---

## Recommended Evaluation Playbook

1. **Image classification baselines** – Fine-tune a small CNN or MobileNet on (a) plain CIFAR-10 subset, (b) augmented dataset via `augmentation_pipeline.dataloader`, and (c) augmented + GAN/DDPM samples mixed in. Track accuracy curves and calibration metrics.
2. **Tabular regression** – Inspect how CTGAN impacts downstream models (Random Forest, XGBoost, etc.) using the metrics exported above. The provided cross-validation helper highlights variance across folds.
3. **Reporting** – Keep generated figures and JSON/CSV metric dumps inside `results/` so they are ignored by Git but easy to cite in a report or notebook.

---

## Contributions

| Name            | Focus Areas                                                               |
|-----------------|---------------------------------------------------------------------------|
| Gabe Graves     | XGBoost baselines, GAN experimentation, updated intro/methods write-up    |
| Lucy Xing       | Data cleaning and preprocessing pipelines                                 |
| Hyuk Lee        | Image augmentation research and visualization                            |
| Hannah Huang    | CNN baselines and training experiments                                    |
| Rohan Nandakumar| Tabular preprocessing, diffusion model research                           |

---

## References

1. Kolesnikov, A. et al. “Big Transfer (BiT): General Visual Representation Learning.” *ECCV* (2020).
2. Kotelnikov, A. et al. “TabDDPM: Modelling Tabular Data with Diffusion Models.” (2022).
3. Mandrekar, J. N. “Receiver operating characteristic curve in diagnostic test assessment.” *J. Thoracic Oncology* (2010).
4. Narkhede, S. “Understanding AUC–ROC Curve.” Towards Data Science (2018).
5. Grinsztajn, L., Oyallon, E., & Varoquaux, G. “Why do tree-based models still outperform deep learning on tabular data?” (2022).
