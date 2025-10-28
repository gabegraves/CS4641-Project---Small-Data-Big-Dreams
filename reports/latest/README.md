## Latest Experiment Snapshots (Oct 28, 2025)

- **Augmentation grids** &rightarrow; `augmentations/` (random crop, color jitter, horizontal flip).
- **Generative models**:
  - `generative/dcgan_epoch_001.png` – DCGAN after one epoch on augmented CIFAR-10. Final losses: `L_G ~= 2.47`, `L_D ~= 0.75`.
  - `generative/ddpm_epoch_001.png` – DDPM after one epoch with 200 diffusion steps.
- **Tabular CTGAN**:
  - `tabular/summary.json` – metrics for baseline vs. synthetic vs. blended regimes.
  - `tabular/ctgan_results.csv` – raw table exported from `src/tabular/ctgan_pipeline.py`.

Regenerate via:

```bash
python -m experiments.augment_cifar10 --augmentations random_crop color_jitter horizontal_flip --num-samples 9
python -m experiments.train_dcgan --epochs 1 --batch-size 64
python -m experiments.train_ddpm --epochs 1 --batch-size 64 --timesteps 200
python -m experiments.run_ctgan --data-path Travel_Times.csv --epochs 30 --synthetic-samples 5000 --splits 5
```
