from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TabularExperimentConfig:
    data_path: Path = Path("Travel_Times.csv")
    target_column: str = "Mean Travel Time (Seconds)"
    categorical_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    ctgan_epochs: int = 300
    synthetic_samples: Optional[int] = None
    results_dir: Path = Path("results/tabular")
    n_splits: int = 5


def _infer_categorical(df: pd.DataFrame, provided: Optional[List[str]]) -> List[str]:
    if provided is not None:
        return provided
    return [
        col
        for col in df.columns
        if df[col].dtype == "object" and col != "Mean Travel Time (Seconds)"
    ]


def load_dataset(
    config: TabularExperimentConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = pd.read_csv(config.data_path)
    categorical_columns = _infer_categorical(df, config.categorical_columns)
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    numeric_columns = X.select_dtypes(exclude="object").columns.tolist()
    return X, y, categorical_columns, numeric_columns


def build_preprocessor(
    categorical_columns: List[str], numeric_columns: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )


def build_regressor(
    categorical_columns: List[str],
    numeric_columns: List[str],
    random_state: int,
) -> Pipeline:
    preprocessor = build_preprocessor(categorical_columns, numeric_columns)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    try:
        rmse = mean_squared_error(y_test, preds, squared=False)  # type: ignore[arg-type]
    except TypeError:
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)

    return {
        "r2": r2_score(y_test, preds),
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, preds),
    }


def run_baseline_and_ctgan(config: Optional[TabularExperimentConfig] = None) -> Dict[str, Dict[str, float]]:
    cfg = config or TabularExperimentConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    X, y, categorical_columns, numeric_columns = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    baseline_model = build_regressor(
        categorical_columns, numeric_columns, cfg.random_state
    )
    baseline_metrics = evaluate_model(baseline_model, X_train, y_train, X_test, y_test)

    # Prepare CTGAN input
    train_df = X_train.copy()
    train_df[cfg.target_column] = y_train
    ctgan = CTGAN(epochs=cfg.ctgan_epochs, verbose=False)
    ctgan.fit(train_df, discrete_columns=categorical_columns)

    num_samples = cfg.synthetic_samples or len(train_df)
    synthetic_df = ctgan.sample(num_samples)
    synthetic_X = synthetic_df.drop(columns=[cfg.target_column])
    synthetic_y = synthetic_df[cfg.target_column]

    synthetic_model = build_regressor(
        categorical_columns, numeric_columns, cfg.random_state
    )
    synthetic_metrics = evaluate_model(
        synthetic_model, synthetic_X, synthetic_y, X_test, y_test
    )

    # Blend original and synthetic
    blended_X = pd.concat([X_train, synthetic_X], axis=0, ignore_index=True)
    blended_y = pd.concat([y_train, synthetic_y], axis=0, ignore_index=True)
    blended_model = build_regressor(
        categorical_columns, numeric_columns, cfg.random_state
    )
    blended_metrics = evaluate_model(
        blended_model, blended_X, blended_y, X_test, y_test
    )

    results = {
        "baseline": baseline_metrics,
        "synthetic_only": synthetic_metrics,
        "blended": blended_metrics,
    }

    metrics_path = cfg.results_dir / "ctgan_results.csv"
    pd.DataFrame(results).to_csv(metrics_path)

    return results


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: List[str],
    numeric_columns: List[str],
    random_state: int,
    n_splits: int,
) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: List[float] = []
    for train_idx, test_idx in kf.split(X):
        model = build_regressor(
            categorical_columns,
            numeric_columns,
            random_state,
        )
        metrics = evaluate_model(
            model,
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
        )
        scores.append(metrics["rmse"])
    return {
        "rmse_mean": float(np.mean(scores)),
        "rmse_std": float(np.std(scores)),
    }


def run_cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train CTGAN and evaluate against the travel time dataset.")
    parser.add_argument("--data-path", type=Path, default=Path("Travel_Times.csv"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--synthetic-samples", type=int)
    parser.add_argument("--target-column", type=str, default="Mean Travel Time (Seconds)")
    parser.add_argument("--results-dir", type=Path, default=Path("results/tabular"))
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    cfg = TabularExperimentConfig(
        data_path=args.data_path,
        ctgan_epochs=args.epochs,
        synthetic_samples=args.synthetic_samples,
        target_column=args.target_column,
        results_dir=args.results_dir,
        n_splits=args.splits,
    )

    results = run_baseline_and_ctgan(cfg)
    X, y, categorical_columns, numeric_columns = load_dataset(cfg)
    cv_stats = cross_validate(
        X,
        y,
        categorical_columns,
        numeric_columns,
        cfg.random_state,
        cfg.n_splits,
    )

    summary = {"results": results, "cross_validation": cv_stats}
    summary_path = cfg.results_dir / "summary.json"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_cli()
