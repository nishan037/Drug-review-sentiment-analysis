"""Evaluation functions for three-class sentiment models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import LABEL_NAMES


def evaluate_predictions(y_true: Any, y_pred: Any) -> tuple[dict[str, Any], np.ndarray]:
    labels = [0, 1, 2]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=[LABEL_NAMES[i] for i in labels],
            zero_division=0, output_dict=True,
        ),
    }
    return metrics, confusion_matrix(y_true, y_pred, labels=labels)


def save_evaluation(metrics: dict[str, Any], matrix: np.ndarray, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(results_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=LABEL_NAMES.values(), columns=LABEL_NAMES.values()).to_csv(
        results_dir / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Purples")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABEL_NAMES.values(),
           yticklabels=LABEL_NAMES.values(), xlabel="Predicted", ylabel="Actual",
           title="TF-IDF + Logistic Regression Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV containing actual and predicted columns")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    metrics, matrix = evaluate_predictions(frame["actual"], frame["predicted"])
    save_evaluation(metrics, matrix, args.output_dir)


if __name__ == "__main__":
    main()
