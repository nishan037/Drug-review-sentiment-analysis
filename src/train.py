"""Train and evaluate the production TF-IDF Logistic Regression pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_predictions, save_evaluation
from src.preprocessing import clean_text, rating_to_label
from src.utils import DEFAULT_MODEL_PATH, set_seed, write_json

REQUIRED_COLUMNS = {"review", "rating"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["review", "rating"]).copy()
    if frame.empty:
        raise ValueError("Dataset has no usable review/rating rows.")
    frame["sentiment"] = frame["rating"].map(rating_to_label)
    return frame


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, tokenizer=str.split, token_pattern=None,
                                  lowercase=False, max_features=20_000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced",
                                           random_state=seed, solver="liblinear")),
    ])


def train(train_path: Path, test_path: Path, model_path: Path, results_dir: Path,
          seed: int = 42, sample_size: int | None = None) -> dict:
    set_seed(seed)
    train_frame, test_frame = load_dataset(train_path), load_dataset(test_path)
    if sample_size:
        train_frame = train_frame.sample(min(sample_size, len(train_frame)), random_state=seed)
        test_frame = test_frame.sample(min(max(30, sample_size // 3), len(test_frame)), random_state=seed)
    pipeline = build_pipeline(seed)
    started = time.perf_counter()
    pipeline.fit(train_frame["review"], train_frame["sentiment"])
    training_seconds = time.perf_counter() - started
    predictions = pipeline.predict(test_frame["review"])
    metrics, matrix = evaluate_predictions(test_frame["sentiment"], predictions)
    metrics.update({"training_seconds": training_seconds, "train_rows": len(train_frame),
                    "test_rows": len(test_frame), "random_seed": seed})
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    save_evaluation(metrics, matrix, results_dir)
    pd.DataFrame({"actual": test_frame["sentiment"].to_numpy(), "predicted": predictions}).to_csv(
        results_dir / "predictions.csv", index=False
    )
    write_json({"model": "TF-IDF + Logistic Regression", "max_features": 20_000,
                "ngram_range": [1, 2], "class_weight": "balanced", "random_seed": seed,
                "train_source": train_path.name, "test_source": test_path.name},
               results_dir / "training_config.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()
    metrics = train(args.train_data, args.test_data, args.model_output, args.results_dir,
                    args.seed, args.sample_size)
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['f1_macro']:.4f}; "
          f"weighted-F1: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
