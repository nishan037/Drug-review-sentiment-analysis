"""Load the production model and make safe sentiment predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.preprocessing import LABEL_NAMES, validate_text
from src.utils import DEFAULT_MODEL_PATH


class SentimentPredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, text: object) -> dict:
        review = validate_text(text)
        label = int(self.pipeline.predict([review])[0])
        result = {"label": label, "sentiment": LABEL_NAMES[label]}
        if hasattr(self.pipeline, "predict_proba"):
            values = self.pipeline.predict_proba([review])[0]
            classes = self.pipeline.classes_
            result["probabilities"] = {LABEL_NAMES[int(cls)]: float(prob)
                                       for cls, prob in zip(classes, values)}
            result["confidence"] = max(result["probabilities"].values())
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(SentimentPredictor(args.model).predict(args.text))


if __name__ == "__main__":
    main()
