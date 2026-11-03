from pathlib import Path

import joblib
import pytest

from src.predict import SentimentPredictor
from src.train import build_pipeline


@pytest.fixture()
def predictor(tmp_path: Path) -> SentimentPredictor:
    texts = ["terrible side effects", "awful pain", "it was okay", "average treatment",
             "excellent medicine", "worked very well"]
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = build_pipeline()
    pipeline.set_params(tfidf__min_df=1)
    pipeline.fit(texts, labels)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    return SentimentPredictor(path)


def test_prediction_has_valid_label_and_probabilities(predictor):
    result = predictor.predict("The medicine worked well")
    assert result["sentiment"] in {"Negative", "Neutral", "Positive"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_prediction_rejects_empty_text(predictor):
    with pytest.raises(ValueError):
        predictor.predict("")
