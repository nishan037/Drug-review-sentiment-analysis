"""Project-wide constants and filesystem helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
