import pytest

from src.preprocessing import clean_text, rating_to_label


@pytest.mark.parametrize(("rating", "expected"), [(1, 0), (4, 0), (5, 1), (6, 1), (7, 2), (10, 2)])
def test_rating_boundaries(rating, expected):
    assert rating_to_label(rating) == expected


@pytest.mark.parametrize("rating", [0, 11])
def test_invalid_rating_range(rating):
    with pytest.raises(ValueError):
        rating_to_label(rating)


def test_clean_text_removes_markup_and_preserves_negation():
    cleaned = clean_text("<b>I did NOT like headaches!</b>")
    assert "not" in cleaned.split()
    assert "<" not in cleaned


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        clean_text("  ")
