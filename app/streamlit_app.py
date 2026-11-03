from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import SentimentPredictor
from src.utils import DEFAULT_MODEL_PATH


def confidence_message(confidence: float) -> tuple[str, str]:
    """Translate a model probability into cautious, plain language."""
    if confidence >= 0.75:
        return "High confidence", "The wording strongly matches this sentiment."
    if confidence >= 0.55:
        return "Moderate confidence", "The review mostly matches this sentiment, but some wording is mixed."
    return "Low confidence", "The review contains mixed or unclear language, so consider the other scores too."


SENTIMENT_EXPLANATIONS = {
    "Negative": "The review mainly describes an unfavourable experience, such as poor results or side effects.",
    "Neutral": "The review appears balanced, uncertain, or neither clearly positive nor negative.",
    "Positive": "The review mainly describes a favourable experience or improvement.",
}

st.set_page_config(page_title="Drug Review AI", page_icon="💊", layout="wide")

st.title("💊 Drug Review AI — Sentiment Analysis")
st.caption("A portfolio demonstration based on an MSc machine learning and deep learning project")

with st.sidebar:
    st.header("About")
    st.markdown("**Nishan Jayathilake**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/nishan037/)")
    st.divider()
    st.write("Production model: TF-IDF + balanced Logistic Regression")
    st.write("Dataset: UCI Drug Review Dataset (Drugs.com), 215,063 reviews")
    st.warning("This model is not clinically validated and does not provide medical advice.")

prediction_tab, comparison_tab, methodology_tab = st.tabs(
    ["Sentiment prediction", "Model comparison", "Methodology"]
)

with prediction_tab:
    st.subheader("Analyse a pharmaceutical review")
    review = st.text_area("Review text", height=150,
                          placeholder="The medication helped my symptoms, but I experienced headaches.")
    if st.button("Analyse sentiment", type="primary"):
        if not review.strip():
            st.error("Enter a review before analysing it.")
        elif not DEFAULT_MODEL_PATH.is_file():
            st.error("The trained model artifact is unavailable. Follow the README training instructions.")
        else:
            try:
                result = SentimentPredictor().predict(review)
                confidence = result.get("confidence", 0.0)
                confidence_label, confidence_explanation = confidence_message(confidence)
                st.success(f"Overall result: **{result['sentiment']}**")
                st.write(SENTIMENT_EXPLANATIONS[result["sentiment"]])

                result_col, confidence_col = st.columns(2)
                result_col.metric("Detected tone", result["sentiment"])
                confidence_col.metric("Model confidence", f"{confidence:.0%}", confidence_label)
                st.caption(confidence_explanation)

                if "probabilities" in result:
                    st.markdown("#### How the model scored the review")
                    probabilities = pd.DataFrame([
                        {"Sentiment": sentiment, "Score": probability,
                         "Easy-to-read score": f"{probability:.0%}"}
                        for sentiment, probability in result["probabilities"].items()
                    ]).sort_values("Score", ascending=False)
                    st.bar_chart(probabilities.set_index("Sentiment")["Score"])
                    st.dataframe(
                        probabilities[["Sentiment", "Easy-to-read score"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "The three scores add up to 100%. The largest score becomes the overall result. "
                        "Close scores mean the wording is more ambiguous."
                    )
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
    st.info("The output reflects a statistical text classifier—not a medical assessment or recommendation.")

with comparison_tab:
    st.subheader("Verified academic experiment results")
    results_path = ROOT / "results" / "model_comparison.csv"
    results = pd.read_csv(results_path)
    display = results.copy()
    for column in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    st.dataframe(display.drop(columns="source"), hide_index=True, use_container_width=True)
    chart = results.set_index("model")[["accuracy", "f1_weighted"]]
    st.bar_chart(chart)
    st.caption("Values are preserved from executed outputs in the original notebook; they have not been fabricated.")
    reproduced_path = ROOT / "results" / "reproduced" / "metrics.json"
    if reproduced_path.is_file():
        import json
        reproduced = json.loads(reproduced_path.read_text(encoding="utf-8"))
        st.subheader("Reproduced production pipeline")
        first, second, third = st.columns(3)
        first.metric("Test accuracy", f"{reproduced['accuracy']:.2%}")
        second.metric("Macro-F1", f"{reproduced['f1_macro']:.2%}")
        third.metric("Weighted F1", f"{reproduced['f1_weighted']:.2%}")
    figure = ROOT / "results" / "reproduced" / "figures" / "confusion_matrix.png"
    if figure.is_file():
        st.image(str(figure), caption="Reproduced production-model confusion matrix")

with methodology_tab:
    st.subheader("Project methodology")
    st.markdown("""
    **Labels:** ratings 1–4 are Negative, 5–6 Neutral, and 7–10 Positive.

    **Preprocessing:** HTML decoding, lowercase conversion, punctuation removal,
    negation-aware stop-word filtering, and Porter stemming.

    **Features and training:** up to 20,000 unigram and bigram TF-IDF features;
    balanced Logistic Regression trained only on the predefined training split.

    **Evaluation:** the untouched predefined test split provides accuracy, macro
    and weighted precision/recall/F1, a classification report, and confusion matrix.
    """)
