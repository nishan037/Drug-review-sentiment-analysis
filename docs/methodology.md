# Methodology

## Data and labels

The project uses the UCI Drug Review Dataset (Drugs.com): 161,297 predefined
training records and 53,766 predefined test records. Ratings 1–4 map to
Negative, 5–6 to Neutral, and 7–10 to Positive. The supplied split is retained.

## Preprocessing

The production pipeline decodes HTML entities, lowercases text, removes markup
and non-alphabetic characters, removes English stop words, and applies Porter
stemming. Unlike the academic notebook, common negations (`not`, `no`, `never`,
and related terms) are retained because removing them can invert sentiment.
The fitted cleaner, TF-IDF vectorizer, and classifier are saved together.

## Models

The academic comparison contains VADER, TF-IDF plus Logistic Regression, LSTM,
GRU, and Bidirectional LSTM. The deployed model is Logistic Regression using up
to 20,000 unigram/bigram TF-IDF features and balanced class weights. It is the
best fit for low-memory CPU inference. Deep models are not loaded by the app.

## Evaluation

Only the training data fits preprocessing and model parameters. The untouched
test split supplies accuracy, macro and weighted precision/recall/F1,
classification report, and confusion matrix. Historical results in
`results/model_comparison.csv` were transcribed from saved original notebook
outputs; newly reproduced results are written separately by `src.train`.

## Limitations

Ratings are used as proxy sentiment labels, class imbalance is substantial,
and drug-review language can contain mixed experiences. Predictions are not a
medical assessment, treatment recommendation, or clinically validated output.
