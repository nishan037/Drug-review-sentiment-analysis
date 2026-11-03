# Model artifacts

`tfidf_logistic_regression.joblib` is the lightweight production pipeline used
by Streamlit. It was trained using the reproducible command in the project
README and is deliberately allowed by `.gitignore` because it is approximately
1.2 MB. Other model binaries remain excluded. Regenerate the artifact whenever
the preprocessing code or pinned scikit-learn version changes.
