from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from config import SLANTS_TRAIN

class SbertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_ = None

    def fit(self, X, y=None):
        if self.model_ is None:
            self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        else:
            texts = np.asarray(X).ravel().astype(str).tolist()
        embs = self.model_.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=True
        )
        return embs

def build():
    df = pd.read_pickle(SLANTS_TRAIN)
    df = df.assign(n=df["R"]+df["L"]).query("n>12").copy()
    X = df[["title","description","category","tags","channel","comments"]].fillna("")
    X["tags"] = X["tags"].apply(lambda x: " ".join(x))
    X["comments"] = X["comments"].apply(lambda x: " ".join(x))
    Xtr, Xte, dtr, dte = train_test_split(X, df, test_size=0.2, random_state=42)
    # expand to binomial rows (y=1 weighted by R, y=0 weighted by L)
    X_exp = pd.concat([Xtr, Xtr], ignore_index=True)
    y_exp = np.r_[np.ones(len(dtr)), np.zeros(len(dtr))]
    w_exp = np.r_[dtr["R"].to_numpy(), dtr["L"].to_numpy()]
    return X_exp, y_exp, w_exp, Xte, dte

X_train, y_train, w_train, X_test, df_test = build()

bart_pipe = Pipeline([("sbert", SbertEncoder()), ("sc", StandardScaler(with_mean=True))])

features = ColumnTransformer(
    transformers=[
        ("title", bart_pipe, "title"),
        ("description", bart_pipe, "description"),
        ("comments", bart_pipe, "comments"),
        ("tags", TfidfVectorizer(), "tags"),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),  # sparse, unscaled
        ("channel",  OneHotEncoder(handle_unknown="ignore"), ["channel"]),   # sparse, unscaled
    ],
)

pipe = Pipeline([
    ("feats", features),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1,
        tol=1e-3,
        max_iter=3000,
        n_jobs=-1,
        verbose=0
    ))
])

pipe.fit(X_train, y_train, clf__sample_weight=w_train)

p_hat = pipe.predict_proba(X_test)[:, 1]
slant_pred = 2*p_hat - 1
y_true = (df_test["R"] - df_test["L"]) / (df_test["R"] + df_test["L"])

print(f"R2: {r2_score(y_true, slant_pred):.4f}  "
      f"MAE: {mean_absolute_error(y_true, slant_pred):.4f}  "
      f"RMSE: {root_mean_squared_error(y_true, slant_pred):.4f}")

print(f"Baseline mean: {np.mean(np.abs(y_true-y_true.mean()))}")
print(f"Baseline median: {np.mean(np.abs(y_true-y_true.median()))}")
