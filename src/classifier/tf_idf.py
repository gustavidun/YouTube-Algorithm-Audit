from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


import pandas as pd
import numpy as np

from dataclasses import asdict
import ast

from config import SLANTS_TRAIN
from data_fetcher import get_videos

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

features = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True), "title"),
        ("description",  TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_df=0.9, sublinear_tf=True), "description"),
        ("tags", TfidfVectorizer(stop_words=None), "tags"),
        ("comments", TfidfVectorizer(stop_words="english"), "comments"),
        ("category",  OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"])
    ],
)

pipe = Pipeline([
    ("feats", features),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=1000))
])

pipe.fit(X_train, y_train, clf__sample_weight=w_train)

pred = pipe.predict(X_test)

p_hat = pipe.predict_proba(X_test)[:, 1]
slant_pred = 2*p_hat - 1
y_true = (df_test["R"] - df_test["L"]) / (df_test["R"] + df_test["L"])

print(f"R2: {r2_score(y_true, slant_pred):.4f}  "
      f"MAE: {mean_absolute_error(y_true, slant_pred):.4f}  "
      f"RMSE: {root_mean_squared_error(y_true, slant_pred):.4f}")

print(f"Baseline mean: {np.mean(np.abs(y_true-y_true.mean()))}")
print(f"Baseline median: {np.mean(np.abs(y_true-y_true.median()))}")