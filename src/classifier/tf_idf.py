from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

from dataclasses import asdict
from functools import lru_cache

from data_fetcher import get_videos

@lru_cache(maxsize=1)
def build_train_test():
    vids = get_videos((-1,1))
    df = pd.DataFrame([asdict(vid) for vid in vids])
    X = df[["title", "description", "category", "tags"]].fillna("")
    X["tags"] = X["tags"].apply(lambda x: " ".join(x))
    y = df["slant"]
    return (X, y)

X, y = build_train_test()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True), "title"),
        ("description",  TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_df=0.9, sublinear_tf=True), "description"),
        ("tags", TfidfVectorizer(stop_words=None), "tags"),
        ("category",  OneHotEncoder(), ["category"]),
    ],
)

pipe = Pipeline([("feats", features), ("reg", Ridge(alpha=0.5))])

g = GridSearchCV(
    pipe,
    {"reg__alpha": np.logspace(-3, 2, 12)},
    cv=5,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    n_jobs=-1,
)

pipe.fit(X_train, y_train).score(X_test, y_test)

pred = pipe.predict(X_test)

r2  = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred)

print(f"Model metrics: {r2}, {mae}, {rmse}")
print(f"Baseline mean: {np.mean(np.abs(y-y.mean()))}")
print(f"Baseline median: {np.mean(np.abs(y-y.median()))}")