from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

import pandas as pd

import config

df = pd.read_pickle(config.SLANT_METADATA)
corpus = 

model = TfidfVectorizer(stop_words="english")  
model.fit_transform(corpus)


