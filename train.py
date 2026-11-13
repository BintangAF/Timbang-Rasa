"""Melatih model analisis sentimen (TF-IDF + Multinomial Naive Bayes) dan menyimpannya."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import bersihkan


def muat(path):
    """Baca TSV tanpa header. File valid punya baris judul ("text\tsentiment"), jadi dibuang."""
    df = pd.read_csv(path, sep="\t", header=None, names=["teks", "label"]).dropna()
    return df[df["label"] != "sentiment"].reset_index(drop=True)


train = muat("data/train_preprocess.tsv")
valid = muat("data/valid_preprocess.tsv")
print("Data latih:", len(train), "| Data uji:", len(valid))
print(train["label"].value_counts().to_string(), "\n")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=bersihkan, ngram_range=(1, 2), min_df=2, sublinear_tf=True
            ),
        ),
        ("clf", MultinomialNB(alpha=0.05)),
    ]
)
model.fit(train["teks"], train["label"])

pred = model.predict(valid["teks"])
print("Akurasi:", round(accuracy_score(valid["label"], pred), 4))
print(classification_report(valid["label"], pred, digits=3))
labels = ["positive", "neutral", "negative"]
print("Confusion matrix (baris=asli, kolom=prediksi)", labels)
print(confusion_matrix(valid["label"], pred, labels=labels))

joblib.dump(model, "model.pkl")
print("\nModel disimpan ke model.pkl")
