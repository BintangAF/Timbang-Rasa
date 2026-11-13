"""Backend Flask: memuat model dan menyediakan API analisis sentimen."""

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

import preprocess  # noqa: F401  (diperlukan agar model.pkl bisa dimuat)

app = Flask(__name__)
model = joblib.load("model.pkl")
tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
fitur = np.array(tfidf.get_feature_names_out())
kelas = list(clf.classes_)

NAMA = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}
MAKS_KARAKTER = 2000


def kata_berpengaruh(teks, label, n=5):
    """Kata dalam teks yang paling mendorong prediksi ke arah label.

    Pada Naive Bayes, tiap kata punya log-probabilitas per kelas. Kata dianggap
    berpengaruh bila log-probabilitasnya di kelas terpilih jauh lebih besar
    daripada rata-rata di kelas lain, dikali bobot TF-IDF kata itu pada teks.
    """
    vektor = tfidf.transform([teks])
    idx = vektor.nonzero()[1]
    if len(idx) == 0:
        return []
    logp = clf.feature_log_prob_
    k = kelas.index(label)
    selisih = logp[k] - np.delete(logp, k, axis=0).mean(axis=0)
    bobot = selisih[idx] * vektor[0, idx].toarray().ravel()
    urut = np.argsort(bobot)[::-1][:n]
    return [fitur[idx[i]] for i in urut if bobot[i] > 0]


@app.get("/")
def beranda():
    return render_template("index.html", maks=MAKS_KARAKTER)


@app.post("/api/analisis")
def analisis():
    data = request.get_json(silent=True) or {}
    teks = (data.get("teks") or "").strip()
    if not teks:
        return jsonify(
            error="Teks masih kosong. Tempel ulasan atau komentar terlebih dahulu."
        ), 400
    if len(teks) > MAKS_KARAKTER:
        return jsonify(
            error=f"Teks terlalu panjang (maksimal {MAKS_KARAKTER} karakter)."
        ), 400

    proba = model.predict_proba([teks])[0]
    label = kelas[int(np.argmax(proba))]
    return jsonify(
        label=label,
        nama=NAMA[label],
        keyakinan=round(float(proba.max()) * 100, 1),
        skor={NAMA[k]: round(float(p) * 100, 1) for k, p in zip(kelas, proba)},
        kata=kata_berpengaruh(teks, label),
    )


if __name__ == "__main__":
    app.run(debug=True)
