# Timbang Rasa — Analisis Sentimen Ulasan/Komentar

Website yang menerima teks ulasan/komentar berbahasa Indonesia, lalu menampilkan
sentimen (Positif / Netral / Negatif) beserta persentase keyakinannya.

- Model: TF-IDF + Multinomial Naive Bayes (scikit-learn)
- Dataset: SmSA dari IndoNLU (11.000 data latih, 1.260 data uji). Akurasi uji: ±87,4%
- Backend: Flask (`app.py`) | Frontend: `templates/index.html`

## Struktur

```
data/            dataset (train_preprocess.tsv, valid_preprocess.tsv)
preprocess.py    pembersihan teks
train.py         melatih dan menyimpan model -> model.pkl
app.py           backend Flask + API /api/analisis
templates/       tampilan web
```

## Menjalankan di komputer

```
python -m venv .venv # buat virtual environment
pip install -r requirements.txt # jalankan setelah mengaktifkan virtual environment
python train.py      # opsional, model.pkl sudah disertakan
python app.py        # buka http://127.0.0.1:5000
```
