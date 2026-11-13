"""Pembersihan teks sederhana. Dipakai saat training dan saat prediksi."""
import re

def bersihkan(teks: str) -> str:
    teks = str(teks).lower()
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)   # hapus URL
    teks = re.sub(r"@\w+|#\w+", " ", teks)               # hapus mention/hashtag
    teks = re.sub(r"(.)\1{2,}", r"\1\1", teks)           # bagus bangeeet -> bagus bangeet
    teks = re.sub(r"[^a-z\s]", " ", teks)                # sisakan huruf saja
    return re.sub(r"\s+", " ", teks).strip()
