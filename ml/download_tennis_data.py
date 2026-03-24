# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
ml/download_tennis_data.py
Скачивает ATP матчи с GitHub (Jeff Sackmann dataset, открытые данные).
Сохраняет в ml/data/tennis_raw.csv
"""
import urllib.request
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
YEARS = list(range(2015, 2025))

def download_atp_data():
    frames = []
    for year in YEARS:
        url = BASE_URL.format(year=year)
        try:
            print(f"  Скачиваем {year}...", end=" ")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(resp, low_memory=False)
            frames.append(df)
            print(f"{len(df)} матчей")
            time.sleep(0.3)
        except Exception as e:
            print(f"ОШИБКА: {e}")

    if not frames:
        print("❌ Данные не загружены")
        return None

    combined = pd.concat(frames, ignore_index=True)
    out = DATA_DIR / "tennis_raw.csv"
    combined.to_csv(out, index=False)
    print(f"\n✅ Сохранено {len(combined)} матчей → {out}")
    return combined

if __name__ == "__main__":
    download_atp_data()
