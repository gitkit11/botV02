# -*- coding: utf-8 -*-
"""
ml/download_data.py — Скачивает исторические данные с football-data.co.uk
=========================================================================
Бесплатно. Без API ключа. CSV файлы.

Лиги:
  E0  = Premier League (Англия)
  SP1 = La Liga (Испания)
  D1  = Bundesliga (Германия)
  I1  = Serie A (Италия)
  F1  = Ligue 1 (Франция)

Сезоны: с 2000-01 по 2024-25
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Лиги которые скачиваем
LEAGUES = {
    "E0":  "Premier League",
    "E1":  "Championship",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
}

# Сезоны (формат: 0001 = 2000-01, 2324 = 2023-24)
def _season_codes(start_year=2005, end_year=2024):
    codes = []
    for y in range(start_year, end_year + 1):
        codes.append(f"{str(y)[-2:]}{str(y+1)[-2:]}")
    return codes

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download_league(league_code: str, seasons: list) -> list:
    """Скачивает CSV для одной лиги по всем сезонам. Возвращает список DataFrame."""
    frames = []
    for season in seasons:
        url = BASE_URL.format(season=season, league=league_code)
        fpath = DATA_DIR / f"{league_code}_{season}.csv"

        # Если уже скачано — читаем из кэша
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, encoding="latin-1")
                if len(df) > 5:
                    df["league"] = league_code
                    df["season"] = season
                    frames.append(df)
                    continue
            except Exception:
                pass

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 500:
                fpath.write_bytes(resp.content)
                df = pd.read_csv(fpath, encoding="latin-1")
                df["league"] = league_code
                df["season"] = season
                frames.append(df)
                print(f"  ✅ {league_code} {season}: {len(df)} матчей")
            else:
                print(f"  ⚠️  {league_code} {season}: нет данных (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  ❌ {league_code} {season}: {e}")

        time.sleep(0.3)  # вежливая задержка

    return frames


def download_all(start_year=2005, end_year=2024) -> pd.DataFrame:
    """Скачивает все лиги и сезоны, возвращает единый DataFrame."""
    seasons = _season_codes(start_year, end_year)
    all_frames = []

    for code, name in LEAGUES.items():
        print(f"\n📥 {name} ({code}) — {len(seasons)} сезонов...")
        frames = download_league(code, seasons)
        all_frames.extend(frames)

    if not all_frames:
        print("❌ Нет данных!")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    out_path = DATA_DIR / "all_matches_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Итого: {len(combined)} матчей → {out_path}")
    return combined


if __name__ == "__main__":
    df = download_all(start_year=2005, end_year=2024)
    print(f"\nКолонки: {list(df.columns[:20])}")
    print(df.head(3))
