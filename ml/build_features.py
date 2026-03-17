# -*- coding: utf-8 -*-
"""
ml/build_features.py v2 — Улучшенные признаки для XGBoost
==========================================================
Новые признаки:
  - Форма дома/в гостях ОТДЕЛЬНО (команды часто лучше дома)
  - Shots on Target (HST, AST) — лучший предиктор голов
  - xG proxy: угловые × 0.05 + удары в створ × 0.35
  - Прогресс сезона (матч-неделя)
  - Разница голов (GD) — сила команды
  - Goals Diff rolling avg
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ─── ELO ──────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame, k=32, base=1500) -> pd.DataFrame:
    elo = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        eh = elo.get(h, base)
        ea = elo.get(a, base)
        home_elos.append(eh)
        away_elos.append(ea)

        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        ftr = row.get("FTR", "")
        if ftr == "H":   s_h, s_a = 1.0, 0.0
        elif ftr == "D": s_h, s_a = 0.5, 0.5
        elif ftr == "A": s_h, s_a = 0.0, 1.0
        else: continue

        elo[h] = eh + k * (s_h - exp_h)
        elo[a] = ea + k * (s_a - (1 - exp_h))

    df = df.copy()
    df["elo_home"] = home_elos
    df["elo_away"] = away_elos
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_win_prob"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))  # ELO implied prob
    return df


# ─── Форма (общая + дома + в гостях) ──────────────────────────────────────────

def compute_form(df: pd.DataFrame, n=5) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # history[team] = список dict с результатами
    history:      dict = {}  # общая
    history_home: dict = {}  # только дома
    history_away: dict = {}  # только в гостях

    cols = {k: [] for k in [
        "form_home", "form_away",
        "form_home_home", "form_away_away",  # форма на своём поле
        "goals_scored_home", "goals_scored_away",
        "goals_conceded_home", "goals_conceded_away",
        "goals_diff_home", "goals_diff_away",
        "sot_home", "sot_away",              # shots on target avg
        "corners_home", "corners_away",
    ]}

    def _avg(hist, field, last_n, default):
        if not hist: return default
        return np.mean([r[field] for r in hist[-last_n:]])

    def _pts(hist, last_n, default=1.5):
        if not hist: return default
        return sum(r["pts"] for r in hist[-last_n:]) / len(hist[-last_n:]) * 3

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        hg = float(row.get("FTHG", 0) or 0)
        ag = float(row.get("FTAG", 0) or 0)
        hst = float(row.get("HST", hg * 3) or hg * 3)   # fallback если нет
        ast = float(row.get("AST", ag * 3) or ag * 3)
        hc  = float(row.get("HC", 5) or 5)
        ac  = float(row.get("AC", 5) or 5)

        ftr = row.get("FTR", "")
        if ftr == "H":   h_pts, a_pts = 1.0, 0.0
        elif ftr == "D": h_pts, a_pts = 0.5, 0.5
        elif ftr == "A": h_pts, a_pts = 0.0, 1.0
        else:            h_pts, a_pts = 0.5, 0.5

        # Записываем ДО обновления
        cols["form_home"].append(_pts(history.get(home, []), n))
        cols["form_away"].append(_pts(history.get(away, []), n))
        cols["form_home_home"].append(_pts(history_home.get(home, []), n))
        cols["form_away_away"].append(_pts(history_away.get(away, []), n))
        cols["goals_scored_home"].append(_avg(history.get(home, []), "scored", n, 1.3))
        cols["goals_scored_away"].append(_avg(history.get(away, []), "scored", n, 1.1))
        cols["goals_conceded_home"].append(_avg(history.get(home, []), "conceded", n, 1.1))
        cols["goals_conceded_away"].append(_avg(history.get(away, []), "conceded", n, 1.3))
        cols["goals_diff_home"].append(_avg(history.get(home, []), "gd", n, 0.2))
        cols["goals_diff_away"].append(_avg(history.get(away, []), "gd", n, -0.2))
        cols["sot_home"].append(_avg(history.get(home, []), "sot", n, 4.5))
        cols["sot_away"].append(_avg(history.get(away, []), "sot", n, 3.5))
        cols["corners_home"].append(_avg(history.get(home, []), "corners", n, 5.0))
        cols["corners_away"].append(_avg(history.get(away, []), "corners", n, 4.5))

        # Обновляем после записи
        for team, pts, scored, conceded, sot, corners, hist_loc in [
            (home, h_pts, hg, ag, hst, hc, history_home),
            (away, a_pts, ag, hg, ast, ac, history_away),
        ]:
            rec = {"pts": pts, "scored": scored, "conceded": conceded,
                   "gd": scored - conceded, "sot": sot, "corners": corners}
            history.setdefault(team, []).append(rec)
            hist_loc.setdefault(team, []).append(rec)

    for col, vals in cols.items():
        df[col] = vals

    df["form_diff"] = df["form_home"] - df["form_away"]
    df["sot_diff"]  = df["sot_home"]  - df["sot_away"]
    return df


# ─── H2H ───────────────────────────────────────────────────────────────────────

def compute_h2h(df: pd.DataFrame, n=5) -> pd.DataFrame:
    df = df.copy()
    h2h_wr, h2h_goals_diff = [], []
    h2h_hist: dict = {}

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = tuple(sorted([home, away]))
        past = h2h_hist.get(key, [])[-n:]

        if not past:
            h2h_wr.append(0.5)
            h2h_goals_diff.append(0.0)
        else:
            hw = sum(1 for r in past if r["winner"] == home)
            h2h_wr.append(hw / len(past))
            h2h_goals_diff.append(np.mean([r["gd"] for r in past]))

        ftr = row.get("FTR", "")
        hg = float(row.get("FTHG", 0) or 0)
        ag = float(row.get("FTAG", 0) or 0)
        winner = home if ftr == "H" else (away if ftr == "A" else "draw")
        h2h_hist.setdefault(key, []).append({
            "winner": winner,
            "gd": hg - ag if ftr != "A" else ag - hg
        })

    df["h2h_home_winrate"] = h2h_wr
    df["h2h_goals_diff"]   = h2h_goals_diff
    return df


# ─── Коэффициенты ──────────────────────────────────────────────────────────────

def compute_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, candidates in [
        ("odds_home", ["B365H", "BbAvH", "PSH", "WHH"]),
        ("odds_draw", ["B365D", "BbAvD", "PSD", "WHD"]),
        ("odds_away", ["B365A", "BbAvA", "PSA", "WHA"]),
    ]:
        for c in candidates:
            if c in df.columns:
                df[col] = pd.to_numeric(df[c], errors="coerce")
                break
        else:
            df[col] = np.nan

    df["imp_home"] = 1.0 / df["odds_home"].replace(0, np.nan)
    df["imp_draw"] = 1.0 / df["odds_draw"].replace(0, np.nan)
    df["imp_away"] = 1.0 / df["odds_away"].replace(0, np.nan)

    total = df["imp_home"].fillna(0) + df["imp_draw"].fillna(0) + df["imp_away"].fillna(0)
    total = total.replace(0, np.nan)
    df["imp_home"] = df["imp_home"] / total
    df["imp_draw"] = df["imp_draw"] / total
    df["imp_away"] = df["imp_away"] / total

    # Overround (чем выше — тем жёстче маржа букмекера)
    df["overround"] = (df["imp_home"].fillna(0) + df["imp_draw"].fillna(0) + df["imp_away"].fillna(0))
    return df


# ─── Сезонный прогресс ────────────────────────────────────────────────────────

def compute_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Матч-неделя внутри сезона (нормализованная 0-1)
    df["match_week_norm"] = (
        df.groupby(["league", "season"])
          .cumcount() / df.groupby(["league", "season"])["Date"].transform("count")
    )
    return df


# ─── Target ────────────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    df["target_home_win"] = (df["FTR"] == "H").astype(int)
    return df


def compute_league_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["league_id"] = df["league"].map({"E0": 0, "SP1": 1, "D1": 2, "I1": 3, "F1": 4}).fillna(0).astype(int)
    return df


# ─── Список признаков ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ELO
    "elo_diff", "elo_home", "elo_away", "elo_win_prob",
    # Форма
    "form_home", "form_away", "form_diff",
    "form_home_home", "form_away_away",
    # Голы
    "goals_scored_home", "goals_scored_away",
    "goals_conceded_home", "goals_conceded_away",
    "goals_diff_home", "goals_diff_away",
    # Shots on Target
    "sot_home", "sot_away", "sot_diff",
    # Угловые
    "corners_home", "corners_away",
    # H2H
    "h2h_home_winrate", "h2h_goals_diff",
    # Коэффициенты
    "imp_home", "imp_draw", "imp_away", "overround",
    # Прочее
    "league_id", "match_week_norm",
]


# ─── Главная функция ──────────────────────────────────────────────────────────

def build_features(raw_path=None) -> pd.DataFrame:
    if raw_path is None:
        raw_path = DATA_DIR / "all_matches_raw.csv"

    print(f"Читаю {raw_path}...")
    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    print(f"   Загружено: {len(df)} строк")

    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    print(f"   После фильтра: {len(df)} матчей")

    print("ELO...")
    df = compute_elo(df)
    print("Форма + SoT...")
    df = compute_form(df)
    print("H2H...")
    df = compute_h2h(df)
    print("Коэффициенты...")
    df = compute_odds_features(df)
    print("Сезон...")
    df = compute_season_features(df)
    print("Target...")
    df = compute_target(df)
    df = compute_league_features(df)

    keep = FEATURE_COLS + ["target", "target_home_win", "league", "season",
                           "HomeTeam", "AwayTeam", "Date",
                           "odds_home", "odds_draw", "odds_away"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["target"] + [c for c in FEATURE_COLS[:10] if c in df.columns])

    out = DATA_DIR / "features.csv"
    df.to_csv(out, index=False)
    print(f"\nFeatures: {len(df)} матчей, {len(FEATURE_COLS)} признаков -> {out}")
    print(f"H={sum(df['target']==0)} | D={sum(df['target']==1)} | A={sum(df['target']==2)}")
    return df


if __name__ == "__main__":
    build_features()
