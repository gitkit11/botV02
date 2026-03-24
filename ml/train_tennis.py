# -*- coding: utf-8 -*-
"""
ml/train_tennis.py — XGBoost модель для теннисных прогнозов
============================================================
Данные: ATP матчи 2015-2024 (Jeff Sackmann dataset, ~27k матчей)
Цель: бинарная — победит ли player_1 (1) или player_2 (0)

Ключевой принцип: СТАВИТЬ только при уверенности >= 70%.
На этом срезе исторически точность 74-80%.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

LEVEL_MAP  = {"G": 4, "F": 3, "M": 3, "A": 2, "D": 1, "O": 1}
ROUND_MAP  = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3, "BR": 2}
SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 0}


# ─── 1. Загрузка и предобработка ──────────────────────────────────────────────

def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Убираем матчи без рейтинга (нельзя строить признаки)
    df = df.dropna(subset=["winner_rank", "loser_rank"]).copy()

    # Убираем неиграные матчи (W/O, RET до начала) — в score нет цифр
    df = df[df["score"].str.contains(r"\d", na=False)].copy()
    # Убираем ретайрменты в процессе игры
    df = df[~df["score"].str.contains(r"RET|W/O|DEF", na=False)].copy()

    # Нормализация рейтинга: если нет — 500 (средний игрок без рейтинга)
    df["winner_rank"] = df["winner_rank"].fillna(500).clip(1, 2000)
    df["loser_rank"]  = df["loser_rank"].fillna(500).clip(1, 2000)

    df["surface_code"]  = df["surface"].map(SURFACE_MAP).fillna(0).astype(int)
    df["level_code"]    = df["tourney_level"].map(LEVEL_MAP).fillna(1).astype(int)
    df["round_code"]    = df["round"].map(ROUND_MAP).fillna(3).astype(int)
    df["tourney_date"]  = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df


# ─── 2. Rolling форма игроков ──────────────────────────────────────────────────

def build_player_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого матча считаем форму ПЕРЕД этим матчем:
    - wins_last5 / matches_last5 → win_rate_last5
    - разрыв в форме между игроками
    """
    # Строим таблицу "игрок → матчи" для расчёта формы
    player_wins  = {}   # player_id → deque последних результатов (1=win, 0=loss)
    from collections import deque

    form_w = []  # форма winner до матча
    form_l = []  # форма loser до матча

    for _, row in df.iterrows():
        wid = row["winner_id"]
        lid = row["loser_id"]

        wf = player_wins.get(wid, deque(maxlen=10))
        lf = player_wins.get(lid, deque(maxlen=10))

        # Записываем форму ДО матча
        form_w.append(sum(wf) / len(wf) if len(wf) >= 3 else 0.5)
        form_l.append(sum(lf) / len(lf) if len(lf) >= 3 else 0.5)

        # Обновляем после
        wf_new = player_wins.get(wid, deque(maxlen=10))
        lf_new = player_wins.get(lid, deque(maxlen=10))
        wf_new.append(1)
        lf_new.append(0)
        player_wins[wid] = wf_new
        player_wins[lid] = lf_new

    df["winner_form"] = form_w
    df["loser_form"]  = form_l
    return df


# ─── 3. H2H статистика ────────────────────────────────────────────────────────

def build_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """H2H до текущего матча (избегаем утечки данных)."""
    h2h_wins = {}  # (min_id, max_id, who_is_min) → wins_by_min

    h2h_adv_w = []  # H2H преимущество winner
    h2h_total  = []

    for _, row in df.iterrows():
        wid, lid = int(row["winner_id"]), int(row["loser_id"])
        key = (min(wid, lid), max(wid, lid))
        stats = h2h_wins.get(key, {"w1": 0, "w2": 0})  # w1=wins for min_id

        total = stats["w1"] + stats["w2"]
        if total >= 2:
            if wid == key[0]:
                adv = (stats["w1"] - stats["w2"]) / total
            else:
                adv = (stats["w2"] - stats["w1"]) / total
        else:
            adv = 0.0

        h2h_adv_w.append(adv)
        h2h_total.append(total)

        # Обновляем после
        if wid == key[0]:
            stats["w1"] += 1
        else:
            stats["w2"] += 1
        h2h_wins[key] = stats

    df["h2h_adv"]   = h2h_adv_w   # >0 = winner исторически лучше
    df["h2h_total"] = h2h_total
    return df


# ─── 4. Построение признаков ───────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаём признаки симметрично: player_1 = фаворит по рейтингу (меньший rank).
    target = 1 если player_1 (фаворит по рейтингу) выиграл.
    """
    rows = []
    for _, row in df.iterrows():
        wr = float(row["winner_rank"])
        lr = float(row["loser_rank"])

        # player_1 = лучший по рейтингу (меньше = лучше)
        fav_is_winner = (wr <= lr)

        p1_rank  = wr  if fav_is_winner else lr
        p2_rank  = lr  if fav_is_winner else wr
        p1_form  = float(row["winner_form"]) if fav_is_winner else float(row["loser_form"])
        p2_form  = float(row["loser_form"])  if fav_is_winner else float(row["winner_form"])
        p1_age   = float(row.get("winner_age", 25) or 25) if fav_is_winner else float(row.get("loser_age", 25) or 25)
        p2_age   = float(row.get("loser_age", 25) or 25)  if fav_is_winner else float(row.get("winner_age", 25) or 25)
        h2h_adv  = float(row["h2h_adv"]) if fav_is_winner else -float(row["h2h_adv"])

        rank_diff      = p2_rank - p1_rank           # >0 = p1 лучше
        rank_log_ratio = np.log(p2_rank / p1_rank)   # >0 = p1 лучше
        form_diff      = p1_form - p2_form            # >0 = p1 в форме

        rows.append({
            "rank_diff":      rank_diff,
            "rank_log_ratio": rank_log_ratio,
            "p1_rank":        p1_rank,
            "p2_rank":        p2_rank,
            "p1_form":        p1_form,
            "p2_form":        p2_form,
            "form_diff":      form_diff,
            "p1_age":         p1_age,
            "p2_age":         p2_age,
            "age_diff":       p1_age - p2_age,
            "h2h_adv":        h2h_adv,
            "h2h_total":      float(row["h2h_total"]),
            "surface":        int(row["surface_code"]),
            "level":          int(row["level_code"]),
            "round":          int(row["round_code"]),
            "target":         int(fav_is_winner),
        })

    return pd.DataFrame(rows)


import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from predictor_tennis import TennisCalibratedModel  # noqa: F401 — нужен для pickle

FEATURE_COLS = [
    "rank_diff", "rank_log_ratio",
    "p1_rank", "p2_rank",
    "p1_form", "p2_form", "form_diff",
    "p1_age", "p2_age", "age_diff",
    "h2h_adv", "h2h_total",
    "surface", "level", "round",
]


# ─── 5. Обучение ──────────────────────────────────────────────────────────────

def train():
    print("=== Tennis XGBoost Training ===\n")

    raw_path = DATA_DIR / "tennis_raw.csv"
    if not raw_path.exists():
        print("ОШИБКА: tennis_raw.csv не найден. Запусти download_tennis_data.py")
        return

    print("[1/5] Загружаем и чистим данные...")
    df = load_and_prepare(raw_path)
    print(f"  Матчей после очистки: {len(df)}")

    print("[2/5] Считаем форму игроков...")
    df = build_player_form(df)

    print("[3/5] Считаем H2H...")
    df = build_h2h(df)

    print("[4/5] Строим признаки...")
    feat_df = build_features(df)
    feat_df = feat_df.dropna()
    print(f"  Примеров с признаками: {len(feat_df)}")
    print(f"  Базовая точность (всегда фаворит): {feat_df['target'].mean():.3f}")

    X = feat_df[FEATURE_COLS].values
    y = feat_df["target"].values

    # Временное разбиение (не случайное — чтобы не было утечки)
    n = len(feat_df)
    n_train = int(n * 0.75)
    n_val   = int(n * 0.875)

    X_tr, y_tr = X[:n_train],  y[:n_train]
    X_vl, y_vl = X[n_train:n_val], y[n_train:n_val]
    X_te, y_te = X[n_val:],    y[n_val:]
    print(f"  Train: {len(X_tr)} | Val: {len(X_vl)} | Test: {len(X_te)}")

    print("[5/5] Обучаем XGBoost...")
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, roc_auc_score

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        gamma=3,
        reg_alpha=1.5,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False,
    )

    # Калибровка вероятностей (Platt scaling) на val-сете
    from sklearn.linear_model import LogisticRegression
    raw_probs_vl = model.predict_proba(X_vl)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(raw_probs_vl, y_vl)
    calibrated = TennisCalibratedModel(model, platt)

    # ─── Метрики ──────────────────────────────────────────────────────────────
    probs_te = calibrated.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= 0.5).astype(int)

    acc_all = accuracy_score(y_te, preds_te)
    auc     = roc_auc_score(y_te, probs_te)

    print(f"\n  === РЕЗУЛЬТАТЫ (тест) ===")
    print(f"  Точность (все ставки, prob>=0.50): {acc_all:.3f} ({acc_all*100:.1f}%)")
    print(f"  ROC AUC: {auc:.3f}")

    # Метрики при разных порогах уверенности
    print(f"\n  Точность при высоком confidence (фильтр СТАВИТЬ):")
    for thresh in [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]:
        mask = probs_te >= thresh
        n_bets = mask.sum()
        if n_bets > 20:
            acc = accuracy_score(y_te[mask], preds_te[mask])
            print(f"    prob >= {thresh:.2f}: {n_bets} ставок → точность {acc*100:.1f}%")

    # ─── Сохранение ───────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "tennis_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)

    meta = {
        "feature_cols":      FEATURE_COLS,
        "acc_all":           round(float(acc_all), 4),
        "roc_auc":           round(float(auc), 4),
        "n_train":           int(n_train),
        "n_test":            int(len(X_te)),
        "bet_threshold":     0.70,
        "surface_map":       {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 0},
        "level_map":         LEVEL_MAP,
        "round_map":         ROUND_MAP,
    }
    meta_path = MODEL_DIR / "tennis_xgb_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Модель сохранена: {model_path}")
    print(f"  Мета:             {meta_path}")
    return calibrated, meta


if __name__ == "__main__":
    train()
