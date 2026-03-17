# -*- coding: utf-8 -*-
"""
ml/dixon_coles.py — Dixon-Coles модель (1997)
==============================================
Золотой стандарт статистического моделирования футбола.

Принцип:
  Каждой команде → attack[i] и defense[i]
  Голы ~ Poisson(attack[home] × defense[away] × home_adv)
  Поправка τ на низкие счета (0-0, 1-0, 0-1, 1-1)

Преимущество над XGBoost:
  - Использует ВСЕ голы, не только W/D/L (в 3x больше информации)
  - Параметры команд обновляются каждую неделю
  - Работает даже с 10 матчами истории
  - Интерпретируемо: attack/defense strength в числах
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─── Функция поправки Dixon-Coles ─────────────────────────────────────────────

def tau(x, y, lam_home, lam_away, rho):
    """
    Поправочный коэффициент для низких счётов.
    Без неё модель переоценивает 0-0 и недооценивает 1-0.
    """
    if x == 0 and y == 0:
        return 1.0 - lam_home * lam_away * rho
    elif x == 1 and y == 0:
        return 1.0 + lam_away * rho
    elif x == 0 and y == 1:
        return 1.0 + lam_home * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dc_log_likelihood(params, teams, matches_arrays, xi=0.0018):
    """
    Векторизованная функция правдоподобия (numpy, без Python-цикла).
    matches_arrays: tuple (hi, ai, hg, ag, weights) — numpy arrays, подготовленные заранее.
    """
    n_teams = len(teams)
    attack   = np.exp(params[:n_teams])
    defense  = np.exp(params[n_teams:2*n_teams])
    home_adv = np.exp(params[2*n_teams])
    rho      = params[2*n_teams + 1]

    hi, ai, hg, ag, weights = matches_arrays

    lam_h = attack[hi] * defense[ai] * home_adv
    lam_a = attack[ai] * defense[hi]

    # tau поправка (только для счётов 0-0, 1-0, 0-1, 1-1)
    tau_vals = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau_vals[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau_vals[m10] = 1.0 + lam_a[m10] * rho
    tau_vals[m01] = 1.0 + lam_h[m01] * rho
    tau_vals[m11] = 1.0 - rho

    # Отфильтровываем tau <= 0
    valid = tau_vals > 0

    # log Poisson PMF = k*log(lam) - lam - log(k!)
    from scipy.special import gammaln
    log_pois_h = hg * np.log(np.maximum(lam_h, 1e-10)) - lam_h - gammaln(hg + 1)
    log_pois_a = ag * np.log(np.maximum(lam_a, 1e-10)) - lam_a - gammaln(ag + 1)

    ll = np.log(np.maximum(tau_vals, 1e-10)) + log_pois_h + log_pois_a
    log_lik = np.sum(weights[valid] * ll[valid])

    return -log_lik


# ─── Обучение ─────────────────────────────────────────────────────────────────

def fit(matches_df: pd.DataFrame, xi=0.0018) -> dict:
    """
    Обучает Dixon-Coles модель.

    matches_df: DataFrame с колонками home, away, hg, ag, date
    xi: скорость затухания (0.0018 = ~385 дней полураспад)

    Возвращает: {"attack": {...}, "defense": {...}, "home_adv": float, "rho": float}
    """
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "hg", "ag"])
    df["date_num"] = (df["date"] - df["date"].min()).dt.days

    teams = sorted(set(df["home"]) | set(df["away"]))
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    print(f"  Команд: {n} | Матчей: {len(df)}")

    # Готовим numpy arrays один раз (не пересоздаём при каждом вызове objective)
    hi_arr = np.array([team_idx[t] for t in df["home"]], dtype=np.int32)
    ai_arr = np.array([team_idx[t] for t in df["away"]], dtype=np.int32)
    hg_arr = df["hg"].values.astype(np.int32)
    ag_arr = df["ag"].values.astype(np.int32)
    today  = df["date_num"].max()
    w_arr  = np.exp(-xi * (today - df["date_num"].values))
    matches_arrays = (hi_arr, ai_arr, hg_arr, ag_arr, w_arr)

    # Начальные параметры: все attack=1, defense=1, home_adv=1.2, rho=-0.1
    x0 = np.zeros(2 * n + 2)
    x0[2 * n]     = np.log(1.2)   # home_adv
    x0[2 * n + 1] = -0.1          # rho

    print("  Оптимизация (векторизовано, ~10-30 сек)...")
    # Нормировка: первая команда = reference (attack[0] фиксирован = 0)
    def _objective(p):
        full_p = np.concatenate([[0.0], p[:n-1],
                                 p[n-1:2*n-1],
                                 p[2*n-1:]])
        return dc_log_likelihood(full_p, teams, matches_arrays, xi)

    x0_reduced = np.concatenate([x0[1:n], x0[n:2*n], x0[2*n:]])
    result = minimize(
        _objective,
        x0_reduced,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-8},
    )
    # Восстанавливаем полный вектор
    params_full = np.concatenate([[0.0], result.x[:n-1],
                                  result.x[n-1:2*n-1],
                                  result.x[2*n-1:]])

    if not result.success:
        print(f"  ⚠️  Оптимизация: {result.message}")

    params = params_full
    attack  = {t: np.exp(params[i])     for i, t in enumerate(teams)}
    defense = {t: np.exp(params[n + i]) for i, t in enumerate(teams)}
    home_adv = np.exp(params[2 * n])
    rho      = params[2 * n + 1]

    return {
        "attack":   attack,
        "defense":  defense,
        "home_adv": home_adv,
        "rho":      rho,
        "teams":    teams,
        "xi":       xi,
        "n_matches": len(df),
    }


# ─── Предсказание ──────────────────────────────────────────────────────────────

def predict_proba(model: dict, home: str, away: str, max_goals=8) -> dict:
    """
    Предсказывает вероятности P(Home Win), P(Draw), P(Away Win).
    Также возвращает lambda_home/away и матрицу счётов.

    Если команда не известна — используем средние значения.
    """
    avg_attack  = np.mean(list(model["attack"].values()))
    avg_defense = np.mean(list(model["defense"].values()))

    lam_h = (model["attack"].get(home, avg_attack)
             * model["defense"].get(away, avg_defense)
             * model["home_adv"])
    lam_a = (model["attack"].get(away, avg_attack)
             * model["defense"].get(home, avg_defense))

    rho = model["rho"]

    # Матрица вероятностей счётов
    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            t = tau(hg, ag, lam_h, lam_a, rho)
            score_matrix[hg, ag] = (t
                                    * poisson.pmf(hg, lam_h)
                                    * poisson.pmf(ag, lam_a))

    # Нормализуем
    score_matrix /= score_matrix.sum()

    p_home = float(np.sum(np.tril(score_matrix, -1)))  # hg > ag
    p_draw = float(np.sum(np.diag(score_matrix)))       # hg == ag
    p_away = float(np.sum(np.triu(score_matrix, 1)))    # hg < ag

    # Топ-5 вероятных счётов
    flat = [(score_matrix[i, j], i, j)
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)]
    flat.sort(reverse=True)
    top_scores = [(f"{hg}:{ag}", round(p * 100, 1)) for p, hg, ag in flat[:5]]

    return {
        "home_win": round(p_home, 4),
        "draw":     round(p_draw, 4),
        "away_win": round(p_away, 4),
        "lambda_home": round(lam_h, 3),
        "lambda_away": round(lam_a, 3),
        "top_scores":  top_scores,
        "home_known":  home in model["attack"],
        "away_known":  away in model["attack"],
    }


def get_team_strength(model: dict, team: str) -> dict:
    """Возвращает силу команды в понятном виде."""
    avg_a = np.mean(list(model["attack"].values()))
    avg_d = np.mean(list(model["defense"].values()))

    atk = model["attack"].get(team, avg_a)
    dfn = model["defense"].get(team, avg_d)

    return {
        "attack":  round(atk, 3),   # >1 = сильная атака
        "defense": round(dfn, 3),   # >1 = слабая оборона
        "attack_rank":  round(atk / avg_a * 100, 0),
        "defense_rank": round(avg_d / dfn * 100, 0),  # выше = лучше
    }


# ─── Сохранение/загрузка ──────────────────────────────────────────────────────

def save_model(model: dict, path=None):
    if path is None:
        path = MODEL_DIR / "dixon_coles.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    # Мета JSON (без больших dict)
    meta = {k: v for k, v in model.items() if k not in ("attack", "defense", "teams")}
    meta["n_teams"] = len(model.get("teams", []))
    with open(MODEL_DIR / "dixon_coles_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Модель сохранена: {path}")


def load_model(path=None) -> dict | None:
    if path is None:
        path = MODEL_DIR / "dixon_coles.pkl"
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Обучение + валидация ─────────────────────────────────────────────────────

def evaluate_model(model: dict, test_df: pd.DataFrame) -> dict:
    """Считает Log Loss и ROI на тестовой выборке."""
    from sklearn.metrics import log_loss

    y_true, y_pred = [], []
    bets = profit = correct = 0

    for _, row in test_df.iterrows():
        pred = predict_proba(model, row["home"], row["away"])
        probs = [pred["home_win"], pred["draw"], pred["away_win"]]

        ftr = row.get("FTR", "")
        actual = {"H": 0, "D": 1, "A": 2}.get(ftr)
        if actual is None:
            continue

        y_true.append(actual)
        y_pred.append(probs)

        # ROI на П1/П2 при EV>5%
        for cls_idx, odds_col in [(0, "odds_home"), (2, "odds_away")]:
            bm_odds = row.get(odds_col, 0)
            if not bm_odds or bm_odds <= 1.0:
                continue
            our_p = probs[cls_idx]
            implied = 1.0 / bm_odds
            ev = (our_p - implied) / implied * 100
            if ev >= 5.0:
                bets += 1
                if actual == cls_idx:
                    profit += bm_odds - 1
                    correct += 1
                else:
                    profit -= 1.0

    ll = log_loss(y_true, y_pred) if y_true else 0
    roi = profit / bets * 100 if bets > 0 else 0
    wr  = correct / bets * 100 if bets > 0 else 0

    print(f"  Log Loss: {ll:.4f}")
    print(f"  ROI (EV>5%): {roi:+.1f}% | {bets} ставок | WR {wr:.1f}%")
    return {"log_loss": round(ll, 4), "roi": round(roi, 2), "bets": bets}


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    DATA_DIR = Path(__file__).parent / "data"
    raw = pd.read_csv(DATA_DIR / "all_matches_raw.csv", encoding="latin-1", low_memory=False)

    # Подготовка
    raw = raw[raw["FTR"].isin(["H", "D", "A"])].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date")

    # Коэффициенты
    for col, candidates in [
        ("odds_home", ["B365H", "BbAvH", "PSH"]),
        ("odds_draw", ["B365D", "BbAvD", "PSD"]),
        ("odds_away", ["B365A", "BbAvA", "PSA"]),
    ]:
        for c in candidates:
            if c in raw.columns:
                raw[col] = pd.to_numeric(raw[c], errors="coerce")
                break

    matches = raw[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                   "odds_home", "odds_draw", "odds_away"]].copy()
    matches.columns = ["date", "home", "away", "hg", "ag", "FTR",
                       "odds_home", "odds_draw", "odds_away"]

    # Train: всё кроме последних 10%
    n = len(matches)
    train = matches.iloc[:int(n * 0.9)]
    test  = matches.iloc[int(n * 0.9):]

    print(f"Dixon-Coles обучение")
    print(f"Train: {len(train)} | Test: {len(test)}")

    model = fit(train)

    print(f"\nПараметры модели:")
    print(f"  Home advantage: {model['home_adv']:.3f}")
    print(f"  Rho (low-score correction): {model['rho']:.4f}")

    # Топ-10 команд по атаке
    atk_sorted = sorted(model["attack"].items(), key=lambda x: x[1], reverse=True)
    print(f"\nТоп-10 атак:")
    for team, val in atk_sorted[:10]:
        print(f"  {team:25s} {val:.3f}")

    print(f"\nОценка на тесте:")
    evaluate_model(model, test)

    # Пример предсказания
    print(f"\nПример: Man City vs Arsenal")
    pred = predict_proba(model, "Man City", "Arsenal")
    print(f"  П1: {pred['home_win']*100:.1f}% | Х: {pred['draw']*100:.1f}% | П2: {pred['away_win']*100:.1f}%")
    print(f"  λ_home: {pred['lambda_home']} | λ_away: {pred['lambda_away']}")
    print(f"  Топ счета: {pred['top_scores']}")

    save_model(model)
