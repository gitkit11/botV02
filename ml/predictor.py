# -*- coding: utf-8 -*-
"""
ml/predictor.py — Интерфейс для использования ML моделей в боте
================================================================
Использование в main.py / football analysis:

    from ml.predictor import get_football_prediction
    pred = get_football_prediction("Man City", "Arsenal", odds)
    # pred["home_win"], pred["draw"], pred["away_win"]
    # pred["lambda_home"], pred["top_scores"]
    # pred["model_used"]
"""

import numpy as np
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"

_dc_model  = None   # Dixon-Coles
_xgb_sport = None   # XGBoost спортивная модель
_xgb_meta  = None   # метаданные

# Нормализация названий команд (API → названия в модели)
_TEAM_NAME_MAP = {
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspurs": "Tottenham",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Sheffield United": "Sheffield Utd",
    "Norwich City": "Norwich",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Paris Saint-Germain": "Paris SG",
    "Paris SG": "Paris SG",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "SS Lazio": "Lazio",
    "AS Roma": "Roma",
    "SSC Napoli": "Napoli",
    "Hellas Verona": "Verona",
    "Olympique Lyon": "Lyon",
    "Olympique Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "Stade Rennais": "Rennes",
    "LOSC Lille": "Lille",
}

def _normalize_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def _load_models():
    global _dc_model, _xgb_sport, _xgb_meta

    # Dixon-Coles
    dc_path = MODEL_DIR / "dixon_coles.pkl"
    if dc_path.exists() and _dc_model is None:
        try:
            with open(dc_path, "rb") as f:
                _dc_model = pickle.load(f)
            logger.info(f"[ML] Dixon-Coles загружен ({_dc_model.get('n_matches',0)} матчей)")
        except Exception as e:
            logger.warning(f"[ML] Dixon-Coles не загружен: {e}")

    # XGBoost
    xgb_path = MODEL_DIR / "football_sport_model.pkl"
    if xgb_path.exists() and _xgb_sport is None:
        try:
            with open(xgb_path, "rb") as f:
                _xgb_sport = pickle.load(f)
            logger.info("[ML] XGBoost Sport загружен")
        except Exception as e:
            logger.warning(f"[ML] XGBoost не загружен: {e}")


def get_football_prediction(
    home_team: str,
    away_team: str,
    bookmaker_odds: dict = None,
    home_elo: float = 1500,
    away_elo: float = 1500,
    home_form: float = 1.5,
    away_form: float = 1.5,
) -> dict:
    """
    Главная функция предсказания для футбола.
    Пробует Dixon-Coles, fallback на XGBoost, fallback на ELO.

    Возвращает:
        home_win, draw, away_win — вероятности (сумма = 1)
        lambda_home, lambda_away — ожидаемые голы
        top_scores — топ-5 счётов с вероятностями
        model_used — какая модель сработала
        ev — expected value для каждого исхода (если есть odds)
    """
    _load_models()

    # Нормализуем названия
    home_team = _normalize_team(home_team)
    away_team = _normalize_team(away_team)

    result = None

    # ── Попытка 1: Dixon-Coles ────────────────────────────────────────────
    if _dc_model is not None:
        try:
            from ml.dixon_coles import predict_proba
            dc = predict_proba(_dc_model, home_team, away_team)
            result = {
                "home_win":    dc["home_win"],
                "draw":        dc["draw"],
                "away_win":    dc["away_win"],
                "lambda_home": dc["lambda_home"],
                "lambda_away": dc["lambda_away"],
                "top_scores":  dc["top_scores"],
                "model_used":  "Dixon-Coles",
                "home_known":  dc["home_known"],
                "away_known":  dc["away_known"],
            }
        except Exception as e:
            logger.warning(f"[ML] Dixon-Coles predict ошибка: {e}")

    # ── Попытка 2: XGBoost ────────────────────────────────────────────────
    if result is None and _xgb_sport is not None:
        try:
            model  = _xgb_sport["model"]
            cals   = _xgb_sport["cals"]
            cols   = _xgb_sport["cols"]

            elo_diff = home_elo - away_elo
            elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

            features = {
                "elo_diff":             elo_diff,
                "elo_home":             home_elo,
                "elo_away":             away_elo,
                "elo_win_prob":         elo_prob,
                "form_home":            home_form,
                "form_away":            away_form,
                "form_diff":            home_form - away_form,
                "form_home_home":       home_form,
                "form_away_away":       away_form,
                "goals_scored_home":    1.5,
                "goals_scored_away":    1.2,
                "goals_conceded_home":  1.1,
                "goals_conceded_away":  1.3,
                "goals_diff_home":      0.3,
                "goals_diff_away":      -0.2,
                "sot_home":             5.0,
                "sot_away":             4.0,
                "sot_diff":             1.0,
                "corners_home":         5.5,
                "corners_away":         4.5,
                "h2h_home_winrate":     0.5,
                "h2h_goals_diff":       0.0,
                "league_id":            0,
                "match_week_norm":      0.5,
            }
            X = np.array([[features.get(c, 0.0) for c in cols]])
            raw = model.predict_proba(X)
            # Применяем калибровку
            cal = np.zeros_like(raw)
            for i, ir in cals.items():
                cal[0, i] = ir.predict(raw[:, i])[0]
            cal /= cal.sum()

            result = {
                "home_win":    round(float(cal[0, 0]), 4),
                "draw":        round(float(cal[0, 1]), 4),
                "away_win":    round(float(cal[0, 2]), 4),
                "lambda_home": None,
                "lambda_away": None,
                "top_scores":  [],
                "model_used":  "XGBoost",
                "home_known":  True,
                "away_known":  True,
            }
        except Exception as e:
            logger.warning(f"[ML] XGBoost predict ошибка: {e}")

    # ── Fallback: ELO Poisson ─────────────────────────────────────────────
    if result is None:
        elo_prob = 1 / (1 + 10 ** (-(home_elo - away_elo) / 400))
        # Простое распределение на основе ELO
        home_win = elo_prob * 0.85  # учёт ничьих
        away_win = (1 - elo_prob) * 0.85
        draw     = 1 - home_win - away_win
        result = {
            "home_win":    round(home_win, 4),
            "draw":        round(draw, 4),
            "away_win":    round(away_win, 4),
            "lambda_home": None,
            "lambda_away": None,
            "top_scores":  [],
            "model_used":  "ELO-fallback",
            "home_known":  False,
            "away_known":  False,
        }

    # ── Expected Value ────────────────────────────────────────────────────
    result["ev"] = {}
    if bookmaker_odds:
        for outcome, bm_key in [("home_win", "home_win"), ("draw", "draw"), ("away_win", "away_win")]:
            bm_odds = bookmaker_odds.get(bm_key, bookmaker_odds.get(outcome.replace("_win", ""), 0))
            if bm_odds and bm_odds > 1.0:
                our_p   = result[outcome]
                implied = 1.0 / bm_odds
                ev = round((our_p - implied) / implied * 100, 1)
                result["ev"][outcome] = {"ev": ev, "odds": bm_odds, "our_prob": round(our_p * 100, 1)}

    return result


def format_ml_block(pred: dict, home_team: str, away_team: str) -> str:
    """
    Форматирует блок ML предсказания для Telegram отчёта.
    """
    model = pred.get("model_used", "ML")
    h = round(pred["home_win"] * 100)
    d = round(pred.get("draw", 0) * 100)
    a = round(pred["away_win"] * 100)

    lines = [f"🔬 <b>{model}:</b> {home_team} {h}% | Х {d}% | {away_team} {a}%"]

    # Ожидаемые голы (Dixon-Coles)
    if pred.get("lambda_home") and pred.get("lambda_away"):
        lh = pred["lambda_home"]
        la = pred["lambda_away"]
        total = round(lh + la, 2)
        lines.append(f"   ⚽ Ожидаемые голы: {lh} + {la} = <b>{total}</b>")

    # Топ счёта
    if pred.get("top_scores"):
        scores_str = "  ".join(f"{s}({p}%)" for s, p in pred["top_scores"][:3])
        lines.append(f"   🎯 Топ счета: {scores_str}")

    # Value bets
    for outcome, data in pred.get("ev", {}).items():
        if data["ev"] >= 5.0:
            name = {"home_win": home_team, "draw": "Ничья", "away_win": away_team}.get(outcome, outcome)
            lines.append(f"   💎 Value: <b>{name}</b> @ {data['odds']} | EV <b>{data['ev']:+.1f}%</b>")

    # Предупреждение если команда неизвестна
    if not pred.get("home_known") or not pred.get("away_known"):
        lines.append(f"   <i>⚠️ Нет истории — среднестатистический прогноз</i>")

    return "\n".join(lines)
