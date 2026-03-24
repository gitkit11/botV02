# -*- coding: utf-8 -*-
"""
ml/predictor_tennis.py — инференс теннисной XGBoost модели
===========================================================
Использование:
    from ml.predictor_tennis import predict_tennis_winner
    result = predict_tennis_winner(
        p1_name="Novak Djokovic", p1_rank=1,
        p2_name="Rafael Nadal",   p2_rank=3,
        surface="Clay",
        level="G",      # G=GrandSlam, M=Masters, A=ATP250/500
        round_str="SF",
        p1_form=0.8,    # доля побед в последних 5-10 матчах (0.0-1.0)
        p2_form=0.6,
        p1_age=36.0, p2_age=37.0,
        h2h_adv=0.1,    # >0 = p1 лучше исторически, от -1 до 1
        h2h_total=5,
    )
    # result: {"winner": "Djokovic", "prob": 0.74, "bet": True, "label": "..."}
"""
import pickle
import json
import numpy as np
from pathlib import Path

_MODEL_DIR  = Path(__file__).parent / "models"
_model      = None
_meta       = None


class TennisCalibratedModel:
    """XGBoost + Platt калибровка. Определена здесь для корректного pickle."""
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        import numpy as _np
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

BET_THRESHOLD = 0.72  # При этом пороге на тесте: 82.1% точность

SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 0}
LEVEL_MAP   = {"G": 4, "F": 3, "M": 3, "A": 2, "D": 1, "O": 1}
ROUND_MAP   = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3, "BR": 2}


def _load_model():
    global _model, _meta
    if _model is not None:
        return _model
    model_path = _MODEL_DIR / "tennis_xgb.pkl"
    meta_path  = _MODEL_DIR / "tennis_xgb_meta.json"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    return _model


def predict_tennis_winner(
    p1_name: str, p1_rank: int,
    p2_name: str, p2_rank: int,
    surface: str = "Hard",
    level: str = "A",
    round_str: str = "R32",
    p1_form: float = 0.5,
    p2_form: float = 0.5,
    p1_age: float = 26.0,
    p2_age: float = 26.0,
    h2h_adv: float = 0.0,
    h2h_total: int = 0,
) -> dict:
    """
    Предсказание победителя теннисного матча.

    p1/p2 порядок не важен — модель сама определяет кто фаворит.
    Возвращает:
        winner     — имя фаворита по модели
        prob       — вероятность победы фаворита (0.5 – 1.0)
        underdog   — имя аутсайдера
        bet        — True если рекомендуется СТАВИТЬ (prob >= threshold)
        label      — строка для отчёта
        threshold  — порог confidence
    """
    model = _load_model()
    if model is None:
        return {"error": "Модель не загружена", "bet": False}

    r1 = max(1, int(p1_rank or 500))
    r2 = max(1, int(p2_rank or 500))

    # Определяем кто player_1 (фаворит по рейтингу)
    fav_is_p1 = (r1 <= r2)
    fav_name  = p1_name if fav_is_p1 else p2_name
    dog_name  = p2_name if fav_is_p1 else p1_name
    fav_rank  = r1 if fav_is_p1 else r2
    dog_rank  = r2 if fav_is_p1 else r1
    fav_form  = p1_form if fav_is_p1 else p2_form
    dog_form  = p2_form if fav_is_p1 else p1_form
    fav_age   = p1_age  if fav_is_p1 else p2_age
    dog_age   = p2_age  if fav_is_p1 else p1_age
    h2h_fav   = h2h_adv if fav_is_p1 else -h2h_adv

    rank_diff      = float(dog_rank - fav_rank)
    rank_log_ratio = float(np.log(max(dog_rank, 1) / max(fav_rank, 1)))

    features = np.array([[
        rank_diff,
        rank_log_ratio,
        float(fav_rank),
        float(dog_rank),
        float(fav_form),
        float(dog_form),
        float(fav_form - dog_form),
        float(fav_age),
        float(dog_age),
        float(fav_age - dog_age),
        float(h2h_fav),
        float(h2h_total),
        float(SURFACE_MAP.get(surface, 0)),
        float(LEVEL_MAP.get(level, 2)),
        float(ROUND_MAP.get(round_str, 3)),
    ]])

    probs = model.predict_proba(features)[0]
    fav_prob = float(probs[1])  # prob that fav wins

    # Если модель неуверена — возвращаем сырую вероятность
    bet = fav_prob >= BET_THRESHOLD

    if bet:
        label = f"🎾 ML: {fav_name} победит | {int(fav_prob*100)}% | {'СТАВИТЬ' if bet else 'ПРОПУСТИТЬ'}"
    else:
        label = f"🎾 ML: {fav_name} фаворит {int(fav_prob*100)}% (ниже порога {int(BET_THRESHOLD*100)}%)"

    return {
        "winner":     fav_name,
        "underdog":   dog_name,
        "prob":       round(fav_prob, 3),
        "prob_pct":   int(fav_prob * 100),
        "bet":        bet,
        "label":      label,
        "threshold":  BET_THRESHOLD,
        "features": {
            "rank_diff":   int(rank_diff),
            "fav_rank":    fav_rank,
            "dog_rank":    dog_rank,
            "surface":     surface,
            "level":       level,
            "fav_form":    round(fav_form, 2),
            "dog_form":    round(dog_form, 2),
            "h2h_adv":     round(h2h_fav, 2),
        }
    }


if __name__ == "__main__":
    # Быстрый тест
    res = predict_tennis_winner(
        p1_name="Djokovic",   p1_rank=2,
        p2_name="Monfils",    p2_rank=45,
        surface="Hard", level="G", round_str="QF",
        p1_form=0.8, p2_form=0.5,
        p1_age=36.0, p2_age=37.0,
        h2h_adv=0.6, h2h_total=20,
    )
    print(res)
