# -*- coding: utf-8 -*-
"""
Тесты для chimera_signal.py
Запуск: python -m pytest tests/test_chimera_signal.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from chimera_signal import (
    compute_chimera_score,
    calibrate_probability,
    _form_score,
    _implied_prob,
    MIN_CHIMERA_SCORE,
    ELO_WEIGHT,
    VALUE_WEIGHT,
)


# ─── _form_score ─────────────────────────────────────────────────────────────

def test_form_score_all_wins():
    assert _form_score("WWWWW") > 0.9

def test_form_score_all_losses():
    assert _form_score("LLLLL") == 0.0

def test_form_score_empty():
    # Нет данных → нейтральный 0.5
    assert _form_score("") == 0.5

def test_form_score_mixed():
    score = _form_score("WDLWW")
    assert 0.0 < score < 1.0

def test_form_score_uses_last5():
    # Берёт последние 5 символов
    s1 = _form_score("WWWWW")
    s2 = _form_score("LLLLLWWWWW")  # хвост WWWWW
    assert s1 == s2


# ─── _implied_prob ────────────────────────────────────────────────────────────

def test_implied_prob_evens():
    # 2.0 → 50%
    assert abs(_implied_prob(2.0) - 0.50) < 0.001

def test_implied_prob_short_fav():
    # 1.5 → 66.7%
    assert abs(_implied_prob(1.5) - 1 / 1.5) < 0.001

def test_implied_prob_invalid():
    # Невалидные кэфы (≤ 1.02) → 0.0 (не ставим)
    assert _implied_prob(0.0) == 0.0
    assert _implied_prob(-1.0) == 0.0
    assert _implied_prob(1.0) == 0.0


# ─── calibrate_probability ────────────────────────────────────────────────────

def test_calibrate_no_data():
    # Без данных калибровки — возвращает оригинал
    assert calibrate_probability(0.6, {}) == 0.6
    assert calibrate_probability(0.6, {"total_checked": 0}) == 0.6

def test_calibrate_positive_bias():
    cal = {"bias": 0.10, "total_checked": 50}
    result = calibrate_probability(0.55, cal)
    # Должно вырасти (мы недооценивали)
    assert result > 0.55

def test_calibrate_negative_bias():
    cal = {"bias": -0.10, "total_checked": 50}
    result = calibrate_probability(0.70, cal)
    # Должно упасть (мы переоценивали)
    assert result < 0.70

def test_calibrate_clamped():
    # Результат всегда в [0.10, 0.90]
    cal = {"bias": 0.50, "total_checked": 50}
    result = calibrate_probability(0.85, cal)
    assert result <= 0.90
    cal2 = {"bias": -0.50, "total_checked": 50}
    result2 = calibrate_probability(0.15, cal2)
    assert result2 >= 0.10


# ─── compute_chimera_score ────────────────────────────────────────────────────

def _make_odds(home=2.0, draw=3.4, away=3.8):
    return {"home_win": home, "draw": draw, "away_win": away}

def test_chimera_returns_sorted():
    """Кандидаты должны быть отсортированы по убыванию score."""
    results = compute_chimera_score(
        home_team="A", away_team="B",
        home_prob=0.55, away_prob=0.25, draw_prob=0.20,
        bookmaker_odds=_make_odds(),
        elo_home=1600, elo_away=1400,
    )
    if len(results) >= 2:
        assert results[0]["chimera_score"] >= results[1]["chimera_score"]

def test_chimera_strong_favorite_wins():
    """Сильный фаворит → П1 должен быть первым кандидатом с высоким chimera_score."""
    results = compute_chimera_score(
        home_team="Топ Команда",
        away_team="Слабая Команда",
        home_prob=0.72,
        away_prob=0.15,
        draw_prob=0.13,
        bookmaker_odds={"home_win": 1.80, "draw": 4.5, "away_win": 5.0},
        home_form="WWWWW",
        away_form="LLLLL",
        elo_home=1800,
        elo_away=1400,
    )
    assert len(results) > 0
    best = results[0]
    assert best["outcome"] == "П1"
    assert best["chimera_score"] > MIN_CHIMERA_SCORE

def test_chimera_equal_teams_low_score():
    """Равные команды с плохим EV → chimera_score ниже порога."""
    results = compute_chimera_score(
        home_team="A", away_team="B",
        home_prob=0.34, away_prob=0.33, draw_prob=0.33,
        bookmaker_odds=_make_odds(home=2.94, draw=2.94, away=2.94),
        elo_home=1500, elo_away=1500,
    )
    for r in results:
        # При равных вероятностях и кэфах с маржой — EV отрицательный
        assert r["chimera_score"] < MIN_CHIMERA_SCORE

def test_chimera_result_structure():
    """Каждый кандидат должен содержать обязательные ключи."""
    results = compute_chimera_score(
        home_team="A", away_team="B",
        home_prob=0.50, away_prob=0.30, draw_prob=0.20,
        bookmaker_odds=_make_odds(),
    )
    required_keys = {"team", "outcome", "chimera_score", "prob", "odds", "ev"}
    for r in results:
        for k in required_keys:
            assert k in r, f"Ключ '{k}' отсутствует в кандидате"

def test_chimera_no_outcome_for_bad_odds():
    """Кэфы <= 1.20 не должны создавать кандидатов."""
    results = compute_chimera_score(
        home_team="A", away_team="B",
        home_prob=0.90, away_prob=0.05, draw_prob=0.05,
        bookmaker_odds={"home_win": 1.05, "draw": 15.0, "away_win": 20.0},
    )
    home_candidates = [r for r in results if r["outcome"] == "П1"]
    assert len(home_candidates) == 0


# ─── Константы ───────────────────────────────────────────────────────────────

def test_min_chimera_score_positive():
    assert MIN_CHIMERA_SCORE > 0

def test_weights_positive():
    assert ELO_WEIGHT > 0
    assert VALUE_WEIGHT > 0
