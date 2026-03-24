# -*- coding: utf-8 -*-
"""
Тесты для math_model.py — ELO, Poisson, форма команды.
Запуск: python -m pytest tests/test_math_model.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from math_model import (
    expected_score,
    update_elo,
    elo_win_probabilities,
    get_form_bonus,
    get_form_string,
    poisson_match_probabilities,
    DEFAULT_ELO,
    K_FACTOR,
)


# ─── expected_score ───────────────────────────────────────────────────────────

def test_expected_score_equal():
    # Равные ELO → ровно 50%
    assert abs(expected_score(1500, 1500) - 0.5) < 0.001

def test_expected_score_stronger_wins_more():
    # Сильная команда ожидает больше очков
    assert expected_score(1700, 1500) > 0.5

def test_expected_score_weaker_expects_less():
    assert expected_score(1300, 1500) < 0.5

def test_expected_score_symmetric():
    e_ab = expected_score(1600, 1400)
    e_ba = expected_score(1400, 1600)
    assert abs(e_ab + e_ba - 1.0) < 0.001


# ─── update_elo ───────────────────────────────────────────────────────────────

def test_elo_winner_gains():
    ratings = {"A": 1500, "B": 1500}
    new = update_elo("A", "B", 2, 0, ratings)
    assert new["A"] > 1500
    assert new["B"] < 1500

def test_elo_loser_drops():
    ratings = {"A": 1700, "B": 1300}
    new = update_elo("A", "B", 0, 1, ratings)  # фаворит проигрывает
    assert new["A"] < 1700
    assert new["B"] > 1300

def test_elo_draw_close_teams():
    ratings = {"A": 1500, "B": 1500}
    new = update_elo("A", "B", 1, 1, ratings)
    # Ничья у равных — почти не меняет (дом преимущество чуть штрафует хозяев)
    assert abs(new["A"] - 1500) < 20
    assert abs(new["B"] - 1500) < 20

def test_elo_unknown_team_gets_default():
    ratings = {}
    new = update_elo("New Team A", "New Team B", 1, 0, ratings)
    assert "New Team A" in new
    assert "New Team B" in new

def test_elo_sum_conserved():
    """Сумма ELO двух команд должна оставаться примерно постоянной."""
    ratings = {"A": 1600, "B": 1400}
    before = ratings["A"] + ratings["B"]
    new = update_elo("A", "B", 2, 1, ratings)
    after = new["A"] + new["B"]
    assert abs(after - before) < 0.5  # округление float


# ─── elo_win_probabilities ────────────────────────────────────────────────────

def test_elo_probs_sum_to_one():
    ratings = {"A": 1600, "B": 1450}
    probs = elo_win_probabilities("A", "B", ratings)
    total = probs["home"] + probs["draw"] + probs["away"]
    assert abs(total - 1.0) < 0.01

def test_elo_probs_home_advantage():
    # При равных командах хозяева имеют преимущество
    ratings = {"A": 1500, "B": 1500}
    probs = elo_win_probabilities("A", "B", ratings)
    assert probs["home"] > probs["away"]

def test_elo_probs_strong_team_favored():
    ratings = {"Strong": 1800, "Weak": 1300}
    probs = elo_win_probabilities("Strong", "Weak", ratings)
    assert probs["home"] > 0.5

def test_elo_probs_keys():
    ratings = {"A": 1500, "B": 1500}
    probs = elo_win_probabilities("A", "B", ratings)
    for key in ("home", "draw", "away"):
        assert key in probs
        assert 0 < probs[key] < 1


# ─── get_form_bonus ───────────────────────────────────────────────────────────

def test_form_bonus_all_wins():
    form = {"Team A": ["W", "W", "W", "W", "W"]}
    bonus = get_form_bonus("Team A", form)
    assert bonus > 0

def test_form_bonus_all_losses():
    form = {"Team A": ["L", "L", "L", "L", "L"]}
    bonus = get_form_bonus("Team A", form)
    assert bonus < 0

def test_form_bonus_no_data():
    assert get_form_bonus("Unknown Team", {}) == 0.0

def test_form_bonus_recent_matters_more():
    # Форма LLLLW (последний матч W) vs WLLLL (первый матч W)
    # Последние матчи имеют больший вес
    form_recent_w = {"T": ["L", "L", "L", "L", "W"]}
    form_old_w    = {"T": ["W", "L", "L", "L", "L"]}
    b_recent = get_form_bonus("T", form_recent_w)
    b_old    = get_form_bonus("T", form_old_w)
    assert b_recent > b_old


# ─── get_form_string ─────────────────────────────────────────────────────────

def test_form_string_returns_last5():
    # ["W","L","D","W","L","W","W"] → последние 5: ["D","W","L","W","W"] → "DWLWW"
    form = {"A": ["W", "L", "D", "W", "L", "W", "W"]}
    s = get_form_string("A", form)
    assert s == "DWLWW"
    assert len(s) == 5
    assert all(c in "WDL" for c in s)

def test_form_string_unknown_team():
    s = get_form_string("Unknown", {})
    assert s == "?????"

def test_form_string_short_history():
    form = {"A": ["W", "D"]}
    s = get_form_string("A", form)
    assert "W" in s or "D" in s


# ─── poisson_match_probabilities ─────────────────────────────────────────────

def test_poisson_probs_sum_to_one():
    probs = poisson_match_probabilities(1.5, 1.1)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 0.01

def test_poisson_strong_attack_favors_home():
    probs = poisson_match_probabilities(2.5, 0.5)  # Хозяева атакуют намного сильнее
    assert probs["home_win"] > probs["away_win"]

def test_poisson_probs_keys():
    probs = poisson_match_probabilities(1.3, 1.3)
    for key in ("home_win", "draw", "away_win"):
        assert key in probs
        assert 0 < probs[key] < 1

def test_poisson_symmetric_teams():
    probs = poisson_match_probabilities(1.3, 1.3)
    assert abs(probs["home_win"] - probs["away_win"]) < 0.1  # примерно равны
