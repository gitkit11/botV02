# -*- coding: utf-8 -*-
"""
Тесты для signal_engine.py
Запуск: python -m pytest tests/test_signal_engine.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from signal_engine import (
    check_football_signal,
    check_cs2_signal,
    _calc_ev,
    _count_wins,
    _kelly,
    FOOTBALL_CFG,
    CS2_CFG,
)


# ─── _calc_ev ────────────────────────────────────────────────────────────────

def test_calc_ev_positive():
    # 60% prob × 2.0 odds - 1 = +0.20
    assert abs(_calc_ev(0.60, 2.0) - 0.20) < 0.001

def test_calc_ev_negative():
    # 40% prob × 1.5 odds - 1 = -0.40
    assert _calc_ev(0.40, 1.5) < 0

def test_calc_ev_invalid_odds():
    assert _calc_ev(0.9, 0.5) == -1.0


# ─── _count_wins ─────────────────────────────────────────────────────────────

def test_count_wins_all():
    assert _count_wins("WWWWW") == 5

def test_count_wins_mixed():
    assert _count_wins("WLDWW") == 3

def test_count_wins_empty():
    assert _count_wins("") == 0

def test_count_wins_lowercase():
    assert _count_wins("wwdll") == 2


# ─── _kelly ──────────────────────────────────────────────────────────────────

def test_kelly_positive():
    k = _kelly(0.60, 2.0)
    assert k > 0

def test_kelly_capped():
    # Очень высокие prob и odds не должны давать больше 15% (default max_kelly)
    k = _kelly(0.99, 10.0)
    assert k <= 15.0

def test_kelly_no_value():
    k = _kelly(0.40, 1.5)  # отрицательный Kelly → 0
    assert k == 0.0


# ─── check_football_signal ───────────────────────────────────────────────────

def _make_football_bookmaker(home=2.1, draw=3.3, away=3.5):
    return {"home_win": home, "draw": draw, "away_win": away}

def test_football_strong_signal():
    """Чёткий фаворит должен получить сигнал."""
    signals = check_football_signal(
        home_team="Большая Команда",
        away_team="Маленькая Команда",
        home_prob=0.65,
        away_prob=0.20,
        draw_prob=0.15,
        bookmaker_odds=_make_football_bookmaker(home=2.0),
        home_form="WWWLW",
        away_form="LLLWL",
        elo_home=1700,
        elo_away=1500,
        ai_agrees=True,
    )
    assert len(signals) > 0
    assert signals[0]["outcome"] == "П1"
    assert signals[0]["score"] >= FOOTBALL_CFG["min_score"]

def test_football_no_signal_low_prob():
    """Слабая вероятность → нет сигнала."""
    signals = check_football_signal(
        home_team="A",
        away_team="B",
        home_prob=0.35,
        away_prob=0.40,
        draw_prob=0.25,
        bookmaker_odds=_make_football_bookmaker(home=2.5, away=2.0),
        elo_home=1500,
        elo_away=1500,
    )
    # Ни один исход не должен быть выше min_score
    for s in signals:
        assert s["score"] < FOOTBALL_CFG["min_score"]

def test_football_no_signal_bad_odds():
    """Кэф вне диапазона + EV отрицательный + нет формы/ELO/AI → нет сигнала для П1."""
    signals = check_football_signal(
        home_team="A",
        away_team="B",
        home_prob=0.80,
        away_prob=0.10,
        draw_prob=0.10,
        bookmaker_odds={"home_win": 1.10, "draw": 8.0, "away_win": 12.0},
        # Нет формы, нет ELO данных, AI не согласен
        elo_home=0,
        elo_away=0,
        ai_agrees=False,
    )
    for s in signals:
        if s["outcome"] == "П1":
            # Кэф 1.10 вне диапазона, EV отрицательный, AI против → ≤ 1 балл
            assert s["score"] < FOOTBALL_CFG["min_score"]

def test_football_returns_list():
    signals = check_football_signal(
        home_team="A", away_team="B",
        home_prob=0.55, away_prob=0.25, draw_prob=0.20,
        bookmaker_odds=_make_football_bookmaker(),
    )
    assert isinstance(signals, list)


# ─── check_cs2_signal ────────────────────────────────────────────────────────

def test_cs2_strong_signal():
    """Явный фаворит в CS2 должен пройти фильтр."""
    signals = check_cs2_signal(
        home_team="Team Vitality",
        away_team="Unknown Team XYZ",
        home_prob=0.70,
        away_prob=0.30,
        bookmaker_odds={"home_win": 1.75, "away_win": 2.10},
        home_form="WWWWL",
        away_form="LLLLL",
        elo_home=1820,
        elo_away=1300,
        mis_home=0.65,
        mis_away=0.35,
        home_avg_rating=1.20,
        away_avg_rating=1.00,
        home_map_winrates={"Nuke": 70.0, "Inferno": 65.0},
        away_map_winrates={"Nuke": 48.0, "Inferno": 45.0},
        home_key_players_form=[1.35, 1.12, 1.08],
        away_key_players_form=[1.00, 0.95, 0.98],
        ai_cs2_agrees=True,
    )
    assert len(signals) > 0
    assert signals[0]["score"] >= CS2_CFG["min_score"]

def test_cs2_no_signal_weak():
    """Слабые показатели → нет сигнала."""
    signals = check_cs2_signal(
        home_team="A",
        away_team="B",
        home_prob=0.48,
        away_prob=0.52,
        bookmaker_odds={"home_win": 2.0, "away_win": 1.8},
    )
    for s in signals:
        assert s["score"] < CS2_CFG["min_score"]

def test_cs2_returns_list():
    signals = check_cs2_signal(
        home_team="A", away_team="B",
        home_prob=0.55, away_prob=0.45,
        bookmaker_odds={"home_win": 1.9, "away_win": 2.0},
    )
    assert isinstance(signals, list)


# ─── Конфиг константы ────────────────────────────────────────────────────────

def test_football_cfg_keys():
    for key in ("min_prob", "min_ev", "min_odds", "max_odds", "min_score"):
        assert key in FOOTBALL_CFG

def test_cs2_cfg_keys():
    for key in ("min_prob", "min_ev", "min_odds", "max_odds", "min_score"):
        assert key in CS2_CFG
