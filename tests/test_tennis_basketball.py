# -*- coding: utf-8 -*-
"""
Тесты для новых функций теннис и баскетбол.
"""

import pytest


# ─── Tennis: form_adjustment ──────────────────────────────────────────────────

def test_form_all_wins():
    from sports.tennis.model import form_adjustment
    result = form_adjustment("WWWWW")
    assert result > 0, "WWWWW должен давать положительный бонус"
    assert result <= 0.05


def test_form_all_losses():
    from sports.tennis.model import form_adjustment
    result = form_adjustment("LLLLL")
    assert result < 0, "LLLLL должен давать отрицательный штраф"
    assert result >= -0.05


def test_form_mixed():
    from sports.tennis.model import form_adjustment
    # "LLLWW" — первый символ = самый последний матч (L).
    # 3 поражения подряд в конце перевешивают 2 старые победы → минус
    result = form_adjustment("LLLWW")
    assert result < 0


def test_form_empty():
    from sports.tennis.model import form_adjustment
    assert form_adjustment("") == 0.0
    assert form_adjustment("?????") == 0.0
    assert form_adjustment(None) == 0.0


def test_form_single_win():
    from sports.tennis.model import form_adjustment
    # Одна победа → положительный результат
    assert form_adjustment("W") > 0


def test_form_recent_wins_weighted_more():
    from sports.tennis.model import form_adjustment
    # WWWLL — последние 3 победы весят больше чем 2 поражения в конце
    v1 = form_adjustment("WWWLL")
    v2 = form_adjustment("LLWWW")
    # v2 — последние две победы, но первые три поражения тяжелее
    # экспоненциальные веса: последний матч (индекс 0) весит 1.0
    # в WWWLL: W(1.0)+W(0.65)+W(0.45)+L(0.30)+L(0.20)
    # в LLWWW: L(1.0)+L(0.65)+W(0.45)+W(0.30)+W(0.20)
    assert v1 > v2  # WWWLL лучше чем LLWWW


# ─── Tennis: rest_days_adjustment ────────────────────────────────────────────

def test_rest_days_fatigue():
    from sports.tennis.model import rest_days_adjustment
    assert rest_days_adjustment(0) == -0.03   # сегодня
    assert rest_days_adjustment(1) == -0.03   # вчера


def test_rest_days_optimal():
    from sports.tennis.model import rest_days_adjustment
    assert rest_days_adjustment(2) == +0.01
    assert rest_days_adjustment(3) == +0.01
    assert rest_days_adjustment(4) == +0.01


def test_rest_days_neutral():
    from sports.tennis.model import rest_days_adjustment
    assert rest_days_adjustment(5)  == 0.0
    assert rest_days_adjustment(10) == 0.0
    assert rest_days_adjustment(13) == 0.0


def test_rest_days_long_break():
    from sports.tennis.model import rest_days_adjustment
    assert rest_days_adjustment(14) == -0.02
    assert rest_days_adjustment(30) == -0.02


def test_rest_days_unknown():
    from sports.tennis.model import rest_days_adjustment
    assert rest_days_adjustment(-1) == 0.0   # нет данных


# ─── Basketball: REST_PENALTY dict и логика _fatigue_penalty ─────────────────

def test_rest_penalty_dict_keys():
    from sports.basketball.core import REST_PENALTY
    assert 0 in REST_PENALTY
    assert 1 in REST_PENALTY
    assert 2 in REST_PENALTY


def test_rest_penalty_values():
    from sports.basketball.core import REST_PENALTY
    assert REST_PENALTY[0] < 0,  "0 дней — должен быть штраф"
    assert REST_PENALTY[1] < 0,  "1 день — должен быть штраф"
    assert REST_PENALTY[2] < 0,  "2 дня — должен быть штраф"
    # Штраф снижается по мере отдыха
    assert REST_PENALTY[0] < REST_PENALTY[1] < REST_PENALTY[2]


def test_rest_penalty_no_penalty_beyond_3():
    from sports.basketball.core import REST_PENALTY
    # 3+ дней отдыха → нет штрафа (ключа нет, fallback = 0.0)
    assert REST_PENALTY.get(3, 0.0) == 0.0
    assert REST_PENALTY.get(99, 0.0) == 0.0


def test_fatigue_no_rest_days_returns_zero():
    """Если нет данных о последней игре → нет штрафа (days=99, не в REST_PENALTY)."""
    from sports.basketball.core import REST_PENALTY
    days = 99  # нет данных = get_rest_days возвращает 99
    result = REST_PENALTY.get(days, 0.0)
    assert result == 0.0
