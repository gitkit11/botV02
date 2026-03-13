# -*- coding: utf-8 -*-
"""
Football Module — Chimera AI
==============================
Независимый модуль для анализа футбольных матчей.

✅ Полностью реализован в v4.4.

Использует:
- The Odds API (матчи, коэффициенты, 10 лиг)
- Understat (xG статистика)
- Пуассон + Dixon-Coles (голевые вероятности)
- ELO рейтинг (реальный, сезон 2024/25)
- Форма последних 5 матчей
- Травмы и дисквалификации (GNews + GPT)
- AI агенты (GPT-4o + Llama 3.3 70B)
- Value Bets (EV > 10%, Kelly %)

Для подключения нового разработчика:
1. Все основные файлы в корне проекта (main.py, agents.py, math_model.py и др.)
2. Этот модуль — точка входа для импорта из других модулей
"""

# Реэкспорт ключевых функций из корневых файлов
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from math_model import poisson_match_probabilities, elo_win_probabilities
from agents import build_math_ensemble, calculate_value_bets
from understat_stats import format_xg_stats, get_team_xg_stats

STATUS = "active"

__all__ = [
    "poisson_match_probabilities",
    "elo_win_probabilities",
    "build_math_ensemble",
    "calculate_value_bets",
    "format_xg_stats",
    "get_team_xg_stats",
]
