# -*- coding: utf-8 -*-
"""
CS2 Module — Chimera AI
========================
Независимый модуль для анализа матчей CS2.

Использует:
- PandaScore API (матчи, команды, статистика)
- Veto-логика (симуляция мап-вето BO3)
- MIS (Map Impact Score) для оценки силы на картах
- AI агенты (GPT-4o + Llama) для тактического анализа
- Золотой сигнал (EV > 15%, уверенность > 60%)

Для подключения нового разработчика:
1. Добавь PANDASCORE_API_KEY в .env
2. Редактируй только файлы в этой папке
3. Публичный интерфейс: get_matches(), analyze_match(), format_report()
"""

from .pandascore import get_cs2_matches_pandascore, get_combined_cs2_matches
from .core import calculate_cs2_win_prob, get_golden_signal, format_cs2_full_report
from .agents import run_cs2_analyst_agent, build_cs2_ensemble
from .veto_logic import simulate_bo3_veto, get_map_impact_score, TEAM_MAP_PREFERENCES

__all__ = [
    "get_combined_cs2_matches",
    "calculate_cs2_win_prob",
    "get_golden_signal",
    "format_cs2_full_report",
    "run_cs2_analyst_agent",
    "build_cs2_ensemble",
    "simulate_bo3_veto",
    "get_map_impact_score",
    "TEAM_MAP_PREFERENCES",
]
