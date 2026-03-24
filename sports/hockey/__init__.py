# -*- coding: utf-8 -*-
"""sports/hockey — публичный API хоккейного модуля."""
from .core import (
    calculate_hockey_win_prob,
    get_hockey_matches,
    get_hockey_odds,
    format_hockey_report,
    HOCKEY_LEAGUES,
)

__all__ = [
    "calculate_hockey_win_prob",
    "get_hockey_matches",
    "get_hockey_odds",
    "format_hockey_report",
    "HOCKEY_LEAGUES",
]
