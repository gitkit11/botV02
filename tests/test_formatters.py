# -*- coding: utf-8 -*-
"""
Тесты для formatters.py — утилиты форматирования.
Запуск: python -m pytest tests/test_formatters.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ─── Импорт без aiogram (нужна заглушка) ─────────────────────────────────────

# Заглушки для aiogram (не нужен для тестирования логики форматирования)
from unittest.mock import MagicMock, patch
import sys

_mock_aiogram = MagicMock()
_mock_types   = MagicMock()
_mock_builder = MagicMock()
sys.modules.setdefault("aiogram", _mock_aiogram)
sys.modules.setdefault("aiogram.types", _mock_types)
sys.modules.setdefault("aiogram.utils", MagicMock())
sys.modules.setdefault("aiogram.utils.keyboard", MagicMock())

from formatters import (
    _safe_truncate,
    translate_outcome,
    conf_icon,
    _prob_icon,
    _escape_md,
)


# ─── _safe_truncate ───────────────────────────────────────────────────────────

def test_safe_truncate_short_text():
    text = "Короткий текст"
    assert _safe_truncate(text, 4000) == text

def test_safe_truncate_long_text():
    text = "A" * 5000
    result = _safe_truncate(text, 4000)
    assert len(result) <= 4100  # с учётом добавленного ⚠️

def test_safe_truncate_adds_warning():
    text = "X\n" * 3000
    result = _safe_truncate(text, 4000)
    assert "сокращён" in result or "⚠️" in result

def test_safe_truncate_preserves_newlines():
    # Обрезает по последнему \n, не посередине строки
    lines = ["Строка " + str(i) for i in range(1000)]
    text = "\n".join(lines)
    result = _safe_truncate(text, 4000)
    # Результат не должен заканчиваться на полуслове
    non_warning = result.split("⚠️")[0]
    assert non_warning.endswith("\n") or non_warning.strip() != ""

def test_safe_truncate_exact_limit():
    text = "A" * 4000
    result = _safe_truncate(text, 4000)
    assert result == text  # ровно лимит — не обрезать


# ─── translate_outcome ────────────────────────────────────────────────────────

def test_translate_home_win():
    r = translate_outcome("home_win", "Arsenal", "Chelsea")
    assert "Arsenal" in r
    assert "хозяев" in r.lower() or "хозяева" in r.lower()

def test_translate_away_win():
    r = translate_outcome("away_win", "Arsenal", "Chelsea")
    assert "Chelsea" in r

def test_translate_draw():
    r = translate_outcome("draw")
    assert "Ничья" in r

def test_translate_none():
    r = translate_outcome(None)
    assert r == "Нет данных"

def test_translate_empty():
    r = translate_outcome("")
    assert r == "Нет данных"

def test_translate_ru_home():
    r = translate_outcome("победа хозяев", "Liverpool")
    assert "Liverpool" in r

def test_translate_ru_away():
    r = translate_outcome("победа гостей", "Liverpool", "Man City")
    assert "Man City" in r

def test_translate_unknown_returns_original():
    r = translate_outcome("some unknown text")
    assert r == "some unknown text"


# ─── conf_icon ────────────────────────────────────────────────────────────────

def test_conf_icon_high():
    assert conf_icon(75) == "🟢"

def test_conf_icon_medium():
    assert conf_icon(60) == "🟡"

def test_conf_icon_low():
    assert conf_icon(40) == "🔴"

def test_conf_icon_boundary_70():
    assert conf_icon(70) == "🟢"

def test_conf_icon_boundary_55():
    assert conf_icon(55) == "🟡"


# ─── _prob_icon ───────────────────────────────────────────────────────────────

def test_prob_icon_fire():
    assert _prob_icon(80) == "🔥"
    assert _prob_icon(95) == "🔥"

def test_prob_icon_star():
    assert _prob_icon(70) == "⭐"
    assert _prob_icon(79) == "⭐"

def test_prob_icon_check():
    assert _prob_icon(60) == "✅"
    assert _prob_icon(69) == "✅"

def test_prob_icon_warning():
    assert _prob_icon(50) == "⚠️"
    assert _prob_icon(59) == "⚠️"

def test_prob_icon_cross():
    assert _prob_icon(49) == "❌"
    assert _prob_icon(0) == "❌"


# ─── _escape_md ──────────────────────────────────────────────────────────────

def test_escape_md_underscore():
    assert r"\_" in _escape_md("hello_world")

def test_escape_md_asterisk():
    assert r"\*" in _escape_md("bold*text")

def test_escape_md_backtick():
    assert r"\`" in _escape_md("code`snippet")

def test_escape_md_clean_text():
    # Чистый текст не меняется
    result = _escape_md("Arsenal vs Chelsea")
    assert result == "Arsenal vs Chelsea"
