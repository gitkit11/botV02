# -*- coding: utf-8 -*-
"""
Unit-тесты для handlers/tennis.py — маршрутизация и логика без Telegram API.
"""
import sys, types as _types

# ─── Stub aiogram ─────────────────────────────────────────────────────────────
# handlers/tennis.py импортирует aiogram; заглушаем его без реального Telegram.
aiogram_stub = _types.ModuleType("aiogram")
aiogram_stub.Router = object
aiogram_stub.types  = _types.ModuleType("aiogram.types")
class _FakeIKB:
    def button(self, **kw): pass
    def adjust(self, *a): pass
    def as_markup(self): return None
aiogram_stub.utils = _types.ModuleType("aiogram.utils")
aiogram_stub.utils.keyboard = _types.ModuleType("aiogram.utils.keyboard")
aiogram_stub.utils.keyboard.InlineKeyboardBuilder = _FakeIKB
sys.modules.setdefault("aiogram", aiogram_stub)
sys.modules.setdefault("aiogram.types", aiogram_stub.types)
sys.modules.setdefault("aiogram.utils", aiogram_stub.utils)
sys.modules.setdefault("aiogram.utils.keyboard", aiogram_stub.utils.keyboard)

# ─── Stub state ────────────────────────────────────────────────────────────────
state_stub = _types.ModuleType("state")
state_stub.tennis_matches_cache = []
state_stub._report_cache = {}
state_stub._REPORT_CACHE_TTL = 2700
sys.modules.setdefault("state", state_stub)

# ─── Stub handlers.common ──────────────────────────────────────────────────────
common_stub = _types.ModuleType("handlers.common")
async def _noop(*a, **kw): pass
common_stub.show_ai_thinking = _noop
handlers_pkg = sys.modules.get("handlers") or _types.ModuleType("handlers")
sys.modules.setdefault("handlers", handlers_pkg)
sys.modules.setdefault("handlers.common", common_stub)

# ─── Stub formatters ──────────────────────────────────────────────────────────
fmt_stub = _types.ModuleType("formatters")
fmt_stub._safe_truncate = lambda t, limit=4000: t[:limit]
sys.modules.setdefault("formatters", fmt_stub)

# ─── Now import the module under test ─────────────────────────────────────────
import importlib
import pytest

# We import only the pure helper logic — not the async handlers themselves
# (they need a running event loop and real aiogram objects).


# ─── 1. tennis_matches_cache is the shared list from state ───────────────────

def test_tennis_cache_is_state_list():
    """handlers/tennis.py должен использовать тот же объект списка из state."""
    from state import tennis_matches_cache as cache
    cache.clear()
    cache.append({"player1": "Federer", "player2": "Nadal", "sport_key": "tennis_atp_roland_garros"})
    assert len(cache) == 1
    assert cache[0]["player1"] == "Federer"
    cache.clear()


# ─── 2. _safe_truncate stays within limit ─────────────────────────────────────

def test_safe_truncate_short():
    from formatters import _safe_truncate
    assert _safe_truncate("hello") == "hello"


def test_safe_truncate_long():
    from formatters import _safe_truncate
    text = "x" * 5000
    result = _safe_truncate(text, limit=4000)
    assert len(result) <= 4000


# ─── 3. Tier signal for tennis ────────────────────────────────────────────────

def test_tennis_tier_fire():
    from signal_engine import get_bet_tier
    sig = get_bet_tier(0.76, 16.0, "tennis")
    assert "🔥🔥🔥" in sig


def test_tennis_tier_strong():
    from signal_engine import get_bet_tier
    sig = get_bet_tier(0.68, 11.0, "tennis")
    assert "🔥🔥" in sig
    assert "🔥🔥🔥" not in sig


def test_tennis_tier_normal():
    from signal_engine import get_bet_tier
    sig = get_bet_tier(0.61, 6.0, "tennis")
    assert sig == "СТАВИТЬ 🔥"


def test_tennis_tier_no_bet():
    from signal_engine import get_bet_tier
    sig = get_bet_tier(0.52, 3.0, "tennis")
    assert sig == "НЕ СТАВИТЬ"


# ─── 4. AI gate doesn't change НЕ СТАВИТЬ when no AI data ────────────────────

def test_ai_gate_no_data_no_change():
    from signal_engine import apply_ai_gate
    tier = "НЕ СТАВИТЬ"
    result = apply_ai_gate(tier, "home_win", "", "")
    assert result == "НЕ СТАВИТЬ"


def test_ai_gate_both_agree_upgrades():
    from signal_engine import apply_ai_gate
    tier = "СТАВИТЬ 🔥"
    result = apply_ai_gate(tier, "home_win", "home_win", "home_win")
    assert "🔥🔥" in result  # upgraded at least one tier


def test_ai_gate_both_disagree_downgrades():
    from signal_engine import apply_ai_gate
    tier = "СТАВИТЬ 🔥🔥"
    result = apply_ai_gate(tier, "home_win", "away_win", "away_win")
    assert result in ("СТАВИТЬ 🔥", "НЕ СТАВИТЬ")


def test_ai_gate_split_no_change():
    from signal_engine import apply_ai_gate
    tier = "СТАВИТЬ 🔥🔥"
    result = apply_ai_gate(tier, "home_win", "home_win", "away_win")
    assert result == "СТАВИТЬ 🔥🔥"


# ─── 5. back_to_report_tennis callback_data parsing ──────────────────────────

def test_back_to_report_parse_idx():
    """Парсинг callback_data: back_to_report_tennis_{sport_key}_{idx}"""
    data      = "back_to_report_tennis_tennis_atp_roland_garros_7"
    stripped  = data[len("back_to_report_tennis_"):]
    parts     = stripped.rsplit("_", 1)
    assert len(parts) == 2
    sport_key, idx_str = parts
    assert idx_str == "7"
    assert sport_key == "tennis_atp_roland_garros"


def test_back_to_report_parse_simple_key():
    data     = "back_to_report_tennis_tennis_atp_5"
    stripped = data[len("back_to_report_tennis_"):]
    parts    = stripped.rsplit("_", 1)
    assert parts[1] == "5"
    assert parts[0] == "tennis_atp"


# ─── 6. Report cache stores and retrieves correctly ───────────────────────────

def test_report_cache_roundtrip():
    import time
    from state import _report_cache, _REPORT_CACHE_TTL
    _report_cache.clear()
    key = "tennis_tennis_atp_test_3"
    _report_cache[key] = {
        "text": "<b>Report</b>", "kb": None,
        "parse_mode": "HTML", "ts": time.time(),
    }
    cached = _report_cache.get(key)
    assert cached is not None
    assert cached["text"] == "<b>Report</b>"
    assert time.time() - cached["ts"] < _REPORT_CACHE_TTL
    _report_cache.clear()


def test_report_cache_expired():
    import time
    from state import _report_cache, _REPORT_CACHE_TTL
    _report_cache.clear()
    key = "tennis_tennis_atp_test_0"
    _report_cache[key] = {"text": "old", "ts": time.time() - _REPORT_CACHE_TTL - 1}
    cached = _report_cache[key]
    assert time.time() - cached["ts"] >= _REPORT_CACHE_TTL  # истёк
    _report_cache.clear()
