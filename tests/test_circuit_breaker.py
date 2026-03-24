# -*- coding: utf-8 -*-
"""
Тесты для circuit_breaker.py.
Запуск: python -m pytest tests/test_circuit_breaker.py -v
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from circuit_breaker import CircuitBreaker, get_breaker, all_statuses


# ─── Базовые состояния ────────────────────────────────────────────────────────

def test_initial_state_closed():
    cb = CircuitBreaker("test_init")
    assert cb.status == "closed"
    assert not cb.is_open()

def test_single_failure_stays_closed():
    cb = CircuitBreaker("test_single", max_failures=3)
    cb.record_failure()
    assert cb.status == "closed"

def test_opens_after_max_failures():
    cb = CircuitBreaker("test_open", max_failures=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.status == "open"
    assert cb.is_open()

def test_success_resets_failures():
    cb = CircuitBreaker("test_reset", max_failures=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()  # счётчик сброшен — ещё не открыт
    assert cb.status == "closed"


# ─── Переходы состояний ───────────────────────────────────────────────────────

def test_closed_to_open():
    cb = CircuitBreaker("test_c2o", max_failures=2)
    cb.record_failure()
    cb.record_failure()
    assert cb.status == "open"

def test_open_to_halfopen_after_timeout():
    cb = CircuitBreaker("test_o2h", max_failures=1, recovery_timeout=0)
    cb.record_failure()
    assert cb.status == "open"
    # После нулевого таймаута → half-open
    result = cb.is_open()
    assert result == False  # half-open не блокирует
    assert cb.status == "half-open"

def test_halfopen_success_closes():
    cb = CircuitBreaker("test_h2c", max_failures=1, recovery_timeout=0)
    cb.record_failure()
    cb.is_open()  # → half-open
    cb.record_success()
    assert cb.status == "closed"

def test_halfopen_failure_reopens():
    cb = CircuitBreaker("test_h2o", max_failures=1, recovery_timeout=0)
    cb.record_failure()
    cb.is_open()  # → half-open
    cb.record_failure()
    assert cb.status == "open"


# ─── Таймер восстановления ────────────────────────────────────────────────────

def test_open_blocks_before_timeout():
    cb = CircuitBreaker("test_block", max_failures=1, recovery_timeout=999)
    cb.record_failure()
    assert cb.is_open() == True

def test_time_until_retry_when_open():
    cb = CircuitBreaker("test_timer", max_failures=1, recovery_timeout=300)
    cb.record_failure()
    remaining = cb.time_until_retry()
    assert 0 < remaining <= 300

def test_time_until_retry_when_closed():
    cb = CircuitBreaker("test_timer_closed", max_failures=3)
    assert cb.time_until_retry() == 0


# ─── Эмодзи статуса ──────────────────────────────────────────────────────────

def test_emoji_closed():
    cb = CircuitBreaker("test_emoji_c", max_failures=3)
    assert cb.status_emoji() == "🟢"

def test_emoji_open():
    cb = CircuitBreaker("test_emoji_o", max_failures=1)
    cb.record_failure()
    assert cb.status_emoji() == "🔴"

def test_emoji_halfopen():
    cb = CircuitBreaker("test_emoji_h", max_failures=1, recovery_timeout=0)
    cb.record_failure()
    cb.is_open()  # → half-open
    assert cb.status_emoji() == "🟡"


# ─── get_breaker + all_statuses ──────────────────────────────────────────────

def test_get_breaker_singleton():
    b1 = get_breaker("singleton_test")
    b2 = get_breaker("singleton_test")
    assert b1 is b2

def test_all_statuses_returns_dict():
    get_breaker("status_test_a")
    get_breaker("status_test_b")
    statuses = all_statuses()
    assert isinstance(statuses, dict)
    assert "status_test_a" in statuses
    assert "status_test_b" in statuses

def test_all_statuses_structure():
    get_breaker("struct_test")
    statuses = all_statuses()
    for name, (status, emoji) in statuses.items():
        assert status in ("closed", "open", "half-open")
        assert emoji in ("🟢", "🟡", "🔴")
