# -*- coding: utf-8 -*-
"""
circuit_breaker.py — Простой circuit breaker для внешних API.

После N ошибок подряд блокирует вызовы на RECOVERY_TIMEOUT секунд.
Состояния: closed (норма) → open (заблокирован) → half-open (проверка).
"""
import time
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, name: str, max_failures: int = 3, recovery_timeout: int = 300):
        self.name = name
        self.max_failures = max_failures
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._open_since: float = 0.0
        self._state = "closed"  # closed | open | half-open

    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._open_since >= self.recovery_timeout:
                self._state = "half-open"
                logger.info(f"[CB:{self.name}] → half-open, проверяем")
                return False
            return True
        return False

    def record_success(self):
        self._failures = 0
        if self._state in ("half-open", "open"):
            self._state = "closed"
            logger.info(f"[CB:{self.name}] → closed (восстановлен)")

    def record_failure(self):
        self._failures += 1
        if self._state == "half-open":
            self._state = "open"
            self._open_since = time.time()
            logger.warning(f"[CB:{self.name}] → open снова (half-open неудача)")
        elif self._failures >= self.max_failures and self._state != "open":
            self._state = "open"
            self._open_since = time.time()
            logger.warning(
                f"[CB:{self.name}] ОТКРЫТ — {self._failures} ошибок подряд. "
                f"Пауза {self.recovery_timeout}s"
            )

    @property
    def status(self) -> str:
        return self._state

    def status_emoji(self) -> str:
        return {"closed": "🟢", "half-open": "🟡", "open": "🔴"}.get(self._state, "❓")

    def time_until_retry(self) -> int:
        """Секунд до следующей попытки (0 если closed/half-open)."""
        if self._state == "open":
            remaining = self.recovery_timeout - (time.time() - self._open_since)
            return max(0, int(remaining))
        return 0


_breakers: dict = {}


def get_breaker(name: str, max_failures: int = 3, recovery_timeout: int = 300) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, max_failures, recovery_timeout)
    return _breakers[name]


def all_statuses() -> dict:
    """Возвращает {name: (status_str, emoji)} для всех зарегистрированных breaker-ов."""
    return {n: (b.status, b.status_emoji()) for n, b in _breakers.items()}
