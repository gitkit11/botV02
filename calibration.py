# -*- coding: utf-8 -*-
"""
calibration.py — Калибровка вероятностей на основе исторических данных Pinnacle
=================================================================================
Применяет поправку: если Pinnacle исторически завышает вероятность 60% → реально 57%,
то мы корректируем нашу вероятность вниз.

Использование:
    from calibration import calibrate_prob
    adjusted = calibrate_prob(0.62)  # → например 0.59

Данные обновляются скриптом: python scripts/build_calibration.py
"""

import os
import json
import time
import logging

logger = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration_table.json"
_CAL_TTL = 3600  # перечитываем файл раз в час

_cal_data: dict = {}
_cal_ts: float = 0.0


def _load_calibration() -> dict:
    global _cal_data, _cal_ts
    if _cal_data and (time.time() - _cal_ts) < _CAL_TTL:
        return _cal_data
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _cal_data = raw.get("table", {})
            _cal_ts = time.time()
            logger.info(f"[Calibration] Загружено {len(_cal_data)} бинов")
    except Exception as e:
        logger.warning(f"[Calibration] Ошибка загрузки: {e}")
    return _cal_data


def calibrate_prob(prob: float) -> float:
    """
    Применяет калибровочную поправку к вероятности.
    Если таблицы нет — возвращает оригинальную вероятность без изменений.

    prob: float от 0.0 до 1.0 (например 0.62)
    returns: скорректированная вероятность
    """
    if not (0.05 < prob < 0.97):
        return prob

    table = _load_calibration()
    if not table:
        return prob  # нет данных — не трогаем

    # Округляем до ближайшего бина (шаг 0.05)
    bin_key = str(round(round(prob * 20) / 20, 2))

    entry = table.get(bin_key)
    if not entry or entry.get("sample_size", 0) < 10:
        return prob  # мало данных в этом бине — не трогаем

    correction = entry.get("correction", 1.0)

    # Ограничиваем поправку — не более ±15% от оригинала
    correction = max(0.85, min(1.15, correction))

    adjusted = prob * correction
    return round(max(0.05, min(0.95, adjusted)), 4)


def calibrate_odds(home_prob: float, draw_prob: float, away_prob: float) -> tuple:
    """
    Калибрует все три вероятности и ренормализует до суммы 1.0.
    Возвращает (home_prob, draw_prob, away_prob).
    """
    h = calibrate_prob(home_prob)
    d = calibrate_prob(draw_prob) if draw_prob > 0 else 0
    a = calibrate_prob(away_prob)

    total = h + d + a
    if total <= 0:
        return home_prob, draw_prob, away_prob

    return round(h / total, 4), round(d / total, 4), round(a / total, 4)


def calibration_info() -> str:
    """Возвращает строку с информацией о таблице калибровки."""
    table = _load_calibration()
    if not table:
        return "Таблица не найдена. Запусти: python scripts/build_calibration.py"

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        built = raw.get("built_at", "?")[:10]
        samples = raw.get("sample_size", "?")
        bins = len(table)
        return f"Калибровка: {bins} бинов, {samples} матчей, обновлено {built}"
    except Exception:
        return f"Калибровка: {len(table)} бинов"
