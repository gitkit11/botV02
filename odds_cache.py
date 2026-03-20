# -*- coding: utf-8 -*-
"""
odds_cache.py — Глобальный кеш The Odds API (файловый + памяти)
================================================================
Все модули используют этот кеш вместо прямых HTTP запросов.

Кеш двухуровневый:
  1. В памяти (_cache) — мгновенный доступ пока бот работает
  2. На диске (odds_cache.json) — переживает перезапуск бота

TTL:
  odds        — 30 минут (коэффициенты на матчи)
  scores      — 15 минут (результаты)
  sports_list — 2 часа   (список активных видов спорта)
"""

import os
import json
import time
import threading
import requests
import logging

logger = logging.getLogger(__name__)

try:
    from config import THE_ODDS_API_KEY as _API_KEY
except ImportError:
    _API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASE_URL = "https://api.the-odds-api.com/v4"

CACHE_TTL       = 1200   # 20 минут (коэффициенты не меняются каждые 10 мин)
SCORES_TTL      = 300    # 5 минут  (результаты обновляются часто)
SPORTS_LIST_TTL = 3600   # 1 час    (список видов спорта меняется редко)

# Регионы — eu+uk достаточно: Pinnacle, Betfair, Betsson и все европейские шарп-буки
# us+au не нужны для европейского спорта и удваивают стоимость запроса
DEFAULT_REGIONS = "eu,uk"
DEFAULT_MARKETS = "h2h,totals,spreads"

# Шарп букмекеры — минимальная маржа, самые точные линии
# Используются для расчёта no-vig вероятностей
SHARP_BOOKS = ["pinnacle", "betfair_ex", "betfair", "matchbook", "smarkets",
               "lowvig", "betsson", "nordicbet"]

CACHE_FILE = "odds_cache.json"

_lock:      threading.Lock = threading.Lock()
_file_lock: threading.Lock = threading.Lock()  # Windows: предотвращает WinError 32
_cache: dict = {}   # key → {"ts": float, "data": any}
_dirty: bool = False  # нужно ли сохранить на диск


# ─── Диск ──────────────────────────────────────────────────────────────────────

def _load_from_disk():
    """Загружает кеш с диска при старте."""
    global _cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Убираем записи старше 24 часов чтобы файл не разбухал
                cutoff = time.time() - 86400
                _cache = {k: v for k, v in data.items()
                          if isinstance(v, dict) and v.get("ts", 0) > cutoff}
                logger.info(f"[OddsCache] Загружено {len(_cache)} записей с диска")
    except Exception as e:
        logger.warning(f"[OddsCache] Ошибка загрузки кеша: {e}")
        _cache = {}


def _save_to_disk():
    """Сохраняет кеш на диск (атомарно через .tmp).
    _file_lock гарантирует что только один поток пишет файл одновременно (Windows).
    """
    global _dirty
    if not _file_lock.acquire(blocking=False):
        # Другой поток уже пишет — пропускаем, данные будут записаны им
        return
    try:
        with _lock:
            data = dict(_cache)
            _dirty = False
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        logger.warning(f"[OddsCache] Ошибка сохранения кеша: {e}")
    finally:
        _file_lock.release()


# Загружаем кеш при импорте модуля
_load_from_disk()


# ─── Внутренний доступ ─────────────────────────────────────────────────────────

def _cache_get(key: str, ttl: int):
    """Возвращает данные из кеша или None если устарели/отсутствуют."""
    with _lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["data"]
    return None


def _cache_set(key: str, data):
    """Сохраняет данные в память и планирует запись на диск."""
    global _dirty
    with _lock:
        _cache[key] = {"ts": time.time(), "data": data}
        _dirty = True
    # Сохраняем на диск в фоновом потоке чтобы не блокировать
    threading.Thread(target=_save_to_disk, daemon=True).start()


def _log_quota(response, sport_key: str = ""):
    """Логирует оставшуюся квоту из заголовков ответа."""
    remaining = response.headers.get("x-requests-remaining", "?")
    used      = response.headers.get("x-requests-used", "?")
    last      = response.headers.get("x-requests-last", "?")
    tag = f" [{sport_key}]" if sport_key else ""
    logger.info(f"[OddsCache]{tag} ЗАПРОС к API: -{last} кредитов | использовано={used} | осталось={remaining}")


# ─── Публичный API ─────────────────────────────────────────────────────────────

def get_odds(sport_key: str, markets: str = DEFAULT_MARKETS,
             regions: str = DEFAULT_REGIONS, ttl: int = CACHE_TTL) -> list:
    """
    Коэффициенты для вида спорта. HTTP запрос только если кеш устарел.
    При ошибке сети — возвращает устаревший кеш (лучше старые данные чем ничего).
    """
    cache_key = f"odds::{sport_key}::{markets}::{regions}"

    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        logger.debug(f"[OddsCache] HIT odds [{sport_key}] ({len(cached)} матчей)")
        return cached

    logger.info(f"[OddsCache] MISS odds [{sport_key}] — делаем HTTP запрос")
    try:
        r = requests.get(
            f"{BASE_URL}/sports/{sport_key}/odds/",
            params={
                "apiKey":     _API_KEY,
                "regions":    regions,
                "markets":    markets,
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        if r.status_code == 401:
            logger.warning(f"[OddsCache] Квота исчерпана (401) для {sport_key}")
            # Возвращаем устаревший кеш если есть
            with _lock:
                entry = _cache.get(cache_key)
                if entry:
                    return entry["data"]
            return []
        if r.status_code == 422:
            return []
        _log_quota(r, sport_key)
        if r.ok:
            data = r.json()
            if isinstance(data, list):
                _cache_set(cache_key, data)
                # Обновляем line_tracker — Pinnacle odds для Варианта A/B
                try:
                    from line_tracker import update_match_odds as _update_lt
                    _sport = sport_key.split("_")[0] if "_" in sport_key else sport_key
                    for _m in data:
                        _mid = _m.get("id")
                        if not _mid:
                            continue
                        for _bk in (_m.get("bookmakers") or []):
                            if _bk.get("key") != "pinnacle":
                                continue
                            _h = _a = _d = None
                            for _mk in (_bk.get("markets") or []):
                                if _mk.get("key") != "h2h":
                                    continue
                                for _oc in (_mk.get("outcomes") or []):
                                    _n = _oc.get("name", "")
                                    _p = _oc.get("price")
                                    if _n == _m.get("home_team"):
                                        _h = _p
                                    elif _n == _m.get("away_team"):
                                        _a = _p
                                    elif _n == "Draw":
                                        _d = _p
                            if _h and _a:
                                _update_lt(_mid, _sport, _m.get("home_team", ""),
                                           _m.get("away_team", ""), _h, _a, _d)
                            break
                except Exception:
                    pass
                return data
    except Exception as e:
        logger.warning(f"[OddsCache] Ошибка get_odds({sport_key}): {e}")
        # При ошибке сети — устаревший кеш
        with _lock:
            entry = _cache.get(cache_key)
            if entry:
                logger.warning(f"[OddsCache] Возвращаем устаревший кеш для {sport_key}")
                return entry["data"]

    return []


def get_scores(sport_key: str, days_from: int = 3,
               ttl: int = SCORES_TTL) -> list:
    """Результаты матчей. HTTP запрос только если кеш устарел."""
    cache_key = f"scores::{sport_key}::{days_from}"

    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        logger.debug(f"[OddsCache] HIT scores [{sport_key}]")
        return cached

    logger.info(f"[OddsCache] MISS scores [{sport_key}] — делаем HTTP запрос")
    try:
        r = requests.get(
            f"{BASE_URL}/sports/{sport_key}/scores/",
            params={
                "apiKey":     _API_KEY,
                "daysFrom":   days_from,
                "dateFormat": "iso",
            },
            timeout=10,
        )
        if r.status_code == 422:
            return []
        _log_quota(r, sport_key)
        if r.ok:
            data = r.json()
            if isinstance(data, list):
                _cache_set(cache_key, data)
                return data
    except Exception as e:
        logger.warning(f"[OddsCache] Ошибка get_scores({sport_key}): {e}")

    return []


def get_sports_list(ttl: int = SPORTS_LIST_TTL) -> list:
    """Список активных видов спорта. HTTP запрос раз в 2 часа."""
    cache_key = "sports_list"

    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        logger.debug(f"[OddsCache] HIT sports_list")
        return cached

    logger.info(f"[OddsCache] MISS sports_list — делаем HTTP запрос")
    try:
        r = requests.get(
            f"{BASE_URL}/sports/",
            params={"apiKey": _API_KEY},
            timeout=10,
        )
        _log_quota(r, "sports_list")
        if r.ok:
            data = r.json()
            if isinstance(data, list):
                _cache_set(cache_key, data)
                return data
    except Exception as e:
        logger.warning(f"[OddsCache] Ошибка get_sports_list: {e}")

    return []


def invalidate(sport_key: str = None):
    """
    Сбрасывает кеш в памяти и на диске.
    sport_key=None — сбрасывает всё.
    """
    global _dirty
    with _lock:
        if sport_key:
            keys_to_del = [k for k in _cache if sport_key in k]
            for k in keys_to_del:
                del _cache[k]
        else:
            _cache.clear()
        _dirty = True
    threading.Thread(target=_save_to_disk, daemon=True).start()


def cache_stats() -> dict:
    """Возвращает статистику кеша: кол-во записей, свежих, размер файла."""
    with _lock:
        now = time.time()
        total = len(_cache)
        fresh_odds   = sum(1 for k, v in _cache.items()
                          if k.startswith("odds::") and (now - v["ts"]) < CACHE_TTL)
        fresh_scores = sum(1 for k, v in _cache.items()
                          if k.startswith("scores::") and (now - v["ts"]) < SCORES_TTL)
    file_size = 0
    try:
        file_size = os.path.getsize(CACHE_FILE) // 1024  # KB
    except Exception:
        pass
    return {
        "total":       total,
        "fresh_odds":  fresh_odds,
        "fresh_scores": fresh_scores,
        "file_kb":     file_size,
    }
