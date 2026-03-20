# -*- coding: utf-8 -*-
"""
Файловый кэш формы теннисистов.
Обновляется фоновой задачей каждые 2 часа.
Chimera-скан читает из кэша мгновенно — без HTTP запросов.
"""

import json
import os
import logging
import time

logger = logging.getLogger(__name__)

CACHE_FILE = "tennis_form_cache.json"
FORM_TTL   = 7200   # 2 часа
H2H_TTL    = 86400  # 24 часа


def _load() -> dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"form": {}, "h2h": {}}


def _save(data: dict):
    try:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        logger.warning(f"[TennisCache] Ошибка сохранения: {e}")


def get_cached_form(player: str) -> dict:
    """Возвращает форму игрока из кэша или {} если устарело."""
    data = _load()
    entry = data.get("form", {}).get(player)
    if entry and time.time() - entry.get("ts", 0) < FORM_TTL:
        return entry
    return {}


def get_cached_h2h(p1: str, p2: str) -> dict:
    """Возвращает H2H из кэша или {} если устарело."""
    key = "___".join(sorted([p1, p2]))
    data = _load()
    entry = data.get("h2h", {}).get(key)
    if entry and time.time() - entry.get("ts", 0) < H2H_TTL:
        return entry
    return {}


def prefetch_tennis_forms(matches: list):
    """
    Основная функция фонового обновления.
    Принимает список матчей [{player1, player2, ...}].
    Вызывает api-tennis.com для каждого игрока/пары и сохраняет в файл.
    """
    try:
        from sports.tennis.api_tennis import get_player_form, get_h2h_by_name
    except ImportError:
        logger.warning("[TennisCache] api_tennis не доступен")
        return

    data = _load()
    form_cache = data.get("form", {})
    h2h_cache  = data.get("h2h", {})
    now = time.time()

    players_done = set()
    h2h_done = set()
    updated = 0

    for m in matches:
        p1 = m.get("player1", "")
        p2 = m.get("player2", "")

        # Форма P1
        if p1 and p1 not in players_done:
            existing = form_cache.get(p1, {})
            if now - existing.get("ts", 0) > FORM_TTL:
                try:
                    result = get_player_form(p1)
                    if result:
                        result["ts"] = now
                        form_cache[p1] = result
                        updated += 1
                except Exception:
                    pass
            players_done.add(p1)

        # Форма P2
        if p2 and p2 not in players_done:
            existing = form_cache.get(p2, {})
            if now - existing.get("ts", 0) > FORM_TTL:
                try:
                    result = get_player_form(p2)
                    if result:
                        result["ts"] = now
                        form_cache[p2] = result
                        updated += 1
                except Exception:
                    pass
            players_done.add(p2)

        # H2H
        if p1 and p2:
            key = "___".join(sorted([p1, p2]))
            if key not in h2h_done:
                existing = h2h_cache.get(key, {})
                if now - existing.get("ts", 0) > H2H_TTL:
                    try:
                        result = get_h2h_by_name(p1, p2)
                        if result:
                            result["ts"] = now
                            h2h_cache[key] = result
                            updated += 1
                    except Exception:
                        pass
                h2h_done.add(key)

    data["form"] = form_cache
    data["h2h"]  = h2h_cache
    _save(data)
    logger.info(f"[TennisCache] Обновлено {updated} записей для {len(players_done)} игроков")
