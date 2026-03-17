# -*- coding: utf-8 -*-
"""
sports/tennis/matches.py — Получение теннисных матчей из The Odds API
"""

import requests
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    from config import THE_ODDS_API_KEY
except ImportError:
    import os
    THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASE_URL = "https://api.the-odds-api.com/v4"

# Кэш матчей: {sport_key: (timestamp, [matches])}
_matches_cache: dict = {}
CACHE_TTL = 1800  # 30 минут


def get_active_tennis_sports() -> List[str]:
    """Возвращает список активных теннисных турниров из The Odds API."""
    try:
        r = requests.get(
            f"{BASE_URL}/sports/",
            params={"apiKey": THE_ODDS_API_KEY},
            timeout=10
        )
        data = r.json()
        return [
            s["key"] for s in data
            if "tennis" in s["key"].lower() and s.get("active", False)
        ]
    except Exception as e:
        logger.warning(f"[Tennis] Ошибка получения видов спорта: {e}")
        return []


def get_tennis_matches(sport_key: str = None) -> List[Dict]:
    """
    Получает теннисные матчи с коэффициентами.
    Если sport_key не указан — сканирует все активные турниры.

    Возвращает список матчей в формате:
    {
      "player1": "Jannik Sinner",
      "player2": "Carlos Alcaraz",
      "odds_p1": 1.65,
      "odds_p2": 2.25,
      "sport_key": "tennis_atp_indian_wells",
      "commence_time": "2025-03-15T18:00:00Z",
    }
    """
    if sport_key:
        sport_keys = [sport_key]
    else:
        sport_keys = get_active_tennis_sports()

    if not sport_keys:
        return []

    all_matches = []

    for sk in sport_keys:
        # Проверяем кэш
        cached = _matches_cache.get(sk)
        if cached and (time.time() - cached[0]) < CACHE_TTL:
            all_matches.extend(cached[1])
            continue

        try:
            r = requests.get(
                f"{BASE_URL}/sports/{sk}/odds/",
                params={
                    "apiKey":     THE_ODDS_API_KEY,
                    "regions":    "eu",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=10
            )
            raw = r.json()
            if not isinstance(raw, list):
                continue

            matches = []
            for event in raw:
                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    continue

                home = event.get("home_team", "")
                away = event.get("away_team", "")
                if not home or not away:
                    continue

                # Берём первого доступного букмекера
                # Матчим по имени — outcomes могут быть не в порядке home/away
                odds_p1 = odds_p2 = 0.0

                def _name_matches(outcome_name: str, player_name: str) -> bool:
                    """Сравнивает имена с учётом разных форматов (точно или частично)."""
                    a = outcome_name.lower().strip()
                    b = player_name.lower().strip()
                    if a == b:
                        return True
                    # Частичное совпадение: одно содержит другое
                    if a in b or b in a:
                        return True
                    # Совпадение по фамилии (последнее слово)
                    a_last = a.split()[-1] if a.split() else a
                    b_last = b.split()[-1] if b.split() else b
                    if len(a_last) >= 4 and a_last == b_last:
                        return True
                    return False

                for bm in bookmakers:
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            for o in outcomes:
                                name = o.get("name", "")
                                price = o.get("price", 0)
                                if _name_matches(name, home):
                                    odds_p1 = price
                                elif _name_matches(name, away):
                                    odds_p2 = price
                            # Fallback по позиции — только если ОБА имени не распознаны
                            if not odds_p1 and not odds_p2 and len(outcomes) >= 2:
                                # Определяем кто фаворит по позиции (The Odds API обычно даёт home первым)
                                odds_p1 = outcomes[0].get("price", 0)
                                odds_p2 = outcomes[1].get("price", 0)
                                logger.debug(
                                    f"[Tennis] Fallback по позиции: {home} @ {odds_p1}, {away} @ {odds_p2}"
                                )
                            break
                    if odds_p1 and odds_p2:
                        break

                if not odds_p1 or not odds_p2:
                    continue

                # Считаем сколько букмекеров дают h2h на этот матч
                bm_count = sum(
                    1 for bm in bookmakers
                    for mkt in bm.get("markets", [])
                    if mkt.get("key") == "h2h"
                )

                matches.append({
                    "player1":        home,
                    "player2":        away,
                    "odds_p1":        round(odds_p1, 2),
                    "odds_p2":        round(odds_p2, 2),
                    "sport_key":      sk,
                    "commence_time":  event.get("commence_time", ""),
                    "event_id":       event.get("id", ""),
                    "bookmakers_count": bm_count,
                })

            _matches_cache[sk] = (time.time(), matches)
            all_matches.extend(matches)
            logger.info(f"[Tennis] {sk}: {len(matches)} матчей")

        except Exception as e:
            logger.warning(f"[Tennis] Ошибка {sk}: {e}")
            continue

    return all_matches
