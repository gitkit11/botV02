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
    """Возвращает список активных теннисных турниров из The Odds API (с кешем 1 час)."""
    try:
        from odds_cache import get_sports_list as _get_sports_list
        data = _get_sports_list()
    except ImportError:
        try:
            r = requests.get(f"{BASE_URL}/sports/", params={"apiKey": THE_ODDS_API_KEY}, timeout=10)
            data = r.json()
        except Exception as e:
            logger.warning(f"[Tennis] Ошибка получения видов спорта: {e}")
            return []

    if not isinstance(data, list):
        return []
    return [
        s["key"] for s in data
        if isinstance(s, dict) and "tennis" in s.get("key", "").lower() and s.get("active", False)
    ]


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
            # Используем глобальный кеш если доступен
            try:
                from odds_cache import get_odds as _get_odds
                raw = _get_odds(sk, markets="h2h,totals", ttl=CACHE_TTL)
            except ImportError:
                r = requests.get(
                    f"{BASE_URL}/sports/{sk}/odds/",
                    params={"apiKey": THE_ODDS_API_KEY, "regions": "eu,uk,us,au",
                            "markets": "h2h,totals", "oddsFormat": "decimal"},
                    timeout=10
                )
                raw = r.json()
            if not isinstance(raw, list):
                continue

            SHARP_BOOKS = ["pinnacle", "betfair_ex", "betfair", "matchbook",
                           "betsson", "marathonbet", "nordicbet"]

            def _name_matches(outcome_name: str, player_name: str) -> bool:
                a = outcome_name.lower().strip()
                b = player_name.lower().strip()
                if a == b: return True
                if a in b or b in a: return True
                a_last = a.split()[-1] if a.split() else a
                b_last = b.split()[-1] if b.split() else b
                if len(a_last) >= 4 and a_last == b_last: return True
                return False

            matches = []
            for event in raw:
                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    continue

                home = event.get("home_team", "")
                away = event.get("away_team", "")
                if not home or not away:
                    continue

                # Собираем коэффициенты от всех/шарп букмекеров
                sharp_p1, sharp_p2 = [], []
                all_p1,   all_p2   = [], []
                pinnacle_p1 = pinnacle_p2 = 0.0
                # Тотал геймов: line → (over_odds, under_odds)
                total_lines: dict = {}  # {22.5: {"over": 1.85, "under": 1.85}}

                for bm in bookmakers:
                    bm_key = bm.get("key", "").lower()
                    is_sharp = any(s in bm_key for s in SHARP_BOOKS)
                    for market in bm.get("markets", []):
                        if market.get("key") == "totals":
                            for o in market.get("outcomes", []):
                                pt    = float(o.get("point", 0) or 0)
                                name  = o.get("name", "").lower()
                                price = float(o.get("price", 0) or 0)
                                if pt > 0 and price >= 1.02:
                                    if pt not in total_lines:
                                        total_lines[pt] = {}
                                    side = "over" if "over" in name else "under" if "under" in name else None
                                    if side and side not in total_lines[pt]:
                                        total_lines[pt][side] = price
                            continue
                        if market.get("key") != "h2h":
                            continue
                        outcomes = market.get("outcomes", [])
                        h = a_odds = 0.0
                        for o in outcomes:
                            name  = o.get("name", "")
                            price = float(o.get("price", 0) or 0)
                            if price < 1.02:
                                continue
                            if _name_matches(name, home):
                                h = price
                            elif _name_matches(name, away):
                                a_odds = price
                        # Fallback по позиции
                        if not h and not a_odds and len(outcomes) >= 2:
                            h      = float(outcomes[0].get("price", 0) or 0)
                            a_odds = float(outcomes[1].get("price", 0) or 0)
                        if h >= 1.02 and a_odds >= 1.02:
                            all_p1.append(h); all_p2.append(a_odds)
                            if is_sharp:
                                sharp_p1.append(h); sharp_p2.append(a_odds)
                            if "pinnacle" in bm_key:
                                pinnacle_p1, pinnacle_p2 = h, a_odds
                        break

                if not all_p1:
                    continue

                # Медиана по шарп-букмекерам, иначе по всем
                src1 = sharp_p1 if sharp_p1 else all_p1
                src2 = sharp_p2 if sharp_p2 else all_p2
                src1.sort(); src2.sort()
                mid = len(src1) // 2
                odds_p1 = round(src1[mid], 2)
                odds_p2 = round(src2[mid], 2)

                # No-vig вероятность из Pinnacle (или медианы шарп)
                nv_p1 = nv_p2 = 0.0
                nv_src1 = pinnacle_p1 or odds_p1
                nv_src2 = pinnacle_p2 or odds_p2
                if nv_src1 and nv_src2:
                    imp1 = 1 / nv_src1
                    imp2 = 1 / nv_src2
                    total = imp1 + imp2
                    if total > 0:
                        nv_p1 = round(imp1 / total, 4)
                        nv_p2 = round(imp2 / total, 4)

                # Считаем сколько букмекеров дают h2h на этот матч
                bm_count = len(all_p1)

                # Выбираем лучшую линию тотала геймов (ближайшую к ATP~22.5 / WTA~19.5)
                bm_total_line   = 0.0
                bm_total_over   = 0.0
                bm_total_under  = 0.0
                if total_lines:
                    best_pt = min(total_lines.keys(), key=lambda x: abs(x - 22.5))
                    tl = total_lines[best_pt]
                    if tl.get("over") and tl.get("under"):
                        bm_total_line  = best_pt
                        bm_total_over  = tl["over"]
                        bm_total_under = tl["under"]

                matches.append({
                    "player1":          home,
                    "player2":          away,
                    "odds_p1":          round(odds_p1, 2),
                    "odds_p2":          round(odds_p2, 2),
                    "no_vig_p1":        nv_p1,
                    "no_vig_p2":        nv_p2,
                    "pinnacle_p1":      round(pinnacle_p1, 2) if pinnacle_p1 else 0,
                    "pinnacle_p2":      round(pinnacle_p2, 2) if pinnacle_p2 else 0,
                    "sport_key":        sk,
                    "commence_time":    event.get("commence_time", ""),
                    "event_id":         event.get("id", ""),
                    "bookmakers_count": bm_count,
                    "bm_total_line":    bm_total_line,
                    "bm_total_over":    bm_total_over,
                    "bm_total_under":   bm_total_under,
                })

            _matches_cache[sk] = (time.time(), matches)
            all_matches.extend(matches)
            logger.info(f"[Tennis] {sk}: {len(matches)} матчей")

        except Exception as e:
            logger.warning(f"[Tennis] Ошибка {sk}: {e}")
            continue

    return all_matches
