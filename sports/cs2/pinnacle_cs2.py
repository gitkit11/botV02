# -*- coding: utf-8 -*-
"""
Pinnacle публичный API — коэффициенты CS2 без ключа.
Pinnacle — острейший букмекер мира, его линии используют как эталон.
"""

import requests
import time
import logging

logger = logging.getLogger(__name__)

BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "X-Device-UUID": "chimera-ai-bot-v1",
}

# Кеш: обновляем не чаще 1 раза в 15 минут
_cache: dict = {}
_cache_time: float = 0
CACHE_TTL = 900  # 15 минут


def _american_to_decimal(american: int) -> float:
    """Конвертация американских коэффициентов в десятичные (европейские)."""
    if american > 0:
        return round(1 + american / 100, 3)
    elif american < 0:
        return round(1 + 100 / abs(american), 3)
    return 0.0


def _get(url: str, params: dict = None) -> list | dict | None:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=12)
        if r.ok:
            return r.json()
        logger.debug(f"[Pinnacle] {url} → {r.status_code}")
    except Exception as e:
        logger.warning(f"[Pinnacle] Ошибка запроса: {e}")
    return None


def get_cs2_leagues() -> list:
    """Возвращает список CS2 лиг из Pinnacle (sport_id=12 = esports)."""
    data = _get(f"{BASE}/sports/12/leagues", {"all": "false"})
    if not data:
        return []
    return [
        l for l in data
        if any(x in l.get("name", "").lower() for x in ["counter", "cs2", "csgo", "cs:go"])
    ]


def get_cs2_odds() -> dict:
    """
    Основная функция. Возвращает словарь:
    {
      "Team1_vs_Team2": {
        "home": "Team1",
        "away": "Team2",
        "odds_home": 1.83,
        "odds_away": 1.99,
        "league": "CS2 - ESL Pro League",
        "start_time": "2026-03-15T18:00:00Z",
        "matchup_id": 12345,
      },
      ...
    }
    Ключ = "home_vs_away" (нижний регистр).
    """
    global _cache, _cache_time
    if _cache and (time.time() - _cache_time) < CACHE_TTL:
        return _cache

    result = {}
    leagues = get_cs2_leagues()

    for league in leagues[:10]:  # топ-10 лиг
        lid = league["id"]
        league_name = league.get("name", "CS2")

        # Матчи лиги
        matchups = _get(f"{BASE}/leagues/{lid}/matchups")
        if not matchups:
            continue

        # Только реальные матчи (2 команды, не special)
        real_matchups = [
            m for m in matchups
            if len(m.get("participants", [])) == 2
            and m.get("type") == "matchup"
            and not m.get("isLive")
        ]
        if not real_matchups:
            continue

        # Коэффициенты для лиги
        markets = _get(f"{BASE}/leagues/{lid}/markets/straight")
        if not markets:
            continue

        # Индекс: matchup_id → moneyline (period=0 = весь матч)
        odds_by_matchup: dict = {}
        for market in markets:
            if market.get("type") != "moneyline":
                continue
            if market.get("isAlternate"):
                continue
            if market.get("period") not in (0, 4):  # 0=весь матч, 4=серия
                continue
            mid = market.get("matchupId")
            if mid in odds_by_matchup:
                # Предпочитаем period=0 (весь матч)
                if market.get("period") == 0:
                    pass
                else:
                    continue
            prices = {p["designation"]: p["price"] for p in market.get("prices", []) if "designation" in p and "price" in p}
            h_price = prices.get("home")
            a_price = prices.get("away")
            if h_price is None or a_price is None:
                continue
            odds_by_matchup[mid] = {
                "odds_home": _american_to_decimal(h_price),
                "odds_away": _american_to_decimal(a_price),
                "period": market.get("period", 0),
            }

        # Собираем итоговый результат
        for m in real_matchups:
            mid = m["id"]
            parts = m.get("participants", [])
            home = parts[0].get("name", "")
            away = parts[1].get("name", "")
            odds = odds_by_matchup.get(mid)
            if not odds or not home or not away:
                continue

            key = f"{home.lower()}_vs_{away.lower()}"
            result[key] = {
                "home": home,
                "away": away,
                "odds_home": odds["odds_home"],
                "odds_away": odds["odds_away"],
                "league": league_name,
                "start_time": m.get("startTime", ""),
                "matchup_id": mid,
            }

    _cache = result
    _cache_time = time.time()
    logger.info(f"[Pinnacle CS2] Загружено {len(result)} матчей с коэффициентами")
    return result


def get_match_odds(home: str, away: str) -> dict | None:
    """
    Ищет коэффициенты для конкретного матча по именам команд.
    Возвращает {"odds_home": 1.83, "odds_away": 1.99} или None.
    """
    all_odds = get_cs2_odds()
    if not all_odds:
        return None

    home_l = home.lower().strip()
    away_l = away.lower().strip()

    # Точное совпадение
    key = f"{home_l}_vs_{away_l}"
    if key in all_odds:
        m = all_odds[key]
        return {"odds_home": m["odds_home"], "odds_away": m["odds_away"]}

    # Обратный порядок
    key2 = f"{away_l}_vs_{home_l}"
    if key2 in all_odds:
        m = all_odds[key2]
        return {"odds_home": m["odds_away"], "odds_away": m["odds_home"]}

    # Нечёткое: ищем по вхождению имён
    for k, m in all_odds.items():
        ph = m["home"].lower()
        pa = m["away"].lower()
        if (home_l in ph or ph in home_l) and (away_l in pa or pa in away_l):
            return {"odds_home": m["odds_home"], "odds_away": m["odds_away"]}
        if (away_l in ph or ph in away_l) and (home_l in pa or pa in home_l):
            return {"odds_home": m["odds_away"], "odds_away": m["odds_home"]}

    return None
