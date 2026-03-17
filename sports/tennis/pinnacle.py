# -*- coding: utf-8 -*-
"""
sports/tennis/pinnacle.py — Коэффициенты на теннис с Pinnacle (guest API, без ключа).

Эндпоинты:
  GET /sports/33/matchups?withSpecials=false  — список матчей
  GET /sports/33/markets/straight             — все котировки одним запросом

Конвертация American odds → Decimal:
  price > 0  →  (price / 100) + 1
  price < 0  →  (100 / abs(price)) + 1
"""

import requests
import time
import logging

logger = logging.getLogger(__name__)

BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://www.pinnacle.com",
    "Referer": "https://www.pinnacle.com/",
}
CACHE_TTL = 1800  # 30 минут
_cache: dict = {}   # {"matchups": (ts, data), "markets": (ts, data)}


def _american_to_decimal(price: int) -> float:
    """Конвертирует American odds в Decimal."""
    if price > 0:
        return round(price / 100 + 1, 3)
    elif price < 0:
        return round(100 / abs(price) + 1, 3)
    return 0.0


def _get(path: str) -> list:
    """GET-запрос к Pinnacle guest API."""
    try:
        r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=12)
        if r.ok:
            return r.json()
        logger.warning(f"[Pinnacle] {path}: HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"[Pinnacle] {path}: {e}")
    return []


def _cached(key: str, path: str) -> list:
    cached = _cache.get(key)
    if cached and (time.time() - cached[0]) < CACHE_TTL:
        return cached[1]
    data = _get(path)
    if data:
        _cache[key] = (time.time(), data)
    return data


def get_tennis_odds() -> list:
    """
    Возвращает список теннисных матчей с коэффициентами Pinnacle.

    Формат каждого матча совместим с форматом sports/tennis/matches.py:
    {
        "player1":     "Jannik Sinner",
        "player2":     "Carlos Alcaraz",
        "odds_p1":     1.65,
        "odds_p2":     2.25,
        "sport_key":   "tennis_pinnacle",
        "tournament":  "ATP Indian Wells",
        "commence_time": "2025-03-15T18:00:00Z",
        "source":      "pinnacle",
    }
    """
    matchups = _cached("matchups", "/sports/33/matchups?withSpecials=false")
    markets  = _cached("markets",  "/sports/33/markets/straight")

    if not matchups or not markets:
        return []

    # Индекс котировок: {matchupId: {home_price, away_price}}
    odds_index: dict = {}
    for m in markets:
        if m.get("type") != "moneyline":
            continue
        if m.get("period") != 0:
            continue
        if m.get("status") != "open":
            continue
        mid = m.get("matchupId")
        if not mid:
            continue
        prices = {p["designation"]: p["price"] for p in m.get("prices", [])}
        home_price = prices.get("home")
        away_price = prices.get("away")
        if home_price is not None and away_price is not None:
            odds_index[mid] = {
                "home": _american_to_decimal(home_price),
                "away": _american_to_decimal(away_price),
            }

    results = []
    for mu in matchups:
        if mu.get("status") != "pending":
            continue
        mid = mu.get("id")
        if not mid or mid not in odds_index:
            continue

        participants = mu.get("participants", [])
        home_p = next((p["name"] for p in participants if p.get("alignment") == "home"), "")
        away_p = next((p["name"] for p in participants if p.get("alignment") == "away"), "")
        if not home_p or not away_p:
            continue

        league = mu.get("league", {})
        tournament = league.get("name", "")
        odds = odds_index[mid]

        results.append({
            "player1":      home_p,
            "player2":      away_p,
            "odds_p1":      odds["home"],
            "odds_p2":      odds["away"],
            "sport_key":    "tennis_pinnacle",
            "tournament":   tournament,
            "commence_time": mu.get("startTime", ""),
            "source":       "pinnacle",
            "tour":         _detect_tour(tournament),
            "surface":      _detect_surface(tournament),
        })

    logger.info(f"[Pinnacle] Теннис: {len(results)} матчей с кэфами")
    return results


def _detect_tour(name: str) -> str:
    n = name.lower()
    if "wta" in n or "women" in n:
        return "wta"
    if "itf" in n:
        return "itf"
    return "atp"


def _detect_surface(name: str) -> str:
    n = name.lower()
    if "clay" in n or "roland" in n or "monte" in n:
        return "clay"
    if "grass" in n or "wimbledon" in n or "queens" in n:
        return "grass"
    return "hard"
