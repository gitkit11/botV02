# -*- coding: utf-8 -*-
"""
sports/tennis/api_tennis.py — Интеграция с api-tennis.com
=========================================================
Даёт:
  - Расписание матчей на 7 дней вперёд (ATP + WTA + Challenger)
  - Рейтинги ATP/WTA в реальном времени
  - H2H статистику игроков
  - Статистику сезона по игрокам

Base URL: https://api.api-tennis.com/tennis/
"""

import requests
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from config import API_TENNIS_KEY
except ImportError:
    import os
    API_TENNIS_KEY = os.getenv("API_TENNIS_KEY", "")

BASE_URL = "https://api.api-tennis.com/tennis/"
CACHE_TTL = 1800  # 30 минут

_cache: dict = {}


def _get(method: str, params: dict) -> list | None:
    """Базовый запрос к api-tennis.com. Возвращает result[] или None."""
    if not API_TENNIS_KEY:
        logger.warning("[API-Tennis] API ключ не задан в .env (API_TENNIS_KEY)")
        return None

    cache_key = f"{method}_{str(sorted(params.items()))}"
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL:
        return cached[1]

    try:
        r = requests.get(
            BASE_URL,
            params={"method": method, "APIkey": API_TENNIS_KEY, **params},
            timeout=15,
        )
        data = r.json()

        # API может возвращать {"error": "0", ...} или {"success": 1, ...}
        is_ok = (data.get("error") in ("0", 0)) or (data.get("success") in (1, "1"))
        if not is_ok:
            msg = data.get("result", [{}])
            if isinstance(msg, list) and msg:
                logger.debug(f"[API-Tennis] {method}: {msg[0].get('msg', data)}")
            return None

        result = data.get("result", [])
        _cache[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        logger.error(f"[API-Tennis] {method} запрос упал: {e}")
        return None


# ─── Матчи / расписание ──────────────────────────────────────────────────────

def get_fixtures(days_ahead: int = 7) -> list:
    """
    Возвращает все матчи на ближайшие N дней (ATP + WTA + Challenger).
    Каждый матч:
    {
      "event_key": "12345",
      "event_date": "2025-03-16",
      "event_time": "14:00",
      "event_first_player":      "Jannik Sinner",
      "event_first_player_key":  "1234",
      "event_second_player":     "Daniil Medvedev",
      "event_second_player_key": "5678",
      "event_status":            "Not Started",
      "tournament_name":         "Indian Wells Masters",
      "tournament_type":         "ATP Masters 1000",
      "league_name":             "ATP",
      "event_final_result":      "-",
    }
    """
    date_start = datetime.now().strftime("%Y-%m-%d")
    date_stop  = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    result = _get("get_fixtures", {
        "date_start": date_start,
        "date_stop":  date_stop,
    })
    return result or []


def get_live_matches() -> list:
    """Возвращает матчи которые идут прямо сейчас."""
    result = _get("get_events", {"live": "1"})
    return result or []


# ─── Рейтинги ────────────────────────────────────────────────────────────────

def get_atp_rankings(limit: int = 100) -> list:
    """
    ATP рейтинг. Каждая запись:
    {"player_name": "Jannik Sinner", "player_key": "1234", "ranking": 1, "ranking_points": 11830}
    """
    result = _get("get_rankings", {"ranking_type": "ATP"})
    return (result or [])[:limit]


def get_wta_rankings(limit: int = 100) -> list:
    """WTA рейтинг."""
    result = _get("get_rankings", {"ranking_type": "WTA"})
    return (result or [])[:limit]


def get_live_rankings() -> dict:
    """
    Возвращает словарь {player_name: rank} для ATP и WTA.
    Кэшируется на 30 минут.
    """
    rankings = {}
    for fetch_fn, prefix in [(get_atp_rankings, ""), (get_wta_rankings, "")]:
        data = fetch_fn(100)
        for entry in data:
            name = entry.get("player_name", "")
            rank = entry.get("ranking")
            if name and rank:
                try:
                    rankings[name] = int(rank)
                except (ValueError, TypeError):
                    pass
    return rankings


# ─── Форма игрока ────────────────────────────────────────────────────────────

_form_cache: dict = {}
_FORM_TTL = 3600  # 1 час

def get_player_form(player_name: str, days: int = 21, last_n: int = 5) -> dict:
    """
    Возвращает форму игрока за последние N дней.
    {
      "form": "WWLWW",        # последние last_n результатов (W/L)
      "win_rate": 0.80,       # процент побед
      "matches": 5,           # количество матчей
      "surface_form": {...}   # форма по покрытиям {"hard": "WWL", "clay": "W"}
    }
    """
    cache_key = f"form_{player_name}_{days}"
    cached = _form_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _FORM_TTL:
        return cached[1]

    result = {"form": "", "win_rate": 0.5, "matches": 0, "surface_form": {}}

    if not API_TENNIS_KEY:
        return result

    try:
        today     = datetime.now()
        date_from = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to   = today.strftime("%Y-%m-%d")

        r = requests.get(
            BASE_URL,
            params={
                "method":     "get_fixtures",
                "APIkey":     API_TENNIS_KEY,
                "date_start": date_from,
                "date_stop":  date_to,
            },
            timeout=20,
        )
        data = r.json()
        is_ok = (data.get("error") in ("0", 0)) or (data.get("success") in (1, "1"))
        if not is_ok:
            return result

        matches = data.get("result", [])
        # API хранит имена как "D. Vekic" — матчим по фамилии
        name_parts = player_name.lower().split()
        last_name  = name_parts[-1] if name_parts else player_name.lower()

        player_results = []
        for m in matches:
            status = m.get("event_status", "").lower()
            if status != "finished":
                continue
            # Только одиночные
            etype = m.get("event_type_type", "").lower()
            if "double" in etype or "mixed" in etype:
                continue

            p1     = m.get("event_first_player",  "").lower()
            p2     = m.get("event_second_player", "").lower()
            winner = m.get("event_winner", "")  # "First Player" или "Second Player"

            is_p1 = last_name in p1
            is_p2 = last_name in p2
            if not is_p1 and not is_p2:
                continue

            won = (is_p1 and winner == "First Player") or (is_p2 and winner == "Second Player")
            surf = _detect_surface_from_name(m.get("tournament_name", ""))
            date = m.get("event_date", "")

            player_results.append({"won": won, "surface": surf, "date": date})

        # Сортируем по дате (новые первыми)
        player_results.sort(key=lambda x: x["date"], reverse=True)
        recent = player_results[:last_n]

        if recent:
            form_str  = "".join("W" if r["won"] else "L" for r in recent)
            wins      = sum(1 for r in recent if r["won"])
            win_rate  = wins / len(recent)

            # Форма по покрытию
            surf_form: dict = {}
            for r in player_results[:10]:
                s = r["surface"]
                surf_form.setdefault(s, []).append("W" if r["won"] else "L")
            surf_form = {s: "".join(v) for s, v in surf_form.items()}

            result = {
                "form":         form_str,
                "win_rate":     round(win_rate, 3),
                "matches":      len(recent),
                "surface_form": surf_form,
            }

    except Exception as e:
        logger.warning(f"[API-Tennis] get_player_form({player_name}): {e}")

    _form_cache[cache_key] = (time.time(), result)
    return result


def get_h2h_by_name(player1: str, player2: str) -> dict:
    """
    H2H по именам (без необходимости знать ключи заранее).
    Возвращает {"p1_wins": N, "p2_wins": N, "total": N, "matches": [...]}
    """
    cache_key = f"h2h_{player1}_{player2}"
    cached = _form_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _FORM_TTL:
        return cached[1]

    result = {"p1_wins": 0, "p2_wins": 0, "total": 0, "matches": []}
    try:
        k1 = find_player_key(player1)
        k2 = find_player_key(player2)
        if k1 and k2:
            h2h = get_h2h(k1, k2)
            if h2h:
                result = h2h
    except Exception as e:
        logger.warning(f"[API-Tennis] get_h2h_by_name: {e}")

    _form_cache[cache_key] = (time.time(), result)
    return result


# ─── H2H ────────────────────────────────────────────────────────────────────

def get_h2h(player1_key: str, player2_key: str) -> dict | None:
    """
    H2H между двумя игроками по их API ключам.
    Возвращает {"p1_wins": 4, "p2_wins": 3, "total": 7, "matches": [...]}
    """
    result = _get("get_H2H", {"p1_id": player1_key, "p2_id": player2_key})
    if not result:
        return None

    p1_wins = p2_wins = 0
    matches = []
    for match in result:
        winner = match.get("event_winner", "")
        first  = match.get("event_first_player", "")
        # winner = "First Player" или "Second Player"
        if winner == "First Player":
            p1_wins += 1
        elif winner == "Second Player":
            p2_wins += 1
        matches.append({
            "date":    match.get("event_date", ""),
            "tournament": match.get("tournament_name", ""),
            "score":   match.get("event_final_result", ""),
            "winner":  first if winner == "First Player" else match.get("event_second_player", ""),
        })

    return {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "total":   p1_wins + p2_wins,
        "matches": matches[:5],
    }


def find_player_key(player_name: str) -> str | None:
    """Ищет ключ игрока по имени."""
    if not player_name or len(player_name.strip()) < 3:
        return None
    result = _get("get_players", {"player_name": player_name})
    if not result:
        return None
    # Ищем точное или частичное совпадение
    name_lower = player_name.lower()
    for p in result:
        pname = p.get("player_name", "").lower()
        if pname == name_lower or name_lower in pname or pname in name_lower:
            return str(p.get("player_key", ""))
    return None


# ─── Форматирование матчей ────────────────────────────────────────────────────

# Типы турниров для фильтрации (приоритет)
PRIORITY_TOURS = [
    "grand slam", "masters 1000", "atp 1000",
    "atp 500", "atp 250", "wta 1000", "wta 500", "wta 250",
    "premier", "international",
]

def get_formatted_matches(days_ahead: int = 7) -> list:
    """
    Возвращает отформатированные матчи готовые для CHIMERA Engine.
    Формат совместим с sports.tennis.matches (player1/player2/odds_p1/odds_p2/sport_key).
    NOTE: api-tennis.com не даёт коэффициенты — они берутся из The Odds API.
          Этот метод даёт расписание для показа всех предстоящих матчей.
    """
    raw = get_fixtures(days_ahead)
    matches = []

    for m in raw:
        p1     = m.get("event_first_player", "").strip()
        p2     = m.get("event_second_player", "").strip()
        status = m.get("event_status", "")
        tour   = m.get("tournament_name", "")
        league = m.get("league_name", m.get("tournament_type", "ATP"))
        date   = m.get("event_date", "")
        time_s = m.get("event_time", "")

        if not p1 or not p2:
            continue
        if status.lower() in ("finished", "cancelled", "postponed"):
            continue

        # Определяем поверхность из названия турнира
        surface = _detect_surface_from_name(tour)
        # Определяем тур (ATP/WTA/etc)
        tour_type = "wta" if "wta" in league.lower() or "wta" in tour.lower() else "atp"
        # Строим sport_key совместимый с нашей системой
        tour_slug = tour.lower().replace(" ", "_").replace("-", "_")[:30]
        sport_key = f"tennis_{tour_type}_{tour_slug}"

        matches.append({
            "player1":       p1,
            "player2":       p2,
            "sport_key":     sport_key,
            "tournament":    tour,
            "league":        league,
            "surface":       surface,
            "tour":          tour_type,
            "commence_time": f"{date}T{time_s}:00Z" if date else "",
            "event_id":      str(m.get("event_key", "")),
            "event_status":  status,
            "odds_p1":       0.0,   # заполняется из The Odds API если есть
            "odds_p2":       0.0,
        })

    return matches


def _detect_surface_from_name(tournament_name: str) -> str:
    """Определяет покрытие по названию турнира."""
    name = tournament_name.lower()
    clay_keywords   = ["clay", "roland garros", "french open", "madrid", "rome",
                       "barcelona", "monte carlo", "monte-carlo", "hamburg",
                       "munich", "estoril", "lyon", "geneva", "marrakech"]
    grass_keywords  = ["wimbledon", "grass", "queens", "halle", "eastbourne",
                       "s-hertogenbosch", "nottingham", "birmingham", "mallorca"]
    for kw in grass_keywords:
        if kw in name:
            return "grass"
    for kw in clay_keywords:
        if kw in name:
            return "clay"
    return "hard"


def format_schedule_text(matches: list) -> str:
    """Форматирует расписание в HTML текст для Telegram."""
    if not matches:
        return "Матчей не найдено."

    # Группируем по турниру
    by_tour: dict = {}
    for m in matches:
        t = m["tournament"]
        if t not in by_tour:
            by_tour[t] = []
        by_tour[t].append(m)

    surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}
    lines = []

    for tour_name, t_matches in list(by_tour.items())[:10]:
        surface  = t_matches[0].get("surface", "hard")
        icon     = surf_icons.get(surface, "🎾")
        tour_lbl = t_matches[0].get("league", "ATP").upper()
        lines.append(f"\n{icon} <b>{tour_lbl} | {tour_name}</b>")

        for m in t_matches[:6]:
            date_str = m.get("commence_time", "")[:10]
            lines.append(f"  • {m['player1']} vs {m['player2']}"
                        + (f"  <i>{date_str}</i>" if date_str else ""))

    return "\n".join(lines)
