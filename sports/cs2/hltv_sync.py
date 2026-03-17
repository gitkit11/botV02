"""
HLTV Sync — синхронный модуль для получения статистики HLTV без asyncio конфликтов.
Использует кэширование и fallback на статические данные.
"""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Путь к кэшу
CACHE_FILE = Path(__file__).parent.parent.parent / "hltv_cache.json"

# Статический кэш — синхронизирован с hltv_stats.py
# При обновлении hltv_stats.py обновляй и этот файл
from .hltv_stats import MAP_STATS as STATIC_MAP_STATS, PLAYER_STATS as STATIC_PLAYER_STATS, TEAM_ALIASES

def load_cache():
    """Загружает кэш из файла."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}")
    return {}

def save_cache(data):
    """Сохраняет кэш в файл."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша: {e}")

def get_team_map_stats(team_name: str) -> dict:
    """Получить винрейты команды по картам (с приоритетом: кэш -> статика)."""
    name = TEAM_ALIASES.get(team_name, team_name)
    cache = load_cache()
    
    # Приоритет: Динамический кэш -> Статический кэш
    return cache.get("maps", {}).get(name) or STATIC_MAP_STATS.get(name, {})

def get_player_stats(team_name: str) -> list:
    """Получить статистику игроков команды."""
    name = TEAM_ALIASES.get(team_name, team_name)
    cache = load_cache()
    return cache.get("players", {}).get(name) or STATIC_PLAYER_STATS.get(name, [])

def update_team_stats(team_name: str, maps: dict = None, players: list = None):
    """Обновляет динамический кэш для команды."""
    name = TEAM_ALIASES.get(team_name, team_name)
    cache = load_cache()
    
    if "maps" not in cache:
        cache["maps"] = {}
    if "players" not in cache:
        cache["players"] = {}
    
    if maps:
        cache["maps"][name] = maps
    if players:
        cache["players"][name] = players
    
    save_cache(cache)

def format_map_stats_for_ai(home_team: str, away_team: str) -> str:
    """Форматирует статистику карт для ИИ-анализа."""
    h_maps = get_team_map_stats(home_team)
    a_maps = get_team_map_stats(away_team)
    
    if not h_maps and not a_maps:
        return "Статистика карт HLTV недоступна."
    
    res = "🗺 Статистика карт (HLTV Winrate):\n"
    all_maps = set(list(h_maps.keys()) + list(a_maps.keys()))
    for m in sorted(all_maps):
        h_wr = h_maps.get(m, "—")
        a_wr = a_maps.get(m, "—")
        res += f"  • {m}: {home_team} {h_wr}% vs {a_wr}% {away_team}\n"
    return res

def format_players_for_ai(home_team: str, away_team: str) -> str:
    """Форматирует статистику игроков для ИИ-анализа."""
    h_players = get_player_stats(home_team)
    a_players = get_player_stats(away_team)
    
    if not h_players and not a_players:
        return "Статистика игроков HLTV недоступна."
    
    res = "👥 Ключевые игроки (HLTV Rating):\n"
    if h_players:
        h_str = ", ".join([f"{p['name']} ({p['rating']})" for p in h_players])
        res += f"  🔹 {home_team}: {h_str}\n"
    if a_players:
        a_str = ", ".join([f"{p['name']} ({p['rating']})" for p in a_players])
        res += f"  🔸 {away_team}: {a_str}\n"
    return res
