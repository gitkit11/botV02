"""
scripts/update_hltv_stats.py — Ежедневное обновление статистики HLTV
=====================================================================
Использует API-зеркало для получения данных без капчи и браузера.
Обновляет динамический кэш в hltv_stats.py.
"""

import json
import logging
import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sports.cs2.hltv_stats import update_team_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Команды для обновления
TEAMS_TO_UPDATE = [
    ("Team Vitality",   9565),
    ("G2 Esports",      5995),
    ("FaZe Clan",       6667),
    ("Natus Vincere",   4608),
    ("Team Spirit",     7020),
    ("MOUZ",            4494),
    ("Heroic",          7175),
    ("Astralis",        4411),
    ("Team Liquid",     5973),
    ("FURIA",           8297),
    ("The MongolZ",     11595),
    ("Cloud9",          5005),
    ("BIG",             8068),
    ("Falcons",         12279),
    ("Eternal Fire",    11518),
    ("paiN",            4773),
]

def get_team_data_api(team_id: int):
    """Получение данных через API-зеркало HLTV."""
    base_url = f"https://hltv-api.vercel.app/api/team/{team_id}"
    
    try:
        response = requests.get(base_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            
            # Извлекаем винрейты карт
            map_stats = {}
            if "mapStats" in data:
                for map_name, stats in data["mapStats"].items():
                    if "winRate" in stats:
                        map_stats[map_name] = float(stats["winRate"])
            
            # Извлекаем игроков
            player_stats = []
            if "players" in data:
                for p in data["players"]:
                    player_stats.append({
                        "name": p.get("name", "Unknown"),
                        "rating": float(p.get("rating", 1.0))
                    })
            
            return map_stats, player_stats
    except Exception as e:
        logger.error(f"Ошибка API для ID {team_id}: {e}")
        
    return None, None

def run_update():
    update_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== HLTV Update started: {update_date} ===")

    success_count = 0
    for team_name, team_id in TEAMS_TO_UPDATE:
        logger.info(f"Обновление {team_name} (ID: {team_id})...")
        
        maps, players = get_team_data_api(team_id)
        
        if maps or players:
            update_team_stats(team_name, maps=maps, players=players)
            logger.info(f"  ✅ {team_name} обновлен")
            success_count += 1
        else:
            logger.warning(f"  ❌ Не удалось получить данные для {team_name}")
            
        time.sleep(1)
        
    logger.info(f"=== Обновление завершено. Успешно: {success_count}/{len(TEAMS_TO_UPDATE)} ===")

if __name__ == "__main__":
    run_update()
