"""
scripts/update_hltv_stats.py — Ежедневное обновление статистики HLTV
=====================================================================
Использует API-зеркало для получения данных без капчи и браузера.
Исправлены ошибки кодировки для Windows (Unicode/cp1252).

Запуск:
    python scripts/update_hltv_stats.py
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

# Настройка логирования с поддержкой UTF-8 в Windows
def setup_logging():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    
    # Исправление для Windows: принудительно используем utf-8 если возможно
    if sys.platform == "win32":
        try:
            import codecs
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        except:
            pass

    root_logger.addHandler(console_handler)
    return logging.getLogger(__name__)

logger = setup_logging()

# ─── Команды для обновления ──────────────────────────────────────────────────
TEAMS_TO_UPDATE = [
    ("Team Vitality",   9565,  "vitality"),
    ("G2 Esports",      5995,  "g2"),
    ("FaZe Clan",       6667,  "faze"),
    ("Natus Vincere",   4608,  "natus-vincere"),
    ("Team Spirit",     7020,  "spirit"),
    ("MOUZ",            4494,  "mouz"),
    ("Heroic",          7175,  "heroic"),
    ("Astralis",        4411,  "astralis"),
    ("Team Liquid",     5973,  "liquid"),
    ("FURIA",           8297,  "furia"),
    ("The MongolZ",     11595, "the-mongolz"),
    ("Cloud9",          5005,  "cloud9"),
    ("BIG",             8068,  "big"),
    ("Falcons",         12279, "falcons"),
]

def get_team_data_api(team_id: int):
    """
    Получение данных через открытое API-зеркало HLTV.
    Это работает без браузера и капчи.
    """
    # Используем публичное зеркало API HLTV
    base_url = f"https://hltv-api.vercel.app/api/team/{team_id}"
    
    try:
        response = requests.get(base_url, timeout=10)
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

def generate_hltv_stats_file(map_results: dict, player_results: dict, update_date: str) -> str:
    content = f'"""\nHLTV Stats — Автоматически обновляемые данные\nДата обновления: {update_date}\n"""\n\n'
    
    content += "MAP_STATS: dict[str, dict[str, float]] = {\n"
    for team, maps in map_results.items():
        content += f'    "{team}": {json.dumps(maps)},\n'
    content += "}\n\n"
    
    content += "PLAYER_STATS: dict[str, list[dict]] = {\n"
    for team, players in player_results.items():
        content += f'    "{team}": {json.dumps(players)},\n'
    content += "}\n\n"
    
    content += "TEAM_ALIASES: dict[str, str] = {\n"
    content += '    "Vitality": "Team Vitality",\n'
    content += '    "G2": "G2 Esports",\n'
    content += '    "FaZe": "FaZe Clan",\n'
    content += '    "NaVi": "Natus Vincere",\n'
    content += '    "Spirit": "Team Spirit",\n'
    content += '    "mousesports": "MOUZ",\n'
    content += '    "Liquid": "Team Liquid",\n'
    content += "}\n"
    
    return content

def run_update():
    update_date = datetime.now().strftime("%Y-%m-%d")
    map_results = {}
    player_results = {}
    
    logger.info(f"=== HLTV Update via API started: {update_date} ===")

    for team_name, team_id, slug in TEAMS_TO_UPDATE:
        logger.info(f"Обновление {team_name}...")
        
        maps, players = get_team_data_api(team_id)
        
        if maps:
            map_results[team_name] = maps
            logger.info(f"  OK: Карты получены")
        if players:
            player_results[team_name] = players
            logger.info(f"  OK: Игроки получены")
            
        time.sleep(1) # Небольшая пауза, чтобы не нагружать API
        
    if map_results or player_results:
        stats_file = PROJECT_ROOT / "sports" / "cs2" / "hltv_stats.py"
        new_content = generate_hltv_stats_file(map_results, player_results, update_date)
        stats_file.write_text(new_content, encoding="utf-8")
        logger.info(f"=== УСПЕХ: hltv_stats.py обновлен ===")
        return True
    else:
        logger.error("=== ОШИБКА: Данные не получены ===")
        return False

if __name__ == "__main__":
    try:
        run_update()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
