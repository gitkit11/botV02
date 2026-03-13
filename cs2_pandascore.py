# -*- coding: utf-8 -*-
import os
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()

def get_cs2_matches_pandascore():
    """
    Получает матчи CS2 (включая Tier-2/3) через PandaScore API.
    Автономный модуль: не зависит от config.py.
    """
    key = os.getenv("PANDASCORE_API_KEY")
    if not key:
        print("[PandaScore] Ошибка: PANDASCORE_API_KEY не найден в .env")
        return []

    url = "https://api.pandascore.co/csgo/matches/upcoming"
    headers = {"Authorization": f"Bearer {key}"}
    params = {"per_page": 50}

    matches = []
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            for item in response.json():
                if item.get('opponents') and len(item['opponents']) >= 2:
                    matches.append(parse_pandascore_item(item, status="UPCOMING"))
        
        # Также берем текущие матчи
        url_running = "https://api.pandascore.co/csgo/matches/running"
        response = requests.get(url_running, headers=headers, params=params)
        if response.status_code == 200:
            for item in response.json():
                if item.get('opponents') and len(item['opponents']) >= 2:
                    matches.append(parse_pandascore_item(item, status="LIVE 🔴"))
        
        return matches
    except Exception as e:
        print(f"[PandaScore] Ошибка: {e}")
        return []

def parse_pandascore_item(item, status):
    """Парсит один матч из формата PandaScore."""
    opponents = item['opponents']
    home = opponents[0]['opponent']['name']
    away = opponents[1]['opponent']['name']
    
    start_time = item.get('begin_at')
    time_str = "—"
    if start_time:
        try:
            dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M")
        except:
            pass

    return {
        "id": item['id'],
        "home": home,
        "away": away,
        "time": f"{time_str} [{status}]",
        "odds": {"home_win": 1.90, "away_win": 1.90},
        "league": item.get('league', {}).get('name', 'Tier-2/3')
    }

def get_combined_cs2_matches():
    """Интерфейс для main.py."""
    return get_cs2_matches_pandascore()
