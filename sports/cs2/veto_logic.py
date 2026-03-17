# -*- coding: utf-8 -*-
import random
from .hltv_stats import MAP_STATS, PLAYER_STATS, TEAM_ALIASES, get_team_map_stats, get_player_stats

# Список официальных карт CS2
ACTIVE_DUTY_POOL = ["Mirage", "Nuke", "Inferno", "Ancient", "Anubis", "Vertigo", "Dust2"]

# Предпочтения команд при вето (0.0 = всегда банят, 1.0 = всегда пикают)
# Основано на winrate по картам и исторических пат-вето
TEAM_MAP_PREFERENCES = {
    "Natus Vincere":  {"Mirage": 0.9, "Nuke": 0.8, "Inferno": 0.6, "Ancient": 0.9, "Anubis": 0.7, "Vertigo": 0.2, "Dust2": 0.5},
    "Team Vitality":  {"Mirage": 0.7, "Nuke": 0.9, "Inferno": 0.8, "Ancient": 0.5, "Anubis": 0.8, "Vertigo": 0.6, "Dust2": 0.9},
    "FaZe Clan":      {"Mirage": 0.8, "Nuke": 0.7, "Inferno": 0.9, "Ancient": 0.6, "Anubis": 0.6, "Vertigo": 0.2, "Dust2": 0.8},
    "G2 Esports":     {"Mirage": 0.9, "Nuke": 0.6, "Inferno": 0.8, "Ancient": 0.7, "Anubis": 0.7, "Vertigo": 0.4, "Dust2": 0.9},
    "Team Spirit":    {"Mirage": 0.8, "Nuke": 0.7, "Inferno": 0.4, "Ancient": 0.9, "Anubis": 0.9, "Vertigo": 0.6, "Dust2": 0.8},
    "MOUZ":           {"Mirage": 0.8, "Nuke": 0.9, "Inferno": 0.7, "Ancient": 0.8, "Anubis": 0.6, "Vertigo": 0.9, "Dust2": 0.5},
    "Heroic":         {"Mirage": 0.6, "Nuke": 0.9, "Inferno": 0.8, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.4, "Dust2": 0.7},
    "Cloud9":         {"Mirage": 0.8, "Nuke": 0.5, "Inferno": 0.6, "Ancient": 0.9, "Anubis": 0.7, "Vertigo": 0.5, "Dust2": 0.7},
    "Team Liquid":    {"Mirage": 0.7, "Nuke": 0.8, "Inferno": 0.6, "Ancient": 0.5, "Anubis": 0.8, "Vertigo": 0.7, "Dust2": 0.6},
    "Astralis":       {"Mirage": 0.7, "Nuke": 0.9, "Inferno": 0.6, "Ancient": 0.8, "Anubis": 0.7, "Vertigo": 0.4, "Dust2": 0.6},
    "ENCE":           {"Mirage": 0.6, "Nuke": 0.8, "Inferno": 0.7, "Ancient": 0.9, "Anubis": 0.5, "Vertigo": 0.6, "Dust2": 0.4},
    "FURIA":          {"Mirage": 0.6, "Nuke": 0.6, "Inferno": 0.9, "Ancient": 0.8, "Anubis": 0.5, "Vertigo": 0.4, "Dust2": 0.7},
    "BIG":            {"Mirage": 0.7, "Nuke": 0.8, "Inferno": 0.6, "Ancient": 0.8, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.5},
    "OG":             {"Mirage": 0.5, "Nuke": 0.6, "Inferno": 0.7, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.6},
    "Complexity":     {"Mirage": 0.6, "Nuke": 0.7, "Inferno": 0.5, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.6, "Dust2": 0.7},
    "Eternal Fire":   {"Mirage": 0.7, "Nuke": 0.8, "Inferno": 0.6, "Ancient": 0.6, "Anubis": 0.7, "Vertigo": 0.6, "Dust2": 0.5},
    "Fnatic":         {"Mirage": 0.6, "Nuke": 0.7, "Inferno": 0.7, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.6},
    "Virtus.pro":     {"Mirage": 0.7, "Nuke": 0.7, "Inferno": 0.6, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.6},
    "3DMAX":          {"Mirage": 0.7, "Nuke": 0.6, "Inferno": 0.8, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.6},
    "paiN Gaming":    {"Mirage": 0.6, "Nuke": 0.5, "Inferno": 0.8, "Ancient": 0.7, "Anubis": 0.6, "Vertigo": 0.5, "Dust2": 0.7},
}

def _normalize_team(team_name: str) -> str:
    return TEAM_ALIASES.get(team_name, team_name)

def _fetch_live_map_stats(team_name: str) -> dict:
    """Получает реальный винрейт по картам из PandaScore для любой команды."""
    try:
        from .pandascore import get_team_map_winrates
        return get_team_map_winrates(team_name, last_n=30)
    except Exception:
        return {}


def simulate_bo3_veto(home_team, away_team, team_map_stats=None):
    pool = list(ACTIVE_DUTY_POOL)
    veto_log = []

    h_name = _normalize_team(home_team)
    a_name = _normalize_team(away_team)

    # Подтягиваем живые данные если нет в статике
    h_live = _fetch_live_map_stats(h_name) if h_name not in TEAM_MAP_PREFERENCES and h_name not in MAP_STATS else {}
    a_live = _fetch_live_map_stats(a_name) if a_name not in TEAM_MAP_PREFERENCES and a_name not in MAP_STATS else {}

    h_pref = {}
    a_pref = {}

    for m in pool:
        h_wr = (team_map_stats or {}).get(h_name, {}).get(m) or MAP_STATS.get(h_name, {}).get(m) or h_live.get(m)
        if h_wr is not None:
            h_pref[m] = h_wr / 100.0
        else:
            h_pref[m] = TEAM_MAP_PREFERENCES.get(h_name, {}).get(m, 0.5)

        a_wr = (team_map_stats or {}).get(a_name, {}).get(m) or MAP_STATS.get(a_name, {}).get(m) or a_live.get(m)
        if a_wr is not None:
            a_pref[m] = a_wr / 100.0
        else:
            a_pref[m] = TEAM_MAP_PREFERENCES.get(a_name, {}).get(m, 0.5)
    
    # 1. Home ban
    h_ban = min(pool, key=lambda m: h_pref.get(m, 0.5))
    pool.remove(h_ban)
    veto_log.append(f"🚫 {home_team} забанил {h_ban}")
    
    # 2. Away ban
    a_ban = min(pool, key=lambda m: a_pref.get(m, 0.5))
    pool.remove(a_ban)
    veto_log.append(f"🚫 {away_team} забанил {a_ban}")
    
    # 3. Home pick
    h_pick = max(pool, key=lambda m: h_pref.get(m, 0.5) - a_pref.get(m, 0.5))
    pool.remove(h_pick)
    veto_log.append(f"✅ {home_team} выбрал {h_pick}")
    
    # 4. Away pick
    a_pick = max(pool, key=lambda m: a_pref.get(m, 0.5) - h_pref.get(m, 0.5))
    pool.remove(a_pick)
    veto_log.append(f"✅ {away_team} выбрал {a_pick}")
    
    # 5. Home ban
    h_ban2 = min(pool, key=lambda m: h_pref.get(m, 0.5))
    pool.remove(h_ban2)
    veto_log.append(f"🚫 {home_team} забанил {h_ban2}")
    
    # 6. Away ban
    a_ban2 = min(pool, key=lambda m: a_pref.get(m, 0.5))
    pool.remove(a_ban2)
    veto_log.append(f"🚫 {away_team} забанил {a_ban2}")
    
    # 7. Decider
    decider = pool[0]
    veto_log.append(f"🎲 Decider: {decider}")
    
    return [h_pick, a_pick, decider], veto_log

def get_map_impact_score(team_name, map_name, team_map_stats=None, live_stats: dict = None):
    name = _normalize_team(team_name)
    wr = (team_map_stats or {}).get(name, {}).get(map_name) or MAP_STATS.get(name, {}).get(map_name)
    if wr is not None:
        return wr / 100.0
    # Живые данные из PandaScore (передаются снаружи чтобы не делать лишних запросов)
    if live_stats and map_name in live_stats:
        return live_stats[map_name] / 100.0
    return TEAM_MAP_PREFERENCES.get(name, {}).get(map_name, 0.5)

def get_team_player_stats(team_name):
    name = _normalize_team(team_name)
    return PLAYER_STATS.get(name, [])
