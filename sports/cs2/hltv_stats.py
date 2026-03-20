"""
HLTV Stats — Модуль для работы со статистикой HLTV.
Поддерживает статический кэш и динамическое обновление.
Данные актуальны на 2024-2025 сезон. Обновляй через update_team_stats() или вручную.
"""
import json
import os
import logging
from .team_registry import TEAM_ALIASES as _REGISTRY_ALIASES, normalize_team_name as _registry_normalize

logger = logging.getLogger(__name__)

CACHE_FILE = "hltv_cache.json"

# Статический кэш (Fallback) — winrate по картам в %
MAP_STATS: dict[str, dict[str, float]] = {
    # ── Tier-1 ──────────────────────────────────────────────────────────────
    "Team Vitality": {
        "Mirage": 67.4, "Nuke": 70.7, "Inferno": 77.8, "Ancient": 50.0,
        "Anubis": 75.0, "Vertigo": 62.0, "Dust2": 90.9,
    },
    "G2 Esports": {
        "Mirage": 66.7, "Nuke": 50.0, "Inferno": 55.6, "Ancient": 45.0,
        "Anubis": 62.5, "Vertigo": 55.0, "Dust2": 60.0,
    },
    "FaZe Clan": {
        "Mirage": 52.4, "Nuke": 47.1, "Inferno": 50.0, "Ancient": 50.0,
        "Anubis": 53.3, "Vertigo": 48.0, "Dust2": 55.6,
    },
    "Natus Vincere": {
        "Mirage": 61.5, "Nuke": 54.5, "Inferno": 58.3, "Ancient": 60.0,
        "Anubis": 57.1, "Vertigo": 45.0, "Dust2": 50.0,
    },
    "Team Spirit": {
        "Mirage": 60.0, "Nuke": 72.7, "Inferno": 65.0, "Ancient": 55.0,
        "Anubis": 58.3, "Vertigo": 65.0, "Dust2": 70.0,
    },
    "MOUZ": {
        "Mirage": 58.0, "Nuke": 72.0, "Inferno": 63.0, "Ancient": 65.0,
        "Anubis": 60.0, "Vertigo": 74.0, "Dust2": 55.0,
    },
    # ── Tier-1 / High Tier-2 ────────────────────────────────────────────────
    "Heroic": {
        "Mirage": 55.0, "Nuke": 70.0, "Inferno": 65.0, "Ancient": 60.0,
        "Anubis": 58.0, "Vertigo": 52.0, "Dust2": 60.0,
    },
    "Cloud9": {
        "Mirage": 62.0, "Nuke": 55.0, "Inferno": 58.0, "Ancient": 65.0,
        "Anubis": 60.0, "Vertigo": 55.0, "Dust2": 60.0,
    },
    "Team Liquid": {
        "Mirage": 60.0, "Nuke": 65.0, "Inferno": 58.0, "Ancient": 55.0,
        "Anubis": 62.0, "Vertigo": 60.0, "Dust2": 58.0,
    },
    "Astralis": {
        "Mirage": 60.0, "Nuke": 72.0, "Inferno": 58.0, "Ancient": 65.0,
        "Anubis": 60.0, "Vertigo": 55.0, "Dust2": 55.0,
    },
    "ENCE": {
        "Mirage": 55.0, "Nuke": 68.0, "Inferno": 60.0, "Ancient": 65.0,
        "Anubis": 55.0, "Vertigo": 58.0, "Dust2": 50.0,
    },
    "FURIA": {
        "Mirage": 55.0, "Nuke": 58.0, "Inferno": 70.0, "Ancient": 65.0,
        "Anubis": 55.0, "Vertigo": 52.0, "Dust2": 62.0,
    },
    "BIG": {
        "Mirage": 58.0, "Nuke": 68.0, "Inferno": 60.0, "Ancient": 65.0,
        "Anubis": 58.0, "Vertigo": 55.0, "Dust2": 52.0,
    },
    # ── Tier-2 ──────────────────────────────────────────────────────────────
    "OG": {
        "Mirage": 52.0, "Nuke": 55.0, "Inferno": 60.0, "Ancient": 58.0,
        "Anubis": 55.0, "Vertigo": 50.0, "Dust2": 55.0,
    },
    "Complexity": {
        "Mirage": 55.0, "Nuke": 58.0, "Inferno": 52.0, "Ancient": 60.0,
        "Anubis": 55.0, "Vertigo": 55.0, "Dust2": 58.0,
    },
    "Eternal Fire": {
        "Mirage": 60.0, "Nuke": 62.0, "Inferno": 58.0, "Ancient": 55.0,
        "Anubis": 60.0, "Vertigo": 55.0, "Dust2": 52.0,
    },
    "Fnatic": {
        "Mirage": 55.0, "Nuke": 60.0, "Inferno": 58.0, "Ancient": 60.0,
        "Anubis": 55.0, "Vertigo": 52.0, "Dust2": 55.0,
    },
    "paiN Gaming": {
        "Mirage": 55.0, "Nuke": 50.0, "Inferno": 65.0, "Ancient": 60.0,
        "Anubis": 55.0, "Vertigo": 52.0, "Dust2": 60.0,
    },
    "NIP": {
        "Mirage": 52.0, "Nuke": 55.0, "Inferno": 55.0, "Ancient": 58.0,
        "Anubis": 53.0, "Vertigo": 50.0, "Dust2": 55.0,
    },
    "Imperial": {
        "Mirage": 55.0, "Nuke": 52.0, "Inferno": 60.0, "Ancient": 55.0,
        "Anubis": 52.0, "Vertigo": 50.0, "Dust2": 58.0,
    },
    "3DMAX": {
        "Mirage": 60.0, "Nuke": 55.0, "Inferno": 62.0, "Ancient": 58.0,
        "Anubis": 55.0, "Vertigo": 52.0, "Dust2": 55.0,
    },
    "Monte": {
        "Mirage": 55.0, "Nuke": 58.0, "Inferno": 55.0, "Ancient": 55.0,
        "Anubis": 52.0, "Vertigo": 55.0, "Dust2": 52.0,
    },
    "Virtus.pro": {
        "Mirage": 58.0, "Nuke": 60.0, "Inferno": 55.0, "Ancient": 58.0,
        "Anubis": 55.0, "Vertigo": 52.0, "Dust2": 55.0,
    },
}

# Рейтинги игроков (HLTV Rating 2.0, приблизительно за 2024-2025)
PLAYER_STATS: dict[str, list[dict]] = {
    "Team Vitality": [
        {"name": "ZywOo",  "rating": 1.35},
        {"name": "apEX",   "rating": 1.05},
        {"name": "flameZ", "rating": 1.12},
        {"name": "mezii",  "rating": 1.08},
        {"name": "Spinx",  "rating": 1.15},
    ],
    "G2 Esports": [
        {"name": "m0NESY",  "rating": 1.28},
        {"name": "NiKo",    "rating": 1.22},
        {"name": "huNter-", "rating": 1.10},
        {"name": "MalbsMd", "rating": 1.15},
        {"name": "Snax",    "rating": 1.02},
    ],
    "FaZe Clan": [
        {"name": "karrigan", "rating": 0.97},
        {"name": "rain",     "rating": 1.08},
        {"name": "broky",    "rating": 1.15},
        {"name": "frozen",   "rating": 1.12},
        {"name": "ropz",     "rating": 1.14},
    ],
    "Natus Vincere": [
        {"name": "iM",       "rating": 1.12},
        {"name": "w0nderful","rating": 1.18},
        {"name": "jL",       "rating": 1.15},
        {"name": "b1t",      "rating": 1.10},
        {"name": "Aleksib",  "rating": 1.02},
    ],
    "Team Spirit": [
        {"name": "donk",    "rating": 1.38},
        {"name": "chopper", "rating": 1.02},
        {"name": "zont1x",  "rating": 1.15},
        {"name": "shalfey", "rating": 1.10},
        {"name": "magixx",  "rating": 1.05},
    ],
    "MOUZ": [
        {"name": "xertioN", "rating": 1.18},
        {"name": "torzsi",  "rating": 1.16},
        {"name": "siuhy",   "rating": 1.10},
        {"name": "Blamef",  "rating": 1.08},
        {"name": "Jimpphat","rating": 1.14},
    ],
    "Heroic": [
        {"name": "stavn",  "rating": 1.22},
        {"name": "cadiaN", "rating": 1.05},
        {"name": "jabbi",  "rating": 1.15},
        {"name": "TeSeS",  "rating": 1.10},
        {"name": "sjuush", "rating": 1.08},
    ],
    "Cloud9": [
        {"name": "sh1ro",       "rating": 1.22},
        {"name": "Ax1Le",       "rating": 1.18},
        {"name": "electroNic",  "rating": 1.12},
        {"name": "HObbit",      "rating": 1.08},
        {"name": "Perfecto",    "rating": 1.05},
    ],
    "Team Liquid": [
        {"name": "oSee",  "rating": 1.15},
        {"name": "NAF",   "rating": 1.14},
        {"name": "EliGE", "rating": 1.14},
        {"name": "jks",   "rating": 1.12},
        {"name": "nitr0", "rating": 1.02},
    ],
    "Astralis": [
        {"name": "device",  "rating": 1.15},
        {"name": "Magisk",  "rating": 1.12},
        {"name": "K0nfig",  "rating": 1.14},
        {"name": "dupreeh", "rating": 1.10},
        {"name": "br0",     "rating": 1.05},
    ],
    "ENCE": [
        {"name": "Gla1ve",   "rating": 1.02},
        {"name": "dycha",    "rating": 1.14},
        {"name": "NertZ",    "rating": 1.15},
        {"name": "maden",    "rating": 1.08},
        {"name": "SunPayus", "rating": 1.13},
    ],
    "FURIA": [
        {"name": "KSCERATO", "rating": 1.18},
        {"name": "yuurih",   "rating": 1.15},
        {"name": "FalleN",   "rating": 1.10},
        {"name": "arT",      "rating": 1.05},
        {"name": "chelo",    "rating": 1.05},
    ],
    "BIG": [
        {"name": "tabseN", "rating": 1.18},
        {"name": "syrsoN", "rating": 1.12},
        {"name": "faveN",  "rating": 1.10},
        {"name": "tiziaN", "rating": 1.08},
        {"name": "prosus", "rating": 1.02},
    ],
    "OG": [
        {"name": "valde",   "rating": 1.08},
        {"name": "degster", "rating": 1.12},
        {"name": "nexa",    "rating": 1.02},
        {"name": "ISSAA",   "rating": 1.05},
        {"name": "niko",    "rating": 1.05},
    ],
    "Complexity": [
        {"name": "hallzerk", "rating": 1.15},
        {"name": "floppy",   "rating": 1.12},
        {"name": "Grim",     "rating": 1.10},
        {"name": "JT",       "rating": 1.08},
        {"name": "nicoodoz", "rating": 1.08},
    ],
    "Eternal Fire": [
        {"name": "xfl0ud",  "rating": 1.18},
        {"name": "woxic",   "rating": 1.15},
        {"name": "Calyx",   "rating": 1.12},
        {"name": "XANTARES","rating": 1.20},
        {"name": "Wicadia", "rating": 1.05},
    ],
    "Fnatic": [
        {"name": "mezii",   "rating": 1.08},
        {"name": "KRIMZ",   "rating": 1.05},
        {"name": "roeJ",    "rating": 1.10},
        {"name": "afro",    "rating": 1.08},
        {"name": "nicoodoz","rating": 1.10},
    ],
    "paiN Gaming": [
        {"name": "hardzao", "rating": 1.12},
        {"name": "biguzera","rating": 1.10},
        {"name": "heat",    "rating": 1.08},
        {"name": "lux1k",   "rating": 1.05},
        {"name": "dav1deuS","rating": 1.15},
    ],
    "Virtus.pro": [
        {"name": "Jame",  "rating": 1.15},
        {"name": "FL1T",  "rating": 1.12},
        {"name": "fame",  "rating": 1.10},
        {"name": "n0rb3r7","rating": 1.08},
        {"name": "electroNic", "rating": 1.05},
    ],
    "3DMAX": [
        {"name": "Ex3rcice", "rating": 1.15},
        {"name": "Graviti",  "rating": 1.10},
        {"name": "Djoko",    "rating": 1.08},
        {"name": "lucky",    "rating": 1.12},
        {"name": "maka",     "rating": 1.05},
    ],
}

# TEAM_ALIASES — импортированы из team_registry как единый источник правды
TEAM_ALIASES = _REGISTRY_ALIASES


def _resolve_hltv_name(team_name: str) -> str:
    """Нормализует имя команды через team_registry, затем проверяет MAP_STATS."""
    canonical = _registry_normalize(team_name)
    # Если после нормализации имя есть в MAP_STATS — возвращаем его
    if canonical in MAP_STATS:
        return canonical
    # Иначе — fuzzy по MAP_STATS/PLAYER_STATS
    try:
        from difflib import get_close_matches
        all_names = list(MAP_STATS.keys()) + list(PLAYER_STATS.keys())
        matches = get_close_matches(canonical, all_names, n=1, cutoff=0.80)
        if matches:
            return matches[0]
        # Пробуем оригинал тоже
        matches = get_close_matches(team_name, all_names, n=1, cutoff=0.80)
        if matches:
            return matches[0]
    except Exception:
        pass
    return canonical


# ── In-memory кеш hltv_cache.json — загружается один раз, не при каждом вызове ──
import time as _time
_hltv_mem_cache: dict = {}
_hltv_mem_ts: float = 0.0
_HLTV_MEM_TTL = 300  # 5 минут — обновляем из файла редко

# ── In-memory кеш результатов get_player_stats — TTL 1 час ──
_player_cache: dict = {}
_PLAYER_CACHE_TTL = 3600  # 1 час

def load_dynamic_cache() -> dict:
    """Загружает кеш с диска. Внутри используй _get_hltv_cache() для in-memory доступа."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _get_hltv_cache() -> dict:
    """In-memory кеш с TTL 5 минут. Не читает файл при каждом вызове."""
    global _hltv_mem_cache, _hltv_mem_ts
    if _hltv_mem_cache and (_time.time() - _hltv_mem_ts) < _HLTV_MEM_TTL:
        return _hltv_mem_cache
    _hltv_mem_cache = load_dynamic_cache()
    _hltv_mem_ts = _time.time()
    return _hltv_mem_cache

def save_dynamic_cache(data):
    global _hltv_mem_cache, _hltv_mem_ts
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Обновляем in-memory кеш сразу чтобы последующие читатели видели свежие данные
        _hltv_mem_cache = data
        _hltv_mem_ts = _time.time()
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша HLTV: {e}")

def get_team_map_stats(team_name: str) -> dict:
    name = _resolve_hltv_name(team_name)
    cache = _get_hltv_cache()  # из памяти, не с диска
    return cache.get("maps", {}).get(name) or MAP_STATS.get(name, {})

def get_player_stats(team_name: str) -> list:
    name = _resolve_hltv_name(team_name)
    # Проверяем in-memory кеш с TTL 1 час
    cached = _player_cache.get(name)
    if cached is not None:
        data, ts = cached
        if (_time.time() - ts) < _PLAYER_CACHE_TTL:
            return data
    # Получаем данные из hltv-кеша или статического словаря
    hltv = _get_hltv_cache()
    result = hltv.get("players", {}).get(name) or PLAYER_STATS.get(name, [])
    # Сохраняем в кеш только если есть данные
    if result:
        _player_cache[name] = (result, _time.time())
    return result

def update_team_stats(team_name: str, maps: dict = None, players: list = None):
    """Обновляет динамический кэш для команды."""
    name = _resolve_hltv_name(team_name)
    cache = load_dynamic_cache()  # здесь нужна свежая версия с диска
    if "maps" not in cache:
        cache["maps"] = {}
    if "players" not in cache:
        cache["players"] = {}
    if maps:
        cache["maps"][name] = maps
    if players:
        cache["players"][name] = players
    save_dynamic_cache(cache)

def format_map_stats_for_ai(home_team: str, away_team: str) -> str:
    h_maps = get_team_map_stats(home_team)
    a_maps = get_team_map_stats(away_team)
    if not h_maps and not a_maps:
        return "Статистика карт HLTV недоступна."
    res = "🗺 Статистика карт (HLTV Winrate %):\n"
    all_maps = sorted(set(list(h_maps.keys()) + list(a_maps.keys())))
    for m in all_maps:
        h_wr = h_maps.get(m, "—")
        a_wr = a_maps.get(m, "—")
        # Добавляем стрелку преимущества
        arrow = ""
        if isinstance(h_wr, (int, float)) and isinstance(a_wr, (int, float)):
            diff = h_wr - a_wr
            if diff >= 10:
                arrow = f" ◀ +{diff:.0f}%"
            elif diff <= -10:
                arrow = f" +{-diff:.0f}% ▶"
        res += f"  • {m}: {home_team} {h_wr}% vs {a_wr}% {away_team}{arrow}\n"
    return res

def format_players_for_ai(home_team: str, away_team: str) -> str:
    h_players = get_player_stats(home_team)
    a_players = get_player_stats(away_team)
    if not h_players and not a_players:
        return "Статистика игроков HLTV недоступна."
    res = "👥 Ключевые игроки (HLTV Rating 2.0):\n"
    if h_players:
        h_avg = sum(p['rating'] for p in h_players) / len(h_players)
        h_str = ", ".join([f"{p['name']} ({p['rating']})" for p in h_players])
        res += f"  🔹 {home_team} [ср. {h_avg:.2f}]: {h_str}\n"
    if a_players:
        a_avg = sum(p['rating'] for p in a_players) / len(a_players)
        a_str = ", ".join([f"{p['name']} ({p['rating']})" for p in a_players])
        res += f"  🔸 {away_team} [ср. {a_avg:.2f}]: {a_str}\n"
    return res
