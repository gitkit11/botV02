# -*- coding: utf-8 -*-
"""
sports/tennis/rankings.py — ATP/WTA рейтинги и статистика поверхностей
Хранит текущие рейтинги и индексы эффективности по поверхностям.
Обновляй ATP_TOP / WTA_TOP раз в несколько недель вручную.
"""

# ─── ATP Топ-100 (март 2025) ────────────────────────────────────────────────
# Формат: "Имя Фамилия": rank
ATP_RANKINGS = {
    "Jannik Sinner":       1,
    "Alexander Zverev":    2,
    "Carlos Alcaraz":      3,
    "Novak Djokovic":      4,
    "Taylor Fritz":        5,
    "Jack Draper":         6,
    "Daniil Medvedev":     7,
    "Casper Ruud":         8,
    "Tommy Paul":          9,
    "Alex de Minaur":      10,
    "Andrey Rublev":       11,
    "Holger Rune":         12,
    "Ugo Humbert":         13,
    "Grigor Dimitrov":     14,
    "Stefanos Tsitsipas":  15,
    "Ben Shelton":         16,
    "Francisco Cerundolo": 17,
    "Sebastian Baez":      18,
    "Arthur Fils":         19,
    "Karen Khachanov":     20,
    "Tomas Martin Etcheverry": 21,
    "Gael Monfils":        22,
    "Felix Auger-Aliassime": 23,
    "Nicolas Jarry":       24,
    "Jakub Mensik":        25,
    "Lorenzo Musetti":     26,
    "Alexei Popyrin":      27,
    "Brandon Nakashima":   28,
    "Matteo Berrettini":   29,
    "Giovanni Mpetshi Perricard": 30,
    "Jiri Lehecka":        31,
    "Flavio Cobolli":      32,
    "Nuno Borges":         33,
    "Luciano Darderi":     34,
    "Tallon Griekspoor":   35,
    "Miomir Kecmanovic":   36,
    "Roberto Bautista Agut": 37,
    "Jordan Thompson":     38,
    "Alejandro Davidovich Fokina": 39,
    "Lorenzo Sonego":      40,
    "David Goffin":        41,
    "Jan-Lennard Struff":  42,
    "Borna Coric":         43,
    "Sebastian Korda":     44,
    "Daniel Altmaier":     45,
    "Adrian Mannarino":    46,
    "Mariano Navone":      47,
    "Tomas Machac":        48,
    "Frances Tiafoe":      49,
    "Gijs Brouwer":        50,
}

# ─── WTA Топ-100 (март 2025) ────────────────────────────────────────────────
WTA_RANKINGS = {
    "Aryna Sabalenka":     1,
    "Iga Swiatek":         2,
    "Coco Gauff":          3,
    "Elena Rybakina":      4,
    "Jessica Pegula":      5,
    "Mirra Andreeva":      6,
    "Emma Navarro":        7,
    "Daria Kasatkina":     8,
    "Barbora Krejcikova":  9,
    "Paula Badosa":        10,
    "Jasmine Paolini":     11,
    "Madison Keys":        12,
    "Liudmila Samsonova":  13,
    "Donna Vekic":         14,
    "Karolina Muchova":    15,
    "Diana Shnaider":      16,
    "Anna Kalinskaya":     17,
    "Elise Mertens":       18,
    "Danielle Collins":    19,
    "Beatriz Haddad Maia": 20,
    "Caroline Wozniacki":  21,
    "Elina Svitolina":     22,
    "Marketa Vondrousova": 23,
    "Ekaterina Alexandrova": 24,
    "Veronika Kudermetova": 25,
    "Victoria Azarenka":   26,
    "Yulia Putintseva":    27,
    "Maria Sakkari":       28,
    "Caroline Garcia":     29,
    "Xinyu Wang":          30,
    "Amanda Anisimova":    31,
    "Naomi Osaka":         32,
    "Clara Tauson":        33,
    "Peyton Stearns":      34,
    "Anastasia Potapova":  35,
    "Magda Linette":       36,
    "Katerina Siniakova":  37,
    "Ons Jabeur":          38,
    "Camila Osorio":       39,
    "Linda Noskova":       40,
    # Дополнительные игроки для точного матчинга
    "Hailey Baptiste":     55,
    "Tatjana Maria":       135,
    "Zeynep Sonmez":       85,
    "Antonia Ruzic":       95,
    "Nikola Bartunkova":   75,
    "Caroline Dolehide":   65,
}

# ─── Индексы по поверхности (0.0 — слабо, 1.0 — доминирование) ──────────────
# Формат: "Игрок": {"hard": 0.8, "clay": 0.6, "grass": 0.7}
# Значение 0.5 = средний уровень, >0.7 = специалист
SURFACE_STRENGTH = {
    # ATP Top-50
    "Jannik Sinner":       {"hard": 0.92, "clay": 0.75, "grass": 0.80},
    "Carlos Alcaraz":      {"hard": 0.85, "clay": 0.95, "grass": 0.90},
    "Novak Djokovic":      {"hard": 0.90, "clay": 0.88, "grass": 0.92},
    "Alexander Zverev":    {"hard": 0.82, "clay": 0.85, "grass": 0.72},
    "Taylor Fritz":        {"hard": 0.80, "clay": 0.60, "grass": 0.78},
    "Jack Draper":         {"hard": 0.78, "clay": 0.70, "grass": 0.82},
    "Daniil Medvedev":     {"hard": 0.88, "clay": 0.62, "grass": 0.70},
    "Holger Rune":         {"hard": 0.78, "clay": 0.82, "grass": 0.72},
    "Casper Ruud":         {"hard": 0.72, "clay": 0.90, "grass": 0.62},
    "Stefanos Tsitsipas":  {"hard": 0.78, "clay": 0.88, "grass": 0.72},
    "Tommy Paul":          {"hard": 0.78, "clay": 0.65, "grass": 0.72},
    "Alex de Minaur":      {"hard": 0.80, "clay": 0.68, "grass": 0.82},
    "Andrey Rublev":       {"hard": 0.80, "clay": 0.78, "grass": 0.70},
    "Ben Shelton":         {"hard": 0.78, "clay": 0.62, "grass": 0.75},
    "Grigor Dimitrov":     {"hard": 0.78, "clay": 0.72, "grass": 0.80},
    "Lorenzo Musetti":     {"hard": 0.70, "clay": 0.82, "grass": 0.68},
    "Ugo Humbert":         {"hard": 0.78, "clay": 0.70, "grass": 0.82},
    "Francisco Cerundolo": {"hard": 0.68, "clay": 0.85, "grass": 0.62},
    "Sebastian Baez":      {"hard": 0.65, "clay": 0.88, "grass": 0.60},
    "Arthur Fils":         {"hard": 0.78, "clay": 0.72, "grass": 0.72},
    "Karen Khachanov":     {"hard": 0.80, "clay": 0.72, "grass": 0.72},
    "Tomas Martin Etcheverry": {"hard": 0.65, "clay": 0.88, "grass": 0.60},
    "Gael Monfils":        {"hard": 0.78, "clay": 0.72, "grass": 0.72},
    "Felix Auger-Aliassime": {"hard": 0.80, "clay": 0.70, "grass": 0.80},
    "Nicolas Jarry":       {"hard": 0.72, "clay": 0.80, "grass": 0.65},
    "Jakub Mensik":        {"hard": 0.75, "clay": 0.70, "grass": 0.72},
    "Alexei Popyrin":      {"hard": 0.78, "clay": 0.65, "grass": 0.75},
    "Brandon Nakashima":   {"hard": 0.80, "clay": 0.65, "grass": 0.72},
    "Matteo Berrettini":   {"hard": 0.75, "clay": 0.78, "grass": 0.88},
    "Giovanni Mpetshi Perricard": {"hard": 0.75, "clay": 0.65, "grass": 0.82},
    "Jiri Lehecka":        {"hard": 0.75, "clay": 0.72, "grass": 0.72},
    "Flavio Cobolli":      {"hard": 0.68, "clay": 0.82, "grass": 0.62},
    "Nuno Borges":         {"hard": 0.70, "clay": 0.80, "grass": 0.68},
    "Luciano Darderi":     {"hard": 0.65, "clay": 0.85, "grass": 0.60},
    "Tallon Griekspoor":   {"hard": 0.75, "clay": 0.70, "grass": 0.72},
    "Miomir Kecmanovic":   {"hard": 0.70, "clay": 0.78, "grass": 0.65},
    "Roberto Bautista Agut": {"hard": 0.78, "clay": 0.80, "grass": 0.75},
    "Jordan Thompson":     {"hard": 0.78, "clay": 0.65, "grass": 0.75},
    "Alejandro Davidovich Fokina": {"hard": 0.68, "clay": 0.85, "grass": 0.65},
    "Lorenzo Sonego":      {"hard": 0.72, "clay": 0.75, "grass": 0.75},
    "Jan-Lennard Struff":  {"hard": 0.72, "clay": 0.72, "grass": 0.78},
    "Sebastian Korda":     {"hard": 0.80, "clay": 0.70, "grass": 0.72},
    "Mariano Navone":      {"hard": 0.65, "clay": 0.88, "grass": 0.58},
    "Tomas Machac":        {"hard": 0.75, "clay": 0.75, "grass": 0.72},
    "Frances Tiafoe":      {"hard": 0.78, "clay": 0.68, "grass": 0.78},
    # WTA Top-40+
    "Aryna Sabalenka":     {"hard": 0.92, "clay": 0.78, "grass": 0.80},
    "Iga Swiatek":         {"hard": 0.85, "clay": 0.97, "grass": 0.72},
    "Coco Gauff":          {"hard": 0.85, "clay": 0.80, "grass": 0.78},
    "Elena Rybakina":      {"hard": 0.85, "clay": 0.72, "grass": 0.92},
    "Jessica Pegula":      {"hard": 0.80, "clay": 0.70, "grass": 0.72},
    "Daria Kasatkina":     {"hard": 0.75, "clay": 0.82, "grass": 0.70},
    "Madison Keys":        {"hard": 0.82, "clay": 0.72, "grass": 0.72},
    "Barbora Krejcikova":  {"hard": 0.75, "clay": 0.80, "grass": 0.85},
    "Mirra Andreeva":      {"hard": 0.78, "clay": 0.80, "grass": 0.72},
    "Emma Navarro":        {"hard": 0.78, "clay": 0.72, "grass": 0.80},
    "Paula Badosa":        {"hard": 0.78, "clay": 0.82, "grass": 0.75},
    "Jasmine Paolini":     {"hard": 0.75, "clay": 0.85, "grass": 0.80},
    "Liudmila Samsonova":  {"hard": 0.82, "clay": 0.72, "grass": 0.72},
    "Donna Vekic":         {"hard": 0.78, "clay": 0.72, "grass": 0.80},
    "Karolina Muchova":    {"hard": 0.78, "clay": 0.82, "grass": 0.78},
    "Diana Shnaider":      {"hard": 0.78, "clay": 0.75, "grass": 0.72},
    "Anna Kalinskaya":     {"hard": 0.80, "clay": 0.72, "grass": 0.72},
    "Elise Mertens":       {"hard": 0.75, "clay": 0.75, "grass": 0.78},
    "Danielle Collins":    {"hard": 0.80, "clay": 0.72, "grass": 0.70},
    "Beatriz Haddad Maia": {"hard": 0.72, "clay": 0.85, "grass": 0.68},
    "Elina Svitolina":     {"hard": 0.78, "clay": 0.78, "grass": 0.78},
    "Marketa Vondrousova": {"hard": 0.72, "clay": 0.80, "grass": 0.88},
    "Ekaterina Alexandrova": {"hard": 0.80, "clay": 0.70, "grass": 0.72},
    "Veronika Kudermetova": {"hard": 0.78, "clay": 0.75, "grass": 0.72},
    "Victoria Azarenka":   {"hard": 0.82, "clay": 0.78, "grass": 0.75},
    "Yulia Putintseva":    {"hard": 0.72, "clay": 0.80, "grass": 0.68},
    "Maria Sakkari":       {"hard": 0.80, "clay": 0.78, "grass": 0.72},
    "Caroline Garcia":     {"hard": 0.80, "clay": 0.75, "grass": 0.80},
    "Amanda Anisimova":    {"hard": 0.80, "clay": 0.72, "grass": 0.78},
    "Naomi Osaka":         {"hard": 0.85, "clay": 0.68, "grass": 0.75},
    "Clara Tauson":        {"hard": 0.75, "clay": 0.72, "grass": 0.78},
    "Ons Jabeur":          {"hard": 0.78, "clay": 0.82, "grass": 0.88},
}

# Поверхность по ключевому слову турнира
TOURNAMENT_SURFACE = {
    "aus_open":      "hard",
    "australian":    "hard",
    "us_open":       "hard",
    "miami":         "hard",
    "indian_wells":  "hard",
    "madrid":        "clay",
    "rome":          "clay",
    "french_open":   "clay",
    "roland_garros": "clay",
    "barcelona":     "clay",
    "monte_carlo":   "clay",
    "wimbledon":     "grass",
    "queens":        "grass",
    "halle":         "grass",
    "eastbourne":    "grass",
}


def get_ranking(player_name: str, tour: str = "atp") -> int:
    """Возвращает рейтинг игрока. Если не найден — 200 (аутсайдер)."""
    db = ATP_RANKINGS if tour == "atp" else WTA_RANKINGS
    # Точное совпадение
    if player_name in db:
        return db[player_name]
    # Совпадение только по фамилии (последнее слово) — НЕ по имени
    # чтобы избежать "Tatjana Maria" → "Maria Sakkari"
    player_last = player_name.lower().split()[-1] if player_name.split() else ""
    if len(player_last) >= 4:
        for name, rank in db.items():
            db_last = name.lower().split()[-1]
            if db_last == player_last:
                return rank
    return 200


def get_surface_strength(player_name: str, surface: str) -> float:
    """Возвращает индекс силы игрока на данной поверхности (0.0-1.0)."""
    data = SURFACE_STRENGTH.get(player_name)
    if data:
        return data.get(surface, 0.65)
    # Если нет данных — среднее
    return 0.65


def detect_surface(sport_key: str) -> str:
    """Определяет поверхность по ключу турнира."""
    key_lower = sport_key.lower()
    for keyword, surface in TOURNAMENT_SURFACE.items():
        if keyword in key_lower:
            return surface
    return "hard"  # по умолчанию


def detect_tour(sport_key: str) -> str:
    """ATP или WTA по ключу турнира."""
    return "wta" if "wta" in sport_key.lower() else "atp"


# ── Персистентный surface ELO (обновляется после каждого матча) ───────────────
import json as _json
import os as _os
import math as _math

_ELO_FILE = _os.path.join(_os.path.dirname(__file__), "..", "..", "elo_tennis.json")
_ELO_K    = 24   # константа Elo для тенниса (чуть выше командных видов)
_surface_elo_cache: dict = {}


def _load_surface_elo() -> dict:
    """Загружает surface ELO из файла. Структура: {player: {hard: elo, clay: elo, grass: elo}}"""
    global _surface_elo_cache
    try:
        with open(_ELO_FILE, "r", encoding="utf-8") as f:
            _surface_elo_cache = _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        _surface_elo_cache = {}
    return _surface_elo_cache


def _save_surface_elo(data: dict):
    """Сохраняет surface ELO на диск."""
    tmp = _ELO_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        _os.replace(tmp, _ELO_FILE)
    except Exception:
        pass


def get_surface_elo(player: str, surface: str, tour: str = "atp") -> float:
    """
    Возвращает surface ELO игрока. Приоритет:
    1. Сохранённый (обновлённый после матчей)
    2. Вычисленный из rank + surface_strength
    """
    if not _surface_elo_cache:
        _load_surface_elo()
    stored = _surface_elo_cache.get(player, {})
    if surface in stored:
        return float(stored[surface])

    # Вычисляем из rank + surface_strength (initial value)
    rank = get_ranking(player, tour)
    base = 2500 - 400 * _math.log10(max(1, rank))
    strength = get_surface_strength(player, surface)
    adjustment = (strength - 0.65) * 600
    return round(base + adjustment, 1)


def update_surface_elo(winner: str, loser: str, surface: str, tour: str = "atp"):
    """
    Обновляет surface ELO после матча (winner победил loser на surface).
    Вызывается из results_tracker после каждого засчитанного результата.
    """
    if not _surface_elo_cache:
        _load_surface_elo()

    w_elo = get_surface_elo(winner, surface, tour)
    l_elo = get_surface_elo(loser,  surface, tour)

    exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400))
    exp_l = 1.0 - exp_w

    new_w = round(w_elo + _ELO_K * (1.0 - exp_w), 1)
    new_l = round(l_elo + _ELO_K * (0.0 - exp_l), 1)

    _surface_elo_cache.setdefault(winner, {})[surface] = new_w
    _surface_elo_cache.setdefault(loser,  {})[surface] = new_l
    _save_surface_elo(_surface_elo_cache)


# Загружаем при импорте
_load_surface_elo()
