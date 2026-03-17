# -*- coding: utf-8 -*-
"""
team_registry.py — Единый реестр данных команд CS2
====================================================
Единственный источник правды для:
- ELO рейтингов CS2 команд
- Алиасов и исторических переименований
- PandaScore ID команд
- LAN бонусов

Все остальные CS2-модули должны импортировать отсюда, а не дублировать данные.
Обновление данных — только здесь.
"""
from difflib import get_close_matches

# ─── ELO рейтинги (обновлено 2024-2025) ──────────────────────────────────────
# Источник: HLTV Ranking + крупные турниры
CS2_ELO: dict[str, int] = {
    # ── Tier-1 ────────────────────────────────────────────────────────────────
    "Team Vitality":            1820,
    "Team Spirit":              1800,
    "Natus Vincere":            1780,
    "FaZe Clan":                1760,
    "G2 Esports":               1750,
    "MOUZ":                     1740,
    # ── High Tier-1 / Low Tier-1 ─────────────────────────────────────────────
    "Heroic":                   1700,
    "Cloud9":                   1680,
    "Team Liquid":              1670,
    "Astralis":                 1660,
    "ENCE":                     1640,
    "FURIA":                    1620,
    "BIG":                      1600,
    "Complexity":               1580,
    "Eternal Fire":             1570,
    # ── Tier-2 ────────────────────────────────────────────────────────────────
    "OG":                       1540,
    "Fnatic":                   1530,
    "Virtus.pro":               1520,
    "3DMAX":                    1510,
    "NIP":                      1500,
    "Monte":                    1490,
    "paiN Gaming":              1480,
    "Imperial":                 1470,
    "9z Team":                  1450,
    "forZe":                    1440,
    "Aurora":                   1430,
    "SAW":                      1420,
    "SINNERS":                  1410,
    "Apeks":                    1400,
    "GamerLegion":              1390,
    "Sprout":                   1380,
    # ── Tier-2/3 ─────────────────────────────────────────────────────────────
    "TheMongolz":               1370,
    "Falcons":                  1360,
    "AMKAL":                    1350,
    "ECSTATIC":                 1340,
    "Metizport":                1330,
    "Sangal":                   1320,
    "MOUZ NXT":                 1310,
    "Lynn Vision":              1300,
    "TYLOO":                    1290,
    "Rare Atom":                1280,
    "ATOX":                     1270,
    "FlyQuest":                 1260,
    "Gaimin Gladiators":        1250,
    "Permitta":                 1220,
    "Natus Vincere Junior":     1210,
    "Spirit Academy":           1200,
    "aTTaX":                    1190,
    "HEET":                     1180,
}

# Дефолт для неизвестных команд — tier-3 уровень
DEFAULT_ELO: int = 1300

# ─── Алиасы и исторические переименования ────────────────────────────────────
# Все варианты → каноническое имя (ключ в CS2_ELO)
TEAM_ALIASES: dict[str, str] = {
    # Сокращения
    "Vitality":              "Team Vitality",
    "Spirit":                "Team Spirit",
    "NaVi":                  "Natus Vincere",
    "Na'Vi":                 "Natus Vincere",
    "NAVI":                  "Natus Vincere",
    "FaZe":                  "FaZe Clan",
    "G2":                    "G2 Esports",
    "Liquid":                "Team Liquid",
    "C9":                    "Cloud9",
    "VP":                    "Virtus.pro",
    "paiN":                  "paiN Gaming",
    "pain":                  "paiN Gaming",
    "pain Gaming":           "paiN Gaming",
    "EF":                    "Eternal Fire",
    "EternalFire":           "Eternal Fire",
    "9z":                    "9z Team",
    "CoL":                   "Complexity",
    "Complexity Gaming":     "Complexity",
    "Complexity GG":         "Complexity",
    "FURIA Esports":         "FURIA",
    "Ninjas in Pyjamas":     "NIP",
    "SINNERS Esports":       "SINNERS",
    "Imperial Esports":      "Imperial",
    "Imperial fe":           "Imperial",
    "FORZE":                 "forZe",
    "FORZE Esports":         "forZe",
    "AMKAL ESPORTS":         "AMKAL",
    "Sangal Esports":        "Sangal",
    "Permitta Esports":      "Permitta",
    "Lynn Vision Gaming":    "Lynn Vision",
    "The Mongolz":           "TheMongolz",
    "Team Falcons":          "Falcons",
    "NAVI Junior":           "Natus Vincere Junior",
    "Team Spirit Academy":   "Spirit Academy",
    "Alternate aTTaX":       "aTTaX",
    # Исторические переименования (старое имя → нынешнее)
    "mousesports":           "MOUZ",
    "mousesports CSGO":      "MOUZ",
    "Mouse":                 "MOUZ",
    "Outsiders":             "Virtus.pro",
    "Gambit Esports":        "Virtus.pro",
    "Gambit":                "Virtus.pro",
    "Virtus.pro Youth":      "Virtus.pro",
    "Hard Legion":           "forZe",
    "Hard Legion Esports":   "forZe",
    "GODSENT":               "Sprout",
    "Copenhagen Flames":     "Astralis",
    "Evil Geniuses":         "NIP",
    "OpTic Gaming":          "Complexity",
    "Global Esports":        "GamerLegion",
    "Extra Salt":            "Team Liquid",
    "B8 Esports":            "AMKAL",
    "Renegades":             "Imperial",
    "Team oNe":              "Imperial",
    "FaZe Academy":          "FaZe Clan",
    "GODSENT":               "Sprout",
}

# ─── PandaScore ID ────────────────────────────────────────────────────────────
# Ключи — канонические имена + алиасы для быстрого поиска
PANDASCORE_IDS: dict[str, int] = {
    "Team Vitality": 3455,       "Vitality": 3455,
    "FaZe Clan": 3212,           "FaZe": 3212,
    "G2 Esports": 3210,          "G2": 3210,
    "Natus Vincere": 3216,       "NaVi": 3216, "Na'Vi": 3216, "NAVI": 3216,
    "Team Spirit": 124523,       "Spirit": 124523,
    "MOUZ": 3240,                "mousesports": 3240, "Mouse": 3240,
    "Heroic": 3246,
    "Astralis": 3209,
    "ENCE": 3251,
    "Cloud9": 3223,              "C9": 3223,
    "Team Liquid": 3213,         "Liquid": 3213,
    "FURIA": 124530,             "FURIA Esports": 124530,
    "BIG": 3248,
    "Virtus.pro": 3217,          "VP": 3217, "Outsiders": 3217,
    "Eternal Fire": 125837,      "EternalFire": 125837,
    "3DMAX": 127915,
    "NIP": 3219,                 "Ninjas in Pyjamas": 3219,
    "Complexity": 3218,          "CoL": 3218, "Complexity Gaming": 3218,
    "OG": 3235,
    "Monte": 129614,
    "SAW": 126611,
    "SINNERS": 126612,           "SINNERS Esports": 126612,
    "Apeks": 126409,
    "GamerLegion": 127073,
    "Sprout": 126210,
    "paiN Gaming": 124527,       "paiN": 124527, "pain": 124527,
    "Imperial": 125000,          "Imperial Esports": 125000,
    "9z Team": 124600,           "9z": 124600,
    "forZe": 126500,             "FORZE": 126500, "FORZE Esports": 126500,
    "Aurora": 128000,
    "Fnatic": 3225,
    "Hard Legion": 126501,       "Hard Legion Esports": 126501,
    "HEET": 127000,
    "ECSTATIC": 128500,
    "Metizport": 129000,
    "AMKAL": 129500,             "AMKAL ESPORTS": 129500,
    "FlyQuest": 126800,
    "Falcons": 130000,           "Team Falcons": 130000,
    "Natus Vincere Junior": 126200, "NAVI Junior": 126200,
    "Spirit Academy": 128800,    "Team Spirit Academy": 128800,
    "Gaimin Gladiators": 129200,
    "MOUZ NXT": 128200,
    "Lynn Vision": 127400,       "Lynn Vision Gaming": 127400,
    "TYLOO": 124800,
    "TheMongolz": 130200,        "The Mongolz": 130200,
    "Rare Atom": 130400,
    "ATOX": 130100,
    "Sangal": 129800,            "Sangal Esports": 129800,
    "Permitta": 128900,          "Permitta Esports": 128900,
}

# ─── LAN бонус/штраф ─────────────────────────────────────────────────────────
# + = лучше на LAN/Major, - = хуже на LAN
LAN_BONUS: dict[str, float] = {
    "Team Spirit":      +0.05,  # доминирует на LAN/Major
    "Natus Vincere":    +0.04,  # исторически силён на Majors
    "G2 Esports":       +0.04,  # результаты на крупных ивентах
    "FaZe Clan":        +0.03,  # LAN-состав
    "Team Vitality":    +0.02,
    "MOUZ":             +0.02,
    "BIG":              -0.04,  # онлайн-специалист, хуже на LAN
    "ENCE":             -0.03,  # лучше в онлайн-лиге
    "Cloud9":           -0.02,
    "Heroic":           -0.02,
}


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def normalize_team_name(name: str) -> str:
    """
    Нормализует любое написание имени команды к каноническому ключу CS2_ELO.
    Порядок: точный → алиас → без учёта регистра → fuzzy.
    """
    if name in CS2_ELO:
        return name
    if name in TEAM_ALIASES:
        canonical = TEAM_ALIASES[name]
        if canonical in CS2_ELO:
            return canonical
    # Без учёта регистра
    name_lower = name.lower()
    for key in CS2_ELO:
        if key.lower() == name_lower:
            return key
    for alias, canonical in TEAM_ALIASES.items():
        if alias.lower() == name_lower and canonical in CS2_ELO:
            return canonical
    # Fuzzy matching — последний шанс
    all_keys = list(CS2_ELO.keys()) + list(TEAM_ALIASES.keys())
    matches = get_close_matches(name, all_keys, n=1, cutoff=0.78)
    if matches:
        matched = matches[0]
        if matched in CS2_ELO:
            return matched
        if matched in TEAM_ALIASES and TEAM_ALIASES[matched] in CS2_ELO:
            return TEAM_ALIASES[matched]
    return name  # оригинал если ничего не нашли


def get_elo(team_name: str) -> int:
    """Возвращает ELO рейтинг команды, с нормализацией имени."""
    return CS2_ELO.get(normalize_team_name(team_name), DEFAULT_ELO)


def get_pandascore_id(team_name: str) -> int | None:
    """Возвращает PandaScore ID команды по любому варианту имени."""
    if team_name in PANDASCORE_IDS:
        return PANDASCORE_IDS[team_name]
    canonical = normalize_team_name(team_name)
    return PANDASCORE_IDS.get(canonical)
