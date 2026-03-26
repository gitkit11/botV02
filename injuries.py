"""
injuries.py — Модуль получения данных о травмах и дисквалификациях.

Использует GNews для поиска актуальных новостей о травмах,
затем GPT-4o-mini для структурированного извлечения имён игроков.

Кэширует результаты на 6 часов чтобы не спамить API.
"""

import json
import os
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Кэш: {team_name: (data, timestamp)}
_injury_cache: dict = {}
CACHE_TTL    = 6 * 3600   # 6 часов
_CACHE_FILE  = os.path.join(os.path.dirname(__file__), "injuries_cache.json")


def _load_cache():
    """Загружает кеш с диска при старте (переживает перезапуск)."""
    global _injury_cache
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cutoff = time.time() - CACHE_TTL
            _injury_cache = {k: tuple(v) for k, v in raw.items() if v[1] > cutoff}
    except Exception:
        _injury_cache = {}


def _save_cache():
    """Сохраняет кеш на диск."""
    try:
        tmp = _CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in _injury_cache.items()},
                      f, ensure_ascii=False)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass


_load_cache()  # загружаем при импорте


def _cache_get(team: str):
    if team in _injury_cache:
        data, ts = _injury_cache[team]
        if (time.time() - ts) < CACHE_TTL:
            return data
    return None


def _cache_set(team: str, data):
    _injury_cache[team] = (data, time.time())
    _save_cache()


def fetch_injury_news(team_name: str, max_results: int = 8) -> list:
    """Ищет свежие новости о травмах команды через GNews.
    Фильтрует статьи — оставляет только те, где упоминается сама команда.
    """
    try:
        from gnews import GNews
        gn = GNews(language='en', country='US', max_results=max_results * 2)
        articles = gn.get_news(f"{team_name} injury suspension doubt miss")
        if not articles:
            return []

        # Оставляем только статьи, где имя команды реально упомянуто в заголовке/описании
        team_lower = team_name.lower()
        # Короткие варианты имени для поиска ("Leeds United" → ["leeds", "united", "leeds united"])
        name_parts = [team_lower] + [p for p in team_lower.split() if len(p) > 3]
        filtered = []
        for art in articles:
            text = (art.get("title", "") + " " + art.get("description", "")).lower()
            if any(part in text for part in name_parts):
                filtered.append(art)
        logger.info(f"[Травмы] {team_name}: {len(articles)} статей → {len(filtered)} релевантных")
        return filtered[:max_results]
    except Exception as e:
        logger.warning(f"[Травмы] GNews ошибка для {team_name}: {e}")
        return []


def extract_injuries_with_ai(team_name: str, news_articles: list) -> dict:
    """
    Использует GPT-4o-mini для извлечения структурированных данных о травмах
    из новостных заголовков.
    Возвращает: {'injured': [...], 'suspended': [...], 'doubts': [...], 'returning': [...]}
    """
    if not news_articles:
        return {"injured": [], "suspended": [], "doubts": [], "returning": []}

    try:
        from openai import OpenAI
        import os
        # Читаем ключ из config.py или переменной среды
        try:
            from config import OPENAI_API_KEY as _oai_key
        except ImportError:
            _oai_key = os.environ.get('OPENAI_API_KEY', '')
        if not _oai_key:
            raise ValueError("OPENAI_API_KEY не задан")
        client = OpenAI(api_key=_oai_key)

        # Собираем заголовки и описания
        headlines = []
        for art in news_articles[:8]:
            title = art.get("title", "")
            desc = art.get("description", "") or ""
            if title:
                headlines.append(f"- {title}. {desc[:100]}")

        news_text = "\n".join(headlines)

        prompt = f"""You are analyzing injury news specifically for {team_name} football club.

Headlines:
{news_text}

CRITICAL RULES:
1. Only include players who play FOR {team_name} — ignore all players from other clubs
2. If a headline mentions a player from another team, skip it entirely
3. Only include players explicitly linked to {team_name} in the same sentence/headline
4. When in doubt — exclude the player (false positives are worse than false negatives)

Return a JSON object:
- "injured": confirmed injured {team_name} players only
- "suspended": suspended/banned {team_name} players only
- "doubts": doubtful/uncertain {team_name} players only
- "returning": returning from injury {team_name} players only

Return ONLY valid JSON, no explanation.
Example: {{"injured": ["Player A"], "suspended": [], "doubts": ["Player B"], "returning": []}}"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        # Извлекаем JSON из ответа
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()

        data = json.loads(raw)
        return {
            "injured": data.get("injured", []),
            "suspended": data.get("suspended", []),
            "doubts": data.get("doubts", []),
            "returning": data.get("returning", []),
        }

    except Exception as e:
        logger.warning(f"[Травмы] AI-извлечение ошибка для {team_name}: {e}")
        return {"injured": [], "suspended": [], "doubts": [], "returning": []}


def get_team_injuries(team_name: str) -> dict:
    """
    Главная функция — получает данные о травмах/дисквалификациях команды.
    Использует кэш 6 часов.

    Возвращает:
    {
        'team': str,
        'injured': [...],
        'suspended': [...],
        'doubts': [...],
        'returning': [...],
        'total_missing': int,
        'impact': str,  # 'high' / 'medium' / 'low' / 'none'
        'news_count': int,
    }
    """
    cached = _cache_get(team_name)
    if cached:
        logger.info(f"[Травмы] Кэш для {team_name}")
        return cached

    logger.info(f"[Травмы] Запрашиваю данные для {team_name}...")
    print(f"[Травмы] Ищу новости о травмах: {team_name}...")

    articles = fetch_injury_news(team_name)
    print(f"[Травмы] Найдено статей: {len(articles)}")

    if not articles:
        result = {
            "team": team_name,
            "injured": [],
            "suspended": [],
            "doubts": [],
            "returning": [],
            "total_missing": 0,
            "impact": "none",
            "news_count": 0,
        }
        _cache_set(team_name, result)
        return result

    extracted = extract_injuries_with_ai(team_name, articles)

    injured = extracted.get("injured", [])
    suspended = extracted.get("suspended", [])
    doubts = extracted.get("doubts", [])
    returning = extracted.get("returning", [])

    total_missing = len(injured) + len(suspended)

    # Оцениваем влияние на матч
    if total_missing >= 4:
        impact = "high"
    elif total_missing >= 2:
        impact = "medium"
    elif total_missing >= 1 or len(doubts) >= 2:
        impact = "low"
    else:
        impact = "none"

    result = {
        "team": team_name,
        "injured": injured,
        "suspended": suspended,
        "doubts": doubts,
        "returning": returning,
        "total_missing": total_missing,
        "impact": impact,
        "news_count": len(articles),
    }

    _cache_set(team_name, result)
    print(f"[Травмы] {team_name}: травмы={injured}, дискв={suspended}, сомн={doubts}")
    return result


def format_injuries_block(home_team: str, away_team: str,
                           home_injuries: dict, away_injuries: dict) -> str:
    """
    Форматирует блок травм/дисквалификаций для отчёта Telegram.
    """
    def _format_team(team_name: str, data: dict) -> str:
        if not data:
            return f" {team_name}: данные недоступны"

        lines = []
        if data.get("injured"):
            lines.append(f"  🤕 Травмы: {', '.join(data['injured'])}")
        if data.get("suspended"):
            lines.append(f"  🟥 Дискв: {', '.join(data['suspended'])}")
        if data.get("doubts"):
            lines.append(f"  ❓ Под вопросом: {', '.join(data['doubts'])}")
        if data.get("returning"):
            lines.append(f"  ✅ Возвращаются: {', '.join(data['returning'])}")

        if not lines:
            return f" {team_name}: все игроки доступны ✅"

        impact = data.get("impact", "none")
        impact_icons = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "✅"}
        icon = impact_icons.get(impact, "")
        missing = data.get("total_missing", 0)
        header = f" {team_name} {icon} ({missing} точно не играет):" if missing > 0 else f" {team_name}:"
        return header + "\n" + "\n".join(lines)

    home_block = _format_team(home_team, home_injuries)
    away_block = _format_team(away_team, away_injuries)

    return f"🏥 *ТРАВМЫ И ДИСКВАЛИФИКАЦИИ:*\n{home_block}\n{away_block}"


def get_match_injuries(home_team: str, away_team: str) -> tuple:
    """
    Получает данные о травмах для обоих команд последовательно.
    Для параллельного запуска используй get_match_injuries_async().
    Возвращает (home_injuries, away_injuries, formatted_block).
    """
    home_injuries = get_team_injuries(home_team)
    away_injuries = get_team_injuries(away_team)
    block = format_injuries_block(home_team, away_team, home_injuries, away_injuries)
    return home_injuries, away_injuries, block


# ── NBA/NHL Injury data via ESPN unofficial API ───────────────────────────────

_ESPN_NBA_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_ESPN_NHL_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
_espn_cache: dict = {}
_ESPN_TTL = 3600  # 1 час

# Алиасы: как ESPN называет команды → наши названия
_ESPN_NBA_ALIASES = {
    "LA Lakers": "Los Angeles Lakers",
    "LA Clippers": "Los Angeles Clippers",
    "GS Warriors": "Golden State Warriors",
    "NY Knicks": "New York Knicks",
    "OKC Thunder": "Oklahoma City Thunder",
    "SA Spurs": "San Antonio Spurs",
    "NO Pelicans": "New Orleans Pelicans",
    "NJ Nets": "Brooklyn Nets",
}


def _fetch_espn_injuries(url: str, cache_key: str) -> dict:
    """Загружает травмы с ESPN API. Возвращает {team_name: [player_names]}."""
    now = time.time()
    if cache_key in _espn_cache and now - _espn_cache[cache_key]["ts"] < _ESPN_TTL:
        return _espn_cache[cache_key]["data"]

    try:
        import requests as _req
        r = _req.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            return {}

        result = {}
        for item in r.json().get("injuries", []):
            team_name = item.get("team", {}).get("displayName", "")
            team_name = _ESPN_NBA_ALIASES.get(team_name, team_name)
            player_name = item.get("athlete", {}).get("displayName", "")
            status = item.get("status", "").lower()
            # Только точно не играющие (out/doubtful/questionable)
            if player_name and status in ("out", "doubtful", "questionable"):
                result.setdefault(team_name, []).append({
                    "name": player_name,
                    "status": status,
                })

        _espn_cache[cache_key] = {"data": result, "ts": now}
        logger.info(f"[ESPN Injuries] Загружено команд: {len(result)}")
        return result
    except Exception as e:
        logger.warning(f"[ESPN Injuries] Ошибка {url}: {e}")
        return {}


def get_nba_injuries(team_name: str) -> dict:
    """
    Получает травмы NBA из ESPN API. Без API-ключа, бесплатно.
    Кэш 1 час. Возвращает тот же формат что get_team_injuries().
    """
    cached = _cache_get(f"nba_{team_name}")
    if cached:
        return cached

    all_injuries = _fetch_espn_injuries(_ESPN_NBA_INJURIES_URL, "nba")

    # Ищем команду (точное совпадение или частичное)
    team_lower = team_name.lower()
    team_data = []
    for espn_team, players in all_injuries.items():
        if team_lower in espn_team.lower() or espn_team.lower() in team_lower:
            team_data = players
            break

    injured   = [p["name"] for p in team_data if p["status"] == "out"]
    doubts    = [p["name"] for p in team_data if p["status"] in ("doubtful", "questionable")]
    missing   = len(injured)
    impact    = "high" if missing >= 2 else ("medium" if missing == 1 else ("low" if doubts else "none"))

    result = {
        "team": team_name,
        "injured": injured,
        "suspended": [],
        "doubts": doubts,
        "returning": [],
        "total_missing": missing,
        "impact": impact,
        "news_count": len(team_data),
        "source": "espn",
    }
    _cache_set(f"nba_{team_name}", result)
    return result


def get_nhl_injuries(team_name: str) -> dict:
    """
    Получает травмы NHL из ESPN API. Без API-ключа, бесплатно.
    Особенно важно для вратарей.
    """
    cached = _cache_get(f"nhl_{team_name}")
    if cached:
        return cached

    all_injuries = _fetch_espn_injuries(_ESPN_NHL_INJURIES_URL, "nhl")

    team_lower = team_name.lower()
    team_data = []
    for espn_team, players in all_injuries.items():
        if team_lower in espn_team.lower() or espn_team.lower() in team_lower:
            team_data = players
            break

    injured  = [p["name"] for p in team_data if p["status"] == "out"]
    doubts   = [p["name"] for p in team_data if p["status"] in ("doubtful", "questionable")]
    missing  = len(injured)
    impact   = "high" if missing >= 2 else ("medium" if missing == 1 else ("low" if doubts else "none"))

    result = {
        "team": team_name,
        "injured": injured,
        "suspended": [],
        "doubts": doubts,
        "returning": [],
        "total_missing": missing,
        "impact": impact,
        "news_count": len(team_data),
        "source": "espn",
    }
    _cache_set(f"nhl_{team_name}", result)
    return result


async def get_match_injuries_async(home_team: str, away_team: str) -> tuple:
    """
    Асинхронная версия: запрашивает травмы для обоих команд параллельно.
    Сокращает время с ~20 сек до ~10 сек.
    """
    import asyncio
    home_injuries, away_injuries = await asyncio.gather(
        asyncio.to_thread(get_team_injuries, home_team),
        asyncio.to_thread(get_team_injuries, away_team)
    )
    block = format_injuries_block(home_team, away_team, home_injuries, away_injuries)
    return home_injuries, away_injuries, block
