# -*- coding: utf-8 -*-
"""
handlers/common.py — вспомогательные функции, используемые всеми хендлерами.
Без зависимостей от main.py — только от внешних модулей.
"""
import asyncio
import logging
import time as _time

from state import (
    matches_cache, _league_matches_cache, _LEAGUE_CACHE_TTL,
    _current_league as _cur_league_ref,
)
from keyboards import FOOTBALL_LEAGUES
from math_model import (
    load_elo_ratings, load_team_form,
    elo_win_probabilities, get_form_string,
)

logger = logging.getLogger(__name__)

# Семафор: не более 5 параллельных AI-анализов
_ai_semaphore = asyncio.Semaphore(5)

# ELO рейтинги и форма команд — загружаем при импорте модуля
_elo_ratings = load_elo_ratings()
_team_form = load_team_form()
print(f"[common] ELO: {len(_elo_ratings)} команд | Форма: {len(_team_form)} команд")

# CS2 белый список лиг
CS2_WHITELIST_LEAGUES = [
    "ESL Pro League", "BLAST", "IEM", "PGL", "Majors", "ESL One",
    "CCT", "Game Masters", "Dust2.dk Ligaen", "Exort Series",
    "NODWIN Clutch Series", "Roman Imperium Cup",
    "Regional", "Open", "Online", "Qualifier", "Championship",
    "League", "Cup", "Series", "Division", "Masters",
]
CS2_TIER3_KEYWORDS = [
    "regional", "open", "qualifier", "division", "roman imperium",
    "nodwin", "exort", "dust2.dk", "game masters",
]

# Текущая выбранная лига (мутируется через get_matches)
_current_league = "soccer_epl"
_last_matches_refresh: float = 0.0


def get_matches(league: str = None, force: bool = False):
    """Получает список матчей через The Odds API (кэш 20 мин)."""
    global matches_cache, _last_matches_refresh, _current_league, _league_matches_cache
    if league:
        _current_league = league
    league_key = _current_league

    if not force:
        cached_entry = _league_matches_cache.get(league_key)
        if cached_entry and (_time.time() - cached_entry["ts"]) < _LEAGUE_CACHE_TTL:
            matches_cache[:] = cached_entry["matches"]
            return cached_entry["matches"]

    if force:
        try:
            from odds_cache import invalidate as _inv
            _inv(league_key)
        except ImportError:
            pass
    try:
        from odds_cache import get_odds as _get_odds
        from datetime import datetime, timezone, timedelta
        data = _get_odds(league_key, markets="h2h,totals,spreads")
        if data:
            now = datetime.now(timezone.utc)
            cutoff = (now - timedelta(hours=3)).isoformat()[:19]
            future = [m for m in data if m.get('commence_time', '') > cutoff]
            result = future[:20]
            _league_matches_cache[league_key] = {"matches": result, "ts": _time.time()}
            matches_cache[:] = result
            _last_matches_refresh = _time.time()
            league_name = dict(FOOTBALL_LEAGUES).get(league_key, league_key)
            print(f"[API] {league_name}: {len(result)} матчей.")
            return result
    except Exception as e:
        print(f"[API Ошибка] {e}")

    cached_entry = _league_matches_cache.get(league_key)
    if cached_entry:
        return cached_entry["matches"]
    return matches_cache


def get_bookmaker_odds(match_data: dict) -> dict:
    """Извлекает коэффициенты из данных матча (Pinnacle → шарп → все буки)."""
    result = {
        "home_win": 0, "draw": 0, "away_win": 0,
        "over_2_5": 0, "under_2_5": 0,
        "over_1_5": 0, "under_1_5": 0,
        "over_3_5": 0, "under_3_5": 0,
        "handicap_home": 0, "handicap_away": 0, "handicap_line": 0,
        "no_vig_home": 0.0, "no_vig_draw": 0.0, "no_vig_away": 0.0,
        "bookmakers_count": 0,
        "pinnacle_home": 0, "pinnacle_draw": 0, "pinnacle_away": 0,
    }

    SHARP_BOOKS = ["pinnacle", "betfair_ex", "betfair", "matchbook",
                   "smarkets", "lowvig", "betsson", "nordicbet", "marathonbet"]

    def _v(v):
        try:
            f = float(v)
            return f if f >= 1.02 else 0.0
        except Exception:
            return 0.0

    try:
        home_team = match_data.get("home_team", "")
        away_team = match_data.get("away_team", "")
        bookmakers = match_data.get("bookmakers", [])
        result["bookmakers_count"] = len(bookmakers)

        sharp_h, sharp_d, sharp_a = [], [], []
        all_h,   all_d,   all_a   = [], [], []

        for bm in bookmakers:
            bm_key = bm.get("key", "").lower()
            is_sharp = any(s in bm_key for s in SHARP_BOOKS)

            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    oc = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    h = _v(oc.get(home_team, 0))
                    a = _v(oc.get(away_team, 0))
                    d = _v(oc.get("Draw", 0))
                    if h and a and d:
                        all_h.append(h); all_d.append(d); all_a.append(a)
                        if is_sharp:
                            sharp_h.append(h); sharp_d.append(d); sharp_a.append(a)
                        if "pinnacle" in bm_key:
                            result["pinnacle_home"] = h
                            result["pinnacle_draw"] = d
                            result["pinnacle_away"] = a

                elif market.get("key") == "totals" and result["over_2_5"] == 0:
                    for o in market.get("outcomes", []):
                        pt    = o.get("point", 0)
                        name  = o.get("name", "")
                        price = _v(o.get("price", 0))
                        if not price:
                            continue
                        if pt == 2.5 and name == "Over"  and not result["over_2_5"]:
                            result["over_2_5"] = price
                        elif pt == 2.5 and name == "Under" and not result["under_2_5"]:
                            result["under_2_5"] = price
                        elif pt == 1.5 and name == "Over"  and not result["over_1_5"]:
                            result["over_1_5"] = price
                        elif pt == 1.5 and name == "Under" and not result["under_1_5"]:
                            result["under_1_5"] = price
                        elif pt == 3.5 and name == "Over"  and not result["over_3_5"]:
                            result["over_3_5"] = price
                        elif pt == 3.5 and name == "Under" and not result["under_3_5"]:
                            result["under_3_5"] = price

                elif market.get("key") == "spreads" and not result["handicap_home"]:
                    for o in market.get("outcomes", []):
                        name  = o.get("name", "")
                        price = _v(o.get("price", 0))
                        line  = o.get("point", 0)
                        if not price:
                            continue
                        if name == home_team:
                            result["handicap_home"] = price
                            result["handicap_line"] = line
                        elif name == away_team:
                            result["handicap_away"] = price

        src_h = sharp_h if sharp_h else all_h
        src_d = sharp_d if sharp_d else all_d
        src_a = sharp_a if sharp_a else all_a

        if src_h:
            src_h.sort(); src_d.sort(); src_a.sort()
            mid = len(src_h) // 2
            result["home_win"] = round(src_h[mid], 3)
            result["draw"]     = round(src_d[mid], 3)
            result["away_win"] = round(src_a[mid], 3)

        nv_h = result["pinnacle_home"] or result["home_win"]
        nv_d = result["pinnacle_draw"] or result["draw"]
        nv_a = result["pinnacle_away"] or result["away_win"]
        if nv_h and nv_d and nv_a:
            imp_h = 1 / nv_h
            imp_d = 1 / nv_d
            imp_a = 1 / nv_a
            total = imp_h + imp_d + imp_a
            if total > 0:
                result["no_vig_home"] = round(imp_h / total, 4)
                result["no_vig_draw"] = round(imp_d / total, 4)
                result["no_vig_away"] = round(imp_a / total, 4)

    except Exception as e:
        print(f"[API Ошибка коэффициентов] {e}")
    return result


def _blend_ai(base_home_prob: float, ai_results: list,
              home_team: str, away_team: str, ai_weight: float = 0.10) -> float:
    """Добавляет AI-вердикты к математической модели с заданным весом."""
    votes, total_conf = 0.0, 0.0
    h_low = home_team.lower()
    a_low = away_team.lower()
    win_keys  = {"home_win", "победа хозяев", "п1", "p1", "home"}
    loss_keys = {"away_win", "победа гостей", "п2", "p2", "away"}
    for res in ai_results:
        if not isinstance(res, dict):
            continue
        outcome = str(res.get("recommended_outcome", res.get("outcome", ""))).lower().strip()
        conf = float(res.get("confidence", res.get("final_confidence_percent", 50))) / 100.0
        if outcome in win_keys or any(t in outcome for t in [h_low[:5]]):
            votes += conf
        elif outcome in loss_keys or any(t in outcome for t in [a_low[:5]]):
            votes += 0.0
        else:
            continue
        total_conf += conf
    if total_conf == 0:
        return base_home_prob
    ai_home_prob = votes / total_conf
    return round(base_home_prob * (1 - ai_weight) + ai_home_prob * ai_weight, 3)


def blend_ai_verdicts(
    h_prob: float, a_prob: float,
    gpt_r: dict, llama_r: dict,
    home_team: str,
    ai_weight: float = 0.12,
) -> tuple:
    """
    Блендирует AI-вердикты GPT и Llama в базовую вероятность модели.
    Используется в basketball и hockey handler-ах после получения AI ответов.

    Возвращает (h_prob_blended, a_prob_blended).
    """
    votes_h  = 0.0
    total_conf = 0.0

    for res in [gpt_r, llama_r]:
        if not isinstance(res, dict):
            continue
        verdict = res.get("verdict", "")
        conf    = float(res.get("confidence", 0)) / 100.0
        if conf < 0.01 or not verdict:
            continue
        if verdict == "home_win":
            votes_h += conf
        elif verdict == "away_win":
            pass  # не голосует за хозяев
        else:
            continue
        total_conf += conf

    if total_conf < 0.01:
        return h_prob, a_prob

    ai_h      = votes_h / total_conf
    blended_h = round(h_prob * (1 - ai_weight) + ai_h * ai_weight, 4)
    blended_h = max(0.05, min(0.95, blended_h))
    blended_a = round(1 - blended_h, 4)
    return blended_h, blended_a


async def show_ai_thinking(msg, home: str, away: str, sport: str = "football"):
    """Живая анимация мышления агентов."""
    sport_icons = {"football": "⚽", "cs2": "🎮", "basketball": "🏀", "tennis": "🎾", "hockey": "🏒"}
    icon = sport_icons.get(sport, "🔮")

    sport_models = {
        "football": (
            "📐 Dixon-Coles\n"
            "📊 ELO + форма\n"
            "🎯 Пуассон / xG\n"
            "🧠 Prophet нейросеть\n"
            "📈 Линия букмекеров"
        ),
        "cs2": (
            "🗺️ MIS — анализ карт\n"
            "📊 ELO + LAN-коэффициент\n"
            "📋 Винрейт (last5 + last20)\n"
            "👤 Рейтинг игроков HLTV\n"
            "🤝 H2H история"
        ),
        "basketball": (
            "📊 ELO рейтинг\n"
            "📈 Линия букмекеров\n"
            "📋 Форма команды\n"
            "🏠 Домашний корт\n"
            "⚡ Back-to-back штраф"
        ),
        "hockey": (
            "📊 ELO рейтинг (NHL/SHL)\n"
            "📈 No-Vig линия (Pinnacle)\n"
            "📋 Форма команды\n"
            "🧊 Преимущество домашнего льда\n"
            "⚡ Back-to-back штраф"
        ),
        "tennis": (
            "🎾 ATP/WTA рейтинг → ELO\n"
            "🏟️ Специализация покрытия\n"
            "📋 Форма (последние матчи)\n"
            "🤝 H2H очные встречи"
        ),
    }

    models_block = sport_models.get(sport, "")
    base = (
        f"<b>{icon} {home}  <code>vs</code>  {away}</b>\n"
        f"<b>🔮 CHIMERA AI</b> — запускаю анализ...\n\n"
        f"<i>Модели:</i>\n{models_block}\n\n"
    )
    steps = [
        ("🐍", "Змея",  "считает математику..."),
        ("🦁", "Лев",   "читает новости и травмы..."),
        ("🐐", "Козёл", "взвешивает всё..."),
        ("🌀", "Тень",  "независимая проверка..."),
    ]
    active = []
    try:
        for emoji, name, action in steps:
            active.append(f"{emoji} <b>{name}:</b> {action}")
            await msg.edit_text(base + "\n".join(active), parse_mode="HTML")
            await asyncio.sleep(0.9)
        done = "\n".join(f"{e} <b>{n}:</b> <i>готово ✓</i>" for e, n, _ in steps)
        await msg.edit_text(
            base + done + "\n\n<i>⚡ Формирую итоговый отчёт...</i>",
            parse_mode="HTML"
        )
        await asyncio.sleep(0.6)
    except Exception as _e:
        logger.debug(f"[ignore] {_e}")
