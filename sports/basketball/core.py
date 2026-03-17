# -*- coding: utf-8 -*-
"""
Баскетбольный движок CHIMERA.
Стратегия:
  - ELO 35% + Odds 35% + Home Court 15% + Form 15%
  - Тоталы: Over/Under с анализом темпа и тренда
  - Back-to-back штраф: -5% если команда играла вчера
  - Форма: последние 5 игр из The Odds API /scores/
"""

import os
import requests
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

try:
    from config import THE_ODDS_API_KEY
except ImportError:
    THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASKETBALL_LEAGUES = [
    ("basketball_nba",        "🏀 NBA"),
    ("basketball_euroleague", "🏆 Евролига"),
]

# ── ELO рейтинги NBA (сезон 2025-26) ─────────────────────────────────────────
NBA_ELO = {
    # Запад
    "Oklahoma City Thunder":      1770,
    "Minnesota Timberwolves":     1720,
    "Houston Rockets":            1710,
    "Los Angeles Lakers":         1680,
    "LA Lakers":                  1680,
    "Denver Nuggets":             1665,
    "Dallas Mavericks":           1650,
    "Golden State Warriors":      1655,
    "Memphis Grizzlies":          1640,
    "Los Angeles Clippers":       1625,
    "LA Clippers":                1625,
    "Phoenix Suns":               1610,
    "Sacramento Kings":           1595,
    "New Orleans Pelicans":       1580,
    "Utah Jazz":                  1530,
    "San Antonio Spurs":          1520,
    "Portland Trail Blazers":     1525,
    # Восток
    "Cleveland Cavaliers":        1760,
    "Boston Celtics":             1750,
    "New York Knicks":            1700,
    "Indiana Pacers":             1690,
    "Milwaukee Bucks":            1660,
    "Miami Heat":                 1645,
    "Orlando Magic":              1630,
    "Atlanta Hawks":              1615,
    "Chicago Bulls":              1600,
    "Detroit Pistons":            1590,
    "Philadelphia 76ers":         1575,
    "Toronto Raptors":            1560,
    "Brooklyn Nets":              1545,
    "Charlotte Hornets":          1535,
    "Washington Wizards":         1510,
}

# ── ELO рейтинги Евролиги (сезон 2025-26) ────────────────────────────────────
EUROLEAGUE_ELO = {
    "FC Barcelona":               1770,
    "Real Madrid":                1755,
    "Olympiacos":                 1730,
    "Fenerbahce SK":              1715,
    "Fenerbahce Beko":            1715,
    "Panathinaikos":              1700,
    "Bayern Munich":              1680,
    "Anadolu Efes":               1660,
    "Efes":                       1660,
    "Maccabi Tel Aviv":           1645,
    "KK Partizan NIS":            1620,
    "Partizan":                   1620,
    "Paris Basketball":           1600,
    "Valencia Basket":            1585,
    "Baskonia":                   1570,
    "AS Monaco":                  1555,
    "Hapoel Tel Aviv":            1540,
    "Virtus Bologna":             1525,
    "Alba Berlin":                1510,
    "Zalgiris Kaunas":            1495,
    "Dubai Basketball":           1475,
    "Crvena Zvezda":              1490,
}

# ── ELO рейтинги NBA G-League / международные ─────────────────────────────────
EXTRA_ELO = {
    # FIBA сборные (для международных турниров)
    "USA":                        1820,
    "Serbia":                     1760,
    "Germany":                    1750,
    "Australia":                  1740,
    "Spain":                      1730,
    "France":                     1720,
    "Canada":                     1710,
    "Greece":                     1700,
    "Slovenia":                   1695,
    "Latvia":                     1680,
    "Lithuania":                  1670,
    "Croatia":                    1660,
    "Italy":                      1650,
    "Argentina":                  1640,
    "Brazil":                     1620,
    "Puerto Rico":                1600,
    # Lega Basket (Италия)
    "EA7 Emporio Armani Milan":   1640,
    "Olimpia Milano":             1640,
    "Dolomiti Energia Trento":    1580,
    "Germani Brescia":            1570,
    # BSL (Турция)
    "Galatasaray":                1590,
    "Besiktas":                   1570,
    "Karsiyaka":                  1550,
}

DEFAULT_ELO       = 1550
HOME_ADVANTAGE_ELO = 50    # домашний корт в баскетболе очень важен
HOME_WIN_RATE_NBA  = 0.595  # исторически НБА: 59.5% дома
B2B_PENALTY        = 0.05   # -5% вероятности за back-to-back

# Градуированный штраф за усталость (дней отдыха → поправка к вероятности)
# 0 дней (сегодня) = -7%, 1 день = -5%, 2 дня = -2%, 3+ = 0%
REST_PENALTY = {0: -0.07, 1: -0.05, 2: -0.02}

# ── Кэш формы и back-to-back (обновляется при каждом запросе) ─────────────────
_form_cache: dict = {}
_b2b_cache:  dict = {}
_rest_days_cache: dict = {}  # team -> days since last game
_cache_ts:   float = 0.0
_CACHE_TTL   = 3600  # 1 час


def _get_elo(team: str, league_key: str = "") -> int:
    """Ищет ELO: сначала файл (после калибровки), потом захардкоженные таблицы."""
    # Попытка загрузить из откалиброванного файла
    try:
        import json
        if os.path.exists("elo_basketball.json"):
            with open("elo_basketball.json", "r", encoding="utf-8") as f:
                live = json.load(f)
            if team in live:
                return live[team]
    except Exception:
        pass

    if "euroleague" in league_key:
        return EUROLEAGUE_ELO.get(team, DEFAULT_ELO)
    return (
        NBA_ELO.get(team)
        or EUROLEAGUE_ELO.get(team)
        or EXTRA_ELO.get(team)
        or DEFAULT_ELO
    )


def _fetch_recent_scores(league_key: str) -> list:
    """Загружает результаты последних 3 дней из The Odds API /scores/."""
    if not THE_ODDS_API_KEY:
        return []
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{league_key}/scores/",
            params={"apiKey": THE_ODDS_API_KEY, "daysFrom": 3},
            timeout=10,
        )
        if r.ok:
            return [m for m in r.json() if m.get("completed")]
    except Exception as e:
        logger.warning(f"[Basketball] scores fetch error: {e}")
    return []


def _build_form_and_b2b(league_key: str):
    """
    Заполняет _form_cache и _b2b_cache на основе последних результатов.
    form_cache[team] = "WWLWL" (последние 5, слева = новее)
    b2b_cache[team]  = True если последний матч был вчера или сегодня
    """
    global _form_cache, _b2b_cache, _cache_ts
    import time
    if time.time() - _cache_ts < _CACHE_TTL:
        return

    scores = _fetch_recent_scores(league_key)
    now = datetime.now(timezone.utc)

    team_results: dict = {}  # team -> [(dt, won)]

    for m in scores:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        raw_scores = m.get("scores") or []
        score_map  = {s["name"]: int(s["score"]) for s in raw_scores if s.get("name") and s.get("score")}
        h_score = score_map.get(home, 0)
        a_score = score_map.get(away, 0)
        if h_score == a_score:
            continue

        # Дата матча
        ct = m.get("commence_time", "")
        try:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except Exception:
            dt = now - timedelta(days=3)

        h_won = h_score > a_score
        team_results.setdefault(home, []).append((dt, h_won))
        team_results.setdefault(away, []).append((dt, not h_won))

    for team, results in team_results.items():
        results.sort(key=lambda x: x[0], reverse=True)  # новее первым
        # Форма: последние 5 с экспоненциальным весом (последний матч = 1.0, -1 = 0.7, -2 = 0.5 …)
        weights = [1.0, 0.7, 0.5, 0.35, 0.25]
        form_str = "".join("W" if w else "L" for _, w in results[:5])
        _form_cache[team] = form_str
        # B2B + дней отдыха
        if results:
            last_dt = results[0][0]
            hours_ago = (now - last_dt).total_seconds() / 3600
            days_rest = int(hours_ago / 24)
            _b2b_cache[team] = hours_ago < 36
            _rest_days_cache[team] = days_rest

    _cache_ts = __import__("time").time()


def get_team_form(team: str, league_key: str = "") -> str:
    """Возвращает строку формы 'WWLWL' или '' если нет данных."""
    _build_form_and_b2b(league_key)
    return _form_cache.get(team, "")


def is_back_to_back(team: str, league_key: str = "") -> bool:
    """True если команда играла вчера/сегодня."""
    _build_form_and_b2b(league_key)
    return _b2b_cache.get(team, False)


def get_rest_days(team: str, league_key: str = "") -> int:
    """Возвращает количество дней отдыха с последней игры (0 = сегодня, 99 = нет данных)."""
    _build_form_and_b2b(league_key)
    return _rest_days_cache.get(team, 99)


def _fatigue_penalty(team: str, league_key: str = "") -> float:
    """
    Штраф за усталость на основе дней отдыха.
    0 дней = -7%, 1 день = -5%, 2 дня = -2%, 3+ = 0%
    """
    days = get_rest_days(team, league_key)
    return REST_PENALTY.get(days, 0.0)


def elo_win_prob(home: str, away: str, league_key: str = "") -> tuple:
    """Вероятность победы по ELO с учётом домашнего корта."""
    h_elo = _get_elo(home, league_key) + HOME_ADVANTAGE_ELO
    a_elo = _get_elo(away, league_key)
    h_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    return round(h_prob, 3), round(1 - h_prob, 3)


def get_basketball_matches(league_key: str) -> list:
    """Загружает матчи из The Odds API для баскетбола."""
    if not THE_ODDS_API_KEY:
        return []
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/",
            params={
                "apiKey":      THE_ODDS_API_KEY,
                "regions":     "eu",
                "markets":     "h2h,totals,spreads",
                "oddsFormat":  "decimal",
            },
            timeout=12,
        )
        if not r.ok:
            logger.warning(f"[Basketball] {league_key} → {r.status_code}")
            return []

        now    = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=2)).isoformat()[:19]
        matches = [m for m in r.json() if m.get("commence_time", "") > cutoff]
        logger.info(f"[Basketball] {league_key}: {len(matches)} матчей")
        return matches[:25]
    except Exception as e:
        logger.error(f"[Basketball] Ошибка {league_key}: {e}")
        return []


def get_basketball_odds(match: dict) -> dict:
    """Извлекает H2H, тотал и фору из матча. Предпочтение Pinnacle."""
    result = {
        "home_win": 0.0, "away_win": 0.0,
        "over": 0.0, "under": 0.0, "total_line": 0.0,
        "spread_home": 0.0, "spread_home_odds": 0.0,
        "spread_away": 0.0, "spread_away_odds": 0.0,
        "bookmaker": "",
    }

    PREFERRED = ["pinnacle", "betfair", "betsson", "1xbet", "unibet"]
    def _pri(bm):
        n = bm.get("key", "").lower()
        for i, p in enumerate(PREFERRED):
            if p in n:
                return i
        return 99

    bookmakers = sorted(match.get("bookmakers", []), key=_pri)
    for bm in bookmakers:
        bm_title = bm.get("title", "")
        for market in bm.get("markets", []):
            if market["key"] == "h2h" and result["home_win"] == 0:
                for o in market["outcomes"]:
                    p = float(o.get("price", 0))
                    if p < 1.02:
                        continue
                    if o["name"] == match.get("home_team"):
                        result["home_win"] = p
                        result["bookmaker"] = bm_title
                    elif o["name"] == match.get("away_team"):
                        result["away_win"] = p

            elif market["key"] == "totals" and result["over"] == 0:
                for o in market["outcomes"]:
                    if o.get("name") == "Over":
                        result["over"]       = float(o.get("price", 0))
                        result["total_line"] = float(o.get("point", 0))
                    elif o.get("name") == "Under":
                        result["under"] = float(o.get("price", 0))

            elif market["key"] == "spreads" and result["spread_home"] == 0:
                for o in market["outcomes"]:
                    if o["name"] == match.get("home_team"):
                        result["spread_home"]      = float(o.get("point", 0))
                        result["spread_home_odds"] = float(o.get("price", 0))
                    elif o["name"] == match.get("away_team"):
                        result["spread_away"]      = float(o.get("point", 0))
                        result["spread_away_odds"] = float(o.get("price", 0))

    return result


def _implied_prob(odds: float) -> float:
    return 1 / odds if odds > 1.02 else 0.0


def calculate_basketball_win_prob(home: str, away: str,
                                   odds: dict = None,
                                   league_key: str = "") -> dict:
    """
    Финальная вероятность победы:
      ELO 35% + Bookmaker Odds 35% + Form/Context 15% + Home Court 15%
      + back-to-back штраф -5% если команда играла вчера
    """
    # Загружаем актуальные веса из signal_engine если есть
    weight_elo  = 0.35
    weight_odds = 0.35
    weight_home = 0.15
    weight_form = 0.15
    try:
        from signal_engine import BASKETBALL_CFG
        weight_elo  = BASKETBALL_CFG.get("weight_elo",  0.35)
        weight_odds = BASKETBALL_CFG.get("weight_odds", 0.35)
        weight_home = BASKETBALL_CFG.get("weight_home", 0.15)
        weight_form = BASKETBALL_CFG.get("weight_form", 0.15)
    except Exception:
        pass

    h_elo_prob, a_elo_prob = elo_win_prob(home, away, league_key)

    # Implied probability из коэффициентов (убираем маржу)
    h_imp = _implied_prob(odds.get("home_win", 0) if odds else 0)
    a_imp = _implied_prob(odds.get("away_win", 0) if odds else 0)
    if h_imp > 0 and a_imp > 0:
        total_imp   = h_imp + a_imp
        h_odds_prob = h_imp / total_imp
        a_odds_prob = a_imp / total_imp
    else:
        h_odds_prob = h_elo_prob
        a_odds_prob = a_elo_prob

    # Форма команд
    h_form = get_team_form(home, league_key)
    a_form = get_team_form(away, league_key)

    def _form_factor(form_str: str) -> float:
        """0.45–0.55 в зависимости от формы."""
        if not form_str:
            return 0.50
        wins = form_str.count("W")
        return 0.40 + wins * 0.03  # 0W=0.40 … 5W=0.55

    h_form_factor = _form_factor(h_form)
    a_form_factor = _form_factor(a_form)
    # Нормализуем форму в вероятность победы
    total_form = h_form_factor + a_form_factor
    h_form_prob = h_form_factor / total_form if total_form > 0 else 0.50

    # Домашнее преимущество
    home_factor = HOME_WIN_RATE_NBA  # 0.595

    # Финальная вероятность (веса из BASKETBALL_CFG)
    h_prob = (
        h_elo_prob  * weight_elo  +
        h_odds_prob * weight_odds +
        home_factor * weight_home +
        h_form_prob * weight_form
    )

    # Градуированный штраф за усталость (заменяет плоский B2B_PENALTY)
    h_fatigue = _fatigue_penalty(home, league_key)
    a_fatigue = _fatigue_penalty(away, league_key)
    # Применяем разницу: если оба устали одинаково — нейтрально
    h_prob += (a_fatigue - h_fatigue)  # хозяин получает выгоду если гость устал больше

    h_prob = min(0.95, max(0.05, round(h_prob, 3)))
    a_prob = round(1 - h_prob, 3)

    h_elo = _get_elo(home, league_key)
    a_elo = _get_elo(away, league_key)
    elo_gap = h_elo - a_elo

    # EV расчёт
    h_ev = round((h_prob * (odds["home_win"] if odds else 0) - 1) * 100, 1) if (odds and odds.get("home_win", 0) > 1.02) else 0.0
    a_ev = round((a_prob * (odds["away_win"] if odds else 0) - 1) * 100, 1) if (odds and odds.get("away_win", 0) > 1.02) else 0.0

    # Анализ тотала
    total_analysis = _analyze_total(h_prob, a_prob, odds, elo_gap)

    # Сигнал
    best_ev   = max(h_ev, a_ev)
    best_pick = home if h_ev >= a_ev else away
    best_odds = (odds["home_win"] if h_ev >= a_ev else odds["away_win"]) if odds else 0
    if best_ev > 5:
        bet_signal = "✅ СТАВИТЬ"
    elif best_ev > 0:
        bet_signal = "⚠️ СЛАБЫЙ СИГНАЛ"
    else:
        bet_signal = "⏸ ПРОПУСТИТЬ"

    return {
        "home_prob":      h_prob,
        "away_prob":      a_prob,
        "elo_home":       h_elo,
        "elo_away":       a_elo,
        "elo_gap":        elo_gap,
        "h_ev":           h_ev,
        "a_ev":           a_ev,
        "best_pick":      best_pick,
        "best_odds":      best_odds,
        "best_ev":        best_ev,
        "bet_signal":     bet_signal,
        "total_analysis": total_analysis,
        "h_elo_prob":     h_elo_prob,
        "h_odds_prob":    h_odds_prob,
        "home_form":       h_form,
        "away_form":       a_form,
        "home_b2b":        is_back_to_back(home, league_key),
        "away_b2b":        is_back_to_back(away, league_key),
        "home_rest_days":  get_rest_days(home, league_key),
        "away_rest_days":  get_rest_days(away, league_key),
        "home_fatigue":    h_fatigue,
        "away_fatigue":    a_fatigue,
    }


def _analyze_total(h_prob: float, a_prob: float, odds: dict, elo_gap: int) -> dict:
    """
    Анализ тотала (Over/Under).
    Равные команды → Over. Явный фаворит → Under.
    """
    if not odds or not odds.get("total_line"):
        return {}

    line  = odds["total_line"]
    over  = odds["over"]
    under = odds["under"]

    if not over or not under:
        return {}

    balance = 1 - abs(h_prob - a_prob) * 2

    if balance > 0.8:
        lean   = "Over"
        reason = "Равные команды — обе будут атаковать"
    elif abs(elo_gap) > 150:
        lean   = "Under"
        reason = "Явный фаворит контролирует темп"
    else:
        lean   = "Over" if over <= under else "Under"
        reason = "Небольшой перекос линии"

    lean_odds = over if lean == "Over" else under
    lean_prob = 0.52 if balance > 0.8 else 0.51
    lean_ev   = round((lean_prob * lean_odds - 1) * 100, 1)

    return {
        "line":       line,
        "lean":       lean,
        "lean_odds":  lean_odds,
        "lean_ev":    lean_ev,
        "lean_prob":  round(lean_prob * 100, 1),
        "reason":     reason,
        "over_odds":  over,
        "under_odds": under,
    }
