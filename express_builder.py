# -*- coding: utf-8 -*-
"""
express_builder.py — Построитель экспресс-ставок на основе математического анализа.

Логика:
  1. Сканирует топ-5 футбольных лиг + NBA (без лишних API-запросов)
  2. Считает вероятности через ELO + неявные вероятности букмекеров
  3. Фильтрует события по EV + вероятности (без signal_engine — он требует AI)
  4. Строит 3 варианта: Надёжный (2 события), Средний (3), Рискованный (4-5)
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

# ── ELO рейтинги — загружаем один раз, обновляем не чаще раза в час ──────────
_elo_cache: dict = {}
_elo_cache_ts: float = 0.0
_ELO_CACHE_TTL = 3600  # 1 час

def _get_elo_ratings() -> dict:
    global _elo_cache, _elo_cache_ts
    if _elo_cache and (time.time() - _elo_cache_ts) < _ELO_CACHE_TTL:
        return _elo_cache
    try:
        from math_model import load_elo_ratings
        _elo_cache = load_elo_ratings()
        _elo_cache_ts = time.time()
    except Exception as e:
        logger.warning(f"[Экспресс] Не удалось загрузить ELO: {e}")
    return _elo_cache

try:
    from config import THE_ODDS_API_KEY
except ImportError:
    THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

# Только топ лиги — экономим API-квоту
SCAN_LEAGUES = [
    ("soccer_epl",                   "football", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 АПЛ"),
    ("soccer_spain_la_liga",         "football", "🇪🇸 Ла Лига"),
    ("soccer_germany_bundesliga",    "football", "🇩🇪 Бундеслига"),
    ("soccer_italy_serie_a",         "football", "🇮🇹 Серия А"),
    ("soccer_uefa_champs_league",    "football", "🏆 ЛЧ"),
    ("basketball_nba",               "basketball", "🏀 NBA"),
]

# Пороги — основной режим
MIN_PROB   = 0.52
MIN_EV     = 0.03   # 3%
MIN_ODDS   = 1.35
MAX_ODDS   = 3.50

# Fallback — если ничего не нашли, смягчаем
FALLBACK_MIN_PROB = 0.50
FALLBACK_MIN_EV   = 0.01


def _get_odds(match: dict) -> dict:
    result = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    try:
        bookmakers = match.get("bookmakers", [])
        PREFERRED = ["pinnacle", "betfair", "betsson", "1xbet", "bet365"]
        def _priority(bm):
            name = bm.get("key", "").lower()
            for i, p in enumerate(PREFERRED):
                if p in name:
                    return i
            return 99
        bookmakers = sorted(bookmakers, key=_priority)
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                home = match.get("home_team", "")
                away = match.get("away_team", "")
                for o in market.get("outcomes", []):
                    price = float(o.get("price", 0))
                    name  = o.get("name", "")
                    if price < 1.02:
                        continue
                    if name == home and not result["home_win"]:
                        result["home_win"] = price
                    elif name == away and not result["away_win"]:
                        result["away_win"] = price
                    elif name == "Draw" and not result["draw"]:
                        result["draw"] = price
            if result["home_win"] and result["away_win"]:
                break
    except Exception:
        pass
    return result


def _implied_probs(odds: dict) -> dict:
    """Неявные вероятности из котировок букмекеров (снимаем маржу)."""
    h = odds.get("home_win", 0)
    a = odds.get("away_win", 0)
    d = odds.get("draw", 0)
    if not h or not a:
        return {}
    raw_h = 1 / h
    raw_a = 1 / a
    raw_d = (1 / d) if d > 1.0 else 0.0
    total = raw_h + raw_a + raw_d
    if total <= 0:
        return {}
    return {
        "home": raw_h / total,
        "away": raw_a / total,
        "draw": raw_d / total,
    }


def _elo_probs(home: str, away: str, sport: str) -> dict:
    """ELO-вероятности как дополнительный сигнал."""
    try:
        if sport == "football":
            from math_model import elo_win_probabilities
            ratings = _get_elo_ratings()  # из кеша, не с диска каждый раз
            p = elo_win_probabilities(home, away, ratings)
            return {"home": p.get("home", 0), "away": p.get("away", 0), "draw": p.get("draw", 0)}
        elif sport == "basketball":
            from sports.basketball.core import _get_elo
            h_elo = _get_elo(home)
            a_elo = _get_elo(away)
            exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - 50) / 400))
            return {"home": exp_h, "away": 1 - exp_h, "draw": 0}
    except Exception:
        pass
    return {}


def _blend(implied: dict, elo: dict) -> dict:
    """Ансамбль: 65% букмекеры + 35% ELO."""
    result = {}
    for k in ("home", "away", "draw"):
        i = implied.get(k, 0)
        e = elo.get(k, 0)
        if e:
            result[k] = 0.65 * i + 0.35 * e
        else:
            result[k] = i
    return result


def _calc_ev(prob: float, odds: float) -> float:
    return prob * odds - 1.0


def _get_totals(match: dict, line: float = 2.5) -> dict:
    """Извлекает кэфы на тотал (Over/Under line) из данных букмекеров."""
    result = {"over": 0.0, "under": 0.0, "line": line}
    try:
        bookmakers = match.get("bookmakers", [])
        PREFERRED = ["pinnacle", "betfair", "betsson", "1xbet", "bet365"]
        def _priority(bm):
            name = bm.get("key", "").lower()
            for i, p in enumerate(PREFERRED):
                if p in name: return i
            return 99
        bookmakers = sorted(bookmakers, key=_priority)
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for o in market.get("outcomes", []):
                    point = float(o.get("point", 0))
                    if abs(point - line) > 0.1:
                        continue
                    name  = o.get("name", "").lower()
                    price = float(o.get("price", 0))
                    if price < 1.02: continue
                    if name == "over"  and not result["over"]:
                        result["over"]  = price
                        result["line"]  = point
                    elif name == "under" and not result["under"]:
                        result["under"] = price
                        result["line"]  = point
            if result["over"] and result["under"]:
                break
    except Exception:
        pass
    return result


def _scan_league(league: str, sport: str, league_name: str,
                 min_prob: float, min_ev: float) -> list:
    candidates = []
    try:
        from odds_cache import get_odds as _fetch_api_odds
        matches = _fetch_api_odds(league, markets="h2h,totals")
        if not matches:
            return []
        logger.info(f"[Экспресс] {league_name}: {len(matches)} матчей")
    except Exception as e:
        logger.warning(f"[Экспресс] {league_name}: {e}")
        return []

    for m in matches[:20]:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        ct   = m.get("commence_time", "")
        if not home or not away:
            continue

        odds = _get_odds(m)
        if not odds["home_win"] or not odds["away_win"]:
            continue

        implied = _implied_probs(odds)
        if not implied:
            continue

        elo = _elo_probs(home, away, sport)
        probs = _blend(implied, elo) if elo else implied

        # ── Победитель ────────────────────────────────────────────────────
        for outcome, prob, odd in [
            ("home_win", probs.get("home", 0), odds["home_win"]),
            ("away_win", probs.get("away", 0), odds["away_win"]),
        ]:
            if odd < MIN_ODDS or odd > MAX_ODDS:
                continue
            if prob < min_prob:
                continue
            ev = _calc_ev(prob, odd)
            if ev < min_ev:
                continue

            label = home if outcome == "home_win" else away
            sport_icon = "🏀" if sport == "basketball" else "⚽"
            candidates.append({
                "match":          f"{home} — {away}",
                "home":           home,
                "away":           away,
                "league":         league_name,
                "outcome":        outcome,
                "label":          f"{sport_icon} {label} (победа)",
                "prob":           round(prob, 4),
                "odds":           round(odd, 2),
                "ev":             round(ev, 4),
                "sport":          sport,
                "commence_time":  ct,
                "bet_type":       "winner",
            })

        # ── Тоталы (только футбол и баскетбол) ───────────────────────────
        if sport in ("football", "basketball"):
            total_line = 2.5 if sport == "football" else 220.5
            totals_odds = _get_totals(m, line=total_line)
            if totals_odds["over"] and totals_odds["under"]:
                actual_line = totals_odds["line"]
                # Расчёт вероятностей тотала из Poisson/ELO
                total_probs = _estimate_totals_prob(home, away, sport, actual_line, probs)
                for t_outcome, t_prob, t_odd in [
                    ("over",  total_probs.get("over",  0), totals_odds["over"]),
                    ("under", total_probs.get("under", 0), totals_odds["under"]),
                ]:
                    if not t_prob or not t_odd or t_odd < 1.30:
                        continue
                    t_ev = _calc_ev(t_prob, t_odd)
                    if t_prob < min_prob or t_ev < min_ev:
                        continue
                    if sport == "football":
                        label = f"⚽ Тотал {'Больше' if t_outcome=='over' else 'Меньше'} {actual_line}"
                    else:
                        label = f"🏀 Тотал {'Больше' if t_outcome=='over' else 'Меньше'} {actual_line}"
                    candidates.append({
                        "match":          f"{home} — {away}",
                        "home":           home,
                        "away":           away,
                        "league":         league_name,
                        "outcome":        f"total_{t_outcome}",
                        "label":          label,
                        "prob":           round(t_prob, 4),
                        "odds":           round(t_odd, 2),
                        "ev":             round(t_ev, 4),
                        "sport":          sport,
                        "commence_time":  ct,
                        "bet_type":       "total",
                        "total_line":     actual_line,
                    })

    return candidates


def _estimate_totals_prob(home: str, away: str, sport: str,
                          line: float, probs: dict) -> dict:
    """
    Оценивает вероятности тотала OVER/UNDER.
    Для футбола: Poisson по ожидаемым голам.
    Для баскетбола: на основе средних очков команды.
    """
    try:
        if sport == "football":
            from math_model import calculate_expected_goals, poisson_match_probabilities
            xg = calculate_expected_goals(home, away)
            if not xg:
                raise ValueError("xG is None")
            # calculate_expected_goals returns (home_xg, away_xg) tuple
            xg_home = xg[0] if isinstance(xg, (tuple, list)) else xg.get("home_xg", 1.3)
            xg_away = xg[1] if isinstance(xg, (tuple, list)) else xg.get("away_xg", 1.1)
            result = poisson_match_probabilities(xg_home, xg_away)
            if line == 2.5:
                return {"over": result.get("over_25", 0.5), "under": result.get("under_25", 0.5)}
            elif line == 1.5:
                return {"over": result.get("over_15", 0.7), "under": 1 - result.get("over_15", 0.7)}
            elif line == 3.5:
                return {"over": result.get("over_35", 0.3), "under": 1 - result.get("over_35", 0.3)}
    except Exception:
        pass

    # Fallback: используем баланс команд
    h = probs.get("home", 0.5)
    a = probs.get("away", 0.5)
    balance = 1 - abs(h - a) * 2  # 0 = явный фаворит, 1 = равные
    p_over = 0.40 + balance * 0.20  # равные → больше голов
    return {"over": round(p_over, 3), "under": round(1 - p_over, 3)}


def scan_all_matches() -> list:
    """Сканирует лиги и возвращает кандидатов, отсортированных по EV."""
    all_candidates = []
    for league, sport, name in SCAN_LEAGUES:
        all_candidates.extend(_scan_league(league, sport, name, MIN_PROB, MIN_EV))

    # Fallback — если мало кандидатов, смягчаем пороги
    if len(all_candidates) < 4:
        logger.info(f"[Экспресс] Fallback: нашли только {len(all_candidates)}, смягчаем пороги")
        all_candidates = []
        for league, sport, name in SCAN_LEAGUES:
            all_candidates.extend(
                _scan_league(league, sport, name, FALLBACK_MIN_PROB, FALLBACK_MIN_EV)
            )

    # Убираем дубликаты матчей, оставляем лучший исход
    seen = {}
    for c in sorted(all_candidates, key=lambda x: x["ev"], reverse=True):
        key = (c["home"], c["away"])
        if key not in seen:
            seen[key] = c
    unique = list(seen.values())
    unique.sort(key=lambda x: x["ev"], reverse=True)
    logger.info(f"[Экспресс] Итого уникальных кандидатов: {len(unique)}")
    return unique


def build_express_variants(candidates: list) -> dict:
    """Строит 3 варианта экспресса."""
    by_prob = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    by_ev   = sorted(candidates, key=lambda x: x["ev"],   reverse=True)

    def _make(events: list):
        if not events:
            return None
        prob  = 1.0
        odds  = 1.0
        for e in events:
            prob *= e["prob"]
            odds *= e["odds"]
        odds = round(odds, 2)
        prob = round(prob, 4)
        ev   = round(_calc_ev(prob, odds), 4)
        b    = odds - 1
        kelly = max(0.0, (prob * b - (1 - prob)) / b) if b > 0 else 0
        kelly_pct = round(min(kelly * 0.25, 0.05) * 100, 1)
        return {
            "events":        events,
            "combined_odds": odds,
            "combined_prob": prob,
            "ev":            ev,
            "kelly_pct":     kelly_pct,
        }

    # Надёжный: 2 события с самой высокой вероятностью
    safe = _make(by_prob[:2]) if len(by_prob) >= 2 else None

    # Средний: 3 события — топ по вероятности
    medium = _make(by_prob[:3]) if len(by_prob) >= 3 else None

    # Рискованный: до 5 событий — смешиваем prob + ev
    risky_pool = list(by_prob[:2])
    seen_r = {(e["home"], e["away"]) for e in risky_pool}
    for e in by_ev:
        if (e["home"], e["away"]) not in seen_r and len(risky_pool) < 5:
            risky_pool.append(e)
            seen_r.add((e["home"], e["away"]))
    risky = _make(risky_pool) if len(risky_pool) >= 4 else None

    return {
        "safe":              safe,
        "medium":            medium,
        "risky":             risky,
        "total_candidates":  len(candidates),
    }


def _stars(ev: float) -> str:
    if ev >= 0.15: return "⭐⭐⭐"
    if ev >= 0.07: return "⭐⭐"
    return "⭐"


def _fmt_match_time(ct: str) -> str:
    """Форматирует время матча для карточки экспресса."""
    if not ct:
        return ""
    try:
        from datetime import datetime, timezone, timedelta
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        dt_msk = dt.astimezone(timezone(timedelta(hours=3)))
        MONTHS = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
        today = datetime.now(timezone.utc).date()
        if dt_msk.date() == today:
            return f"Сегодня {dt_msk.strftime('%H:%M')}"
        return f"{dt_msk.day} {MONTHS[dt_msk.month-1]} {dt_msk.strftime('%H:%M')}"
    except Exception:
        return ""


def format_express_card(variant: dict, title: str, emoji: str) -> str:
    if not variant:
        return ""
    lines = [f"{emoji} <b>{title}</b>", "━━━━━━━━━━━━━━━━━━━━━━━━━"]
    for i, e in enumerate(variant["events"], 1):
        t_str = _fmt_match_time(e.get("commence_time", ""))
        time_part = f"  🕐 {t_str}" if t_str else ""
        lines.append(
            f"<b>{i}. {e['label']}</b>\n"
            f"   <i>{e['match']}</i>  <code>{e['league']}</code>{time_part}\n"
            f"   📊 {int(e['prob']*100)}% | Кэф: <b>{e['odds']}</b> | "
            f"EV: <b>+{int(e['ev']*100)}%</b> {_stars(e['ev'])}"
        )
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"💰 Итоговый кэф: <b>{variant['combined_odds']}</b>")
    lines.append(f"📈 Вероятность: <b>{int(variant['combined_prob']*100)}%</b>")
    ev_sign = "+" if variant["ev"] >= 0 else ""
    lines.append(f"⚡ EV: <b>{ev_sign}{int(variant['ev']*100)}%</b>")
    lines.append(f"🏦 Ставка: <b>{variant['kelly_pct']}% от банка</b>")
    return "\n".join(lines)


def format_all_express(variants: dict) -> list:
    messages = []
    for key, title, emoji in [
        ("safe",   "🟢 Надёжный экспресс",   "🟢"),
        ("medium", "🟡 Средний экспресс",     "🟡"),
        ("risky",  "🔴 Рискованный экспресс", "🔴"),
    ]:
        card = format_express_card(variants.get(key), title, emoji)
        if card:
            messages.append(card)
    return messages
