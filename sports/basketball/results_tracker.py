# -*- coding: utf-8 -*-
"""
Basketball Results Tracker — авто-проверка результатов и обновление ELO.

Как работает:
1. Берёт непроверенные прогнозы баскетбола из БД
2. Запрашивает завершённые матчи через The Odds API /scores/
3. Сопоставляет по именам команд
4. Обновляет is_correct, roi_outcome, result_checked_at
5. Обновляет ELO команд
"""

import os
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ELO_K = 20  # K-фактор для баскетбола
ELO_FILE = "elo_basketball.json"

BASKETBALL_LEAGUES = [
    "basketball_nba",
    "basketball_euroleague",
]


def load_elo() -> dict:
    import json
    if os.path.exists(ELO_FILE):
        try:
            with open(ELO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        from .core import NBA_ELO, EUROLEAGUE_ELO
        elo = {}
        elo.update(NBA_ELO)
        elo.update(EUROLEAGUE_ELO)
        return elo
    except Exception:
        return {}


def save_elo(elo_data: dict):
    import json
    try:
        with open(ELO_FILE, "w", encoding="utf-8") as f:
            json.dump(elo_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[Basketball ELO] Ошибка сохранения: {e}")


def update_elo(winner: str, loser: str, elo_data: dict, default: int = 1550) -> dict:
    w = elo_data.get(winner, default)
    l = elo_data.get(loser, default)
    expected_w = 1 / (1 + 10 ** ((l - w) / 400))
    elo_data[winner] = round(w + ELO_K * (1 - expected_w))
    elo_data[loser]  = round(l + ELO_K * (0 - (1 - expected_w)))
    logger.info(f"[Basketball ELO] {winner}: {w}→{elo_data[winner]} | {loser}: {l}→{elo_data[loser]}")
    return elo_data


def get_finished_basketball_matches() -> list:
    """
    Получает завершённые матчи NBA и Евролиги через The Odds API /scores/.
    Возвращает список: {home, away, winner, home_score, away_score}
    """
    try:
        api_key = os.getenv("THE_ODDS_API_KEY", "")
        if not api_key:
            try:
                from config import THE_ODDS_API_KEY
                api_key = THE_ODDS_API_KEY
            except ImportError:
                pass
        if not api_key:
            logger.warning("[Basketball Tracker] THE_ODDS_API_KEY не задан")
            return []

        results = []
        for league in BASKETBALL_LEAGUES:
            try:
                r = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{league}/scores/",
                    params={"apiKey": api_key, "daysFrom": 2},
                    timeout=12,
                )
                if not r.ok:
                    logger.warning(f"[Basketball Tracker] {league}: {r.status_code}")
                    continue

                for m in r.json():
                    if not m.get("completed"):
                        continue
                    scores = m.get("scores") or []
                    if len(scores) < 2:
                        continue

                    home = m.get("home_team", "")
                    away = m.get("away_team", "")
                    score_map = {s["name"]: int(s["score"]) for s in scores if s.get("name") and s.get("score")}

                    home_score = score_map.get(home, 0)
                    away_score = score_map.get(away, 0)

                    if home_score == away_score:
                        continue

                    winner = home if home_score > away_score else away
                    loser  = away if home_score > away_score else home

                    results.append({
                        "match_id":   m.get("id", ""),
                        "home":       home,
                        "away":       away,
                        "winner":     winner,
                        "loser":      loser,
                        "home_score": home_score,
                        "away_score": away_score,
                    })

            except Exception as e:
                logger.error(f"[Basketball Tracker] Ошибка {league}: {e}")

        logger.info(f"[Basketball Tracker] Найдено завершённых матчей: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"[Basketball Tracker] get_finished: {e}")
        return []


def check_and_update_basketball_results() -> int:
    """
    Основная функция. Вызывается каждый час из main.py.
    Возвращает количество обновлённых прогнозов.
    """
    try:
        from database import get_pending_predictions, update_result
    except ImportError:
        logger.error("[Basketball Tracker] Не удалось импортировать database")
        return 0

    pending = get_pending_predictions("basketball")
    if not pending:
        logger.info("[Basketball Tracker] Нет непроверенных прогнозов баскетбола")
        return 0

    finished = get_finished_basketball_matches()
    if not finished:
        logger.info("[Basketball Tracker] Нет завершённых матчей баскетбола")
        return 0

    # Индекс по ID и по именам команд
    finished_by_id   = {m["match_id"]: m for m in finished}
    finished_by_name = {}
    for m in finished:
        key  = (m["home"].lower().strip(), m["away"].lower().strip())
        key2 = (m["away"].lower().strip(), m["home"].lower().strip())
        finished_by_name[key]  = m
        finished_by_name[key2] = m

    elo_data = load_elo()
    updated  = 0
    now      = datetime.now(timezone.utc).isoformat()

    for pred in pending:
        match_id = str(pred.get("match_id", ""))
        result   = finished_by_id.get(match_id)
        if result is None:
            h = pred.get("home_team", "").lower().strip()
            a = pred.get("away_team", "").lower().strip()
            result = finished_by_name.get((h, a))

        if result is None:
            continue

        winner     = result["winner"]
        home       = result["home"]
        home_score = result["home_score"]
        away_score = result["away_score"]

        actual_outcome = "home_win" if winner == home else "away_win"
        recommended    = pred.get("recommended_outcome") or pred.get("ensemble_best_outcome", "")
        is_correct     = 1 if (recommended and recommended == actual_outcome) else 0

        # ROI: если ставить рекомендовали
        bet_signal = pred.get("bet_signal", "")
        if "СТАВИТЬ" in bet_signal and is_correct:
            odds_key = "bookmaker_odds_home" if actual_outcome == "home_win" else "bookmaker_odds_away"
            win_odds = float(pred.get(odds_key) or 1.85)
            roi = round(win_odds - 1, 3)
        elif "СТАВИТЬ" in bet_signal and not is_correct:
            roi = -1.0
        else:
            roi = None

        try:
            update_result(
                sport="basketball",
                match_id=match_id,
                real_home_score=home_score,
                real_away_score=away_score,
                real_outcome=actual_outcome,
                is_correct=is_correct,
                roi_outcome=roi,
            )
            updated += 1
        except Exception as e:
            logger.error(f"[Basketball Tracker] update_result ошибка: {e}")
            continue

        elo_data = update_elo(result["winner"], result["loser"], elo_data)
        logger.info(
            f"[Basketball Tracker] {home} {home_score}:{away_score} {result['away']} | "
            f"Прогноз {'OK' if is_correct else 'WRONG'}"
        )

    save_elo(elo_data)
    logger.info(f"[Basketball Tracker] Обновлено прогнозов: {updated}")
    return updated
