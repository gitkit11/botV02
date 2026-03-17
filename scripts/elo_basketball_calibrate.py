# -*- coding: utf-8 -*-
"""
elo_basketball_calibrate.py — Рекалибровка ELO рейтингов баскетбола.

Источники:
  1. Наша БД (basketball_predictions) — результаты уже сыгранных матчей
  2. The Odds API /scores/ — последние 3 дня

Запуск: python elo_basketball_calibrate.py
Результат: обновляет elo_basketball.json
Автоматически: каждый понедельник в 3:00 (main.py)
"""

import json
import os
import requests
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ELO_FILE   = "elo_basketball.json"
ELO_K      = 20
ELO_K_SOFT = 10   # Меньший K для старых результатов из БД

try:
    from config import THE_ODDS_API_KEY
except ImportError:
    THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASKETBALL_LEAGUES = ["basketball_nba", "basketball_euroleague"]


def load_elo() -> dict:
    if os.path.exists(ELO_FILE):
        try:
            with open(ELO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Стартовые рейтинги из core.py
    from sports.basketball.core import NBA_ELO, EUROLEAGUE_ELO
    elo = {}
    elo.update(NBA_ELO)
    elo.update(EUROLEAGUE_ELO)
    return elo


def save_elo(elo_data: dict):
    with open(ELO_FILE, "w", encoding="utf-8") as f:
        json.dump(elo_data, f, ensure_ascii=False, indent=2)
    logger.info(f"[ELO] Сохранено {len(elo_data)} команд → {ELO_FILE}")


def update_elo(winner: str, loser: str, elo: dict, default: int = 1550, k: int = ELO_K) -> dict:
    w = elo.get(winner, default)
    l = elo.get(loser, default)
    exp_w = 1 / (1 + 10 ** ((l - w) / 400))
    elo[winner] = round(w + k * (1 - exp_w))
    elo[loser]  = round(l + k * (0 - (1 - exp_w)))
    return elo


def get_results_from_db() -> list:
    """Берёт результаты из нашей БД basketball_predictions."""
    results = []
    try:
        import sqlite3
        db = "chimera_predictions.db"
        if not os.path.exists(db):
            return []
        with sqlite3.connect(db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT home_team, away_team, real_outcome
                FROM basketball_predictions
                WHERE real_outcome IS NOT NULL
                  AND real_outcome NOT IN ('expired', '')
                ORDER BY created_at ASC
            """)
            for home, away, outcome in cursor.fetchall():
                if outcome == "home_win":
                    results.append({"winner": home, "loser": away})
                elif outcome == "away_win":
                    results.append({"winner": away, "loser": home})
        logger.info(f"[ELO] Из БД: {len(results)} результатов")
    except Exception as e:
        logger.error(f"[ELO] Ошибка чтения БД: {e}")
    return results


def get_results_from_api() -> list:
    """Берёт свежие результаты из The Odds API /scores/."""
    if not THE_ODDS_API_KEY:
        logger.warning("[ELO] THE_ODDS_API_KEY не задан")
        return []

    results = []
    for league in BASKETBALL_LEAGUES:
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{league}/scores/",
                params={"apiKey": THE_ODDS_API_KEY, "daysFrom": 3},
                timeout=12,
            )
            if not r.ok:
                logger.warning(f"[ELO] API {league}: {r.status_code}")
                continue

            for m in r.json():
                if not m.get("completed"):
                    continue
                scores = m.get("scores") or []
                if len(scores) < 2:
                    continue

                home = m.get("home_team", "")
                away = m.get("away_team", "")
                score_map = {
                    s["name"]: int(s["score"])
                    for s in scores
                    if s.get("name") and s.get("score")
                }
                h_score = score_map.get(home, 0)
                a_score = score_map.get(away, 0)
                if h_score == a_score:
                    continue

                winner = home if h_score > a_score else away
                loser  = away if h_score > a_score else home
                results.append({"winner": winner, "loser": loser})

            logger.info(f"[ELO] API {league}: {len(results)} результатов")
        except Exception as e:
            logger.error(f"[ELO] API {league} ошибка: {e}")

    return results


def calibrate():
    logger.info("=" * 50)
    logger.info("CHIMERA — Рекалибровка ELO баскетбола")
    logger.info(f"Время: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 50)

    elo = load_elo()
    logger.info(f"[ELO] Загружено рейтингов: {len(elo)}")

    # Результаты из БД (мягкий K — они уже были учтены при матчах)
    db_results = get_results_from_db()
    for r in db_results:
        elo = update_elo(r["winner"], r["loser"], elo, k=ELO_K_SOFT)
    logger.info(f"[ELO] Обработано из БД: {len(db_results)}")

    # Свежие результаты из API (полный K)
    api_results = get_results_from_api()
    for r in api_results:
        elo = update_elo(r["winner"], r["loser"], elo, k=ELO_K)
    logger.info(f"[ELO] Обработано из API: {len(api_results)}")

    # Топ-10 команд после рекалибровки
    top = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("[ELO] Топ-10 команд после рекалибровки:")
    for i, (team, rating) in enumerate(top, 1):
        logger.info(f"  {i:2}. {team:<30} {rating}")

    save_elo(elo)
    return len(db_results) + len(api_results)


if __name__ == "__main__":
    total = calibrate()
    print(f"\nГотово. Обработано матчей: {total}")
