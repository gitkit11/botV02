# -*- coding: utf-8 -*-
"""
CS2 Results Tracker — авто-проверка результатов и само-обучение.

Что делает:
1. Получает завершённые матчи CS2 из PandaScore
2. Сопоставляет с непроверенными прогнозами в БД
3. Обновляет is_correct, roi_outcome, is_total_correct
4. Обновляет ELO команд по факту результата
5. Корректирует map winrates через EMA (exponential moving average)
"""

import json
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── ELO константы ────────────────────────────────────────────────────────────
ELO_K = 24           # K-фактор (меньше чем в футболе — CS2 более стабилен)
ELO_FILE = "elo_cs2.json"   # Отдельный файл от футбольного ELO

# ── Обучение карт: экспоненциальное скользящее среднее ────────────────────────
MAP_ALPHA = 0.12     # Вес нового результата (12%). Больше → быстрее обучается


def load_elo() -> dict:
    """Загружает ELO рейтинги CS2 из файла (fallback: из core.py)."""
    if os.path.exists(ELO_FILE):
        try:
            with open(ELO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: берём из core.py
    try:
        from .core import CS2_ELO, DEFAULT_ELO
        return dict(CS2_ELO)
    except Exception:
        return {}


def save_elo(elo_data: dict):
    """Сохраняет ELO рейтинги CS2 в файл."""
    try:
        with open(ELO_FILE, "w", encoding="utf-8") as f:
            json.dump(elo_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[CS2 ELO] Ошибка сохранения: {e}")


def update_elo(winner: str, loser: str, elo_data: dict, default_elo: int = 1450) -> dict:
    """
    Обновляет ELO рейтинги после матча.
    Формула Эло: R_new = R_old + K * (actual - expected)
    """
    w_elo = elo_data.get(winner, default_elo)
    l_elo = elo_data.get(loser, default_elo)

    expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
    expected_l = 1 - expected_w

    elo_data[winner] = round(w_elo + ELO_K * (1 - expected_w))
    elo_data[loser]  = round(l_elo + ELO_K * (0 - expected_l))

    logger.info(f"[CS2 ELO] {winner}: {w_elo} → {elo_data[winner]} | {loser}: {l_elo} → {elo_data[loser]}")
    return elo_data


def update_map_stats(team: str, map_name: str, won: bool):
    """
    Обновляет winrate команды на карте через EMA.
    won=True → команда выиграла карту, won=False → проиграла.
    """
    try:
        from .hltv_stats import get_team_map_stats, update_team_stats
        current_stats = get_team_map_stats(team)
        if not current_stats:
            current_stats = {}

        old_wr = current_stats.get(map_name, 50.0)
        actual  = 100.0 if won else 0.0
        new_wr  = round(old_wr * (1 - MAP_ALPHA) + actual * MAP_ALPHA, 1)
        current_stats[map_name] = new_wr

        update_team_stats(team, maps=current_stats)
        logger.info(f"[CS2 Map] {team} on {map_name}: {old_wr}% → {new_wr}%")
    except Exception as e:
        logger.error(f"[CS2 Map] Ошибка обновления {team}/{map_name}: {e}")


def get_finished_cs2_matches(hours_back: int = 12, pending_ids: list = None) -> list:
    """
    Получает завершённые матчи CS2 из обоих источников.
    pending_ids: список числовых PandaScore ID из pending predictions — точечный запрос.
    """
    results = []
    seen_ids = set()

    # Источник 1: Esports Data API (Tier 1-3)
    for m in _get_finished_esports_data():
        mid = m["match_id"]
        if mid not in seen_ids:
            results.append(m)
            seen_ids.add(mid)

    # Источник 2: PandaScore — точечно по нужным ID или последние 100
    numeric_ids = [i for i in (pending_ids or []) if str(i).isdigit()]
    for m in _get_finished_pandascore(match_ids=numeric_ids if numeric_ids else None):
        mid = m["match_id"]
        if mid not in seen_ids:
            results.append(m)
            seen_ids.add(mid)

    logger.info(f"[CS2 Tracker] Всего завершённых матчей: {len(results)}")
    return results


def _get_finished_esports_data() -> list:
    """Esports Data API через RapidAPI — покрывает Tier 1-3 CS2."""
    try:
        import requests
        key = os.getenv("RAPID_API_KEY", "")
        if not key:
            return []
        host = "esports-data.p.rapidapi.com"
        headers = {"x-rapidapi-key": key, "x-rapidapi-host": host}
        r = requests.get(
            f"https://{host}/matches/recent",
            headers=headers,
            params={"game": "csgo", "per_page": 50},
            timeout=12,
        )
        if not r.ok:
            logger.warning(f"[CS2 Tracker] Esports Data API: {r.status_code}")
            return []

        results = []
        for m in r.json().get("data", []):
            if m.get("status") != "completed":
                continue
            home = (m.get("team1_name") or "").strip()
            away = (m.get("team2_name") or "").strip()
            if not home or not away:
                continue

            details = m.get("details") or {}
            winner_name = (details.get("winner") or "").strip()

            # Парсим счёт "2-1", "1-0" и т.д.
            score_str = m.get("score", "0-0")
            try:
                parts = score_str.split("-")
                home_score = int(parts[0])
                away_score = int(parts[1])
            except Exception:
                home_score, away_score = 0, 0

            if winner_name == home:
                winner, loser = home, away
            elif winner_name == away:
                winner, loser = away, home
            elif home_score > away_score:
                winner, loser = home, away
            elif away_score > home_score:
                winner, loser = away, home
            else:
                continue

            results.append({
                "match_id":   str(m.get("source_id") or m.get("id", "")),
                "home":       home,
                "away":       away,
                "winner":     winner,
                "loser":      loser,
                "home_score": home_score,
                "away_score": away_score,
                "maps":       [],
            })

        logger.info(f"[CS2 Tracker] Esports Data: {len(results)} завершённых матчей")
        return results
    except Exception as e:
        logger.error(f"[CS2 Tracker] Esports Data API ошибка: {e}")
        return []


def _get_finished_pandascore(match_ids: list = None) -> list:
    """
    PandaScore: ищет результаты.
    Если переданы match_ids — делает точечный запрос по конкретным ID.
    Иначе — запрашивает последние 100 завершённых матчей.
    """
    try:
        from .pandascore import _request_with_retry, PANDASCORE_BASE
        if match_ids:
            # Точечный запрос по ID — намного точнее, не зависит от сортировки
            ids_str = ",".join(str(i) for i in match_ids if str(i).isdigit())
            if not ids_str:
                return []
            r = _request_with_retry(
                f"{PANDASCORE_BASE}/matches",
                params={"filter[id]": ids_str, "per_page": 50},
            )
        else:
            r = _request_with_retry(
                f"{PANDASCORE_BASE}/matches/past",
                params={"per_page": 100, "sort": "-end_at"},
            )
        if not r or not r.ok:
            return []

        results = []
        for m in r.json():
            if m.get("status") != "finished":
                continue
            opponents = m.get("opponents", [])
            if len(opponents) < 2:
                continue
            home = opponents[0]["opponent"]["name"]
            away = opponents[1]["opponent"]["name"]
            winner_id = m.get("winner_id")
            home_id   = opponents[0]["opponent"]["id"]
            away_id   = opponents[1]["opponent"]["id"]
            if winner_id == home_id:
                winner, loser = home, away
                home_score, away_score = 2, (1 if m.get("number_of_games", 3) == 3 else 0)
            elif winner_id == away_id:
                winner, loser = away, home
                home_score, away_score = (1 if m.get("number_of_games", 3) == 3 else 0), 2
            else:
                continue
            results.append({
                "match_id":   str(m["id"]),
                "home":       home, "away": away,
                "winner":     winner, "loser": loser,
                "home_score": home_score, "away_score": away_score,
                "maps":       [],
            })
        return results
    except Exception as e:
        logger.error(f"[CS2 Tracker] PandaScore fallback ошибка: {e}")
        return []


def check_and_update_cs2_results() -> int:
    """
    Основная функция авто-обучения. Вызывается по расписанию.

    1. Берёт непроверенные CS2 прогнозы из БД
    2. Ищет их среди завершённых матчей PandaScore
    3. Обновляет результаты в БД
    4. Обновляет ELO и карты

    Возвращает количество обновлённых записей.
    """
    try:
        from database import get_pending_predictions, update_result
    except ImportError:
        logger.error("[CS2 Tracker] Не удалось импортировать database")
        return 0

    pending = get_pending_predictions("cs2")
    if not pending:
        logger.info("[CS2 Tracker] Нет непроверенных прогнозов CS2")
        return 0

    # Числовые ID для точечного запроса к PandaScore (только настоящие ID)
    numeric_ids = [str(p["match_id"]) for p in pending if str(p.get("match_id","")).isdigit()]
    # Всегда запрашиваем последние матчи по имени — основной путь для chimera_* ID
    finished = get_finished_cs2_matches(hours_back=72, pending_ids=numeric_ids if numeric_ids else None)
    if not finished:
        logger.info("[CS2 Tracker] Нет завершённых матчей CS2 за 48 часов")
        return 0

    # normalize_team_name: "G2" → "G2 Esports", "NaVi" → "Natus Vincere" и т.д.
    try:
        from .team_registry import normalize_team_name as _norm_cs2
    except Exception:
        def _norm_cs2(n):
            return n.lower().strip()

    def _cs2_key(name: str) -> str:
        return _norm_cs2(name).lower().strip()

    elo_data = load_elo()
    finished_by_id   = {m["match_id"]: m for m in finished}
    # Fallback: по нормализованным именам команд
    finished_by_name = {}
    for m in finished:
        h_key = _cs2_key(m["home"])
        a_key = _cs2_key(m["away"])
        finished_by_name[(h_key, a_key)] = m
        finished_by_name[(a_key, h_key)] = m  # обе стороны

    updated = 0

    for pred in pending:
        match_id = str(pred.get("match_id", ""))
        result = finished_by_id.get(match_id)
        if result is None:
            h = _cs2_key(pred.get("home_team", ""))
            a = _cs2_key(pred.get("away_team", ""))
            result = finished_by_name.get((h, a))
            if result:
                logger.info(f"[CS2 Tracker] Совпадение по именам: {pred.get('home_team')} vs {pred.get('away_team')}")

        if result is None:
            continue

        winner  = result["winner"]
        home    = result["home"]
        away    = result["away"]
        h_score = result["home_score"]
        a_score = result["away_score"]

        # Какой исход бот предсказал (fallback на ensemble_best_outcome)
        recommended = pred.get("recommended_outcome") or pred.get("ensemble_best_outcome", "")
        actual_outcome = "home_win" if winner == home else "away_win"
        is_correct = 1 if (recommended and recommended == actual_outcome) else 0

        # ROI (только на основе реальных коэффициентов; без кэфа — None, не считаем)
        if is_correct:
            odds_key = "bookmaker_odds_home" if actual_outcome == "home_win" else "bookmaker_odds_away"
            raw_odds = pred.get(odds_key)
            if raw_odds and float(raw_odds) > 1.01:
                roi = round(float(raw_odds) - 1, 3)
            else:
                roi = None   # нет реального кэфа — не учитываем в ROI статистике
        else:
            roi = -1.0

        # Тотал: is_total_correct
        total_pred = pred.get("total_prediction", "")
        maps_count = h_score + a_score   # 2 (2:0) или 3 (2:1)
        actual_total = "OVER 2.5" if maps_count >= 3 else "UNDER 2.5"
        is_total_correct = None
        if total_pred:
            is_total_correct = 1 if total_pred == actual_total else 0

        # Записываем в БД
        try:
            update_result(
                sport="cs2",
                match_id=match_id,
                real_home_score=h_score,
                real_away_score=a_score,
                real_outcome=actual_outcome,
                is_correct=is_correct,
                roi_outcome=roi,
            )
            # Обновляем поле тотала если колонка есть
            _update_total_result(match_id, is_total_correct, actual_total)
            updated += 1
        except Exception as e:
            logger.error(f"[CS2 Tracker] Ошибка update_result для {match_id}: {e}")
            continue

        # ── ELO обновление ───────────────────────────────────────────────────
        elo_data = update_elo(winner, result["loser"], elo_data)

        # ── Карты обновление ─────────────────────────────────────────────────
        for game in result.get("maps", []):
            map_name   = game.get("map")
            map_winner = game.get("winner")
            if not map_name or not map_winner:
                continue
            map_loser = away if map_winner == home else home
            update_map_stats(map_winner, map_name, won=True)
            update_map_stats(map_loser,  map_name, won=False)

        logger.info(
            f"[CS2 Tracker] ✅ {home} {h_score}:{a_score} {away} | "
            f"Прогноз {'✓' if is_correct else '✗'} | Тотал {'✓' if is_total_correct == 1 else ('✗' if is_total_correct == 0 else '—')}"
        )

    save_elo(elo_data)
    logger.info(f"[CS2 Tracker] Обновлено прогнозов: {updated}")
    return updated


def _update_total_result(match_id: str, is_total_correct: Optional[int], actual_total: str):
    """Обновляет поля тотала (если колонки есть в БД)."""
    try:
        import sqlite3
        from database import DB_FILE
        with sqlite3.connect(DB_FILE) as conn:
            # Проверяем что колонки существуют
            cols = [r[1] for r in conn.execute("PRAGMA table_info(cs2_predictions)").fetchall()]
            if "is_total_correct" not in cols:
                return
            conn.execute(
                "UPDATE cs2_predictions SET is_total_correct=?, actual_total=? WHERE match_id=?",
                (is_total_correct, actual_total, match_id),
            )
            conn.commit()
    except Exception:
        pass  # Колонки могут ещё не существовать — нестрашно


def get_cs2_bet_stats() -> dict:
    """
    Возвращает статистику бота по CS2 с разбивкой по типам ставок.
    Используется для отображения в Telegram.
    """
    try:
        import sqlite3
        from database import DB_FILE
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            # Общая статистика
            total     = c.execute("SELECT COUNT(*) FROM cs2_predictions").fetchone()[0]
            checked   = c.execute("SELECT COUNT(*) FROM cs2_predictions WHERE is_correct IS NOT NULL").fetchone()[0]
            wins      = c.execute("SELECT COUNT(*) FROM cs2_predictions WHERE is_correct = 1").fetchone()[0]
            roi_sum   = c.execute("SELECT SUM(roi_outcome) FROM cs2_predictions WHERE roi_outcome IS NOT NULL").fetchone()[0] or 0

            # Тотал
            cols = [r[1] for r in c.execute("PRAGMA table_info(cs2_predictions)").fetchall()]
            total_checked = total_wins_t = 0
            if "is_total_correct" in cols:
                total_checked = c.execute("SELECT COUNT(*) FROM cs2_predictions WHERE is_total_correct IS NOT NULL").fetchone()[0]
                total_wins_t  = c.execute("SELECT COUNT(*) FROM cs2_predictions WHERE is_total_correct = 1").fetchone()[0]

            # Последние 10
            recent = c.execute("""
                SELECT home_team, away_team, real_home_score, real_away_score,
                       recommended_outcome, is_correct, created_at
                FROM cs2_predictions
                WHERE is_correct IS NOT NULL
                ORDER BY result_checked_at DESC
                LIMIT 10
            """).fetchall()

            # По месяцам
            monthly = c.execute("""
                SELECT strftime('%Y-%m', created_at) as month,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as wins,
                       SUM(COALESCE(roi_outcome, 0)) as roi
                FROM cs2_predictions
                WHERE is_correct IS NOT NULL
                GROUP BY month
                ORDER BY month DESC
                LIMIT 6
            """).fetchall()

        accuracy      = wins / checked * 100 if checked > 0 else 0
        total_acc     = total_wins_t / total_checked * 100 if total_checked > 0 else 0

        return {
            "total":          total,
            "checked":        checked,
            "wins":           wins,
            "accuracy":       round(accuracy, 1),
            "roi":            round(roi_sum, 2),
            "total_checked":  total_checked,
            "total_wins":     total_wins_t,
            "total_accuracy": round(total_acc, 1),
            "recent":         [dict(r) for r in recent],
            "monthly":        [dict(r) for r in monthly],
        }
    except Exception as e:
        logger.error(f"[CS2 Stats] Ошибка: {e}")
        return {"error": str(e)}
