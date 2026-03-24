# -*- coding: utf-8 -*-
"""
agent_memory.py — Память агентов Chimera AI
============================================
Извлекает из БД историю прогнозов по командам и формирует
контекст для агентов GPT/Llama чтобы они учитывали прошлые ошибки.
"""

import sqlite3
from typing import Optional

DB_FILE = "chimera_predictions.db"


def _get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_team_memory(team_name: str, sport: str = "football", limit: int = 5) -> dict:
    """
    Возвращает историю прогнозов на команду из БД.
    Используется чтобы агенты знали насколько точно мы предсказывали эту команду раньше.
    """
    table = f"{sport}_predictions"
    result = {
        "team": team_name,
        "total": 0,
        "correct": 0,
        "accuracy": 0,
        "recent": [],
        "summary": "",
    }
    try:
        with _get_conn() as conn:
            rows = conn.execute(f"""
                SELECT home_team, away_team, recommended_outcome,
                       is_correct, real_home_score, real_away_score,
                       bookmaker_odds_home, bookmaker_odds_away,
                       ensemble_home, ensemble_away, created_at
                FROM {table}
                WHERE (home_team LIKE ? OR away_team LIKE ?)
                  AND is_correct IN (0, 1)
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{team_name}%", f"%{team_name}%", limit)).fetchall()

            if not rows:
                return result

            result["total"] = len(rows)
            result["correct"] = sum(1 for r in rows if r["is_correct"] == 1)
            result["accuracy"] = round(result["correct"] / result["total"] * 100)

            for r in rows:
                is_home = team_name.lower() in r["home_team"].lower()
                opponent = r["away_team"] if is_home else r["home_team"]
                our_pred = r["recommended_outcome"]
                score = f"{r['real_home_score']}:{r['real_away_score']}" if r["real_home_score"] is not None else "?"
                outcome_icon = "✅" if r["is_correct"] == 1 else "❌"
                result["recent"].append({
                    "opponent": opponent,
                    "prediction": our_pred,
                    "correct": r["is_correct"],
                    "score": score,
                    "icon": outcome_icon,
                })

            # Формируем текстовое резюме для промпта агента
            recent_str = " | ".join(
                f"{r['icon']} vs {r['opponent'][:10]} ({r['score']})"
                for r in result["recent"]
            )
            acc = result["accuracy"]
            if acc >= 70:
                confidence_note = "наши прогнозы на эту команду точные"
            elif acc >= 50:
                confidence_note = "прогнозы на эту команду средней точности"
            else:
                confidence_note = "прогнозы на эту команду часто ошибались — будь осторожен"

            result["summary"] = (
                f"История прогнозов на {team_name}: {result['correct']}/{result['total']} верных ({acc}%) — {confidence_note}. "
                f"Последние: {recent_str}"
            )

    except Exception:
        pass

    return result


def get_match_memory_context(home_team: str, away_team: str, sport: str = "football") -> str:
    """
    Возвращает строку контекста для инжекции в промпт агента.
    Включает историю прогнозов по обеим командам.
    """
    home_mem = get_team_memory(home_team, sport)
    away_mem = get_team_memory(away_team, sport)

    parts = []
    if home_mem["total"] > 0:
        parts.append(home_mem["summary"])
    if away_mem["total"] > 0:
        parts.append(away_mem["summary"])

    if not parts:
        return ""

    return "\n\nПАМЯТЬ АГЕНТА (история прошлых прогнозов):\n" + "\n".join(parts)


def get_h2h_memory(home_team: str, away_team: str, sport: str = "football") -> str:
    """
    Ищет прошлые прогнозы на конкретную пару команд.
    """
    table = f"{sport}_predictions"
    try:
        with _get_conn() as conn:
            rows = conn.execute(f"""
                SELECT recommended_outcome, is_correct,
                       real_home_score, real_away_score, created_at
                FROM {table}
                WHERE (home_team LIKE ? AND away_team LIKE ?)
                   OR (home_team LIKE ? AND away_team LIKE ?)
                ORDER BY created_at DESC
                LIMIT 3
            """, (
                f"%{home_team}%", f"%{away_team}%",
                f"%{away_team}%", f"%{home_team}%",
            )).fetchall()

            if not rows:
                return ""

            lines = []
            for r in rows:
                icon = "✅" if r["is_correct"] == 1 else "❌"
                score = f"{r['real_home_score']}:{r['real_away_score']}" if r["real_home_score"] is not None else "?"
                lines.append(f"{icon} Прогноз: {r['recommended_outcome']} | Счёт: {score}")

            return f"\nНаши прошлые прогнозы на этот матч:\n" + "\n".join(lines)

    except Exception:
        return ""
