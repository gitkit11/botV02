# -*- coding: utf-8 -*-
"""
sports/tennis — Теннисный модуль Chimera AI
===========================================
Поддерживает ATP и WTA турниры через The Odds API.

Публичный интерфейс:
  get_tennis_matches()       — матчи + коэффициенты
  analyze_tennis_match()     — полный CHIMERA анализ матча
  scan_tennis_signals()      — сканирование всех матчей, возврат кандидатов
"""

from sports.tennis.matches import get_tennis_matches
from sports.tennis.rankings import detect_surface, detect_tour
from sports.tennis.model import calculate_tennis_probs, compute_tennis_chimera_score


STATUS = "active"


def analyze_tennis_match(
    player1: str,
    player2: str,
    odds_p1: float,
    odds_p2: float,
    sport_key: str = "",
    p1_form: str = "?????",
    p2_form: str = "?????",
    h2h_p1_wins: int = 0,
    h2h_total: int = 0,
    line_movement: dict = None,
) -> dict:
    """
    Полный анализ теннисного матча.
    Возвращает вероятности, CHIMERA Score кандидатов и текстовый резюме.
    """
    surface = detect_surface(sport_key)

    probs = calculate_tennis_probs(
        player1=player1, player2=player2,
        sport_key=sport_key, surface=surface,
        p1_form=p1_form, p2_form=p2_form,
        h2h_p1_wins=h2h_p1_wins, h2h_total=h2h_total,
    )

    candidates = compute_tennis_chimera_score(
        player1=player1, player2=player2,
        prob_p1=probs["p1_win"], prob_p2=probs["p2_win"],
        odds_p1=odds_p1, odds_p2=odds_p2,
        p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
        surface=surface,
        p1_form=p1_form, p2_form=p2_form,
        h2h_p1_wins=h2h_p1_wins, h2h_total=h2h_total,
        line_movement=line_movement,
        sport_key=sport_key,
    )

    return {
        "probs":      probs,
        "candidates": candidates,
        "surface":    surface,
    }


def scan_tennis_signals() -> list:
    """
    Сканирует все активные теннисные турниры.
    Возвращает список CHIMERA кандидатов (все матчи, все исходы).
    """
    matches = get_tennis_matches()
    if not matches:
        return []

    all_candidates = []

    for m in matches:
        try:
            surface = detect_surface(m["sport_key"])

            # Движение линий
            movement = {}
            try:
                from line_movement import make_match_key, record_odds, get_movement
                mkey = make_match_key(m["player1"], m["player2"], m.get("commence_time", ""))
                record_odds(mkey, {"home_win": m["odds_p1"], "away_win": m["odds_p2"]})
                movement = get_movement(mkey, {"home_win": m["odds_p1"], "away_win": m["odds_p2"]})
            except Exception:
                pass

            # Реальная форма и H2H из api-tennis.com
            p1_form = p2_form = "?????"
            h2h_p1_wins = h2h_total = 0
            try:
                from sports.tennis.api_tennis import get_player_form, get_h2h_by_name
                f1 = get_player_form(m["player1"])
                f2 = get_player_form(m["player2"])
                p1_form = f1.get("form", "?????") or "?????"
                p2_form = f2.get("form", "?????") or "?????"
                h2h = get_h2h_by_name(m["player1"], m["player2"])
                h2h_p1_wins = h2h.get("p1_wins", 0)
                h2h_total   = h2h.get("total", 0)
            except Exception:
                pass

            probs = calculate_tennis_probs(
                player1=m["player1"], player2=m["player2"],
                sport_key=m["sport_key"], surface=surface,
                p1_form=p1_form, p2_form=p2_form,
                h2h_p1_wins=h2h_p1_wins, h2h_total=h2h_total,
            )

            candidates = compute_tennis_chimera_score(
                player1=m["player1"], player2=m["player2"],
                prob_p1=probs["p1_win"], prob_p2=probs["p2_win"],
                odds_p1=m["odds_p1"], odds_p2=m["odds_p2"],
                p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
                surface=surface,
                p1_form=p1_form, p2_form=p2_form,
                h2h_p1_wins=h2h_p1_wins, h2h_total=h2h_total,
                line_movement=movement,
                sport_key=m["sport_key"],
            )
            # Добавляем время матча в каждый кандидат
            for c in candidates:
                c["commence_time"] = m.get("commence_time", "")
            all_candidates.extend(candidates)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Tennis scan] {m.get('player1')} vs {m.get('player2')}: {e}")
            continue

    all_candidates.sort(key=lambda x: x["chimera_score"], reverse=True)
    return all_candidates
