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

    # Ограничиваем до 15 лучших матчей для скана (по EV кэфа)
    matches = sorted(matches, key=lambda x: abs(x.get("odds_p1", 2) - 2), reverse=True)[:15]

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

            # Читаем форму и H2H из фонового кэша (мгновенно, без HTTP)
            p1_form = p2_form = "?????"
            h2h_p1_wins = h2h_total = 0
            try:
                from sports.tennis.form_cache import get_cached_form, get_cached_h2h
                c1 = get_cached_form(m["player1"])
                c2 = get_cached_form(m["player2"])
                p1_form = c1.get("form", "?????") or "?????"
                p2_form = c2.get("form", "?????") or "?????"
                ch2h = get_cached_h2h(m["player1"], m["player2"])
                h2h_p1_wins = ch2h.get("p1_wins", 0)
                h2h_total   = ch2h.get("total", 0)
            except Exception:
                pass

            probs = calculate_tennis_probs(
                player1=m["player1"], player2=m["player2"],
                sport_key=m["sport_key"], surface=surface,
                p1_form=p1_form, p2_form=p2_form,
                h2h_p1_wins=h2h_p1_wins, h2h_total=h2h_total,
                odds_p1=m.get("odds_p1", 0),
                odds_p2=m.get("odds_p2", 0),
                no_vig_p1=m.get("no_vig_p1", 0.0),
                no_vig_p2=m.get("no_vig_p2", 0.0),
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
            # Тотал геймов с реальной букмекерской линией
            from sports.tennis.model import predict_tennis_game_totals
            from sports.tennis.rankings import detect_tour
            tour = detect_tour(m["sport_key"])
            game_totals = predict_tennis_game_totals(
                p1_win=probs["p1_win"], p2_win=probs["p2_win"],
                p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
                surface=surface, tour=tour,
                bm_total_line=m.get("bm_total_line", 0.0),
                bm_total_over=m.get("bm_total_over", 0.0),
                bm_total_under=m.get("bm_total_under", 0.0),
            )
            ml_bet    = probs.get("ml_bet", False)
            ml_result = probs.get("ml_result")

            # ML фильтр: если ML недоступна или рекомендует — добавляем
            # Если ML доступна но не рекомендует — снижаем chimera_score на 10 (мягкий штраф вместо блока)
            if probs.get("ml_available") and not ml_bet:
                for c in candidates:
                    c["chimera_score"] = max(0, c.get("chimera_score", 0) - 10)

            # Добавляем время матча, тоталы и ML данные в каждый кандидат
            for c in candidates:
                c["commence_time"] = m.get("commence_time", "")
                c["game_totals"]   = game_totals
                c["ml_bet"]        = ml_bet
                c["ml_prob"]       = ml_result.get("prob_pct") if ml_result else None
                c["ml_label"]      = ml_result.get("label", "") if ml_result else ""
            all_candidates.extend(candidates)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Tennis scan] {m.get('player1')} vs {m.get('player2')}: {e}")
            continue

    all_candidates.sort(key=lambda x: x["chimera_score"], reverse=True)
    return all_candidates
