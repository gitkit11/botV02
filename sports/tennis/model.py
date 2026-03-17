# -*- coding: utf-8 -*-
"""
sports/tennis/model.py — Математическая модель теннисных матчей
===============================================================
Факторы (в порядке важности):
1. Разница рейтингов (основной предиктор)
2. Поверхность (специализация игрока)
3. Форма (последние 5 матчей)
4. H2H (очные встречи)
"""

import math
from sports.tennis.rankings import get_ranking, get_surface_strength, detect_tour


def rank_to_elo(rank: int) -> float:
    """
    Конвертирует ATP/WTA рейтинг в Elo-подобный балл.
    Топ-1 ≈ 2400, топ-10 ≈ 2200, топ-50 ≈ 1900, #200 ≈ 1500.
    """
    if rank <= 0:
        rank = 200
    # Логарифмическая шкала: топ игроки расставлены дальше
    return round(2500 - 400 * math.log10(max(1, rank)), 1)


def surface_adjusted_elo(player: str, base_elo: float, surface: str) -> float:
    """
    Корректирует Elo с учётом специализации на поверхности.
    Специалист по глине (+150 на clay, -100 на hard/grass).
    """
    strength = get_surface_strength(player, surface)
    # Нейтральный уровень = 0.65, диапазон коррекции ±200 Elo
    adjustment = (strength - 0.65) * 600
    return base_elo + adjustment


def win_probability_from_elo(elo_a: float, elo_b: float) -> float:
    """Вероятность победы A над B по формуле Elo."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def form_adjustment(form_str: str) -> float:
    """
    Бонус/штраф к вероятности на основе формы с экспоненциальным весом.
    Последний матч важнее всего: веса [1.0, 0.65, 0.45, 0.30, 0.20].
    'WWWWW' → ~+0.05, 'LLLLL' → ~-0.05
    """
    if not form_str or form_str == "?????":
        return 0.0
    results = form_str.upper()[-5:]
    weights = [1.0, 0.65, 0.45, 0.30, 0.20]  # [последний, ..., самый старый]
    weighted_wins  = 0.0
    total_weight   = 0.0
    for i, ch in enumerate(results):
        w = weights[i] if i < len(weights) else 0.10
        if ch == 'W':
            weighted_wins += w
        total_weight += w
    if total_weight == 0:
        return 0.0
    win_rate = weighted_wins / total_weight
    # Отклонение от нейтральной 0.5 → корректировка ±0.05
    return round((win_rate - 0.5) * 0.10, 4)


def h2h_adjustment(h2h_wins: int, h2h_total: int) -> float:
    """
    Корректировка вероятности на основе H2H.
    Если P1 выиграл 7 из 10 встреч → +0.04
    Если 3 из 10 → -0.04
    Минимум 3 встречи для учёта.
    """
    if h2h_total < 3:
        return 0.0
    rate = h2h_wins / h2h_total
    return round((rate - 0.5) * 0.08, 4)


def rest_days_adjustment(days_since_last_match: int) -> float:
    """
    Штраф/бонус за дни отдыха.
    0-1 дней (или турнирный марафон): -0.03
    2 дня: нейтрально
    5+ дней (долгий перерыв): -0.01 (потеря ритма)
    Оптимально 2-4 дня: +0.01
    """
    if days_since_last_match < 0:
        return 0.0
    if days_since_last_match <= 1:
        return -0.03   # усталость/нет восстановления
    if days_since_last_match <= 4:
        return +0.01   # оптимальный отдых
    if days_since_last_match >= 14:
        return -0.02   # долгий перерыв — потеря игрового ритма
    return 0.0


def calculate_tennis_probs(
    player1: str,
    player2: str,
    sport_key: str = "",
    surface: str = "hard",
    p1_form: str = "?????",
    p2_form: str = "?????",
    h2h_p1_wins: int = 0,
    h2h_total: int = 0,
    p1_rest_days: int = -1,
    p2_rest_days: int = -1,
) -> dict:
    """
    Главная функция — вычисляет вероятности победы для теннисного матча.

    Возвращает:
    {
      "p1_win": 0.62, "p2_win": 0.38,
      "p1_rank": 5, "p2_rank": 23,
      "p1_elo": 2180, "p2_elo": 2050,
      "surface": "hard",
      "tour": "atp",
      "rank_gap": 18,
    }
    """
    tour = detect_tour(sport_key) if sport_key else "atp"

    # Рейтинги
    p1_rank = get_ranking(player1, tour)
    p2_rank = get_ranking(player2, tour)

    # Базовый Elo
    p1_elo_base = rank_to_elo(p1_rank)
    p2_elo_base = rank_to_elo(p2_rank)

    # Коррекция поверхности
    p1_elo = surface_adjusted_elo(player1, p1_elo_base, surface)
    p2_elo = surface_adjusted_elo(player2, p2_elo_base, surface)

    # Базовая вероятность
    p1_win = win_probability_from_elo(p1_elo, p2_elo)

    # Форма
    p1_form_adj = form_adjustment(p1_form)
    p2_form_adj = form_adjustment(p2_form)
    p1_win += (p1_form_adj - p2_form_adj)

    # H2H
    p1_win += h2h_adjustment(h2h_p1_wins, h2h_total)

    # Дни отдыха (если переданы)
    if p1_rest_days >= 0:
        p1_win += rest_days_adjustment(p1_rest_days)
    if p2_rest_days >= 0:
        p1_win -= rest_days_adjustment(p2_rest_days)  # если P2 устал — P1 выигрывает

    # Ограничиваем [0.10, 0.82] — не позволяем модели быть слишком уверенной
    p1_win = max(0.10, min(0.82, p1_win))
    p2_win = 1.0 - p1_win

    return {
        "p1_win":       round(p1_win, 4),
        "p2_win":       round(p2_win, 4),
        "p1_rank":      p1_rank,
        "p2_rank":      p2_rank,
        "p1_elo":       round(p1_elo, 0),
        "p2_elo":       round(p2_elo, 0),
        "surface":      surface,
        "tour":         tour,
        "rank_gap":     abs(p1_rank - p2_rank),
        "p1_form":      p1_form,
        "p2_form":      p2_form,
        "p1_rest_days": p1_rest_days,
        "p2_rest_days": p2_rest_days,
    }


def predict_tennis_game_totals(
    p1_win: float,
    p2_win: float,
    p1_rank: int = 100,
    p2_rank: int = 100,
    surface: str = "hard",
    tour: str = "atp",
    best_of: int = 3,
) -> dict:
    """
    Предсказывает тотал геймов в матче (OVER/UNDER).

    Стандартные линии: BO3 → 20.5, 21.5, 22.5; BO5 → 34.5, 35.5
    Логика:
    - Большой разрыв рейтингов → доминация → UNDER (короткий матч 6-2, 6-3)
    - Равные игроки → OVER (тесные сеты, тайбрейки)
    - Clay → OVER (длинные розыгрыши)
    - Grass → UNDER (сервисные доминации, короткие геймы)
    """
    gap = abs(p1_win - p2_win)  # 0 = равные, 0.5+ = явный фаворит

    # Стандартная линия по туру
    if best_of == 5:
        base_line = 34.5
    elif tour == "wta":
        base_line = 19.5  # женские матчи короче
    else:
        base_line = 21.5  # ATP BO3

    # Базовая вероятность UNDER (равные → OVER, неравные → UNDER)
    # gap=0.00 → p_under≈0.35 (равные, много геймов)
    # gap=0.30 → p_under≈0.55 (один явно сильнее)
    # gap=0.50 → p_under≈0.65 (очень неравные)
    p_under = 0.35 + gap * 0.60
    p_under = max(0.28, min(0.72, p_under))

    # Коррекция поверхности
    surface_adj = {
        "clay":  -0.08,   # глина → длиннее (OVER)
        "grass": +0.08,   # трава → короче (UNDER)
        "hard":  0.00,
    }
    p_under += surface_adj.get(surface, 0.0)
    p_under = max(0.25, min(0.75, p_under))

    # Коррекция рейтинга: если оба топ-20 → ожидаем жёсткую борьбу (OVER)
    if p1_rank <= 20 and p2_rank <= 20:
        p_under -= 0.05  # топ-матч → OVER
    # Если один из них топ-5 → может доминировать (UNDER)
    elif min(p1_rank, p2_rank) <= 5 and max(p1_rank, p2_rank) >= 30:
        p_under += 0.05

    p_under = max(0.25, min(0.75, p_under))
    p_over  = 1.0 - p_under
    prediction = f"UNDER {base_line}" if p_under > p_over else f"OVER {base_line}"

    # Причина
    surface_names = {"hard": "хард", "clay": "глина", "grass": "трава"}
    surf_name = surface_names.get(surface, surface)
    if gap >= 0.30:
        reason = f"Явный фаворит (разрыв {int(gap*100)}%) — ожидается быстрый матч"
    elif surface == "clay":
        reason = f"Глина ({surf_name}) — длинные розыгрыши, много геймов"
    elif p_under < 0.45:
        reason = "Равные игроки — ожидаются тайбрейки и много геймов"
    else:
        reason = f"Небольшой разрыв в классе — стандартный по объёму матч"

    return {
        "prediction":   prediction,
        "line":         base_line,
        "under_prob":   round(p_under, 2),
        "over_prob":    round(p_over, 2),
        "confidence":   round(max(p_under, p_over) * 100),
        "reason":       reason,
        "surface":      surface,
    }


def compute_tennis_chimera_score(
    player1: str,
    player2: str,
    prob_p1: float,
    prob_p2: float,
    odds_p1: float,
    odds_p2: float,
    p1_rank: int = 100,
    p2_rank: int = 100,
    surface: str = "hard",
    p1_form: str = "?????",
    p2_form: str = "?????",
    h2h_p1_wins: int = 0,
    h2h_total: int = 0,
    line_movement: dict = None,
    sport_key: str = "",
) -> list:
    """
    Аналог compute_chimera_score для тенниса.
    Возвращает список кандидатов (обычно 2: P1 и P2).
    """
    from line_movement import get_movement_score

    RANK_WEIGHT    = 25  # max 25 pts — разрыв рейтингов
    SURFACE_WEIGHT = 20  # max 20 pts — специализация на поверхности
    VALUE_WEIGHT   = 30  # max 30 pts — ценность кэфа
    PROB_WEIGHT    = 15  # max 15 pts — сила вероятности
    H2H_WEIGHT     = 10  # max 10 pts — H2H история

    candidates = []

    for player, prob, odds, rank_fav, rank_opp, form, h2h_wins, outcome_key in [
        (player1, prob_p1, odds_p1, p1_rank, p2_rank, p1_form, h2h_p1_wins, "P1"),
        (player2, prob_p2, odds_p2, p2_rank, p1_rank, p2_form, h2h_total - h2h_p1_wins, "P2"),
    ]:
        if not odds or odds <= 1.10 or prob <= 0:
            continue

        implied = 1.0 / odds

        # 1. Рейтинговое преимущество (чем ниже rank = лучше)
        rank_adv = max(0, rank_opp - rank_fav)  # положительный если фаворит
        rank_pts = min(RANK_WEIGHT, rank_adv / 4.0)

        # 2. Поверхность
        surf_str = get_surface_strength(player, surface)
        surf_pts = max(0, (surf_str - 0.55) * SURFACE_WEIGHT / 0.40)

        # 3. Ценность кэфа
        value = prob - implied
        value_pts = max(0, min(VALUE_WEIGHT, value * 300))

        # 4. Сила вероятности
        prob_pts = max(0, min(PROB_WEIGHT, (prob - 0.50) * 100))

        # 5. H2H
        h2h_pts = 0.0
        if h2h_total >= 3:
            rate = h2h_wins / h2h_total
            h2h_pts = max(0, min(H2H_WEIGHT, (rate - 0.33) * H2H_WEIGHT / 0.30))

        # 6. Движение линии
        line_pts = 0.0
        if line_movement:
            outcome_map = {"P1": "home_win", "P2": "away_win"}
            line_pts = get_movement_score(line_movement, outcome_map.get(outcome_key, ""))

        chimera_score = rank_pts + surf_pts + value_pts + prob_pts + h2h_pts + line_pts

        ev = round((prob * odds - 1) * 100, 1)
        kelly_raw = max(0, (prob * odds - 1) / (odds - 1)) * 100 if odds > 1 else 0
        kelly = round(min(kelly_raw, 20.0), 1)  # максимум 20% банка

        surface_emoji = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}.get(surface, "🎾")

        candidates.append({
            "sport":         "tennis",
            "home":          player1,
            "away":          player2,
            "outcome":       outcome_key,
            "team":          player,
            "odds":          round(odds, 2),
            "prob":          round(prob * 100, 1),
            "implied_prob":  round(implied * 100, 1),
            "ev":            ev,
            "kelly":         kelly,
            "chimera_score": round(chimera_score, 1),
            "rank_pts":      round(rank_pts, 1),
            "surf_pts":      round(surf_pts, 1),
            "value_pts":     round(value_pts, 1),
            "prob_pts":      round(prob_pts, 1),
            "h2h_pts":       round(h2h_pts, 1),
            "line_pts":      round(line_pts, 1),
            "rank":          rank_fav,
            "opp_rank":      rank_opp,
            "surface":       surface,
            "surface_emoji": surface_emoji,
            "form":          form[-5:] if form and form != "?????" else "",
            "h2h_wins":      h2h_wins,
            "h2h_total":     h2h_total,
            "sport_key":     sport_key,
            "ai_confirmed":  None,
            "ai_confidence": None,
            "ai_reason":     None,
            "elo_pts":       round(rank_pts, 1),   # alias для format_chimera_signals
            "form_pts":      round(surf_pts, 1),
            "xg_pts":        0.0,
            "h2h_pts":       round(h2h_pts, 1),
            "elo_gap":       abs(rank_fav - rank_opp),
            "league":        sport_key,
        })

    candidates.sort(key=lambda x: x["chimera_score"], reverse=True)
    return candidates
