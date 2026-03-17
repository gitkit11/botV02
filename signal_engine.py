# -*- coding: utf-8 -*-
"""
signal_engine.py — Движок сигналов CHIMERA AI v2
==================================================
Система баллов: сигнал выдаётся когда набрано MIN_SCORE баллов из MAX_SCORE.
Это реалистичнее чем требовать ВСЕ условия одновременно.

Каждое условие даёт 1 балл. Минимум для сигнала: 4 из 6 (футбол), 6 из 10 (CS2).
"""

from typing import Optional, List, Dict
from config_thresholds import FOOTBALL_CFG, CS2_CFG, BASKETBALL_CFG

# Пороги импортированы из config_thresholds.py — редактируй там, не здесь.


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _count_wins(form_str: str) -> int:
    if not form_str:
        return 0
    return form_str.upper().count("W")

def _calc_ev(prob: float, odds: float) -> float:
    if odds <= 1.0:
        return -1.0
    return prob * odds - 1.0

def _kelly(prob: float, odds: float, max_kelly: float = 0.15) -> float:
    b = odds - 1
    q = 1 - prob
    k = (prob * b - q) / b if b > 0 else 0
    return round(max(0.0, min(k, max_kelly)) * 100, 1)

def _strength(score: int, max_score: int, ev: float) -> str:
    ratio = score / max_score
    if ratio >= 0.85 and ev >= 0.20:
        return "🔥🔥 СИЛЬНЫЙ"
    elif ratio >= 0.70 and ev >= 0.12:
        return "🔥 ХОРОШИЙ"
    else:
        return "✅ ОБЫЧНЫЙ"


# ─── Футбол ──────────────────────────────────────────────────────────────────

def check_football_signal(
    home_team: str,
    away_team: str,
    home_prob: float,
    away_prob: float,
    draw_prob: float,
    bookmaker_odds: dict,
    home_form: str = "",
    away_form: str = "",
    elo_home: float = 0,
    elo_away: float = 0,
    ai_agrees: Optional[bool] = None,
    ensemble_prob: Optional[float] = None,
) -> list[dict]:
    c = FOOTBALL_CFG
    signals = []

    candidates = [
        ("П1", home_prob, bookmaker_odds.get("home_win", 0), home_team, home_form, elo_home, elo_away),
        ("П2", away_prob, bookmaker_odds.get("away_win", 0), away_team, away_form, elo_away, elo_home),
        ("Х",  draw_prob, bookmaker_odds.get("draw", 0),     "Ничья",   "",        0,        0),
    ]

    for outcome, prob, odds, team, form, elo_fav, elo_opp in candidates:
        if odds <= 1.0 or prob <= 0:
            continue

        ev_prob = ensemble_prob if ensemble_prob is not None else prob
        ev = _calc_ev(ev_prob, odds)
        score = 0
        checks = []

        # 1. Вероятность
        if prob >= c["min_prob"]:
            score += 1
            checks.append(f"Вероятность {int(prob*100)}% ✅")
        else:
            checks.append(f"Вероятность {int(prob*100)}% ❌")

        # 2. EV
        if ev >= c["min_ev"]:
            score += 1
            checks.append(f"EV +{round(ev*100,1)}% ✅")
        else:
            checks.append(f"EV {round(ev*100,1)}% ❌")

        # 3. Коэффициент
        if c["min_odds"] <= odds <= c["max_odds"]:
            score += 1
            checks.append(f"Кэф {odds} ✅")
        else:
            checks.append(f"Кэф {odds} ❌")

        # 4. Форма (только П1/П2)
        if outcome != "Х" and form:
            wins = _count_wins(form[-5:])
            if wins >= c["min_form_wins"]:
                score += 1
                checks.append(f"Форма {form[-5:]} ({wins}/5) ✅")
            else:
                checks.append(f"Форма {form[-5:]} ({wins}/5) ❌")
        elif outcome != "Х":
            checks.append("Форма: нет данных ⚪")

        # 5. ELO
        if elo_fav > 0 and elo_opp > 0:
            gap = elo_fav - elo_opp
            if gap >= c["min_elo_gap"]:
                score += 1
                checks.append(f"ELO +{gap} ✅")
            else:
                checks.append(f"ELO {gap} ❌")
        else:
            checks.append("ELO: нет данных ⚪")

        # 6. ИИ согласен
        if ai_agrees is True:
            score += 1
            checks.append("ИИ согласен ✅")
        elif ai_agrees is False:
            checks.append("ИИ не согласен ❌")
        # если None — не считаем

        max_score = 5 if outcome == "Х" else 6  # без формы для ничьей

        if score >= c["min_score"]:
            sig = {
                "sport": "football",
                "home": home_team,
                "away": away_team,
                "outcome": outcome,
                "team": team,
                "odds": odds,
                "prob": round(prob * 100, 1),
                "ev": round(ev * 100, 1),
                "kelly": _kelly(ev_prob, odds),
                "score": score,
                "max_score": max_score,
                "strength": _strength(score, max_score, ev),
                "checks": checks,
            }
            if ensemble_prob is not None:
                sig["ai_source"] = "ensemble"
            signals.append(sig)

    return signals


# ─── CS2 ─────────────────────────────────────────────────────────────────────

def check_cs2_signal(
    home_team: str,
    away_team: str,
    home_prob: float,
    away_prob: float,
    bookmaker_odds: dict,
    home_form: str = "",
    away_form: str = "",
    elo_home: float = 0,
    elo_away: float = 0,
    mis_home: float = 0,
    mis_away: float = 0,
    home_avg_rating: float = 0,
    away_avg_rating: float = 0,
    home_map_winrates: Dict[str, float] = None,
    away_map_winrates: Dict[str, float] = None,
    predicted_maps: List[str] = None,
    home_key_players_form: List[float] = None,
    away_key_players_form: List[float] = None,
    ai_cs2_agrees: Optional[bool] = None,
) -> list[dict]:
    c = CS2_CFG
    signals = []

    candidates = [
        ("П1", home_prob, bookmaker_odds.get("home_win", 0), home_team, home_form, elo_home, elo_away, mis_home, mis_away, home_avg_rating, away_avg_rating, home_map_winrates, away_map_winrates, predicted_maps, home_key_players_form, away_key_players_form),
        ("П2", away_prob, bookmaker_odds.get("away_win", 0), away_team, away_form, elo_away, elo_home, mis_away, mis_home, away_avg_rating, home_avg_rating, away_map_winrates, home_map_winrates, predicted_maps, away_key_players_form, home_key_players_form),
    ]

    for outcome, prob, odds, team, form, elo_fav, elo_opp, mis_fav, mis_opp, rat_fav, rat_opp, fav_map_winrates, opp_map_winrates, p_maps, fav_key_players, opp_key_players in candidates:
        if odds <= 1.0 or prob <= 0:
            continue

        ev = _calc_ev(prob, odds)
        score = 0
        checks = []

        # 1. Вероятность
        if prob >= c["min_prob"]:
            score += 1
            checks.append(f"Вероятность {int(prob*100)}% ✅")
        else:
            checks.append(f"Вероятность {int(prob*100)}% ❌")

        # 2. EV
        if ev >= c["min_ev"]:
            score += 1
            checks.append(f"EV +{round(ev*100,1)}% ✅")
        else:
            checks.append(f"EV {round(ev*100,1)}% ❌")

        # 3. Коэффициент
        if c["min_odds"] <= odds <= c["max_odds"]:
            score += 1
            checks.append(f"Кэф {odds} ✅")
        else:
            checks.append(f"Кэф {odds} ❌")

        # 4. Форма
        if form:
            wins = _count_wins(form[-5:])
            if wins >= c["min_form_wins"]:
                score += 1
                checks.append(f"Форма {form[-5:]} ({wins}/5) ✅")
            else:
                checks.append(f"Форма {form[-5:]} ({wins}/5) ❌")
        else:
            checks.append("Форма: нет данных ⚪")

        # 5. ELO
        if elo_fav > 0 and elo_opp > 0:
            gap = elo_fav - elo_opp
            if gap >= c["min_elo_gap"]:
                score += 1
                checks.append(f"ELO +{gap} ✅")
            else:
                checks.append(f"ELO {gap} ❌")
        else:
            checks.append("ELO: нет данных ⚪")

        # 6. MIS (карты)
        if mis_fav > 0 and mis_opp > 0:
            mis_gap = mis_fav - mis_opp
            if mis_gap >= c["min_mis_gap"]:
                score += 1
                checks.append(f"Карты +{round(mis_gap*100,1)}% ✅")
            else:
                checks.append(f"Карты {round(mis_gap*100,1)}% ❌")
        else:
            checks.append("Карты: нет данных ⚪")

        # 7. Рейтинг игроков
        if rat_fav > 0 and rat_opp > 0:
            rat_gap = rat_fav - rat_opp
            if rat_gap >= c["min_rating_gap"]:
                score += 1
                checks.append(f"Рейтинг +{round(rat_gap,2)} ✅")
            else:
                checks.append(f"Рейтинг {round(rat_gap,2)} ❌")
        else:
            checks.append("Рейтинг: нет данных ⚪")

        # 8. Преимущество на ключевых картах (Map Pool Advantage)
        if fav_map_winrates and opp_map_winrates and p_maps:
            fav_avg_winrate = sum(fav_map_winrates.get(m, 0) for m in p_maps) / len(p_maps) if p_maps else 0
            opp_avg_winrate = sum(opp_map_winrates.get(m, 0) for m in p_maps) / len(p_maps) if p_maps else 0
            map_advantage = fav_avg_winrate - opp_avg_winrate
            if map_advantage >= c["min_map_advantage"]:
                score += 1
                checks.append(f"Преимущество на картах +{round(map_advantage*100,1)}% ✅")
            else:
                checks.append(f"Преимущество на картах {round(map_advantage*100,1)}% ❌")
        else:
            checks.append("Преимущество на картах: нет данных ⚪")

        # 9. Форма ключевых игроков (Key Player Form)
        if fav_key_players and opp_key_players:
            fav_kp_avg = sum(fav_key_players) / len(fav_key_players) if fav_key_players else 0
            opp_kp_avg = sum(opp_key_players) / len(opp_key_players) if opp_key_players else 0
            kp_advantage = fav_kp_avg - opp_kp_avg
            if kp_advantage >= c["min_key_player_advantage"]:
                score += 1
                checks.append(f"Ключевые игроки +{round(kp_advantage,2)} ✅")
            else:
                checks.append(f"Ключевые игроки {round(kp_advantage,2)} ❌")
        else:
            checks.append("Ключевые игроки: нет данных ⚪")

        # 10. Подтверждение от АИ
        if ai_cs2_agrees is True:
            score += 1
            checks.append("ИИ CS2 согласен ✅")
        elif ai_cs2_agrees is False:
            checks.append("ИИ CS2 не согласен ❌")

        max_score = 10

        if score >= c["min_score"]:
            signals.append({
                "sport": "cs2",
                "home": home_team,
                "away": away_team,
                "outcome": outcome,
                "team": team,
                "odds": odds,
                "prob": round(prob * 100, 1),
                "ev": round(ev * 100, 1),
                "kelly": _kelly(prob, odds),
                "score": score,
                "max_score": max_score,
                "strength": _strength(score, max_score, ev),
                "checks": checks,
            })

    return signals

def format_signal(sig: dict) -> str:
    """Форматирует сигнал в красивый текст для Telegram."""
    emoji = "⚽️ ФУТБОЛ" if sig['sport'] == "football" else "🎮 CS2"
    
    text = [
        f"<b>{emoji} | {sig['strength']}</b>",
        f"🏆 {sig['home']} vs {sig['away']}",
        f"🎯 Прогноз: <b>{sig['team']} ({sig['outcome']})</b>",
        "",
        f"📊 Вероятность: {sig['prob']}%",
        f"📈 Коэффициент: {sig['odds']}",
        f"💰 Ценность (EV): +{sig['ev']}%",
        f"⚖️ Ставка: {sig['kelly']}% банка ({('3u' if sig['kelly'] >= 4 else '2u' if sig['kelly'] >= 2 else '1u')})",
        "",
        f"📝 <b>Анализ ({sig['score']}/{sig['max_score']} баллов):</b>"
    ]
    
    for check in sig['checks']:
        text.append(f"• {check}")
        
    text.append("\n#ChimeraAI #Прогноз")
    return "\n".join(text)

def predict_cs2_totals(
    home_prob: float,
    away_prob: float,
    home_map_stats: Dict[str, float] = None,
    away_map_stats: Dict[str, float] = None,
    predicted_maps: List[str] = None,
) -> dict:
    """
    Предсказывает тотал карт в BO3 (over/under 2.5).

    Логика:
    - Большой разрыв вероятностей → фаворит сметёт 2:0 → UNDER 2.5
    - Примерно равные команды → затяжная серия 2:1 → OVER 2.5
    - Учёт карт: если обе команды имеют свою сильную карту в пуле → OVER

    Реальная статистика CS2 BO3:
    - При разнице вероятностей 20%+: 2:0 случается ~55-65% матчей
    - При примерно равных командах: 2:1 случается ~60% матчей
    """
    stronger_prob = max(home_prob, away_prob)

    # Базовая вероятность UNDER 2.5 (2:0)
    # stronger_prob=0.50 → p_under≈0.28 (равные команды почти никогда 2:0)
    # stronger_prob=0.65 → p_under≈0.47
    # stronger_prob=0.70 → p_under≈0.55
    # stronger_prob=0.80 → p_under≈0.68
    p_under = 0.28 + (stronger_prob - 0.50) * 1.60
    p_under = max(0.20, min(0.78, p_under))

    # Коррекция по картам: если обе команды сильны на разных картах → OVER
    map_diversity = 0.0
    if home_map_stats and away_map_stats and predicted_maps:
        for m in predicted_maps:
            h_wr = home_map_stats.get(m, 50.0) / 100.0
            a_wr = away_map_stats.get(m, 50.0) / 100.0
            # Карта спорная (<10% разрыв) → обе равны → склоняется к OVER
            if abs(h_wr - a_wr) < 0.10:
                map_diversity += 0.05
            # Карта явно принадлежит одной команде (>20%) → склоняется к UNDER
            elif abs(h_wr - a_wr) > 0.20:
                map_diversity -= 0.03
        p_under -= map_diversity

    p_under = max(0.20, min(0.78, p_under))
    p_over  = 1.0 - p_under
    is_under = p_under > p_over

    # Причина
    gap_pct = int((stronger_prob - 0.50) * 200)
    if is_under and gap_pct >= 30:
        reason = f"Явный фаворит (разрыв +{gap_pct}%) — скорее всего зачистка 2:0"
    elif is_under:
        reason = f"Небольшое преимущество фаворита — возможен 2:0"
    elif map_diversity > 0.05:
        reason = "Обе команды сильны на своих картах — ожидается 2:1"
    else:
        reason = "Равные команды — серия скорее всего затянется до 3-й карты"

    return {
        "prediction":   "UNDER 2.5" if is_under else "OVER 2.5",
        "under_prob":   round(p_under, 2),
        "over_prob":    round(p_over,  2),
        "confidence":   round(max(p_under, p_over) * 100),
        "reason":       reason,
    }


def predict_cs2_round_totals(
    home_prob: float,
    away_prob: float,
    home_map_stats: Dict[str, float] = None,
    away_map_stats: Dict[str, float] = None,
    predicted_maps: List[str] = None,
    match_format: str = "best_of_3",
) -> dict:
    """
    Предсказывает тотал раундов/карт для CS2.

    Для BO3:
    - Тотал карт: OVER/UNDER 2.5 (3 карты = OVER, 2 карты = UNDER)
    - Тотал раундов первой карты: OVER/UNDER 25.5 (стандартный рынок)

    Статистика: в среднем ~26-27 раундов на карту в профессиональном CS2.
    Команды с высоким eco-винрейтом → UNDER (доминируют экономику → короткие карты).
    Равные команды → OVER (много overtime-раундов).
    """
    stronger_prob = max(home_prob, away_prob)
    weaker_prob   = min(home_prob, away_prob)
    gap = stronger_prob - weaker_prob

    # ── Тотал раундов на карте OVER/UNDER 25.5 ────────────────────────────────
    # gap=0 (50/50) → много раундов, gap=0.30+ → доминация, мало раундов
    # Базовая вероятность что карта будет SHORT (< 25.5 раундов)
    # Реальная статистика: ~40% карт заканчиваются 16:8 или быстрее (под 25.5)
    p_rounds_under = 0.35 + gap * 0.80
    p_rounds_under = max(0.22, min(0.72, p_rounds_under))

    # Коррекция по картам: если первая карта спорная (обе команды ≥50% WR) → OVER
    first_map_adjustment = 0.0
    if home_map_stats and away_map_stats and predicted_maps:
        first_map = predicted_maps[0] if predicted_maps else None
        if first_map:
            h_wr = home_map_stats.get(first_map, 50.0) / 100.0
            a_wr = away_map_stats.get(first_map, 50.0) / 100.0
            if h_wr >= 0.50 and a_wr >= 0.50:
                first_map_adjustment = -0.06  # обе сильны → OVER раундов
            elif abs(h_wr - a_wr) > 0.25:
                first_map_adjustment = +0.05  # один явно сильнее → UNDER

    p_rounds_under += first_map_adjustment
    p_rounds_under = max(0.22, min(0.72, p_rounds_under))
    p_rounds_over  = 1.0 - p_rounds_under
    rounds_pred    = "UNDER 25.5" if p_rounds_under > p_rounds_over else "OVER 25.5"

    # ── Тотал карт OVER/UNDER 2.5 (из predict_cs2_totals — переиспользуем) ──
    p_maps_under = 0.28 + (stronger_prob - 0.50) * 1.60
    p_maps_under = max(0.20, min(0.78, p_maps_under))
    if home_map_stats and away_map_stats and predicted_maps:
        map_diversity = 0.0
        for m in predicted_maps:
            h_wr = home_map_stats.get(m, 50.0) / 100.0
            a_wr = away_map_stats.get(m, 50.0) / 100.0
            if abs(h_wr - a_wr) < 0.10:
                map_diversity += 0.05
            elif abs(h_wr - a_wr) > 0.20:
                map_diversity -= 0.03
        p_maps_under -= map_diversity
        p_maps_under = max(0.20, min(0.78, p_maps_under))

    p_maps_over  = 1.0 - p_maps_under
    maps_pred    = "UNDER 2.5 карты" if p_maps_under > p_maps_over else "OVER 2.5 карты"

    # Причина
    if gap >= 0.25:
        rounds_reason = f"Явный фаворит (разрыв {int(gap*100)}%) — ожидается быстрая карта"
    elif p_rounds_over > p_rounds_under:
        rounds_reason = "Равные команды — ожидается много раундов, возможен overtime"
    else:
        rounds_reason = f"Небольшое преимущество — карта может быть короткой"

    return {
        # Тотал карт (BO3)
        "maps_prediction":    maps_pred,
        "maps_under_prob":    round(p_maps_under, 2),
        "maps_over_prob":     round(p_maps_over, 2),
        "maps_confidence":    round(max(p_maps_under, p_maps_over) * 100),
        # Тотал раундов (первая карта)
        "rounds_prediction":  rounds_pred,
        "rounds_under_prob":  round(p_rounds_under, 2),
        "rounds_over_prob":   round(p_rounds_over, 2),
        "rounds_confidence":  round(max(p_rounds_under, p_rounds_over) * 100),
        "rounds_reason":      rounds_reason,
    }


def get_cs2_ranked_bets(
    home_team: str,
    away_team: str,
    home_prob: float,
    away_prob: float,
    bookmaker_odds: dict,
    totals_data: dict = None,
    home_form: str = "",
    away_form: str = "",
) -> list:
    """
    Возвращает список ставок, отсортированных по ожидаемой ценности (EV).

    Типы ставок:
    1. П1 / П2 — победитель матча (из bookmaker_odds)
    2. Тотал < 2.5 / > 2.5 — количество карт (типичный кэф ~1.85)
    3. Фора -1.5 / +1.5 — для явных фаворитов/аутсайдеров (типичный кэф ~2.0 / ~1.40)

    Кэфы на тотал и фору берутся из bookmaker_odds если есть, иначе оценочные.
    """
    bets = []

    h_odds = bookmaker_odds.get("home_win", 0)
    a_odds = bookmaker_odds.get("away_win", 0)

    # ── 1. Победитель ────────────────────────────────────────────────────────
    for prob, odds, team, outcome in [
        (home_prob, h_odds, home_team, "П1"),
        (away_prob, a_odds, away_team, "П2"),
    ]:
        if odds > 1.0:
            ev = prob * odds - 1
            kelly = _kelly(prob, odds)
            if ev > 0.03:
                bets.append({
                    "type":     outcome,
                    "label":    f"{team} победит",
                    "prob":     round(prob * 100, 1),
                    "odds":     odds,
                    "ev":       round(ev * 100, 1),
                    "kelly":    kelly,
                    "priority": ev * prob,
                    "note":     "",
                })

    # ── 2. Тотал карт ────────────────────────────────────────────────────────
    if totals_data:
        pred = totals_data["prediction"]
        is_under = "UNDER" in pred
        total_prob = totals_data["under_prob"] if is_under else totals_data["over_prob"]
        # Используем кэф из bookmaker_odds если есть, иначе типичный 1.85
        if is_under:
            total_odds = bookmaker_odds.get("under_2_5", bookmaker_odds.get("total_under", 1.85))
        else:
            total_odds = bookmaker_odds.get("over_2_5", bookmaker_odds.get("total_over", 1.85))

        ev = total_prob * total_odds - 1
        kelly = _kelly(total_prob, total_odds)
        if ev > 0.02:
            label = f"Тотал карт Меньше 2.5 (завершится 2:0)" if is_under else f"Тотал карт Больше 2.5 (завершится 2:1)"
            bets.append({
                "type":     pred,
                "label":    label,
                "prob":     round(total_prob * 100, 1),
                "odds":     total_odds,
                "ev":       round(ev * 100, 1),
                "kelly":    kelly,
                "priority": ev * total_prob,
                "note":     totals_data.get("reason", ""),
            })

    # ── 3. Фора -1.5 для явного фаворита ─────────────────────────────────────
    # Фора -1.5 выигрывает только при 2:0. P(2:0) ≈ p_under (если фаворит)
    stronger_team  = home_team  if home_prob >= away_prob else away_team
    stronger_prob2 = max(home_prob, away_prob)
    if stronger_prob2 >= 0.65 and totals_data:
        hc_prob = totals_data["under_prob"]  # вероятность 2:0 ≈ вероятность UNDER
        hc_odds = bookmaker_odds.get("handicap_minus", 2.05)  # типичный кэф фора -1.5
        ev = hc_prob * hc_odds - 1
        kelly = _kelly(hc_prob, hc_odds)
        if ev > 0.04:
            bets.append({
                "type":     "Фора -1.5",
                "label":    f"{stronger_team} с форой -1.5 (победа 2:0)",
                "prob":     round(hc_prob * 100, 1),
                "odds":     hc_odds,
                "ev":       round(ev * 100, 1),
                "kelly":    kelly,
                "priority": ev * hc_prob * 0.9,  # слегка ниже приоритет чем прямая
                "note":     "Требует победы всухую, проверь кэф на своём БК",
            })

    # Сортируем по EV * вероятность (ожидаемая прибыль)
    bets.sort(key=lambda x: x["priority"], reverse=True)
    return bets


def format_signals_list(signals: list[dict]) -> str:
    """Форматирует список сигналов в красивый текст для Telegram."""
    if not signals:
        return "📊 Сигналов не найдено. Рынок спокоен или нет ценных ставок."
    
    text = [f"🎯 <b>Найдено сигналов: {len(signals)}</b>\n"]
    
    # Сортируем по EV (ценность)
    sorted_signals = sorted(signals, key=lambda x: x.get('ev', 0), reverse=True)
    
    for i, sig in enumerate(sorted_signals[:10], 1):  # Показываем топ-10
        emoji = "⚽️" if sig['sport'] == "football" else "🎮"
        text.append(f"{i}. {emoji} <b>{sig['home']} vs {sig['away']}</b>")
        text.append(f"   Прогноз: {sig['team']} ({sig['outcome']}) | EV: +{sig['ev']}% | Кэф: {sig['odds']}")
        text.append(f"   Сила: {sig['strength']} | Баллы: {sig['score']}/{sig['max_score']}\n")
    
    text.append("\n#ChimeraAI #Сигналы")
    return "\n".join(text)
