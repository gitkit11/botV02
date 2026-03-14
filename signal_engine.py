# -*- coding: utf-8 -*-
"""
signal_engine.py — Движок сигналов CHIMERA AI v2
==================================================
Система баллов: сигнал выдаётся когда набрано MIN_SCORE баллов из MAX_SCORE.
Это реалистичнее чем требовать ВСЕ условия одновременно.

Каждое условие даёт 1 балл. Минимум для сигнала: 4 из 6 (футбол), 6 из 10 (CS2).
"""

from typing import Optional, List, Dict

# ─── Пороги ──────────────────────────────────────────────────────────────────

FOOTBALL_CFG = {
    "min_prob":     0.55,   # Вероятность минимум 55%
    "min_ev":       0.07,   # EV минимум 7%
    "min_odds":     1.45,   # Не брать очевидных фаворитов
    "max_odds":     4.00,
    "min_form_wins": 3,     # Минимум 3 победы из последних 5
    "min_elo_gap":  40,     # Разница ELO минимум 40 очков
    "min_score":    4,      # Минимум 4 балла из 6 для сигнала
}

CS2_CFG = {
    "min_prob":             0.55,
    "min_ev":               0.07,
    "min_odds":             1.45,
    "max_odds":             3.50,
    "min_form_wins":        3,
    "min_elo_gap":          30,
    "min_mis_gap":          0.03,   # Преимущество по картам минимум 3%
    "min_rating_gap":       0.06, # Разница рейтингов игроков
    "min_map_advantage":    0.10,  # Новый: мин. преимущество на картах (в %)
    "min_key_player_advantage": 0.15, # Новый: мин. преимущество ключевых игроков (в %)
    "min_score":            6,     # Изменено: Минимум 6 баллов из 10 для сигнала
}


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

        if score >= c["min_score"] and ev > 0:
            signals.append({
                "sport": "football",
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
    home_map_winrates: Dict[str, float] = None, # Новый параметр
    away_map_winrates: Dict[str, float] = None, # Новый параметр
    predicted_maps: List[str] = None,          # Новый параметр
    home_key_players_form: List[float] = None, # Новый параметр
    away_key_players_form: List[float] = None, # Новый параметр
    ai_cs2_agrees: Optional[bool] = None,      # Новый параметр
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

        # 8. Преимущество на ключевых картах (Map Pool Advantage) - НОВЫЙ
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

        # 9. Форма ключевых игроков (Key Player Form) - НОВЫЙ
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

        # 10. Подтверждение от быстрого ИИ-агента (AI CS2 Agrees) - НОВЫЙ
        if ai_cs2_agrees is True:
            score += 1
            checks.append("ИИ CS2 согласен ✅")
        elif ai_cs2_agrees is False:
            checks.append("ИИ CS2 не согласен ❌")
        else:
            checks.append("ИИ CS2: нет данных ⚪")

        max_score = 10 # Обновлено до 10 критериев

        if score >= c["min_score"] and ev > 0:
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


# ─── Форматирование ───────────────────────────────────────────────────────────

def format_signal(signal: dict) -> str:
    icon = "🎮" if signal["sport"] == "cs2" else "⚽"
    passing = [c for c in signal["checks"] if "✅" in c]
    lines = [
        f"{icon} *{signal["home"]} vs {signal["away"]}*",
        f"",
        f"{signal["strength"]}  ({signal["score"]}/{signal["max_score"]} факторов)",
        f"",
        f"📌 *{signal["outcome"]}* — {signal["team"]}",
        f"💰 Кэф: *{signal["odds"]}*  |  Вероятность: *{signal["prob"]}%*",
        f"📈 EV: *+{signal["ev"]}%*  |  Ставка: *{signal["kelly"]}% банка*",
        f"",
        f"✅ Факторы за:",
    ]
    for c in passing:
        lines.append(f"  • {c.replace(" ✅","")}")
    return "\n".join(lines)


def format_signals_list(signals: list[dict], title: str = "📡 СИГНАЛЫ ДНЯ") -> str:
    if not signals:
        return (
            f"{title}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"❌ Сегодня нет сигналов.\n\n"
            f"Матчи не набрали достаточно факторов:\n"
            f"• Нужно 6+ из 10 условий для CS2\n"
            f"• Вероятность > 55%\n"
            f"• EV > 7%\n\n"
            f"_Попробуй позже — матчи обновляются._"
        )

    lines = [
        f"{title}",
        f"━━━━━━━━━━━━━━━━━━━━━━━",
        f"Найдено: *{len(signals)}* сигнал(а)",
        f"",
    ]
    for i, sig in enumerate(signals, 1):
        lines.append(f"*— Сигнал {i} —*")
        lines.append(format_signal(sig))
        lines.append("")

    lines.append(f"⚠️ _Управляй банком: не более {signals[0]["kelly"]}% на сигнал_")
    return "\n".join(lines)


# ─── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sigs = check_football_signal(
        "Manchester City", "Burnley",
        home_prob=0.68, away_prob=0.16, draw_prob=0.16,
        bookmaker_odds={"home_win": 1.72, "draw": 4.10, "away_win": 5.20},
        home_form="WWWLW", away_form="LLLWL",
        elo_home=1880, elo_away=1640,
    )
    print("=== Футбол ===")
    print(format_signals_list(sigs))

    sigs2 = check_cs2_signal(
        "Team Vitality", "Astralis",
        home_prob=0.72, away_prob=0.28,
        bookmaker_odds={"home_win": 1.62, "away_win": 2.35},
        home_form="WWWWL", away_form="LLWLL",
        elo_home=1820, elo_away=1610,
        mis_home=0.57, mis_away=0.43,
        home_avg_rating=1.19, away_avg_rating=1.05,
        home_map_winrates={
            "Inferno": 0.75, "Mirage": 0.65, "Nuke": 0.80, "Overpass": 0.70,
            "Vertigo": 0.60, "Ancient": 0.55, "Anubis": 0.68
        },
        away_map_winrates={
            "Inferno": 0.60, "Mirage": 0.50, "Nuke": 0.65, "Overpass": 0.55,
            "Vertigo": 0.45, "Ancient": 0.40, "Anubis": 0.58
        },
        predicted_maps=["Nuke", "Inferno", "Mirage"],
        home_key_players_form=[1.30, 1.25, 1.18],
        away_key_players_form=[1.10, 1.05, 0.98],
        ai_cs2_agrees=True,
    )
    print("=== CS2 ===")
    print(format_signals_list(sigs2))
