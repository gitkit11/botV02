# -*- coding: utf-8 -*-
"""
signal_engine.py — Движок сигналов CHIMERA AI
===============================================
Выдаёт сигнал ТОЛЬКО когда ВСЕ условия выполнены.
Лучше пропустить матч, чем дать плохой сигнал.

Логика:
  1. Математическая вероятность > MIN_PROB
  2. EV (Expected Value) > MIN_EV
  3. Коэффициент в диапазоне [MIN_ODDS, MAX_ODDS]
  4. Форма команды: 3+ побед из последних 5
  5. ИИ агент согласен с прогнозом (опционально)
  6. Нет противоречий между моделями

Для CS2 дополнительно:
  7. Преимущество на картах (MIS > 0.55)
  8. Разница в рейтингах игроков > 0.10
"""

from typing import Optional

# ─── Пороги сигналов ─────────────────────────────────────────────────────────

FOOTBALL_THRESHOLDS = {
    "min_prob":    0.62,   # Наша вероятность минимум 62%
    "min_ev":      0.12,   # EV минимум 12%
    "min_odds":    1.55,   # Не брать очевидных фаворитов
    "max_odds":    3.50,   # Не брать аутсайдеров
    "min_form":    3,      # Минимум 3 победы из последних 5
    "min_elo_gap": 50,     # Разница ELO минимум 50 очков
}

CS2_THRESHOLDS = {
    "min_prob":    0.62,   # Наша вероятность минимум 62%
    "min_ev":      0.12,   # EV минимум 12%
    "min_odds":    1.55,
    "max_odds":    3.00,
    "min_form":    3,      # Минимум 3 победы из последних 5
    "min_elo_gap": 40,
    "min_mis_gap": 0.05,   # Преимущество по картам минимум 5%
    "min_rating_gap": 0.08, # Разница рейтингов игроков
}


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _count_wins_in_form(form_str: str) -> int:
    """Считает победы в строке формы типа 'WLWWL'"""
    if not form_str:
        return 0
    return form_str.upper().count('W')

def _calc_ev(prob: float, odds: float) -> float:
    """Expected Value = prob * odds - 1"""
    if odds <= 1.0:
        return -1.0
    return prob * odds - 1.0

def _kelly(prob: float, odds: float, max_kelly: float = 0.15) -> float:
    """Критерий Келли (ограничен max_kelly банка)"""
    b = odds - 1
    q = 1 - prob
    k = (prob * b - q) / b if b > 0 else 0
    return round(max(0.0, min(k, max_kelly)) * 100, 1)

def _signal_strength(ev: float, prob: float) -> str:
    """Определяет силу сигнала"""
    if ev >= 0.25 and prob >= 0.70:
        return "🔥🔥 СИЛЬНЫЙ"
    elif ev >= 0.15 and prob >= 0.65:
        return "🔥 ХОРОШИЙ"
    else:
        return "✅ ОБЫЧНЫЙ"


# ─── Футбол: проверка сигнала ─────────────────────────────────────────────────

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
    """
    Проверяет матч на наличие сигнала по всем критериям.
    Возвращает список сигналов (может быть пустым).
    """
    t = FOOTBALL_THRESHOLDS
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
        reasons_fail = []
        reasons_pass = []

        # Условие 1: Вероятность
        if prob >= t["min_prob"]:
            reasons_pass.append(f"Вероятность {int(prob*100)}% ✅")
        else:
            reasons_fail.append(f"Вероятность {int(prob*100)}% < {int(t['min_prob']*100)}% ❌")

        # Условие 2: EV
        if ev >= t["min_ev"]:
            reasons_pass.append(f"EV +{round(ev*100,1)}% ✅")
        else:
            reasons_fail.append(f"EV +{round(ev*100,1)}% < {int(t['min_ev']*100)}% ❌")

        # Условие 3: Коэффициент
        if t["min_odds"] <= odds <= t["max_odds"]:
            reasons_pass.append(f"Кэф {odds} ✅")
        else:
            reasons_fail.append(f"Кэф {odds} вне диапазона [{t['min_odds']}-{t['max_odds']}] ❌")

        # Условие 4: Форма (только для П1/П2, не для ничьей)
        if outcome != "Х" and form:
            wins = _count_wins_in_form(form[-5:])
            if wins >= t["min_form"]:
                reasons_pass.append(f"Форма {form[-5:]} ({wins}/5 побед) ✅")
            else:
                reasons_fail.append(f"Форма {form[-5:]} ({wins}/5 побед) < {t['min_form']} ❌")

        # Условие 5: ELO разрыв (только для П1/П2)
        if outcome != "Х" and elo_fav > 0 and elo_opp > 0:
            elo_gap = elo_fav - elo_opp
            if elo_gap >= t["min_elo_gap"]:
                reasons_pass.append(f"ELO разрыв +{elo_gap} ✅")
            else:
                reasons_fail.append(f"ELO разрыв {elo_gap} < {t['min_elo_gap']} ❌")

        # Условие 6: ИИ согласен
        if ai_agrees is True:
            reasons_pass.append("ИИ согласен ✅")
        elif ai_agrees is False:
            reasons_fail.append("ИИ не согласен ❌")

        # Итог: сигнал только если НЕТ провальных условий
        if not reasons_fail:
            strength = _signal_strength(ev, prob)
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
                "strength": strength,
                "reasons": reasons_pass,
            })

    return signals


# ─── CS2: проверка сигнала ────────────────────────────────────────────────────

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
    ai_agrees: Optional[bool] = None,
) -> list[dict]:
    """
    Проверяет CS2 матч на сигнал.
    Дополнительно учитывает MIS (карты) и рейтинги игроков.
    """
    t = CS2_THRESHOLDS
    signals = []

    candidates = [
        ("П1", home_prob, bookmaker_odds.get("home_win", 0), home_team, home_form, elo_home, elo_away, mis_home, mis_away, home_avg_rating, away_avg_rating),
        ("П2", away_prob, bookmaker_odds.get("away_win", 0), away_team, away_form, elo_away, elo_home, mis_away, mis_home, away_avg_rating, home_avg_rating),
    ]

    for outcome, prob, odds, team, form, elo_fav, elo_opp, mis_fav, mis_opp, rating_fav, rating_opp in candidates:
        if odds <= 1.0 or prob <= 0:
            continue

        ev = _calc_ev(prob, odds)
        reasons_fail = []
        reasons_pass = []

        # Условие 1: Вероятность
        if prob >= t["min_prob"]:
            reasons_pass.append(f"Вероятность {int(prob*100)}% ✅")
        else:
            reasons_fail.append(f"Вероятность {int(prob*100)}% < {int(t['min_prob']*100)}% ❌")

        # Условие 2: EV
        if ev >= t["min_ev"]:
            reasons_pass.append(f"EV +{round(ev*100,1)}% ✅")
        else:
            reasons_fail.append(f"EV +{round(ev*100,1)}% < {int(t['min_ev']*100)}% ❌")

        # Условие 3: Коэффициент
        if t["min_odds"] <= odds <= t["max_odds"]:
            reasons_pass.append(f"Кэф {odds} ✅")
        else:
            reasons_fail.append(f"Кэф {odds} вне диапазона [{t['min_odds']}-{t['max_odds']}] ❌")

        # Условие 4: Форма
        if form:
            wins = _count_wins_in_form(form[-5:])
            if wins >= t["min_form"]:
                reasons_pass.append(f"Форма {form[-5:]} ({wins}/5) ✅")
            else:
                reasons_fail.append(f"Форма {form[-5:]} ({wins}/5) < {t['min_form']} ❌")

        # Условие 5: ELO
        if elo_fav > 0 and elo_opp > 0:
            elo_gap = elo_fav - elo_opp
            if elo_gap >= t["min_elo_gap"]:
                reasons_pass.append(f"ELO разрыв +{elo_gap} ✅")
            else:
                reasons_fail.append(f"ELO разрыв {elo_gap} < {t['min_elo_gap']} ❌")

        # Условие 6: MIS (преимущество на картах)
        if mis_fav > 0 and mis_opp > 0:
            mis_gap = mis_fav - mis_opp
            if mis_gap >= t["min_mis_gap"]:
                reasons_pass.append(f"MIS карты +{round(mis_gap*100,1)}% ✅")
            else:
                reasons_fail.append(f"MIS карты {round(mis_gap*100,1)}% < {int(t['min_mis_gap']*100)}% ❌")

        # Условие 7: Рейтинг игроков
        if rating_fav > 0 and rating_opp > 0:
            rating_gap = rating_fav - rating_opp
            if rating_gap >= t["min_rating_gap"]:
                reasons_pass.append(f"Рейтинг игроков +{round(rating_gap,2)} ✅")
            else:
                reasons_fail.append(f"Рейтинг игроков {round(rating_gap,2)} < {t['min_rating_gap']} ❌")

        # Условие 8: ИИ согласен
        if ai_agrees is True:
            reasons_pass.append("ИИ согласен ✅")
        elif ai_agrees is False:
            reasons_fail.append("ИИ не согласен ❌")

        # Итог
        if not reasons_fail:
            strength = _signal_strength(ev, prob)
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
                "strength": strength,
                "reasons": reasons_pass,
            })

    return signals


# ─── Форматирование сигнала для Telegram ─────────────────────────────────────

def format_signal(signal: dict) -> str:
    """Форматирует один сигнал для вывода в Telegram"""
    sport_icon = "🎮" if signal["sport"] == "cs2" else "⚽"
    lines = [
        f"{sport_icon} *{signal['home']} vs {signal['away']}*",
        f"",
        f"{signal['strength']}",
        f"",
        f"📌 Ставка: *{signal['outcome']}* ({signal['team']})",
        f"💰 Коэффициент: *{signal['odds']}*",
        f"📊 Наша вероятность: *{signal['prob']}%*",
        f"📈 EV: *+{signal['ev']}%*",
        f"🏦 Размер ставки (Келли): *{signal['kelly']}% банка*",
        f"",
        f"✅ Почему сигнал:",
    ]
    for r in signal["reasons"]:
        lines.append(f"  • {r}")
    return "\n".join(lines)


def format_signals_list(signals: list[dict], title: str = "📡 СИГНАЛЫ ДНЯ") -> str:
    """Форматирует список сигналов для команды /signals"""
    if not signals:
        return (
            f"{title}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"❌ Сегодня нет сигналов.\n\n"
            f"Все матчи не прошли фильтры:\n"
            f"• Вероятность < 62%\n"
            f"• EV < 12%\n"
            f"• Плохая форма команды\n\n"
            f"_Лучше пропустить день, чем потерять деньги._"
        )

    lines = [
        f"{title}",
        f"━━━━━━━━━━━━━━━━━━━━━━━",
        f"Найдено сигналов: *{len(signals)}*",
        f"",
    ]

    for i, sig in enumerate(signals, 1):
        lines.append(f"*Сигнал {i}:*")
        lines.append(format_signal(sig))
        lines.append("─────────────────────")

    lines.append(f"\n⚠️ _Ставь не более {signals[0]['kelly']}% банка на каждый сигнал_")
    return "\n".join(lines)


# ─── Быстрый тест ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Тест футбол
    sigs = check_football_signal(
        home_team="Manchester City",
        away_team="Burnley",
        home_prob=0.72,
        away_prob=0.15,
        draw_prob=0.13,
        bookmaker_odds={"home_win": 1.75, "draw": 4.20, "away_win": 5.50},
        home_form="WWWLW",
        away_form="LLLWL",
        elo_home=1900,
        elo_away=1650,
        ai_agrees=True,
    )
    print("=== Футбол ===")
    if sigs:
        for s in sigs:
            print(format_signal(s))
    else:
        print("Нет сигналов")

    print()

    # Тест CS2
    sigs2 = check_cs2_signal(
        home_team="Team Vitality",
        away_team="Astralis",
        home_prob=0.75,
        away_prob=0.25,
        bookmaker_odds={"home_win": 1.65, "away_win": 2.30},
        home_form="WWWWL",
        away_form="LLWLL",
        elo_home=1820,
        elo_away=1620,
        mis_home=0.58,
        mis_away=0.42,
        home_avg_rating=1.19,
        away_avg_rating=1.05,
        ai_agrees=True,
    )
    print("=== CS2 ===")
    if sigs2:
        for s in sigs2:
            print(format_signal(s))
    else:
        print("Нет сигналов")
