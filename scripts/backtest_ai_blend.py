# -*- coding: utf-8 -*-
"""
scripts/backtest_ai_blend.py
============================
Симуляция: сравниваем точность и ROI
  A) Только математическая модель (ELO + Odds + Home + Form)
  B) Модель + AI-блендирование (GPT 12% веса)

Запуск: python scripts/backtest_ai_blend.py
"""

import sqlite3
import os
import sys

# Чтобы импортировать из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "chimera_predictions.db")

AI_WEIGHT = 0.12  # тот же вес что в blend_ai_verdicts


# ── Парсим вердикт из текста GPT/Llama ────────────────────────────────────────

def _parse_verdict_from_text(text: str, home: str, away: str) -> str:
    """
    Определяет кого поддержал AI по тексту.
    Возвращает 'home_win', 'away_win' или '' если неясно.
    """
    if not text or text.startswith("❌") or "недоступна" in text.lower():
        return ""

    txt = text.lower()
    h_short = home.split()[-1].lower()   # последнее слово (напр. "celtics")
    a_short = away.split()[-1].lower()

    # Ищем слова победы рядом с именем команды
    win_keywords = ["победит", "фаворит", "преимуществ", "выиграет",
                    "лучше", "превосходств", "победа", "выигрыш",
                    "favour", "win", "advantage", "better", "stronger"]

    h_score = 0
    a_score = 0

    for kw in win_keywords:
        # Ищем в окне ±100 символов от ключевого слова
        idx = 0
        while True:
            pos = txt.find(kw, idx)
            if pos == -1:
                break
            window = txt[max(0, pos - 80):pos + 80]
            if h_short in window:
                h_score += 1
            if a_short in window:
                a_score += 1
            idx = pos + 1

    # Специальные паттерны
    if "хозяев" in txt or "домашней команды" in txt or "home team" in txt:
        h_score += 2
    if "гостей" in txt or "гостевой" in txt or "away team" in txt:
        a_score += 2

    # Итог
    if h_score > a_score:
        return "home_win"
    elif a_score > h_score:
        return "away_win"
    return ""  # неопределённость


def _blend_prob(h_prob: float, gpt_verdict: str, llama_verdict: str,
                home: str, away: str, weight: float = AI_WEIGHT) -> float:
    """Применяет AI-блендирование к базовой вероятности."""
    votes_h = 0.0
    conf_sum = 0.0
    default_conf = 0.65  # средняя уверенность AI если нет данных

    for verdict in [gpt_verdict, llama_verdict]:
        if verdict == "home_win":
            votes_h += default_conf
            conf_sum += default_conf
        elif verdict == "away_win":
            conf_sum += default_conf

    if conf_sum < 0.01:
        return h_prob  # AI не дал чёткого сигнала — модель без изменений

    ai_h = votes_h / conf_sum
    blended = h_prob * (1 - weight) + ai_h * weight
    return round(max(0.05, min(0.95, blended)), 4)


# ── Загрузка и симуляция ──────────────────────────────────────────────────────

def run_backtest(sport: str = "basketball"):
    table = f"{sport}_predictions"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"""
            SELECT home_team, away_team,
                   ensemble_home, ensemble_away,
                   gpt_verdict, llama_verdict,
                   recommended_outcome, real_outcome, is_correct,
                   bookmaker_odds_home, bookmaker_odds_away,
                   bet_signal
            FROM {table}
            WHERE real_outcome IS NOT NULL
              AND real_outcome != 'expired'
              AND real_outcome != ''
        """).fetchall()
    except Exception as e:
        print(f"Ошибка: {e}")
        conn.close()
        return
    conn.close()

    if not rows:
        print(f"Нет завершённых матчей в {table}")
        return

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {sport.upper()}  |  {len(rows)} матчей")
    print(f"{'='*60}\n")

    results_model = []  # (correct, roi)
    results_ai    = []
    results_bk    = []  # букмекерская implied prob

    no_ai_count = 0

    for row in rows:
        home  = row["home_team"] or ""
        away  = row["away_team"] or ""
        h_prob_model = float(row["ensemble_home"] or 0.5)
        a_prob_model = float(row["ensemble_away"] or 0.5)
        real  = row["real_outcome"]
        odds_h = float(row["bookmaker_odds_home"] or 0)
        odds_a = float(row["bookmaker_odds_away"] or 0)
        gpt_txt   = row["gpt_verdict"]   or ""
        llama_txt = row["llama_verdict"] or ""

        # ── Сценарий A: чистая модель ──────────────────────────────────────
        pred_model = "home_win" if h_prob_model >= a_prob_model else "away_win"
        correct_model = int(pred_model == real)
        odds_model = odds_h if pred_model == "home_win" else odds_a
        roi_model  = (float(odds_model) - 1) if (correct_model and odds_model > 1.01) else (-1.0 if odds_model > 1.01 else 0)

        results_model.append((correct_model, roi_model, odds_model))

        # ── Сценарий B: модель + AI блендирование ─────────────────────────
        gpt_verdict   = _parse_verdict_from_text(gpt_txt, home, away)
        llama_verdict = _parse_verdict_from_text(llama_txt, home, away)

        has_ai = bool(gpt_verdict or llama_verdict)
        if not has_ai:
            no_ai_count += 1

        h_prob_ai = _blend_prob(h_prob_model, gpt_verdict, llama_verdict, home, away)
        pred_ai = "home_win" if h_prob_ai >= (1 - h_prob_ai) else "away_win"
        correct_ai = int(pred_ai == real)
        odds_ai = odds_h if pred_ai == "home_win" else odds_a
        roi_ai  = (float(odds_ai) - 1) if (correct_ai and odds_ai > 1.01) else (-1.0 if odds_ai > 1.01 else 0)

        results_ai.append((correct_ai, roi_ai, odds_ai))

        # ── Сценарий C: чистый букмекер (baseline) ────────────────────────
        if odds_h > 1.01 and odds_a > 1.01:
            bk_pred = "home_win" if (1/odds_h) > (1/odds_a) else "away_win"
            correct_bk = int(bk_pred == real)
            odds_bk = odds_h if bk_pred == "home_win" else odds_a
            roi_bk  = (float(odds_bk) - 1) if (correct_bk and odds_bk > 1.01) else -1.0
            results_bk.append((correct_bk, roi_bk, odds_bk))

    def _stats(results, label):
        n = len(results)
        if n == 0:
            return
        wins  = sum(r[0] for r in results)
        total_roi = sum(r[1] for r in results if r[2] > 1.01)
        bets  = sum(1 for r in results if r[2] > 1.01)
        acc   = wins / n * 100
        roi_pct = total_roi / bets * 100 if bets > 0 else 0

        # Стрик
        streak = cur = 0
        last = None
        for r in results:
            if r[0] == last:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 1
                last = r[0]

        bar_filled = round(acc / 10)
        bar = "▓" * bar_filled + "░" * (10 - bar_filled)
        print(f"  {label}")
        print(f"    [{bar}] {acc:.1f}%  ({wins}/{n} угадано)")
        print(f"    ROI: {roi_pct:+.1f}%  |  Лучший стрик: {streak}")
        print()

    print("─── Все матчи ───────────────────────────────────────")
    _stats(results_model, "A) Только модель (ELO + Odds + Home + Form)")
    _stats(results_ai,    "B) Модель + AI блендирование (12% вес)")
    if results_bk:
        _stats(results_bk, "C) Чистый букмекер (baseline)")

    # Только матчи где AI дал чёткий сигнал
    ai_clear_model = [(results_model[i][0], results_model[i][1], results_model[i][2])
                      for i, row in enumerate(rows)
                      if _parse_verdict_from_text(row["gpt_verdict"] or "", row["home_team"] or "", row["away_team"] or "")
                      or _parse_verdict_from_text(row["llama_verdict"] or "", row["home_team"] or "", row["away_team"] or "")]
    ai_clear_ai   = [results_ai[i]
                     for i, row in enumerate(rows)
                     if _parse_verdict_from_text(row["gpt_verdict"] or "", row["home_team"] or "", row["away_team"] or "")
                     or _parse_verdict_from_text(row["llama_verdict"] or "", row["home_team"] or "", row["away_team"] or "")]

    if ai_clear_model:
        print(f"─── Только матчи с чётким AI сигналом ({len(ai_clear_model)}/{len(rows)}) ──")
        _stats(ai_clear_model, "A) Только модель")
        _stats(ai_clear_ai,    "B) Модель + AI")

    # Матчи где AI СОГЛАСИЛСЯ с моделью vs РАЗОШЁЛСЯ
    agree_model = []
    agree_ai    = []
    disagree_model = []
    disagree_ai    = []

    for i, row in enumerate(rows):
        gv = _parse_verdict_from_text(row["gpt_verdict"] or "", row["home_team"] or "", row["away_team"] or "")
        if not gv:
            continue
        model_pred = "home_win" if float(row["ensemble_home"] or 0.5) >= float(row["ensemble_away"] or 0.5) else "away_win"
        if gv == model_pred:
            agree_model.append(results_model[i])
            agree_ai.append(results_ai[i])
        else:
            disagree_model.append(results_model[i])
            disagree_ai.append(results_ai[i])

    if agree_model:
        print(f"─── Модель и AI СОГЛАСНЫ ({len(agree_model)} матчей) ──────────────")
        _stats(agree_model, "A) Только модель")
        _stats(agree_ai,    "B) Модель + AI")

    if disagree_model:
        print(f"─── Модель и AI РАСХОДЯТСЯ ({len(disagree_model)} матчей) ──────────")
        _stats(disagree_model, "A) Только модель")
        _stats(disagree_ai,    "B) Модель + AI (AI перевешивает)")

    print(f"\nМатчей без AI данных: {no_ai_count}/{len(rows)}")
    print(f"Вес AI в блендировании: {AI_WEIGHT*100:.0f}%")


if __name__ == "__main__":
    sport = sys.argv[1] if len(sys.argv) > 1 else "basketball"
    run_backtest(sport)
    if sport == "basketball":
        print("\n")
        run_backtest("football")
