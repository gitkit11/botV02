# -*- coding: utf-8 -*-
"""
scripts/backtest_ai.py — Бэктест стратегий: модель vs модель+AI.

Запуск: python scripts/backtest_ai.py
"""
import sys, io, sqlite3, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DB = "chimera_predictions.db"

# ─── Парсинг AI-вердикта из любого формата ────────────────────────────────────

def parse_verdict(text, home_team, away_team):
    """
    Возвращает 'home_win' | 'away_win' | 'draw' | None.
    Понимает:
      • структурированные: 'home_win', 'away_win'
      • русские: 'Победа хозяев', 'Победа гостей', 'Ничья'
      • прозу: ищет имена команд в тексте фаворита
    """
    if not text or not text.strip():
        return None
    t = text.strip()

    # Структурированный формат
    if t in ("home_win", "away_win", "draw"):
        return t

    # Русские метки
    ru_map = {
        "победа хозяев": "home_win",
        "победа гостей": "away_win",
        "ничья":         "draw",
        "пропустить":    None,
        "недоступна":    None,
        "тень недоступна": None,
        "error":         None,
    }
    tl = t.lower()
    for key, val in ru_map.items():
        if key in tl:
            return val

    # Проза: ищем, кто явный фаворит по контексту
    # Простая эвристика: кто упоминается раньше как «явный фаворит» / «являются фаворитом»
    h_lower = home_team.lower().split()[-1]   # фамилия / последнее слово
    a_lower = away_team.lower().split()[-1]
    tl = tl.replace("(llama недоступна", "").replace("(тень недоступна", "")

    # Найдём первое вхождение каждого в тексте
    h_pos = next((tl.find(w) for w in home_team.lower().split() if tl.find(w) >= 0), -1)
    a_pos = next((tl.find(w) for w in away_team.lower().split() if tl.find(w) >= 0), -1)

    if h_pos == -1 and a_pos == -1:
        return None

    # Ключевые слова победителя рядом с именем
    win_keywords = ["фаворит", "победит", "преимущест", "выиграет", "явным", "более высок"]
    for kw in win_keywords:
        kw_pos = tl.find(kw)
        if kw_pos == -1:
            continue
        # Кто ближе к этому ключевому слову?
        if h_pos >= 0 and a_pos >= 0:
            return "home_win" if abs(h_pos - kw_pos) < abs(a_pos - kw_pos) else "away_win"
        elif h_pos >= 0:
            return "home_win"
        elif a_pos >= 0:
            return "away_win"

    # Последняя попытка: кто упоминается первым?
    if h_pos >= 0 and a_pos >= 0:
        return "home_win" if h_pos < a_pos else "away_win"
    return None


# ─── Расчёт ROI ───────────────────────────────────────────────────────────────

def calc_roi(wins, total, avg_odds, stake=1.0):
    if total == 0:
        return 0.0
    profit = wins * (avg_odds - 1) * stake - (total - wins) * stake
    return round(profit / (total * stake) * 100, 1)


# ─── Загрузка и анализ таблицы ────────────────────────────────────────────────

def analyze_table(conn, table, sport):
    c = conn.cursor()
    c.execute(f"""
        SELECT home_team, away_team,
               gpt_verdict, llama_verdict,
               recommended_outcome, real_outcome,
               is_correct, bet_signal,
               ensemble_home, ensemble_away,
               bookmaker_odds_home, bookmaker_odds_away
        FROM {table}
        WHERE real_outcome IS NOT NULL AND real_outcome != 'expired'
        ORDER BY id
    """)
    rows = c.fetchall()

    print(f"\n{'='*60}")
    print(f"  {sport.upper()}  ({table})  — {len(rows)} матчей с результатом")
    print(f"{'='*60}")

    # ── Стратегия 1: Модель (ensemble) — ставим на recommended_outcome ────────
    s1_total = s1_wins = 0
    s1_odds  = []

    # ── Стратегия 2: Модель + оба AI согласны ─────────────────────────────────
    s2_total = s2_wins = 0
    s2_odds  = []

    # ── Стратегия 3: Модель + хотя бы 1 AI согласен ──────────────────────────
    s3_total = s3_wins = 0
    s3_odds  = []

    # ── Стратегия 4: AI единогласны, модель не важна ──────────────────────────
    s4_total = s4_wins = 0
    s4_odds  = []

    # ── Стратегия 5: AI против модели (оба AI не согласны с моделью) ──────────
    s5_total = s5_wins = 0  # ставим на то, что говорят AI, против модели
    s5_odds  = []

    detail_s2 = []
    detail_s5 = []

    for row in rows:
        home, away, gpt_raw, llama_raw, rec, real, is_correct, signal, ens_h, ens_a, odds_h, odds_a = row

        # Парсим AI вердикты
        gpt_v   = parse_verdict(gpt_raw,   home, away)
        llama_v = parse_verdict(llama_raw, home, away)

        # Коэффициент на рекомендованный исход
        if rec == "home_win":
            bet_odds = odds_h or 0
        elif rec == "away_win":
            bet_odds = odds_a or 0
        else:
            bet_odds = 0

        # Результат
        won = (is_correct == 1)

        # ── Стратегия 1: просто модель ──────────────────────────────────────
        if bet_odds > 1.0:
            s1_total += 1
            s1_odds.append(bet_odds)
            if won:
                s1_wins += 1

        # ── Стратегия 2: модель + оба AI согласны ───────────────────────────
        both_agree_with_model = (gpt_v == rec and llama_v == rec)
        if bet_odds > 1.0 and gpt_v is not None and llama_v is not None and both_agree_with_model:
            s2_total += 1
            s2_odds.append(bet_odds)
            if won:
                s2_wins += 1
            detail_s2.append(f"  {'✅' if won else '❌'} {home} vs {away}: rec={rec}, gpt={gpt_v}, llama={llama_v}, odds={bet_odds}")

        # ── Стратегия 3: модель + хотя бы 1 AI ─────────────────────────────
        one_agrees = ((gpt_v == rec) or (llama_v == rec)) and (gpt_v is not None or llama_v is not None)
        if bet_odds > 1.0 and one_agrees:
            s3_total += 1
            s3_odds.append(bet_odds)
            if won:
                s3_wins += 1

        # ── Стратегия 4: оба AI единогласны ─────────────────────────────────
        if gpt_v is not None and llama_v is not None and gpt_v == llama_v and bet_odds > 1.0:
            s4_total += 1
            s4_odds.append(bet_odds)
            if gpt_v == real or (gpt_v == rec and won):
                s4_wins += 1

        # ── Стратегия 5: AI против модели ───────────────────────────────────
        # Оба AI выбирают другую команду — ставим на AI, а не модель
        if gpt_v is not None and llama_v is not None and gpt_v == llama_v and gpt_v != rec:
            ai_bet = gpt_v
            ai_odds = odds_h if ai_bet == "home_win" else odds_a
            if ai_odds and ai_odds > 1.0:
                s5_total += 1
                s5_odds.append(ai_odds)
                ai_won = (real == ai_bet)
                if ai_won:
                    s5_wins += 1
                detail_s5.append(
                    f"  {'✅' if ai_won else '❌'} {home} vs {away}: "
                    f"model={rec} | AI={ai_bet} | real={real} | odds={ai_odds}"
                )

    # ── Вывод ────────────────────────────────────────────────────────────────
    avg = lambda lst: round(sum(lst)/len(lst), 2) if lst else 0

    def show(name, total, wins, odds_list):
        if total == 0:
            print(f"  {name:<40} нет данных")
            return
        acc  = round(wins / total * 100, 1)
        roi  = calc_roi(wins, total, avg(odds_list))
        print(f"  {name:<40} {wins}/{total} ({acc}%) | avg кэф {avg(odds_list)} | ROI {roi:+.1f}%")

    show("1. Модель (ensemble)",               s1_total, s1_wins, s1_odds)
    show("2. Модель + ОБА AI согласны",         s2_total, s2_wins, s2_odds)
    show("3. Модель + хотя бы 1 AI",            s3_total, s3_wins, s3_odds)
    show("4. Оба AI единогласны (любой исход)", s4_total, s4_wins, s4_odds)
    show("5. Оба AI против модели",             s5_total, s5_wins, s5_odds)

    if detail_s2:
        print(f"\n  [Стратегия 2 — детали]")
        for d in detail_s2:
            print(d)

    if detail_s5:
        print(f"\n  [Стратегия 5 — AI vs модель]")
        for d in detail_s5:
            print(d)

    return {
        "sport": sport,
        "s1": (s1_total, s1_wins, avg(s1_odds)),
        "s2": (s2_total, s2_wins, avg(s2_odds)),
        "s3": (s3_total, s3_wins, avg(s3_odds)),
        "s4": (s4_total, s4_wins, avg(s4_odds)),
        "s5": (s5_total, s5_wins, avg(s5_odds)),
    }


# ─── Сводная таблица по всем видам спорта ─────────────────────────────────────

def summary(results):
    print(f"\n{'='*60}")
    print("  ИТОГОВЫЙ ВЫВОД ПО ВСЕМ ВИДАМ СПОРТА")
    print(f"{'='*60}")

    totals = {i: [0, 0, []] for i in range(1, 6)}
    for r in results:
        for i, key in enumerate(["s1", "s2", "s3", "s4", "s5"], 1):
            total, wins, a_odds = r[key]
            totals[i][0] += total
            totals[i][1] += wins
            totals[i][2].extend([a_odds] * total if a_odds > 0 else [])

    names = {
        1: "Модель (ensemble)",
        2: "Модель + ОБА AI согласны",
        3: "Модель + хотя бы 1 AI",
        4: "Оба AI единогласны",
        5: "AI против модели",
    }
    avg = lambda lst: round(sum(lst)/len(lst), 2) if lst else 0

    for i in range(1, 6):
        total, wins, odds_list = totals[i]
        if total == 0:
            print(f"  {names[i]:<40} нет данных")
            continue
        acc = round(wins / total * 100, 1)
        a   = avg(odds_list)
        roi = calc_roi(wins, total, a)
        print(f"  {names[i]:<40} {wins}/{total} ({acc}%) | ROI {roi:+.1f}%")

    print()
    print("  ВЫВОД:")
    t1, w1, _ = totals[1]
    t2, w2, _ = totals[2]
    t3, w3, _ = totals[3]
    if t2 > 0 and t1 > 0:
        acc1 = w1 / t1 * 100
        acc2 = w2 / t2 * 100
        diff = acc2 - acc1
        sign = "+" if diff > 0 else ""
        print(f"  Добавление ОБОИХ AI: точность {sign}{diff:.1f}% (матчей: {t1} → {t2})")
        if diff > 0:
            print(f"  ✅ AI фильтр работает — убирает плохие ставки")
        else:
            print(f"  ⚠️ AI не улучшает результат на текущей выборке")
    if t3 > 0 and t1 > 0:
        acc1 = w1 / t1 * 100
        acc3 = w3 / t3 * 100
        diff3 = acc3 - acc1
        sign = "+" if diff3 > 0 else ""
        print(f"  Добавление 1 AI:     точность {sign}{diff3:.1f}% (матчей: {t1} → {t3})")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn    = sqlite3.connect(DB)
    results = []

    tables = [
        ("football_predictions",   "Футбол"),
        ("basketball_predictions", "Баскетбол"),
        ("tennis_predictions",     "Теннис"),
        ("cs2_predictions",        "CS2"),
        ("hockey_predictions",     "Хоккей"),
    ]

    for tbl, sport in tables:
        try:
            res = analyze_table(conn, tbl, sport)
            results.append(res)
        except Exception as e:
            print(f"[{sport}] Ошибка: {e}")

    summary(results)
    conn.close()
