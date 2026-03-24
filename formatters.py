# formatters.py — функции форматирования отчётов и вспомогательные утилиты

import html as _html
from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder


def _safe_truncate(text: str, limit: int = 4000) -> str:
    """Обрезает текст до лимита Telegram, сохраняя целые строки."""
    if len(text) <= limit:
        return text
    footer = "\n\n⚠️ <i>Отчёт сокращён</i>"
    cut = limit - len(footer)
    truncated = text[:cut]
    last_newline = truncated.rfind('\n')
    if last_newline > cut - 200:
        truncated = truncated[:last_newline]
    return truncated + footer


def translate_outcome(text, home_team="Хозяева", away_team="Гости"):
    """Переводит исход с английского на русский с названиями команд."""
    if not text:
        return "Нет данных"
    text_lower = text.lower()
    if "home" in text_lower and "win" in text_lower:
        return f"{home_team} (хозяева)"
    if "away" in text_lower and "win" in text_lower:
        return f"{away_team} (гости)"
    if "draw" in text_lower or "ничья" in text_lower:
        return "Ничья"
    if "хозяев" in text_lower or "хозяева" in text_lower:
        return f"{home_team} (хозяева)"
    if "гостей" in text_lower or "гость" in text_lower or "гости" in text_lower:
        return f"{away_team} (гости)"
    if "победа хозяев" in text_lower:
        return f"{home_team} (хозяева)"
    if "победа гостей" in text_lower:
        return f"{away_team} (гости)"
    return text


def conf_icon(c):
    """Иконка уверенности."""
    if c >= 70: return "🟢"
    elif c >= 55: return "🟡"
    return "🔴"


def reliability_fires(prob_pct: float) -> str:
    """Единая шкала надёжности 1-5 огней для всех спортов.
    prob_pct — вероятность победителя в процентах (0-100).
    """
    if prob_pct >= 85:
        return "🔥🔥🔥🔥🔥 Топ сигнал"
    elif prob_pct >= 75:
        return "🔥🔥🔥🔥 Высокая"
    elif prob_pct >= 65:
        return "🔥🔥🔥 Хорошая"
    elif prob_pct >= 55:
        return "🔥🔥 Средняя"
    else:
        return "🔥 Слабая"


def _prob_icon(prob: float) -> str:
    """Эмодзи-шкала уверенности по вероятности."""
    if prob >= 80: return "🔥"
    elif prob >= 70: return "⭐"
    elif prob >= 60: return "✅"
    elif prob >= 50: return "⚠️"
    return "❌"


def _escape_md(text: str) -> str:
    """Экранирует спецсимволы для Markdown V1 в Telegram."""
    for ch in ('_', '*', '`', '['):
        text = text.replace(ch, "\\" + ch)
    return text


def format_main_report(home_team, away_team, prophet_data, oracle_results, gpt_result, llama_result,
                       mixtral_result=None, poisson_probs=None, elo_probs=None, ensemble_probs=None,
                       home_xg_stats=None, away_xg_stats=None, value_bets=None, injuries_block=None,
                       match_time=None, chimera_verdict_block="", ml_block="",
                       bookmaker_odds=None, movement_block=""):
    """Форматирует главный отчёт анализа матча с полным математическим анализом."""

    _pd = prophet_data if prophet_data is not None else [0.33, 0.33, 0.34]
    home_prob = _pd[1] * 100
    draw_prob = _pd[0] * 100
    away_prob = _pd[2] * 100

    home_sentiment_score = oracle_results.get(home_team, {}).get('sentiment', 0)
    away_sentiment_score = oracle_results.get(away_team, {}).get('sentiment', 0)
    home_sentiment_label = "🟢 Позитивный" if home_sentiment_score > 0.1 else ("🔴 Негативный" if home_sentiment_score < -0.1 else "⚪ Нейтральный")
    away_sentiment_label = "🟢 Позитивный" if away_sentiment_score > 0.1 else ("🔴 Негативный" if away_sentiment_score < -0.1 else "⚪ Нейтральный")

    gpt_verdict_raw = gpt_result.get("recommended_outcome", "Нет данных")
    gpt_verdict = translate_outcome(gpt_verdict_raw, home_team, away_team)
    gpt_confidence = gpt_result.get("final_confidence_percent", 0)
    gpt_summary = gpt_result.get("final_verdict_summary", "")
    gpt_odds = gpt_result.get("bookmaker_odds", 0)
    gpt_stake = gpt_result.get("recommended_stake_percent", 0)
    gpt_ev = gpt_result.get("expected_value_percent", 0)
    bet_signal = gpt_result.get("bet_signal", "ПРОПУСТИТЬ")
    signal_reason = gpt_result.get("signal_reason", "")

    llama_verdict_raw = llama_result.get("recommended_outcome", "Нет данных")
    llama_verdict = translate_outcome(llama_verdict_raw, home_team, away_team)
    llama_confidence = llama_result.get("final_confidence_percent", gpt_confidence)
    llama_summary = llama_result.get("analysis_summary", "")
    llama_total = llama_result.get("total_goals_prediction", "—")
    llama_total_reason = llama_result.get("total_goals_reasoning", "")
    llama_btts = llama_result.get("both_teams_to_score_prediction", "—")

    # Mixtral агент
    mixtral_verdict_raw = ""
    mixtral_verdict = ""
    mixtral_confidence = 0
    mixtral_summary = ""
    if mixtral_result and not mixtral_result.get('error'):
        mixtral_verdict_raw = mixtral_result.get("recommended_outcome", "")
        mixtral_verdict = translate_outcome(mixtral_verdict_raw, home_team, away_team)
        mixtral_confidence = mixtral_result.get("final_confidence_percent", 0)
        mixtral_summary = mixtral_result.get("analysis_summary", "")

    # Консенсус всех агентов
    all_verdicts = [v for v in [gpt_verdict_raw, llama_verdict_raw, mixtral_verdict_raw] if v]
    def _outcome_key(v):
        v = v.lower()
        if 'хозяев' in v or 'home' in v: return 'home'
        if 'гостей' in v or 'away' in v: return 'away'
        return 'draw'
    outcome_counts = {}
    for v in all_verdicts:
        k = _outcome_key(v)
        outcome_counts[k] = outcome_counts.get(k, 0) + 1
    max_count = max(outcome_counts.values()) if outcome_counts else 0
    if max_count >= 2:
        agreement_text = f"✅ {max_count}/{len(all_verdicts)} агентов согласны!"
    else:
        agreement_text = "⚠️ Агенты расходятся во мнениях"

    # Проверяем конфликт между ансамблем и GPT
    conflict_warning = ""
    ensemble_best_key = None  # будет установлен ниже в блоке ансамбля
    if ensemble_probs:
        _probs_tmp = {k: ensemble_probs.get(k, 0) for k in ['home', 'draw', 'away']}
        ensemble_best_key = max(_probs_tmp, key=_probs_tmp.get)
    if ensemble_best_key:
        gpt_key = _outcome_key(gpt_verdict_raw)
        if gpt_key != ensemble_best_key:
            ens_label = {'home': f'П1 ({home_team})', 'draw': 'Ничья', 'away': f'П2 ({away_team})'}[ensemble_best_key]
            conflict_warning = f"⚠️ КОНФЛИКТ: Химера рекомендует {gpt_verdict}, а математика указывает на {ens_label}"

    if bet_signal == "СТАВИТЬ":
        signal_icon = "🔥 СИГНАЛ: СТАВИТЬ!"
    elif value_bets:
        # Главный исход не рекомендован, но есть value ставки
        best_vb = value_bets[0]
        signal_icon = f"💰 СИГНАЛ: VALUE СТАВКА! {best_vb['outcome']} @ {best_vb['odds']} (EV: +{best_vb['ev']}%)"
    else:
        signal_icon = "❌ НЕ СТАВИТЬ"

    # xG блок
    xg_block = ""
    if home_xg_stats or away_xg_stats:
        xg_lines = ["📊 *xG СТАТИСТИКА (Understat):*"]
        if home_xg_stats:
            xg_lines.append(
                f"🏠 {home_team}: xG={home_xg_stats.get('avg_xg_last5','?')} | "
                f"xGA={home_xg_stats.get('avg_xga_last5','?')} | "
                f"Форма: {home_xg_stats.get('form_last5','?')}"
            )
        if away_xg_stats:
            xg_lines.append(
                f"✈️ {away_team}: xG={away_xg_stats.get('avg_xg_last5','?')} | "
                f"xGA={away_xg_stats.get('avg_xga_last5','?')} | "
                f"Форма: {away_xg_stats.get('form_last5','?')}"
            )
        xg_block = "\n".join(xg_lines)

    # Пуассон блок
    poisson_block = ""
    if poisson_probs:
        # Индикатор источника данных
        src = poisson_probs.get('data_source', 'fallback')
        if src == 'understat':
            src_icon = "🟢"  # зелёный = реальные данные
            src_label = "Understat ✅"
        elif src == 'partial':
            src_icon = "🟡"  # жёлтый = частичные данные
            src_label = "частичные данные ⚠️"
        else:
            src_icon = "🔴"  # красный = резервные значения
            src_label = "среднелиговые ❌"
        home_exp = poisson_probs.get('home_exp', '?')
        away_exp = poisson_probs.get('away_exp', '?')
        poisson_block = (
            f"🎯 *ПУАССОН (xG-модель):* {src_icon} _{src_label}_\n"
            f" Хозяев xG: {home_exp} | Гости xG: {away_exp}\n"
            f" П1: {round(poisson_probs['home_win']*100)}% | Х: {round(poisson_probs['draw']*100)}% | П2: {round(poisson_probs['away_win']*100)}%\n"
            f" Тотал >2.5: {round(poisson_probs['over_25']*100)}% | Обе забьют: {round(poisson_probs['btts']*100)}%\n"
            f" Счёт: {poisson_probs['most_likely_score']} ({round(poisson_probs['most_likely_score_prob']*100)}%)"
        )

    # ELO блок
    elo_block = ""
    if elo_probs:
        h_form = elo_probs.get('home_form', '')
        a_form = elo_probs.get('away_form', '')
        h_bonus = elo_probs.get('home_form_bonus', 0)
        a_bonus = elo_probs.get('away_form_bonus', 0)
        h_bonus_str = f" ({h_bonus:+.0f})" if h_bonus != 0 else ""
        a_bonus_str = f" ({a_bonus:+.0f})" if a_bonus != 0 else ""
        form_line = ""
        if h_form and h_form != '?????':
            form_line = f"\n Форма: {home_team}: {h_form}{h_bonus_str} | {away_team}: {a_form}{a_bonus_str}"
        elo_block = (
            f"⚡ *ELO РЕЙТИНГ + ФОРМА:*\n"
            f" {home_team}: {elo_probs.get('home_elo',1500)} | {away_team}: {elo_probs.get('away_elo',1500)}{form_line}\n"
            f" ELO П1: {round(elo_probs.get('home',0)*100)}% | Х: {round(elo_probs.get('draw',0)*100)}% | П2: {round(elo_probs.get('away',0)*100)}%"
        )

    # Ансамбль блок
    ensemble_block = ""
    ensemble_best_key = None
    if ensemble_probs:
        probs = {k: ensemble_probs.get(k, 0) for k in ['home', 'draw', 'away']}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_key, best_val = sorted_probs[0]
        second_val = sorted_probs[1][1]
        gap = best_val - second_val

        best_outcome_map = {'home': f'П1 ({home_team})', 'draw': 'Ничья', 'away': f'П2 ({away_team})'}
        best_outcome_label = best_outcome_map[best_key]
        best_prob = round(best_val * 100)
        ensemble_best_key = best_key

        # Индикатор уверенности преимущества
        if gap >= 0.10:
            conf_label = "🟢 чёткое преимущество"
        elif gap >= 0.05:
            conf_label = "🟡 небольшое преимущество"
        else:
            conf_label = "🔴 равный матч"

        weights = ensemble_probs.get('weights', {})
        w_str = f"Пуассон {round(weights.get('poisson',0)*100)}%+ELO {round(weights.get('elo',0)*100)}%+AI {round(weights.get('ai',0)*100)}%+Бук {round(weights.get('bookmaker',0)*100)}%+Пр {round(weights.get('prophet',0)*100)}%" if weights else ""
        ensemble_block = (
            f"🔢 *АНСАМБЛЬ (взвешенный):*\n"
            f" П1: {round(probs['home']*100)}% | Х: {round(probs['draw']*100)}% | П2: {round(probs['away']*100)}%\n"
            f" {conf_label}: *{best_outcome_label}* ({best_prob}%)\n"
            f" _Веса: {w_str}_"
        )

    # Value bets блок
    value_block = ""
    if value_bets:
        vlines = ["💰 *VALUE СТАВКИ (ансамбль):*"]
        for vb in value_bets[:3]:  # Показываем топ-3
            vlines.append(
                f" ✅ *{vb['outcome']}* @ {vb['odds']} — наша вероятность: {vb['our_prob']}% vs бук: {vb['book_prob']}%"
            )
            vlines.append(
                f"   EV: +{vb['ev']}% | Келли: {vb['kelly']}% от банка"
            )
        value_block = "\n".join(vlines)

    # Mixtral блок
    mixtral_block = ""
    if mixtral_result and not mixtral_result.get('error') and mixtral_summary:
        mixtral_block = f"🌀 *Тень:*\n_{mixtral_summary}_\n\n⚔️ Вердикт: {mixtral_verdict} ({mixtral_confidence}%)"

    # Собираем блоки
    math_section = ""
    math_parts = [p for p in [xg_block, poisson_block, elo_block, ensemble_block] if p]
    if math_parts:
        math_section = "\n\n".join(math_parts)

    # Форматируем дату матча
    match_time_str = ""
    if match_time:
        try:
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(match_time.replace('Z', '+00:00'))
            # Переводим в Московское время (UTC+3)
            moscow_tz = timezone(timedelta(hours=3))
            dt_moscow = dt.astimezone(moscow_tz)
            match_time_str = dt_moscow.strftime('%d.%m.%Y %H:%M МСК')
        except Exception:
            match_time_str = str(match_time)[:16]

    # Строка коэффициентов для шапки
    _odds = bookmaker_odds or {}
    _h_odds = _odds.get("home_win") or _odds.get("pinnacle_home")
    _d_odds = _odds.get("draw")     or _odds.get("pinnacle_draw")
    _a_odds = _odds.get("away_win") or _odds.get("pinnacle_away")

    # Валидация коэффициентов — если любой > 30, данные скорее всего битые
    _odds_suspicious = any(o and float(o) > 30.0 for o in [_h_odds, _d_odds, _a_odds] if o)
    if _odds_suspicious:
        value_bets = []  # не показывать value ставки с битыми коэффициентами

    if _h_odds and _a_odds and not _odds_suspicious:
        _odds_line = f"💰 П1 *{_h_odds}* · Х *{_d_odds}* · П2 *{_a_odds}*" if _d_odds else f"💰 П1 *{_h_odds}* · П2 *{_a_odds}*"
    elif _odds_suspicious:
        _odds_line = "⚠️ _Коэффициенты недоступны_"
    else:
        _odds_line = ""

    # Надёжность — 5-уровневая шкала 🔥
    # Берём среднее между уверенностью GPT и согласием агентов
    _agree_bonus = 5 if max_count >= 2 else 0
    _reliability = reliability_fires(gpt_confidence + _agree_bonus)

    # Блок вердикта
    if bet_signal == "СТАВИТЬ":
        _verdict_header = "✅ СТАВИТЬ"
        _verdict_bet = f"💰 *{gpt_verdict}* @ {gpt_odds} | EV: +{gpt_ev:.1f}% | Банк: {gpt_stake:.1f}%"
    elif value_bets:
        best_vb = value_bets[0]
        _verdict_header = "💎 VALUE СТАВКА"
        _verdict_bet = f"💰 *{best_vb['outcome']}* @ {best_vb['odds']} | EV: +{best_vb['ev']}%"
    else:
        _verdict_header = "❌ НЕ СТАВИТЬ"
        _verdict_bet = f"_{signal_reason or 'Нет ценности в текущих коэффициентах'}_"

    # Ансамбль одной строкой
    _ens_line = ""
    if ensemble_probs:
        _ep = ensemble_probs
        _ens_line = (f"🔢 Ансамбль: П1 *{round(_ep.get('home',0)*100)}%* · "
                     f"Х *{round(_ep.get('draw',0)*100)}%* · П2 *{round(_ep.get('away',0)*100)}%*")

    # ELO одной строкой
    _elo_line = ""
    if elo_probs:
        _hf = elo_probs.get('home_form',''); _af = elo_probs.get('away_form','')
        _form = f" | Форма: {_hf} · {_af}" if _hf and _hf != '?????' else ""
        _elo_line = (f"⚡ ELO: {home_team} {elo_probs.get('home_elo',1500)} · "
                     f"{away_team} {elo_probs.get('away_elo',1500)}{_form}")

    # Пуассон одной строкой
    _poi_line = ""
    if poisson_probs:
        _poi_line = (f"🎯 Пуассон: П1 {round(poisson_probs['home_win']*100)}% · "
                     f"Х {round(poisson_probs['draw']*100)}% · П2 {round(poisson_probs['away_win']*100)}% "
                     f"| Тотал >2.5: {round(poisson_probs['over_25']*100)}% "
                     f"| Счёт: {poisson_probs['most_likely_score']}")

    # Тотал и BTTS от Тени
    _total_line = ""
    if llama_total and llama_total != "—":
        _total_line = f"⚽ Тотал: *{llama_total}*"
        if llama_btts and llama_btts != "—":
            _total_line += f" | ОЗ: *{llama_btts}*"

    # Экранируем имена команд для Markdown V1
    _ht = _escape_md(home_team)
    _at = _escape_md(away_team)

    # Блок деталей (второстепенное)
    _details_parts = []
    _details_parts.append(f"📊 Пророк: П1 {home_prob:.0f}% · Х {draw_prob:.0f}% · П2 {away_prob:.0f}%")
    _details_parts.append(f"🗣 Оракул: {_ht} {home_sentiment_label} · {_at} {away_sentiment_label}")
    if home_xg_stats or away_xg_stats:
        _xg_h = home_xg_stats.get('avg_xg_last5','?') if home_xg_stats else '?'
        _xg_a = away_xg_stats.get('avg_xg_last5','?') if away_xg_stats else '?'
        _details_parts.append(f"📈 xG: {_ht} {_xg_h} · {_at} {_xg_a}")
    _details = "\n".join(_details_parts)

    report = f"""🏆 *CHIMERA AI — АНАЛИЗ МАТЧА*
━━━━━━━━━━━━━━━━━━━━━━━━━
⚽ *{_ht} vs {_at}*
{(f'📅 {match_time_str}') if match_time_str else ''}{(chr(10) + _odds_line) if _odds_line else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 *Победит: {gpt_verdict}* — уверенность {gpt_confidence}%
{_reliability} | {agreement_text}
🎯 *{_verdict_header}*
{_verdict_bet}
{('⚽ ' + _total_line) if _total_line else ''}{(chr(10) + value_block) if value_block else ''}{(chr(10) + movement_block) if movement_block else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *АНАЛИЗ*
━━━━━━━━━━━━━━━━━━━━━━━━━
{_ens_line}
{_elo_line}
{_poi_line}
{(chr(10) + '━━━━━━━━━━━━━━━━━━━━━━━━━') if gpt_summary or llama_summary else ''}
{('🐍🦁🐐 *Химера:* _' + gpt_summary + '_') if gpt_summary else ''}
{('🌀 *Тень:* _' + llama_summary + '_') if llama_summary else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━
{_details}
{(chr(10) + injuries_block) if injuries_block else ''}
"""
    def _h2m(s):
        return (s.replace("<b>", "*").replace("</b>", "*")
                 .replace("<i>", "_").replace("</i>", "_")
                 .replace("<code>", "`").replace("</code>", "`"))
    if ml_block:
        report += "\n" + _h2m(ml_block)
    if chimera_verdict_block:
        report += "\n" + _h2m(chimera_verdict_block)
    return report.strip()


def format_goals_report(home_team, away_team, goals_result, bookmaker_odds=None, poisson_probs=None):
    """Форматирует отчёт по рынку голов."""
    _ht = _escape_md(home_team); _at = _escape_md(away_team)
    if bookmaker_odds is None:
        bookmaker_odds = {}
    summary = goals_result.get("summary", "")
    over_2_5 = goals_result.get("total_over_2_5", "—")
    over_2_5_conf = goals_result.get("total_over_2_5_confidence", 0)
    over_2_5_reason = goals_result.get("total_over_2_5_reason", "")
    over_1_5 = goals_result.get("total_over_1_5", "—")
    over_1_5_conf = goals_result.get("total_over_1_5_confidence", 0)
    btts = goals_result.get("btts", "—")
    btts_conf = goals_result.get("btts_confidence", 0)
    btts_reason = goals_result.get("btts_reason", "")
    first_goal = goals_result.get("first_goal", "—")
    best_bet = goals_result.get("best_goals_bet", "")

    # Реальные коэффициенты из API
    real_over_2_5  = bookmaker_odds.get("over_2_5", 0)
    real_under_2_5 = bookmaker_odds.get("under_2_5", 0)
    real_over_1_5  = bookmaker_odds.get("over_1_5", 0)

    # EV для тотала 2.5 на основе Пуассон-вероятности vs букмекерский кэф
    ev_block = ""
    if poisson_probs and real_over_2_5 and real_under_2_5:
        p_over  = poisson_probs.get("over_25", 0)
        p_under = poisson_probs.get("under_25", 0)
        if p_over > 0 and p_under > 0:
            ev_over  = round((p_over  * real_over_2_5  - 1) * 100, 1)
            ev_under = round((p_under * real_under_2_5 - 1) * 100, 1)
            best_ev = max(ev_over, ev_under)
            if best_ev > 0:
                side = f"Больше 2.5 @ {real_over_2_5}" if ev_over >= ev_under else f"Меньше 2.5 @ {real_under_2_5}"
                ev_val = ev_over if ev_over >= ev_under else ev_under
                ev_block = f"\n🎯 *ЦЕННОСТЬ (EV):* {side} → EV *+{ev_val}%* (Пуассон: {int(p_over*100)}%/{int(p_under*100)}%)"
            else:
                ev_block = f"\n📊 _(нет ценности: EV Больше={ev_over:+.1f}% / Меньше={ev_under:+.1f}%)_"

    # Строки с коэффициентами
    odds_2_5_str = f" | Кэф: Больше {real_over_2_5} / Меньше {real_under_2_5}" if real_over_2_5 else ""
    odds_1_5_str = f" | Кэф: {real_over_1_5}" if real_over_1_5 else ""

    return f"""
⚽ *АНАЛИЗ ГОЛОВ — {_ht} vs {_at}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ ГОЛОВ:*
{conf_icon(over_2_5_conf)} Тотал 2.5: *{over_2_5}* ({over_2_5_conf}%){odds_2_5_str}
_{over_2_5_reason}_{ev_block}

{conf_icon(over_1_5_conf)} Тотал 1.5: *{over_1_5}* ({over_1_5_conf}%){odds_1_5_str}

🥅 *ОБЕ ЗАБЬЮТ:*
{conf_icon(btts_conf)} Обе забьют: *{btts}* ({btts_conf}%)
_{btts_reason}_

🎯 *КТО ЗАБЬЁТ ПЕРВЫМ:* {first_goal}

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА ГОЛЫ:*
_{best_bet}_
""".strip()


def format_corners_report(home_team, away_team, corners_result):
    """Форматирует отчёт по рынку угловых."""
    _ht = _escape_md(home_team); _at = _escape_md(away_team)
    summary = corners_result.get("summary", "")
    total = corners_result.get("total_corners_over_9_5", "—")
    total_conf = corners_result.get("total_corners_confidence", 0)
    total_reason = corners_result.get("total_corners_reason", "")
    home_c = corners_result.get("home_corners_over_4_5", "—")
    away_c = corners_result.get("away_corners_over_4_5", "—")
    winner = corners_result.get("corners_winner", "—")
    best_bet = corners_result.get("best_corners_bet", "")

    return f"""
🚩 *АНАЛИЗ УГЛОВЫХ — {_ht} vs {_at}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ УГЛОВЫХ:*
{conf_icon(total_conf)} Тотал 9.5: *{total}* ({total_conf}%)
_{total_reason}_

🏠 {_ht}: *{home_c}* угловых
✈️ {_at}: *{away_c}* угловых
🏆 Больше угловых: *{winner}*

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА УГЛОВЫЕ:*
_{best_bet}_
""".strip()


def format_cards_report(home_team, away_team, cards_result):
    """Форматирует отчёт по рынку карточек."""
    _ht = _escape_md(home_team); _at = _escape_md(away_team)
    summary = cards_result.get("summary", "")
    total = cards_result.get("total_cards_over_3_5", "—")
    total_conf = cards_result.get("total_cards_confidence", 0)
    total_reason = cards_result.get("total_cards_reason", "")
    red = cards_result.get("red_card", "—")
    red_conf = cards_result.get("red_card_confidence", 0)
    more_cards = cards_result.get("more_cards_team", "—")
    best_bet = cards_result.get("best_cards_bet", "")

    return f"""
🟨 *АНАЛИЗ КАРТОЧЕК — {_ht} vs {_at}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ КАРТОЧЕК:*
{conf_icon(total_conf)} Тотал 3.5: *{total}* ({total_conf}%)
_{total_reason}_

🟥 Красная карточка: *{red}* ({red_conf}%)
🟨 Больше карточек: *{more_cards}*

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА КАРТОЧКИ:*
_{best_bet}_
""".strip()


def format_handicap_report(home_team, away_team, handicap_result):
    """Форматирует отчёт по рынку гандикапов."""
    _ht = _escape_md(home_team); _at = _escape_md(away_team)
    summary = handicap_result.get("summary", "")
    ah_home = handicap_result.get("asian_handicap_home", "—")
    ah_home_conf = handicap_result.get("asian_handicap_home_confidence", 0)
    ah_away = handicap_result.get("asian_handicap_away", "—")
    ah_away_conf = handicap_result.get("asian_handicap_away_confidence", 0)
    dc = handicap_result.get("double_chance", "—")
    dc_reason = handicap_result.get("double_chance_reason", "")
    best_bet = handicap_result.get("best_handicap_bet", "")

    return f"""
⚖️ *ГАНДИКАПЫ — {_ht} vs {_at}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *АЗИАТСКИЙ ГАНДИКАП:*
{conf_icon(ah_home_conf)} {_ht} -0.5: *{ah_home}* ({ah_home_conf}%)
{conf_icon(ah_away_conf)} {_at} +0.5: *{ah_away}* ({ah_away_conf}%)

🎯 *ДВОЙНОЙ ШАНС:*
Рекомендация: *{dc}*
_{dc_reason}_

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА ГАНДИКАП:*
_{best_bet}_
""".strip()


def _format_chimera_page(candidates: list, idx: int, bankroll: float = 0) -> str:
    """Форматирует одну страницу карусели — полный формат как в format_chimera_signals."""
    from chimera_signal import _format_match_time, score_label, _format_totals_block
    c = candidates[idx]
    total = len(candidates)
    sp = c.get("sport", "football")
    sp_emoji = {"cs2": "🎮", "tennis": "🎾", "basketball": "🏀"}.get(sp, "⚽")

    t_str, t_live = _format_match_time(c.get("commence_time", ""))
    live_tag = "🟢 LIVE" if t_live else (f"🕐 {t_str}" if t_str else "")
    time_line = f"\n{live_tag}" if live_tag else ""

    score = c.get("chimera_score", 0)
    label = score_label(score)

    if idx == 0:
        header = f"🎯 <b>CHIMERA SIGNAL — ЛУЧШАЯ СТАВКА ДНЯ</b>"
    else:
        header = f"📋 <b>ВАРИАНТ {idx + 1} из {total}</b>"

    matchup = f"<b>{c.get('home','')} vs {c.get('away','')}</b>"
    bet_line = f"📌 Ставка: <b>{c.get('team','')} ({c.get('outcome','')})</b>"

    lines = [
        header, "",
        f"<b>{label} [{score:.0f}/100]</b>{time_line}", "",
        f"{sp_emoji} | {matchup}",
        bet_line,
        f"💰 Кэф: <b>{c.get('odds','?')}</b> | Наша вероятность: <b>{c.get('prob',0)}%</b> {_prob_icon(c.get('prob',0))}",
        f"📈 EV: <b>{c.get('ev',0):+.1f}%</b> | Ставь: <b>{c.get('kelly',0):.1f}%</b> банка",
    ]

    # AI блок
    if c.get("ai_confirmed") is True:
        llama_agrees = c.get("llama_agrees")
        ai_header = "🐉 Химера единогласна (Змея + Лев + Козёл + Тень)" if llama_agrees else "🐍🦁🐐 Химера подтверждает"
        ai_reason   = _html.escape(str(c.get("ai_reason", "") or ""))
        llama_logic = _html.escape(str(c.get("llama_logic", "") or ""))
        llama_warn  = _html.escape(str(c.get("llama_warning", "") or ""))
        lines += ["", f"<b>{ai_header} ({c.get('ai_confidence',0)}% уверенности):</b>",
                  f"<i>🐍🦁🐐 Химера: «{ai_reason}»</i>"]
        if llama_logic:
            lines.append(f"<i>🌀 Тень: «{llama_logic}»</i>")
        if llama_warn:
            lines.append(f"⚠️ Риск: {llama_warn}")
    elif c.get("ai_confirmed") is False:
        lines.append("\n⚠️ AI сомневается в этой ставке")
    elif c.get("ai_confirmed") is None:
        if c.get("ai_reason"):
            lines.append(f"\n🤖 <i>AI: «{_html.escape(str(c['ai_reason']))}»</i>")
        else:
            lines.append("\n🤖 <i>AI не выбрал этот вариант как лучший</i>")

    # Детали CHIMERA Score
    score_lines = [
        "", "📊 <b>Детали CHIMERA Score:</b>",
        f"├ ELO преимущество: {c.get('elo_pts',0):+.0f} pts" +
            (f" (разрыв: {c['elo_gap']} очков)" if c.get('elo_gap') else ""),
        f"├ Форма команды: {c.get('form_pts',0):+.0f} pts" +
            (f" ({c['form']})" if c.get('form') else ""),
        f"├ Ценность кэфа: {c.get('value_pts',0):+.0f} pts" +
            f" ({c.get('prob',0)}% vs бук {c.get('implied_prob',0)}%)",
        f"├ Сила прогноза: {c.get('prob_pts',0):+.0f} pts",
    ]
    if c.get("xg_pts", 0):
        score_lines.append(f"├ xG качество: {c['xg_pts']:+.0f} pts")
    if c.get("line_pts", 0):
        icon = "📉" if c["line_pts"] > 0 else "⚠️"
        score_lines.append(f"├ {icon} Движение линии: {c['line_pts']:+.0f} pts")
    if c.get("h2h_pts", 0):
        score_lines.append(f"├ ⚔️ H2H история: {c['h2h_pts']:+.0f} pts")
    score_lines[-1] = score_lines[-1].replace("├", "└")
    lines += score_lines

    totals_block = _format_totals_block(c)
    if totals_block:
        lines += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━", totals_block]

    result = "\n".join(lines)
    if len(result) > 4000:
        result = result[:3990] + "\n…"
    return result


def _build_chimera_carousel_kb(candidates: list, idx: int, user_id: int) -> types.InlineKeyboardMarkup:
    """Клавиатура карусели: ◀️ счётчик ▶️ + кнопка 'Я поставил'."""
    total = len(candidates)
    c = candidates[idx]
    kelly = c.get("kelly", 2)
    units = 3 if kelly >= 4 else (2 if kelly >= 2 else 1)

    nav_row = []
    if idx > 0:
        nav_row.append(types.InlineKeyboardButton(text="◀️", callback_data=f"chimera_page_{idx-1}"))
    nav_row.append(types.InlineKeyboardButton(text=f"{idx+1} / {total}", callback_data="chimera_noop"))
    if idx < total - 1:
        nav_row.append(types.InlineKeyboardButton(text="▶️", callback_data=f"chimera_page_{idx+1}"))

    # Кнопка ставки — всегда через индекс, pred_id получаем лениво при нажатии
    bet_row = [types.InlineKeyboardButton(
        text=f"✅ Я поставил {units}u — записать",
        callback_data=f"chimera_bet_{idx}_{units}"
    )]

    return types.InlineKeyboardMarkup(inline_keyboard=[nav_row, bet_row])


def _build_chimera_kb(top_candidates, top_pred_id, top_sport, top_odds, user_id):
    """Обёртка — строит карусель с idx=0 + кнопка обновить."""
    if not top_candidates:
        return None
    kb = _build_chimera_carousel_kb(top_candidates, 0, user_id)
    rows = list(kb.inline_keyboard) + [[
        types.InlineKeyboardButton(text="🔄 Обновить сигналы", callback_data="chimera_refresh")
    ]]
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


def _make_loss_explanation(rec_outcome: str, real_outcome: str, home: str, away: str) -> str:
    """Краткое объяснение почему прогноз не сыграл."""
    import random
    if rec_outcome == "home_win" and real_outcome == "draw":
        opts = [
            f"{home} не смог реализовать преимущество — матч завершился ничьёй",
            f"Не хватило одного гола: {home} сравнял счёт, но дожать не смог",
            f"Ничья вместо победы {home} — бывает, модель не угадала интенсивность матча",
        ]
    elif rec_outcome == "home_win" and real_outcome == "away_win":
        opts = [
            f"{away} переиграл фаворита — один из тех дней когда статистика уходит на второй план",
            f"Неожиданная победа {away}. Возможно сказались факторы вне модели: усталость, тактика, удача",
            f"{home} не оправдал ожиданий. {away} был готов лучше",
        ]
    elif rec_outcome == "away_win" and real_outcome == "draw":
        opts = [
            f"{away} не смог дожать — матч завершился ничьёй вместо победы гостей",
            f"Домашние стены помогли {home} устоять. Ничья вместо победы {away}",
            f"Счёт сравнялся, но {away} не хватило для полной победы",
        ]
    elif rec_outcome == "away_win" and real_outcome == "home_win":
        opts = [
            f"{home} удержал преимущество своего поля. Домашний фактор сработал сильнее модели",
            f"Неожиданная победа {home} — гости не смогли реализовать перевес на бумаге",
            f"{home} победил вопреки прогнозу — вероятно сыграли мотивация и домашняя поддержка",
        ]
    elif rec_outcome in ("team1_win", "p1_win") and real_outcome in ("team2_win", "p2_win"):
        opts = [
            f"{away} показал более сильную игру в этот день",
            f"Неожиданный результат: {away} переиграл фаворита",
            f"{away} превзошёл ожидания — один из тех матчей где класс уступил форме дня",
        ]
    elif rec_outcome in ("team2_win", "p2_win") and real_outcome in ("team1_win", "p1_win"):
        opts = [
            f"{home} удивил своей игрой и переиграл гостей",
            f"Неожиданная победа {home} — прогноз не учёл их сегодняшнюю форму",
            f"{home} оказался сильнее — дисперсия в действии",
        ]
    else:
        opts = [
            "Матч сложился не в нашу пользу — такое бывает на дистанции",
            "Результат не совпал с прогнозом. Продолжаем анализировать",
            "Не угадали — это часть процесса. Важна дистанция, а не один матч",
        ]
    return random.choice(opts)
