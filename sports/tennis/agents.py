# -*- coding: utf-8 -*-
"""
sports/tennis/agents.py — AI агенты для анализа теннисных матчей
=================================================================
GPT-4.1-mini (Стратег) + Llama 3.3 70B (Тактик) — два независимых мнения.
"""

import json
import logging

logger = logging.getLogger(__name__)

try:
    from agents import client as _gpt_client, groq_client as _groq_client
except ImportError:
    _gpt_client = None
    _groq_client = None


def _call_ai(prompt: str, client, model: str, timeout: int = 30) -> str:
    """Вызывает AI и возвращает текстовый ответ."""
    if not client:
        return f"❌ Клиент {model} не инициализирован"
    try:
        is_groq = "llama" in model.lower() or "groq" in str(type(client)).lower()
        system_msg = (
            "Ты — профессиональный аналитик тенниса для беттинга. "
            "Пиши строго на русском языке. Запрещено использовать иероглифы."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
            timeout=timeout,
        )
        text = response.choices[0].message.content or ""

        # Фильтр CJK символов (Llama иногда пишет иероглифы)
        import re
        text = re.sub(
            r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u3040-\u309f\u30a0-\u30ff]',
            '', text
        ).strip()
        return text
    except Exception as e:
        return f"❌ Ошибка {model}: {str(e)[:120]}"


def run_tennis_gpt_agent(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    p1_form: str = "", p2_form: str = "",
    p1_rank: int = 100, p2_rank: int = 100,
    game_totals: dict = None,
) -> str:
    """
    GPT-4.1-mini — Стратег.
    Даёт структурированный анализ теннисного матча.
    """
    surf_names  = {"hard": "хард (твёрдое покрытие)", "clay": "грунт", "grass": "трава"}
    surf_ru     = surf_names.get(surface, "хард")
    tour_upper  = tour.upper()

    h2h_block = ""
    if h2h_total >= 3:
        h2h_block = f"\nОчные встречи (H2H): {player1} {h2h_p1_wins}:{h2h_total - h2h_p1_wins} {player2}"

    form_block = ""
    if p1_form and p1_form != "?????":
        form_block += f"\nФорма {player1}: {p1_form}"
    if p2_form and p2_form != "?????":
        form_block += f"\nФорма {player2}: {p2_form}"

    # Блок тоталов с реальной букмекерской линией
    totals_block = ""
    if game_totals:
        line = game_totals.get("line", 22.5)
        pred = game_totals.get("prediction", "")
        conf = game_totals.get("confidence", 0)
        if game_totals.get("has_real_line"):
            ev = game_totals.get("ev", 0)
            bm_over = game_totals.get("bm_over_odds", 0)
            bm_under = game_totals.get("bm_under_odds", 0)
            totals_block = (
                f"\nБУКМЕКЕРСКАЯ ЛИНИЯ ТОТАЛА: {line} геймов "
                f"(Б={bm_over} / М={bm_under})"
                f"\nМОДЕЛЬ: {pred} {line} ({conf}% уверенности, EV={ev:+.1f}%)"
            )
        elif pred:
            totals_block = f"\nМОДЕЛЬ ТОТАЛА: {pred} {line} ({conf}% уверенности)"

    prompt = f"""Ты — профессиональный аналитик тенниса для беттинга. Анализируй строго и честно.

МАТЧ ({tour_upper}): {player1} vs {player2}
ПОКРЫТИЕ: {surf_ru}
РЕЙТИНГ: {player1} #{p1_rank} | {player2} #{p2_rank}
КЭФЫ: {player1}={odds_p1} | {player2}={odds_p2}
НАШИ ВЕРОЯТНОСТИ: {player1}={round(probs['p1_win']*100)}% | {player2}={round(probs['p2_win']*100)}%{h2h_block}{form_block}{totals_block}

Дай структурированный анализ:

**ФАВОРИТ:** [игрок] — [главная причина в 1 предложении]

**ПОКРЫТИЕ:** Кому это покрытие подходит больше и почему? (стиль игры, специализация)

**ФОРМА И H2H:** Что говорит текущая форма? Есть ли психологическое преимущество по H2H?

**ПРОГНОЗ:** [игрок] победит с вероятностью [X%]. Счёт: [2:0 / 2:1]

**ТОТАЛЫ:** {f"Линия букмекера {line} геймов — наша модель говорит {pred}. " if game_totals else "Тотал геймов "}[Больше/Меньше] — объясни: темп игры, стиль (подача vs задняя линия), покрытие.

Будь конкретным. Не пиши воду. Не давай рекомендацию ставить или нет — это решает математическая модель."""

    return _call_ai(prompt, _gpt_client, "gpt-4.1-mini")


def run_tennis_llama_agent(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    p1_form: str = "", p2_form: str = "",
    p1_rank: int = 100, p2_rank: int = 100,
    gpt_verdict: str = "",
    game_totals: dict = None,
) -> str:
    """
    Llama 3.3 70B — Независимый тактик.
    Смотрит на матч критически, ищет риски которые GPT мог пропустить.
    """
    surf_names = {"hard": "хард", "clay": "грунт", "grass": "трава"}
    surf_ru    = surf_names.get(surface, "хард")
    tour_upper = tour.upper()

    h2h_block = ""
    if h2h_total >= 3:
        h2h_block = f"\nH2H: {player1} {h2h_p1_wins}:{h2h_total - h2h_p1_wins} {player2}"

    gpt_block = ""
    if gpt_verdict and not gpt_verdict.startswith("❌"):
        short = gpt_verdict[:200].replace('\n', ' ')
        gpt_block = f"\n\nМнение GPT (кратко): «{short}...»"

    totals_block = ""
    if game_totals:
        line = game_totals.get("line", 22.5)
        pred = game_totals.get("prediction", "")
        if game_totals.get("has_real_line"):
            ev = game_totals.get("ev", 0)
            totals_block = f"\nБУКМЕКЕРСКАЯ ЛИНИЯ: {line} геймов | Модель: {pred} (EV={ev:+.1f}%)"
        elif pred:
            totals_block = f"\nМодель тотала: {pred} {line} геймов"

    prompt = f"""Ты — независимый тактический аналитик тенниса. Смотри критически, не копируй чужие выводы.

МАТЧ ({tour_upper}): {player1} vs {player2}
ПОКРЫТИЕ: {surf_ru}
РЕЙТИНГ: #{p1_rank} vs #{p2_rank}
КЭФЫ: {player1}={odds_p1} | {player2}={odds_p2}{h2h_block}{totals_block}{gpt_block}

Дай НЕЗАВИСИМУЮ оценку:

**ТАКТИКА:** Как каждый игрок будет строить игру на {surf_ru}? Кому стиль игры подходит больше?

**РИСКИ ДЛЯ ФАВОРИТА:** Что может пойти не так? (усталость, травма, психология под давлением)

**КЛЮЧЕВОЙ ФАКТОР:** Один главный аспект, который решит исход матча

**ТОТАЛЫ:** {f"Линия {line} — соглашаешься с моделью ({pred})?" if game_totals else "Ожидается долгий матч или быстрый разгром?"}

**ИТОГ:** [игрок] победит | Уверенность: [X%]

Если есть серьёзные риски — укажи их прямо. Не соглашайся с GPT просто так. Не давай беттинг-рекомендацию — это решает математическая модель."""

    return _call_ai(prompt, _groq_client, "llama-3.3-70b-versatile", timeout=35)


def run_tennis_chimera_agents(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    p1_rank: int = 100, p2_rank: int = 100,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    p1_form: str = "", p2_form: str = "",
    gpt_text: str = "",
    llama_text: str = "",
) -> dict:
    """
    Запускает CHIMERA Multi-Agent анализ для теннисного матча.
    gpt_text / llama_text — уже готовые ответы агентов, передаём в market_verdict
    чтобы финальный вердикт читал реальный анализ, а не запускал новых агентов.
    """
    try:
        from chimera_multi_agent import run_agent_market_verdict, bayesian_combine, format_verdict_block
    except ImportError:
        return {}

    surf_names = {"hard": "хард", "clay": "грунт", "grass": "трава"}
    surf_ru = surf_names.get(surface, "хард")
    h2h_str = f"H2H: {player1} {h2h_p1_wins}:{h2h_total-h2h_p1_wins} {player2}" if h2h_total >= 3 else "H2H: мало данных"
    form_str = ""
    if p1_form and p1_form != "?????":
        form_str += f"\nФорма {player1}: {p1_form}"
    if p2_form and p2_form != "?????":
        form_str += f"\nФорма {player2}: {p2_form}"

    match_info = (
        f"МАТЧ ({tour.upper()}): {player1} vs {player2}\n"
        f"ПОКРЫТИЕ: {surf_ru}\n"
        f"РЕЙТИНГ: {player1} #{p1_rank} | {player2} #{p2_rank}\n"
        f"КОЭФФИЦИЕНТЫ: {player1}={odds_p1} | {player2}={odds_p2}\n"
        f"{h2h_str}{form_str}"
    )

    math_probs = {"home": probs.get("p1_win", 0.5), "away": probs.get("p2_win", 0.5)}
    bookmaker_odds = {"home": odds_p1, "away": odds_p2}
    favorite = player1 if math_probs["home"] >= math_probs["away"] else player2

    # Используем уже готовые тексты GPT и Llama как статистик и скептик
    # Это позволяет market_verdict читать реальный анализ агентов
    stat_text  = gpt_text   if gpt_text   and not gpt_text.startswith("❌")   else ""
    skept_text = llama_text if llama_text and not llama_text.startswith("❌") else ""

    market_verdict = run_agent_market_verdict(
        "tennis", match_info, math_probs, bookmaker_odds, stat_text, skept_text
    )

    agent_results = {
        "statistician":  stat_text,
        "skeptic":       skept_text,
        "market_verdict": market_verdict,
    }

    final_probs = bayesian_combine(math_probs, stat_text, skept_text)
    verdict_block = format_verdict_block(agent_results, final_probs, bookmaker_odds, favorite)

    return {
        "agent_results": agent_results,
        "final_probs": final_probs,
        "verdict_block": verdict_block,
    }


def format_tennis_full_report(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    gpt_text: str, llama_text: str,
    candidates: list,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    sport_key: str = "",
    chimera_verdict_block: str = "",
    commence_time: str = None,
) -> str:
    """Собирает полный HTML-отчёт по теннисному матчу."""

    surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}
    surf_names = {"hard": "Хард", "clay": "Грунт", "grass": "Трава"}
    surf_icon  = surf_icons.get(surface, "🎾")
    surf_name  = surf_names.get(surface, "Хард")
    tour_upper = tour.upper()

    # Турнир из sport_key
    tournament_name = sport_key.replace(f"tennis_{tour}_", "").replace("_", " ").title() if sport_key else ""

    p1_win_pct = round(probs["p1_win"] * 100)
    p2_win_pct = round(probs["p2_win"] * 100)
    p1_rank    = probs.get("p1_rank", "?")
    p2_rank    = probs.get("p2_rank", "?")

    # Форматируем дату/время
    time_str = ""
    if commence_time:
        try:
            from datetime import datetime, timezone, timedelta
            _dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            _dt_msk = _dt.astimezone(timezone(timedelta(hours=3)))
            MONTHS = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
            time_str = f"📅 {_dt_msk.day} {MONTHS[_dt_msk.month-1]} {_dt_msk.year}, {_dt_msk.strftime('%H:%M')} МСК"
        except Exception:
            pass

    # Тотал геймов
    totals_line = ""
    try:
        from sports.tennis.model import predict_tennis_game_totals
        best_of = 5 if "grand_slam" in sport_key.lower() else 3
        _totals = predict_tennis_game_totals(
            p1_win=probs["p1_win"], p2_win=probs["p2_win"],
            p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
            surface=surface, tour=tour, best_of=best_of,
        )
        if _totals.get("confidence", 0) >= 55:
            totals_line = (
                f"\n📊 <b>Тотал геймов:</b> {_totals['prediction']} "
                f"(<b>{_totals['confidence']}%</b>) — <i>{_totals['reason']}</i>"
            )
    except Exception:
        pass

    # Определяем вердикт и надёжность
    best_cand = candidates[0] if candidates else None

    # Фаворит (для ПРОПУСТИТЬ тоже нужен)
    _winner     = player1 if p1_win_pct >= p2_win_pct else player2
    _winner_pct = max(p1_win_pct, p2_win_pct)
    _loser_pct  = min(p1_win_pct, p2_win_pct)

    if best_cand and best_cand.get("ev", 0) > 2:
        _winner_odds = best_cand["odds"]
        _ev          = best_cand["ev"]
        _kelly       = best_cand["kelly"]
        _u           = "3u" if _kelly >= 4 else ("2u" if _kelly >= 2 else "1u")
        _verdict_hdr = "✅ СТАВИТЬ"
        _verdict_bet = f"💰 <b>{_winner}</b> @ {_winner_odds} | EV: +{_ev:.1f}% | Банк: {_kelly:.1f}% ({_u})"
        try:
            from formatters import reliability_fires as _rf
            _reliability = _rf(_winner_pct)
        except Exception:
            _reliability = "🔥🔥🔥 Высокая"
    else:
        _verdict_hdr = "❌ НЕ СТАВИТЬ"
        # Определяем конкретную причину пропуска
        try:
            from config_thresholds import TENNIS_CFG as _tc
            _fav_odds = float(odds_p2) if _winner == player2 else float(odds_p1)
            _bk_implied = round(100 / _fav_odds, 1) if _fav_odds > 1 else 0
            _raw_ev   = _fav_odds * (_winner_pct / 100) - 1
            if _fav_odds < _tc.get("min_odds", 1.60):
                _skip_reason = (
                    f"Кэф {_fav_odds} слишком низкий.\n"
                    f"Наша модель: {_winner_pct}%, букмекер закладывает {_bk_implied}%.\n"
                    f"Нет преимущества — такие ставки не окупаются в долгосрок."
                )
            elif _raw_ev < _tc.get("min_ev", 0.12):
                _skip_reason = (
                    f"Наша модель: {_winner_pct}%, букмекер: {_bk_implied}%.\n"
                    f"EV {round(_raw_ev*100,1)}% — маржа мала, не перекрывает риск."
                )
            elif _winner_pct < _tc.get("min_prob", 0.65) * 100:
                _skip_reason = (
                    f"Уверенность {_winner_pct}% недостаточна.\n"
                    f"Нужно минимум {round(_tc.get('min_prob',0.65)*100)}% для входа."
                )
            else:
                _skip_reason = "Сигналы не набирают минимальный балл для входа"
        except Exception:
            _skip_reason = "Условия для ставки не выполнены"
        _verdict_bet = f"🏆 <b>{_winner}</b> выиграет — но ставить не рекомендуем\n<i>📋 {_skip_reason}</i>"
        _reliability = "⚠️ Модель уверена — ставки нет"

    lines = [
        f"{surf_icon} <b>{tour_upper} {tournament_name} | {surf_name}</b>",
        f"🎾 <b>{player1}</b> (#{p1_rank})  vs  <b>{player2}</b> (#{p2_rank})",
    ]
    if time_str:
        lines.append(time_str)
    lines += [
        f"💰 {player1}={odds_p1} · {player2}={odds_p2}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🎯 <b>ВЕРДИКТ: {_verdict_hdr}</b>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🏆 <b>{_winner}</b> — {_winner_pct}% | {_loser_pct}%",
        f"{_reliability}",
        _verdict_bet,
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 <b>АНАЛИЗ:</b>",
        f"⚡ ELO: {player1}={probs.get('p1_elo', '?')} · {player2}={probs.get('p2_elo', '?')}",
        f"🎯 Вероятности: <b>{player1} {p1_win_pct}%</b> · <b>{player2} {p2_win_pct}%</b>",
    ]

    if h2h_total >= 3:
        lines.append(f"⚔️ H2H: {player1} <b>{h2h_p1_wins}:{h2h_total - h2h_p1_wins}</b> {player2} (последние {h2h_total})")

    if totals_line:
        lines.append(totals_line)

    # Value bets
    best_value = [c for c in candidates if c.get("ev", 0) > 2]
    if best_value:
        lines.append(f"")
        lines.append(f"💎 <b>VALUE СТАВКИ:</b>")
        for c in best_value[:2]:
            _u = '3u' if c['kelly'] >= 4 else ('2u' if c['kelly'] >= 2 else '1u')
            kelly_str = f" | Келли: {c['kelly']:.1f}% банка ({_u})" if c['kelly'] > 0 else ""
            lines.append(
                f"  ✅ <b>{c['team']}</b> @ {c['odds']} "
                f"| Наша вер.: {c['prob']}% vs бук: {c['implied_prob']}% "
                f"| EV: <b>{c['ev']:+.1f}%</b>{kelly_str}"
            )

    # GPT (усекаем до 700 символов чтобы не превысить лимит Telegram)
    _gpt = gpt_text[:700] + "…" if gpt_text and len(gpt_text) > 700 else gpt_text
    lines.append(f"")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🦁 <b>Химера (анализ):</b>")
    if _gpt and not _gpt.startswith("❌"):
        lines.append(_gpt)
    else:
        lines.append(f"<i>{_gpt}</i>")

    # Llama (усекаем до 700 символов)
    _llama = llama_text[:700] + "…" if llama_text and len(llama_text) > 700 else llama_text
    lines.append(f"")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🌀 <b>Тень (независимый взгляд):</b>")
    if _llama and not _llama.startswith("❌"):
        lines.append(_llama)
    else:
        lines.append(f"<i>{_llama}</i>")

    # CHIMERA Score топ кандидата
    if candidates:
        top = candidates[0]
        lines.append(f"")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🏆 <b>CHIMERA Score: {top['chimera_score']:.0f}/100</b>")
        lines.append(
            f"├ Рейтинг: +{top.get('rank_pts', 0):.0f} | "
            f"Поверхность: +{top.get('surf_pts', 0):.0f} | "
            f"Ценность: +{top.get('value_pts', 0):.0f}"
        )
        if top.get("line_pts", 0):
            icon = "📉" if top["line_pts"] > 0 else "⚠️"
            lines.append(f"└ {icon} Движение линии: {top['line_pts']:+.0f} pts")

    # CHIMERA Multi-Agent Verdict (усекаем если слишком длинный)
    if chimera_verdict_block:
        _cvb = chimera_verdict_block[:800] + "…" if len(chimera_verdict_block) > 800 else chimera_verdict_block
        lines.append(_cvb)

    report = "\n".join(lines)

    # Финальная защита: Telegram лимит 4096 символов
    if len(report) > 4000:
        report = report[:3950] + "\n\n<i>…(отчёт сокращён)</i>"

    return report
