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

    prompt = f"""Ты — профессиональный аналитик тенниса для беттинга. Анализируй строго и честно.

МАТЧ ({tour_upper}): {player1} vs {player2}
ПОКРЫТИЕ: {surf_ru}
РЕЙТИНГ: {player1} #{p1_rank} | {player2} #{p2_rank}
КЭФЫ: {player1}={odds_p1} | {player2}={odds_p2}
НАШИ ВЕРОЯТНОСТИ: {player1}={round(probs['p1_win']*100)}% | {player2}={round(probs['p2_win']*100)}%{h2h_block}{form_block}

Дай структурированный анализ:

**ФАВОРИТ:** [игрок] — [главная причина в 1 предложении]

**ПОКРЫТИЕ:** Кому это покрытие подходит больше и почему? (стиль игры, специализация)

**ФОРМА И H2H:** Что говорит текущая форма? Есть ли психологическое преимущество по H2H?

**ПРОГНОЗ:** [игрок] победит с вероятностью [X%]. Счёт: [2:0 / 2:1]

**ТОТАЛЫ:** Тотал геймов [Больше/Меньше] 22.5 — [почему: играют в атаку или держат подачу?]

**СТАВКА:** [СТАВИТЬ на X @ кэф Y / ПРОПУСТИТЬ] — причина в 1 предложении

Будь конкретным. Не пиши воду."""

    return _call_ai(prompt, _gpt_client, "gpt-4.1-mini")


def run_tennis_llama_agent(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    p1_form: str = "", p2_form: str = "",
    p1_rank: int = 100, p2_rank: int = 100,
    gpt_verdict: str = "",
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
        # Берём только первые 200 символов вердикта GPT
        short = gpt_verdict[:200].replace('\n', ' ')
        gpt_block = f"\n\nМнение GPT (кратко): «{short}...»"

    prompt = f"""Ты — независимый тактический аналитик тенниса. Смотри критически, не копируй чужие выводы.

МАТЧ ({tour_upper}): {player1} vs {player2}
ПОКРЫТИЕ: {surf_ru}
РЕЙТИНГ: #{p1_rank} vs #{p2_rank}
КЭФЫ: {player1}={odds_p1} | {player2}={odds_p2}{h2h_block}{gpt_block}

Дай НЕЗАВИСИМУЮ оценку:

**ТАКТИКА:** Как каждый игрок будет строить игру на {surf_ru}? Кому стиль игры подходит больше?

**РИСКИ ДЛЯ ФАВОРИТА:** Что может пойти не так? (усталость, травма, психология под давлением)

**КЛЮЧЕВОЙ ФАКТОР:** Один главный аспект, который решит исход матча

**ТОТАЛЫ:** Ожидается ли долгий матч (много геймов) или быстрый разгром?

**ИТОГ:** [игрок] победит | Уверенность: [X%] | Если не ставить — почему?

Если есть серьёзные сомнения — скажи прямо. Не соглашайся с GPT просто так."""

    return _call_ai(prompt, _groq_client, "llama-3.3-70b-versatile", timeout=35)


def run_tennis_chimera_agents(
    player1: str, player2: str,
    probs: dict, odds_p1: float, odds_p2: float,
    surface: str, tour: str,
    p1_rank: int = 100, p2_rank: int = 100,
    h2h_p1_wins: int = 0, h2h_total: int = 0,
    p1_form: str = "", p2_form: str = "",
) -> dict:
    """
    Запускает CHIMERA Multi-Agent анализ для теннисного матча.
    Возвращает dict с результатами агентов и финальным вердиктом.
    """
    try:
        from chimera_multi_agent import run_all_agents, bayesian_combine, format_verdict_block
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

    agent_results = run_all_agents("tennis", match_info, math_probs, bookmaker_odds, favorite)
    final_probs = bayesian_combine(math_probs, agent_results.get("statistician", ""), agent_results.get("skeptic", ""))
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

    lines = [
        f"{surf_icon} <b>{tour_upper} {tournament_name} | {surf_name}</b>",
        f"",
        f"🎾 <b>{player1}</b> (#{p1_rank})  vs  <b>{player2}</b> (#{p2_rank})",
    ]
    if time_str:
        lines.append(time_str)
    lines += [
        f"",
        f"📊 <b>МАТЕМАТИЧЕСКИЙ АНАЛИЗ:</b>",
        f"⚡ Рейтинговый ELO: {player1}={probs.get('p1_elo', '?')} | {player2}={probs.get('p2_elo', '?')}",
        f"🎯 Вероятности: <b>{player1} {p1_win_pct}%</b> | <b>{player2} {p2_win_pct}%</b>",
        f"💰 Коэффициенты: {player1}={odds_p1} | {player2}={odds_p2}",
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
    lines.append(f"🧠 <b>GPT-4.1 (Стратег):</b>")
    if _gpt and not _gpt.startswith("❌"):
        lines.append(_gpt)
    else:
        lines.append(f"<i>{_gpt}</i>")

    # Llama (усекаем до 700 символов)
    _llama = llama_text[:700] + "…" if llama_text and len(llama_text) > 700 else llama_text
    lines.append(f"")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🤖 <b>Llama 3.3 70B (Тактик):</b>")
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
