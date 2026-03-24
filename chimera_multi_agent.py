# -*- coding: utf-8 -*-
"""
chimera_multi_agent.py — Bayesian Multi-Agent Engine
=====================================================
Архитектура:
  Llama (бесплатно, Groq):
    - Агент Статистик  → анализирует цифры, форму, H2H
    - Агент Скептик    → ищет почему фаворит может проиграть

  GPT-4.1-mini (платно, минимум токенов):
    - Агент Маркет+Вердикт → движение линий + финальное решение

  Bayesian Combiner:
    - Взвешивает математику + мнения агентов
    - Выдаёт одну финальную вероятность

Работает для всех спортов: football / cs2 / tennis
"""

import re
import time
import threading
import logging

logger = logging.getLogger(__name__)

# Импорт клиентов из основного agents.py
try:
    from agents import client as _gpt_client, groq_client as _groq_client
except ImportError:
    _gpt_client = None
    _groq_client = None


def _clean_cjk(text: str) -> str:
    return re.sub(
        r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]',
        '', text
    ).strip()


def _call_llama(prompt: str, max_tokens: int = 400) -> str:
    """Вызов Llama 3.3 через Groq."""
    if not _groq_client:
        return "❌ Groq недоступен"
    try:
        resp = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "Ты — спортивный аналитик. Отвечай строго на русском языке. "
                    "Запрещено использовать иероглифы. Будь конкретным и кратким."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=30,
        )
        text = resp.choices[0].message.content or ""
        return _clean_cjk(text).strip()
    except Exception as e:
        return f"❌ Llama: {str(e)[:80]}"


def _call_gpt(prompt: str, max_tokens: int = 300) -> str:
    """Вызов GPT-4.1-mini — только для важного."""
    if not _gpt_client:
        return "❌ GPT недоступен"
    try:
        resp = _gpt_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "Ты — топовый аналитик ставок. Отвечай строго на русском языке. "
                    "Будь конкретным. Максимум 3 предложения на каждый пункт."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=25,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"❌ GPT: {str(e)[:80]}"


# ─── Агент Статистик (Llama) ──────────────────────────────────────────────────

def run_agent_statistician(sport: str, match_info: str, math_probs: dict) -> str:
    """
    Llama: анализирует только цифры — форму, H2H, тренды.
    Возвращает текстовый анализ + вероятностную оценку.
    """
    sport_names = {"football": "футбол", "cs2": "CS2", "tennis": "теннис"}
    sport_ru = sport_names.get(sport, sport)

    prob_lines = " | ".join(f"{k}={round(v*100)}%" for k, v in math_probs.items())

    prompt = f"""Ты — Агент Статистик. Анализируй ТОЛЬКО цифры и статистику. Никаких общих фраз.

ВИД СПОРТА: {sport_ru}
{match_info}
МАТЕМАТИЧЕСКИЕ ВЕРОЯТНОСТИ: {prob_lines}

Твой анализ:
**СТАТИСТИКА:** Что говорят цифры? Форма, H2H, тренды последних матчей.
**ВЕРОЯТНОСТЬ:** Согласен или нет с математикой? Если нет — почему?
**ВЫВОД:** [команда/игрок] — статистический фаворит с [X]% уверенностью.

Максимум 5 строк. Только факты."""

    return _call_llama(prompt, max_tokens=350)


# ─── Агент Скептик (Llama) ────────────────────────────────────────────────────

def run_agent_skeptic(sport: str, match_info: str, favorite: str) -> str:
    """
    Llama: ищет причины почему фаворит ПРОИГРАЕТ. Devil's advocate.
    """
    sport_names = {"football": "футбол", "cs2": "CS2", "tennis": "теннис"}
    sport_ru = sport_names.get(sport, sport)

    prompt = f"""Ты — Агент Скептик. Твоя задача найти причины почему ФАВОРИТ проиграет. Играй роль devil's advocate.

ВИД СПОРТА: {sport_ru}
{match_info}
ФАВОРИТ ПО МАТЕМАТИКЕ: {favorite}

Найди слабые места фаворита:
**РИСКИ:** Топ-3 причины почему {favorite} может проиграть.
**АУТСАЙДЕР:** Что есть у аутсайдера чего нет у фаворита?
**ВЫВОД:** Стоит ли снижать уверенность в фаворите? [Да/Нет] — почему.

Максимум 5 строк. Будь честным, не выдумывай."""

    return _call_llama(prompt, max_tokens=300)


# ─── Агент Маркет + Вердикт (GPT) ────────────────────────────────────────────

def run_agent_market_verdict(
    sport: str,
    match_info: str,
    math_probs: dict,
    bookmaker_odds: dict,
    stat_analysis: str,
    skeptic_analysis: str,
) -> str:
    """
    GPT-4.1-mini: смотрит на движение линий + выносит финальный CHIMERA VERDICT.
    Минимум токенов — только самое важное.
    """
    sport_names = {"football": "футбол", "cs2": "CS2", "tennis": "теннис"}
    sport_ru = sport_names.get(sport, sport)

    prob_lines = " | ".join(f"{k}={round(v*100)}%" for k, v in math_probs.items())
    odds_lines = " | ".join(f"{k}={v}" for k, v in bookmaker_odds.items() if v)

    # Обрезаем тексты агентов чтобы не тратить токены GPT
    stat_short = stat_analysis[:200] if stat_analysis and not stat_analysis.startswith("❌") else "нет данных"
    skep_short = skeptic_analysis[:200] if skeptic_analysis and not skeptic_analysis.startswith("❌") else "нет данных"

    prompt = f"""Ты — финальный арбитр. У тебя уже есть мнения двух аналитиков. Вынеси CHIMERA VERDICT.

ВИД СПОРТА: {sport_ru}
{match_info}
МАТЕМАТИКА: {prob_lines}
КОЭФФИЦИЕНТЫ: {odds_lines}

СТАТИСТИК сказал: {stat_short}
СКЕПТИК сказал: {skep_short}

Твои задачи:
1. МАРКЕТ: Есть ли value bet? Сравни наши вероятности с коэффициентами. Value = наша вер. > 1/коэф на 5%+
2. ВЕРДИКТ: Одно предложение — кто победит и с какой вероятностью. НЕ пиши ставить или пропустить — это решает математическая модель по EV.

Формат:
**МАРКЕТ:** [есть value @ кэф X / нет value / коэффициент справедливый]
**CHIMERA VERDICT:** [команда/игрок] победит — [X]%

Максимум 4 строки."""

    return _call_gpt(prompt, max_tokens=250)


# ─── Параллельный запуск агентов ─────────────────────────────────────────────

def run_all_agents(
    sport: str,
    match_info: str,
    math_probs: dict,
    bookmaker_odds: dict,
    favorite: str,
) -> dict:
    """
    Запускает Статистика и Скептика параллельно (threading),
    затем GPT для финального вердикта.
    Возвращает dict с результатами всех агентов.
    """
    results = {}

    # Параллельный запуск двух Llama агентов
    def _run_stat():
        results["statistician"] = run_agent_statistician(sport, match_info, math_probs)

    def _run_skep():
        results["skeptic"] = run_agent_skeptic(sport, match_info, favorite)

    t1 = threading.Thread(target=_run_stat)
    t2 = threading.Thread(target=_run_skep)
    t1.start()
    t2.start()
    t1.join(timeout=40)
    t2.join(timeout=40)

    # GPT только после того как Llama агенты завершились
    results["market_verdict"] = run_agent_market_verdict(
        sport, match_info, math_probs, bookmaker_odds,
        results.get("statistician", ""),
        results.get("skeptic", ""),
    )

    return results


# ─── Bayesian Combiner ────────────────────────────────────────────────────────

def bayesian_combine(math_probs: dict, agent_text: str, skeptic_text: str) -> dict:
    """
    Простое байесовское обновление вероятностей на основе мнений агентов.
    Парсит % из текстов агентов и делает взвешенное среднее.

    math_probs: {"home": 0.55, "draw": 0.25, "away": 0.20} или {"home": 0.60, "away": 0.40}
    Возвращает обновлённые вероятности.
    """
    # Пробуем извлечь вероятность из текста агентов
    def _extract_prob(text: str) -> float | None:
        if not text or text.startswith("❌"):
            return None
        matches = re.findall(r'(\d{2,3})\s*%', text)
        if matches:
            # Берём последнее упоминание % (обычно итоговый вывод)
            val = int(matches[-1])
            if 10 <= val <= 95:
                return val / 100.0
        return None

    # Скептик снижает уверенность в фаворите
    skeptic_penalty = 0.0
    if skeptic_text and not skeptic_text.startswith("❌"):
        if "да" in skeptic_text.lower() and "снизить" in skeptic_text.lower():
            skeptic_penalty = 0.03  # -3% у фаворита
        elif "серьёзн" in skeptic_text.lower() or "высок" in skeptic_text.lower():
            skeptic_penalty = 0.05  # -5%

    # Обновляем вероятности: math 60% + агент 25% + скептик корректировка
    updated = dict(math_probs)
    agent_prob = _extract_prob(agent_text)

    keys = list(math_probs.keys())
    if agent_prob and len(keys) >= 1:
        # Находим фаворита (максимальная вероятность)
        fav_key = max(math_probs, key=math_probs.get)
        agent_fav_prob = agent_prob
        agent_weight = 0.25
        math_weight = 0.75

        new_fav = math_probs[fav_key] * math_weight + agent_fav_prob * agent_weight
        new_fav = max(0.05, min(0.95, new_fav - skeptic_penalty))

        # Перераспределяем остаток
        delta = math_probs[fav_key] - new_fav
        updated[fav_key] = round(new_fav, 3)
        others = [k for k in keys if k != fav_key]
        if others:
            per_other = delta / len(others)
            for k in others:
                updated[k] = round(math_probs[k] + per_other, 3)

    # Нормализуем
    total = sum(updated.values())
    if total > 0:
        updated = {k: round(v / total, 3) for k, v in updated.items()}

    return updated


# ─── Форматирование CHIMERA VERDICT блока ────────────────────────────────────

def format_verdict_block(
    agents_results: dict,
    final_probs: dict,
    bookmaker_odds: dict,
    favorite_name: str,
) -> str:
    """
    Генерирует HTML блок CHIMERA VERDICT для добавления в конец любого отчёта.
    """
    stat = agents_results.get("statistician", "")
    skep = agents_results.get("skeptic", "")
    verdict = agents_results.get("market_verdict", "")

    # Вычисляем финальную вероятность фаворита
    fav_key = max(final_probs, key=final_probs.get)
    fav_prob = round(final_probs[fav_key] * 100)

    # Нормализуем ключи odds: home_win→home, away_win→away, draw→draw
    _key_map = {"home_win": "home", "away_win": "away", "home": "home", "away": "away", "draw": "draw"}
    norm_odds = {}
    for k, v in bookmaker_odds.items():
        mapped = _key_map.get(k)
        if mapped and v and v > 1.0:
            norm_odds[mapped] = v

    # Ищем лучший value bet
    best_value = None
    for outcome_key, odds_val in norm_odds.items():
        our_prob = final_probs.get(outcome_key, 0)
        if not our_prob:
            continue
        implied = 1.0 / odds_val
        ev = (our_prob - implied) / implied * 100
        if ev >= 3.0:
            if best_value is None or ev > best_value["ev"]:
                best_value = {"outcome": outcome_key, "odds": odds_val, "ev": round(ev, 1), "prob": round(our_prob*100)}

    lines = [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🏆 <b>CHIMERA MULTI-AGENT VERDICT</b>",
    ]

    # Статистик — коротко
    if stat and not stat.startswith("❌"):
        stat_short = stat.split("\n")[0].replace("**СТАТИСТИКА:**", "").replace("**", "").strip()
        if stat_short:
            lines.append(f"📊 <b>Лев:</b> {stat_short[:120]}")

    # Скептик — коротко
    if skep and not skep.startswith("❌"):
        skep_lines = [l for l in skep.split("\n") if "риск" in l.lower() or "**риски" in l.lower()]
        if skep_lines:
            skep_short = skep_lines[0].replace("**РИСКИ:**", "").replace("**", "").strip()
            lines.append(f"⚠️ <b>Козёл:</b> {skep_short[:120]}")

    # Маркет
    if best_value:
        lines.append(f"💎 <b>Ценность:</b> @ {best_value['odds']} | наша вер. {best_value['prob']}% vs бук. {round(100/best_value['odds'])}% | EV <b>{best_value['ev']:+.1f}%</b>")
    else:
        lines.append("💎 <b>Ценность:</b> value bet не найден")

    # Финальный вердикт — сигнал всегда из математики (EV), не из GPT-текста
    signal = "🟢 СТАВИТЬ" if best_value else "⏸ НЕТ VALUE"
    if best_value:
        signal_detail = f" @ {best_value['odds']} | EV {best_value['ev']:+.1f}%"
    else:
        signal_detail = ""

    # Дополняем текстом GPT если он есть (только прогноз без беттинг-решения)
    gpt_pred = ""
    if verdict and not verdict.startswith("❌"):
        for line in verdict.split("\n"):
            if "chimera verdict" in line.lower() or "вердикт" in line.lower():
                clean = line.replace("**CHIMERA VERDICT:**", "").replace("**", "").strip()
                # Убираем если GPT всё же написал СТАВИТЬ/ПРОПУСТИТЬ
                for drop in ["| СТАВИТЬ", "| ПРОПУСТИТЬ", "| СТАВИТЬ", "| ПРОПУСТИТЬ"]:
                    clean = clean.split(drop)[0].strip().rstrip("|").strip()
                gpt_pred = clean
                break

    # Вердикт не дублируем — он уже показан в шапке отчёта

    return "\n".join(lines)
