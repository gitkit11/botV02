# -*- coding: utf-8 -*-
"""
agents.py — Три независимых ИИ-агента для анализа футбола (Chimera AI)
========================================================================

Архитектура "Трех голов":
1. СТАТИСТИК (GPT-4o) — анализирует только цифры и вероятности
2. РАЗВЕДЧИК (GPT-4o) — анализирует новости, травмы, мотивацию
3. АРБИТР (GPT-4o) — синтезирует мнения и выносит финальный вердикт
4. LLAMA (Llama 3.3 70B через Groq) — независимое второе мнение

Каждый агент работает независимо. Если один из них недоступен — система сигнализирует об ошибке,
но НЕ подменяет его другим агентом (это нарушает суть "разногласия мнений").
"""

import os
import re
from openai import OpenAI, APIStatusError
from groq import Groq
import json
import time
import logging
import threading
from collections import deque


def _clean_cjk(text: str) -> str:
    """Удаляет китайские/японские/корейские символы из строки."""
    # CJK Unified Ideographs, CJK Symbols, Hiragana, Katakana, Hangul
    cleaned = re.sub(
        r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
        r'\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]',
        '', text
    )
    # Убираем лишние пробелы после удаления
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned


def _sanitize_json_strings(obj):
    """Рекурсивно чистит все строковые значения в JSON от CJK символов."""
    if isinstance(obj, dict):
        return {k: _sanitize_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json_strings(i) for i in obj]
    elif isinstance(obj, str):
        return _clean_cjk(obj)
    return obj

logger = logging.getLogger(__name__)

# --- 1. Настройка клиентов ---
try:
    from config import OPENAI_API_KEY, GROQ_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# OpenAI клиент для GPT-4o (Статистик, Разведчик, Арбитр)
client = None
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("[agents] OpenAI client initialized.")
except Exception as e:
    print(f"[agents] CRITICAL: Failed to init OpenAI client: {e}")
    client = None

# Groq клиент для Llama 3.3 70B (независимое мнение)
groq_client = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[agents] Groq client initialized.")
    else:
        print("[agents] Groq API key not found, Llama agent unavailable.")
except Exception as e:
    groq_client = None
    print(f"[agents] CRITICAL: Failed to init Groq client: {e}")

# --- Groq rate limiter (макс 25 RPM, оставляем запас) ---
_groq_lock = threading.Lock()
_groq_timestamps: deque = deque()  # время успешных запросов за последние 60 сек
_GROQ_MAX_RPM = 25


def _groq_wait():
    """Ждёт, если в последнюю минуту уже отправлено _GROQ_MAX_RPM запросов."""
    while True:
        with _groq_lock:
            now = time.time()
            # Убираем записи старше 60 секунд
            while _groq_timestamps and _groq_timestamps[0] < now - 60:
                _groq_timestamps.popleft()
            if len(_groq_timestamps) < _GROQ_MAX_RPM:
                _groq_timestamps.append(now)
                return
            # Ждём до освобождения слота
            wait_until = _groq_timestamps[0] + 60.01
        sleep_for = wait_until - time.time()
        if sleep_for > 0:
            logger.info(f"[Groq] Rate limit: жду {sleep_for:.1f}с...")
            time.sleep(sleep_for)


# --- 2. Функция-помощник для вызова ИИ ---
def call_ai(prompt, client_instance, model, retries=2):
    """
    Отправляет промпт в указанную модель и возвращает ответ в формате JSON.
    
    ВАЖНО: Если модель недоступна, возвращаем ошибку, НЕ подменяем другой моделью!
    """
    if not client_instance:
        error_msg = f"[ОШИБКА] Клиент для модели {model} не инициализирован!"
        print(error_msg)
        return {"error": error_msg, "status": "unavailable"}
    
    is_groq = isinstance(client_instance, Groq)

    for attempt in range(retries):
        try:
            print(f"[{model}] Отправляю запрос (попытка {attempt+1})...")

            if is_groq:
                _groq_wait()
                response = client_instance.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            "Ты — эксперт мирового класса по ставкам на футбол. "
                            "Отвечай ТОЛЬКО валидным JSON объектом. "
                            "СТРОГО пиши ВСЕ текстовые поля ТОЛЬКО на русском языке. "
                            "ЗАПРЕЩЕНО использовать китайские, японские, корейские иероглифы "
                            "и любые символы не из кириллицы/латиницы/цифр. "
                            "Будь конкретным и аналитичным."
                        )},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    timeout=30
                )
            else:
                response = client_instance.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Ты — эксперт мирового класса по ставкам на футбол. Отвечай ТОЛЬКО валидным JSON объектом. Все текстовые поля пиши на русском языке. Будь конкретным и аналитичным."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    timeout=30
                )
            
            result = json.loads(response.choices[0].message.content)
            # Для Groq (Llama) чистим иероглифы из всех строковых полей
            if is_groq:
                result = _sanitize_json_strings(result)
            print(f"[{model}] ✅ Ответ получен: {str(result)[:100]}...")
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            err_str = str(e)
            print(f"[{model} ОШИБКА попытка {attempt+1}] {error_type}: {err_str[:100]}")
            logger.warning(f"[AI] Retry {attempt+1}/{retries} для {model}: {err_str[:200]}")

            # 403 — сразу raise, не retry
            if "403" in err_str:
                if is_groq:
                    print(f"[{model}] ⚠️ Groq API вернула 403 (Access Denied). Проверьте:")
                    print(f"    1. API ключ в GROQ_API_KEY")
                    print(f"    2. Региональные ограничения")
                    print(f"    3. Лимит запросов")
                return {"error": f"403: Access Denied", "status": "blocked", "model": model}

            # 429 / rate_limit — ждём дольше и повторяем
            if "429" in err_str or "rate_limit" in err_str.lower():
                sleep_sec = 5 if not is_groq else 15 * (attempt + 1)
                print(f"[{model}] ⚠️ Rate limit (429). Жду {sleep_sec}с...")
                time.sleep(sleep_sec)
                continue

            # 500 / 503 — сервер временно недоступен, короткая пауза
            if "500" in err_str or "503" in err_str:
                print(f"[{model}] ⚠️ Сервер недоступен (5xx). Жду 3с...")
                time.sleep(3)
                continue

            if attempt < retries - 1:
                time.sleep(2)
                
    # Если все попытки исчерпаны — возвращаем ошибку (БЕЗ подмены на другую модель!)
    error_msg = f"[{model}] ❌ Все попытки исчерпаны. Модель недоступна."
    print(error_msg)
    return {"error": error_msg, "status": "unavailable", "model": model}

# --- 3. Специализированные ИИ-агенты (основной анализ) ---

def run_statistician_agent(prophet_data, team_stats_text=None,
                           poisson_probs=None, elo_probs=None,
                           home_form=None, away_form=None, h2h_data=None):
    """
    🦁 ЛЕВ (GPT-4.1-mini)
    Анализирует цифры: Prophet, Пуассон/xG, ELO, форму, H2H.
    """
    _pd = prophet_data if prophet_data is not None else [0.33, 0.33, 0.34]

    # Блок Пуассон+xG
    poisson_block = ""
    if poisson_probs:
        poisson_block = (
            f"\nПУАССОН (xG-модель): П1={round(poisson_probs.get('home_win',0)*100)}% | "
            f"Х={round(poisson_probs.get('draw',0)*100)}% | П2={round(poisson_probs.get('away_win',0)*100)}%"
            f"\n  xG хозяев={poisson_probs.get('home_xg','?')} | xG гостей={poisson_probs.get('away_xg','?')}"
            f"\n  Тотал >2.5: {round(poisson_probs.get('over_25',0)*100)}% | Обе забьют: {round(poisson_probs.get('btts',0)*100)}%"
            f"\n  Вероятный счёт: {poisson_probs.get('most_likely_score','?')}"
        )

    # Блок ELO
    elo_block = ""
    if elo_probs:
        elo_block = (
            f"\nELO: П1={round(elo_probs.get('home',0)*100)}% | "
            f"Х={round(elo_probs.get('draw',0)*100)}% | П2={round(elo_probs.get('away',0)*100)}%"
        )

    # Форма
    form_block = ""
    if home_form:
        form_block += f"\nФорма хозяев (последние 5): {home_form}"
    if away_form:
        form_block += f"\nФорма гостей (последние 5): {away_form}"

    # H2H
    h2h_block = ""
    if h2h_data and h2h_data.get("total", 0) >= 3:
        h2h_block = f"\nH2H (последние {h2h_data['total']}): хозяева {h2h_data.get('home_wins',0)} побед, ничьих {h2h_data.get('draws',0)}, гостей {h2h_data.get('away_wins',0)}"

    stats_block = f"\nСтатистика сезона (API-Football):\n{team_stats_text}" if team_stats_text else ""

    # Память агента — история прошлых прогнозов на эти команды
    memory_block = ""
    try:
        from agent_memory import get_match_memory_context
        memory_block = get_match_memory_context(home_team, away_team, "football")
    except Exception:
        pass

    prompt = f"""Ты — лучший в мире футбольный статистик. Анализируй только числовые данные, никакой воды.

ДАННЫЕ ДЛЯ АНАЛИЗА:
Нейросеть Пророк (10 сезонов АПЛ): П1={_pd[1]:.1%} | Х={_pd[0]:.1%} | П2={_pd[2]:.1%}{poisson_block}{elo_block}{form_block}{h2h_block}{stats_block}{memory_block}

ЗАДАЧИ:
1. Сопоставь все три модели (Пророк + Пуассон + ELO) — они согласуются или расходятся?
2. Что говорит форма и H2H? Подтверждают или опровергают математику?
3. Если xG данные есть — используй их для прогноза тотала голов.
4. Дай итоговую статистическую оценку.

Формат ответа (только JSON):
{{
  "analysis_summary": "Статистическое резюме (2-3 предложения). Используй конкретные числа из данных.",
  "models_agreement": "согласуются" или "расходятся",
  "home_win_prob": <0.0-1.0>,
  "draw_prob": <0.0-1.0>,
  "away_win_prob": <0.0-1.0>,
  "total_goals_lean": "Больше 2.5" или "Меньше 2.5",
  "total_goals_confidence": <40-85>,
  "match_balance": "равный" или "лёгкое преимущество хозяев" или "явный фаворит хозяева" или "лёгкое преимущество гостей" или "явный фаворит гости"
}}"""
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_scout_agent(home_team, away_team, news_summary):
    """
    🦁 ЛЕВ (GPT-4o)
    Анализирует новости, травмы, мотивацию, моральный дух.
    """
    prompt = f"""
    Ты — лучший спортивный аналитик. Находишь скрытые факторы, невидимые в статистике.
    Матч: {home_team} vs {away_team}

    Новостной фон:
    {news_summary}

    Задача:
    1. Найди ключевые качественные факторы: травмы, моральный дух, мотивация, конфликты, усталость от плотного графика
    2. Оцени как новостной фон влияет на вероятность каждого исхода
    3. Дай оценку настроения каждой команды

    Формат ответа (только JSON):
    {{
      "analysis_summary": "Ключевые выводы из новостей (2-3 предложения).",
      "home_team_sentiment": <число от -1.0 до 1.0>,
      "away_team_sentiment": <число от -1.0 до 1.0>,
      "key_factor": "Самый важный фактор влияющий на матч (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_arbitrator_agent(stats_result, scout_result, bookmaker_odds,
                         poisson_probs=None, elo_probs=None):
    """
    🐍 ЗМЕЙ (GPT-4.1-mini)
    Синтезирует Льва + Скаута + математику, выносит финальный вердикт.
    """
    # Математический блок для арбитра
    math_block = ""
    if poisson_probs:
        math_block += (
            f"\nПУАССОН xG: П1={round(poisson_probs.get('home_win',0)*100)}% | "
            f"Х={round(poisson_probs.get('draw',0)*100)}% | П2={round(poisson_probs.get('away_win',0)*100)}%"
            f" | >2.5: {round(poisson_probs.get('over_25',0)*100)}%"
        )
    if elo_probs:
        math_block += (
            f"\nELO: П1={round(elo_probs.get('home',0)*100)}% | "
            f"П2={round(elo_probs.get('away',0)*100)}%"
        )

    # Согласие моделей
    models_agreement = stats_result.get('models_agreement', 'неизвестно')
    total_lean = stats_result.get('total_goals_lean', '')
    total_conf = stats_result.get('total_goals_confidence', 0)

    prompt = f"""Ты — финальный Арбитр, мастер-аналитик ставок. Синтезируй все данные и вынеси окончательное решение.

МАТЕМАТИЧЕСКИЕ МОДЕЛИ:{math_block if math_block else " нет данных"}

ОТЧЁТ СТАТИСТИКА (Лев):
- Резюме: {stats_result.get('analysis_summary', 'Нет данных')}
- П1={stats_result.get('home_win_prob',0.33):.1%} | Х={stats_result.get('draw_prob',0.33):.1%} | П2={stats_result.get('away_win_prob',0.33):.1%}
- Модели: {models_agreement} | Тотал: {total_lean} ({total_conf}%)

ОТЧЁТ РАЗВЕДЧИК (Козёл):
- Резюме: {scout_result.get('analysis_summary', 'Нет данных')}
- Ключевой фактор: {scout_result.get('key_factor', 'Нет данных')}
- Настроение хозяев: {scout_result.get('home_team_sentiment', 0.0):.1f} | Гостей: {scout_result.get('away_team_sentiment', 0.0):.1f}

КОЭФФИЦИЕНТЫ: П1={bookmaker_odds.get('home_win',0)} | Х={bookmaker_odds.get('draw',0)} | П2={bookmaker_odds.get('away_win',0)}

ЗАДАЧИ:
1. Взвесь математику (70%) и новостной фон (30%)
2. Если модели расходятся — объясни почему и какой доверяешь больше
3. Найди Value Bet: твоя вер. > 1/коэф на 5%+ = value. Формула Келли: ((Вер × Кэф) - 1) / (Кэф - 1) × 0.5
4. ВАЖНО: Рекомендуй СТАВИТЬ только при уверенности ≥ 58% И EV > 5%

Формат ответа (только JSON):
{{
  "final_verdict_summary": "Финальное решение (2-3 предложения). Объясни почему именно этот исход.",
  "recommended_outcome": "Победа хозяев" или "Ничья" или "Победа гостей",
  "final_confidence_percent": <0-100>,
  "bookmaker_odds": <кэф на рекомендованный исход>,
  "expected_value_percent": <EV в %, 0 если нет>,
  "recommended_stake_percent": <Kelly % от банка, 0 если нет ценности>,
  "bet_signal": "СТАВИТЬ" или "ПРОПУСТИТЬ",
  "signal_reason": "Причина в 1 предложении"
}}"""
    return call_ai(prompt, client, "gpt-4.1-mini")

# --- 4. Llama Агент (НЕЗАВИСИМОЕ МНЕНИЕ, БЕЗ FALLBACK) ---

def run_llama_agent(home_team, away_team, prophet_data, news_summary, bookmaker_odds,
                    team_stats_text=None, poisson_probs=None, elo_probs=None):
    """
    🌀 ТЕНЬ (Llama 3.3 70B через Groq)
    Даёт НЕЗАВИСИМОЕ второе мнение. Если недоступна — возвращаем ошибку, НЕ подменяем на GPT!
    """
    if not groq_client:
        return {"error": "[Llama] Groq не инициализирован", "status": "unavailable", "model": "llama-3.3-70b-versatile"}

    _pd = prophet_data if prophet_data is not None else [0.33, 0.33, 0.34]

    # Математический блок для Llama
    math_block = f"Нейросеть: П1={_pd[1]:.1%} | Х={_pd[0]:.1%} | П2={_pd[2]:.1%}"
    if poisson_probs:
        math_block += (
            f"\nПуассон xG: П1={round(poisson_probs.get('home_win',0)*100)}% | "
            f"П2={round(poisson_probs.get('away_win',0)*100)}% | "
            f">2.5={round(poisson_probs.get('over_25',0)*100)}% | "
            f"BTTS={round(poisson_probs.get('btts',0)*100)}%"
            f"\nxG: хозяев={poisson_probs.get('home_xg','?')} | гостей={poisson_probs.get('away_xg','?')}"
        )
    if elo_probs:
        math_block += f"\nELO: П1={round(elo_probs.get('home',0)*100)}% | П2={round(elo_probs.get('away',0)*100)}%"

    stats_block = f"\nСтатистика API-Football:\n{team_stats_text}" if team_stats_text else ""

    prompt = f"""Ты — независимый футбольный аналитик. Дай СВОЙ прогноз, не копируй чужие выводы.
Матч: {home_team} (хозяева) vs {away_team} (гости)

МАТЕМАТИЧЕСКИЕ ДАННЫЕ:
{math_block}

НОВОСТИ: {news_summary[:400] if news_summary else 'нет'}
КОЭФФИЦИЕНТЫ: П1={bookmaker_odds.get('home_win',0)} | Х={bookmaker_odds.get('draw',0)} | П2={bookmaker_odds.get('away_win',0)}{stats_block}

ЗАДАЧИ:
1. Дай НЕЗАВИСИМЫЙ прогноз со своими вероятностями — не просто повторяй математику
2. Тотал голов: используй xG данные если есть (xG хозяев + xG гостей = ожидаемый тотал)
3. Обе забьют: смотри на xG гостей — если >0.8, вероятно забьют
4. Если математические модели расходятся с коэффициентами — это важный сигнал, объясни

Формат ответа (только JSON):
{{
  "analysis_summary": "Твой независимый анализ (2-3 предложения с конкретными числами).",
  "recommended_outcome": "Победа хозяев" или "Ничья" или "Победа гостей",
  "home_win_prob": <0.0-1.0>,
  "draw_prob": <0.0-1.0>,
  "away_win_prob": <0.0-1.0>,
  "final_confidence_percent": <0-100>,
  "total_goals_prediction": "Больше 2.5" или "Меньше 2.5",
  "total_goals_reasoning": "Причина с опорой на xG или статистику (1 предложение)",
  "both_teams_to_score_prediction": "Да" или "Нет",
  "btts_reasoning": "Причина (1 предложение)"
}}"""
    return call_ai(prompt, groq_client, "llama-3.3-70b-versatile")

# --- 5. Дополнительные рыночные агенты (Маркет-мейкеры) ---


def run_football_chimera_agents(
    home_team: str, away_team: str,
    math_probs: dict,
    bookmaker_odds: dict,
    news_summary: str = "",
    stats_text: str = "",
    gpt_summary: str = "",
    llama_summary: str = "",
) -> dict:
    """
    Запускает CHIMERA Multi-Agent анализ для футбольного матча.
    math_probs: {"home": 0.45, "draw": 0.28, "away": 0.27}
    bookmaker_odds: {"home": 1.85, "draw": 3.40, "away": 4.20}
    gpt_summary: уже готовый текст от GPT-арбитра (передаётся из main.py)
    llama_summary: уже готовый текст от Llama (передаётся из main.py)
    """
    try:
        from chimera_multi_agent import run_all_agents, run_agent_market_verdict, bayesian_combine, format_verdict_block
    except ImportError:
        return {}

    match_info = (
        f"МАТЧ: {home_team} (хозяева) vs {away_team} (гости)\n"
        f"КОЭФФИЦИЕНТЫ: П1={bookmaker_odds.get('home_win', bookmaker_odds.get('home',0))} | "
        f"Х={bookmaker_odds.get('draw',0)} | П2={bookmaker_odds.get('away_win', bookmaker_odds.get('away',0))}"
    )
    if news_summary:
        match_info += f"\nНОВОСТИ: {news_summary[:300]}"
    if stats_text:
        match_info += f"\nСТАТИСТИКА: {stats_text[:300]}"
    # Память агента — история прогнозов на эти команды
    try:
        from agent_memory import get_match_memory_context, get_h2h_memory
        _mem = get_match_memory_context(home_team, away_team, "football")
        _h2h_mem = get_h2h_memory(home_team, away_team, "football")
        if _mem:
            match_info += _mem
        if _h2h_mem:
            match_info += _h2h_mem
    except Exception:
        pass

    # Нормализуем ключи
    probs = {
        "home": math_probs.get("home", math_probs.get("home_win", 0.34)),
        "draw": math_probs.get("draw", 0.33),
        "away": math_probs.get("away", math_probs.get("away_win", 0.33)),
    }
    odds = {
        "home": bookmaker_odds.get("home_win", bookmaker_odds.get("home", 0)),
        "draw": bookmaker_odds.get("draw", 0),
        "away": bookmaker_odds.get("away_win", bookmaker_odds.get("away", 0)),
    }

    # Фаворит
    favorite_map = {"home": home_team, "draw": "Ничья", "away": away_team}
    fav_key = max(probs, key=probs.get)
    favorite = favorite_map.get(fav_key, home_team)

    # Если переданы готовые тексты агентов — используем их напрямую,
    # иначе запускаем внутренних агентов (fallback)
    stat_text  = gpt_summary   if gpt_summary   and not gpt_summary.startswith("❌")   else ""
    skept_text = llama_summary if llama_summary and not llama_summary.startswith("❌") else ""

    if stat_text or skept_text:
        market_verdict = run_agent_market_verdict(
            "football", match_info, probs, odds, stat_text, skept_text
        )
        agent_results = {
            "statistician":   stat_text,
            "skeptic":        skept_text,
            "market_verdict": market_verdict,
        }
    else:
        agent_results = run_all_agents("football", match_info, probs, odds, favorite)

    final_probs  = bayesian_combine(probs, agent_results.get("statistician", ""), agent_results.get("skeptic", ""))
    verdict_block = format_verdict_block(agent_results, final_probs, odds, favorite)

    return {
        "agent_results": agent_results,
        "final_probs": final_probs,
        "verdict_block": verdict_block,
    }


def run_mixtral_agent(home_team, away_team, stats, news_summary="", bookmaker_odds=None,
                      team_stats_text="", poisson_probs=None, elo_probs=None, *args, **kwargs):
    """
    Агент Mixtral 8x7B через Groq — третье независимое мнение.
    Фокус: тактические паттерны, исторические тенденции стиля игры.
    """
    if not groq_client:
        return {"error": "Groq недоступен", "status": "unavailable"}

    odds_text = ""
    if bookmaker_odds:
        odds_text = (
            f"Букмекерские коэффициенты: "
            f"П1={bookmaker_odds.get('home_win', 0):.2f}, "
            f"X={bookmaker_odds.get('draw', 0):.2f}, "
            f"П2={bookmaker_odds.get('away_win', 0):.2f}"
        )

    elo_text = ""
    if elo_probs:
        elo_text = (
            f"ELO-вероятности: П1={elo_probs.get('home', 0)*100:.1f}%, "
            f"X={elo_probs.get('draw', 0)*100:.1f}%, "
            f"П2={elo_probs.get('away', 0)*100:.1f}%"
        )

    prompt = f"""Ты — тактический аналитик футбола. Анализируй ТОЛЬКО тактику и стиль игры.

Матч: {home_team} vs {away_team}
{elo_text}
{odds_text}
{f"Новости: {news_summary}" if news_summary else ""}
{f"Статистика: {team_stats_text[:300]}" if team_stats_text else ""}

Ответь строго в JSON:
{{
  "recommended_outcome": "home_win" | "draw" | "away_win",
  "final_confidence_percent": число 45-80,
  "analysis_summary": "1-2 предложения про тактику и стиль игры",
  "tactical_edge": "у кого тактическое преимущество и почему"
}}"""

    result = call_ai(prompt, groq_client, "llama3-70b-8192")
    if result.get("error"):
        return result
    return _sanitize_json_strings(result)


def run_goals_market_agent(home_team, away_team, news_summary="", bookmaker_odds=None, poisson_probs=None):
    """Анализирует рынок голов: тотал 2.5/1.5, обе забьют, кто забьёт первым."""
    if bookmaker_odds is None:
        bookmaker_odds = {}

    # Блок математических вероятностей
    math_block = ""
    if poisson_probs:
        math_block = (
            f"\nМАТЕМАТИКА (Пуассон+xG):"
            f"\n  xG хозяев={poisson_probs.get('home_xg','?')} | xG гостей={poisson_probs.get('away_xg','?')}"
            f"\n  Вер. >2.5: {round(poisson_probs.get('over_25',0)*100)}% | >1.5: {round(poisson_probs.get('over_15',0)*100)}%"
            f"\n  Обе забьют: {round(poisson_probs.get('btts',0)*100)}%"
            f"\n  Вероятный счёт: {poisson_probs.get('most_likely_score','?')} ({round(poisson_probs.get('most_likely_score_prob',0)*100)}%)"
        )

    prompt = f"""Ты — эксперт по рынку голов в футболе.
Матч: {home_team} vs {away_team}
Коэффициенты: Тотал>2.5={bookmaker_odds.get('over_2_5',0)} | Тотал<2.5={bookmaker_odds.get('under_2_5',0)} | Тотал>1.5={bookmaker_odds.get('over_1_5',0)}{math_block}
Новости: {news_summary[:250] if news_summary else 'нет'}

ВАЖНО: Если есть данные Пуассон/xG — они математически точнее чем просто мнение. Используй их как основу.
EV тотала = (наша_вер × кэф - 1). Рекомендуй только если EV > 0.

Ответь строго в JSON:
{{
  "summary": "анализ атаки/защиты обеих команд с опорой на xG (2 предложения)",
  "total_over_2_5": "Больше" или "Меньше",
  "total_over_2_5_confidence": число 40-85,
  "total_over_2_5_reason": "причина с xG числами если есть (1 предложение)",
  "total_over_1_5": "Больше" или "Меньше",
  "total_over_1_5_confidence": число 40-90,
  "btts": "Да" или "Нет",
  "btts_confidence": число 40-85,
  "btts_reason": "причина с опорой на xG гостей (1 предложение)",
  "first_goal": "Хозяева" или "Гости" или "Неопределённо",
  "best_goals_bet": "лучшая ставка с EV расчётом если кэфы есть (1 предложение)"
}}"""
    result = call_ai(prompt, client, "gpt-4.1-mini")
    return _sanitize_json_strings(result) if not result.get("error") else result


def run_corners_market_agent(home_team, away_team, news_summary="", bookmaker_odds=None):
    """Анализирует рынок угловых: тотал 9.5, кто возьмёт больше."""
    if bookmaker_odds is None:
        bookmaker_odds = {}
    prompt = f"""Ты — эксперт по рынку угловых в футболе.
Матч: {home_team} vs {away_team}
Новости: {news_summary[:300] if news_summary else 'нет'}

Ответь строго в JSON:
{{
  "summary": "краткий анализ угловых (2 предложения)",
  "total_corners_over_9_5": "Больше" или "Меньше",
  "total_corners_confidence": число 40-80,
  "total_corners_reason": "почему (1 предложение)",
  "home_corners_over_4_5": "Больше" или "Меньше",
  "away_corners_over_4_5": "Больше" или "Меньше",
  "corners_winner": "{home_team}" или "{away_team}" или "Равно",
  "best_corners_bet": "лучшая ставка на угловые (1 предложение)"
}}"""
    result = call_ai(prompt, client, "gpt-4.1-mini")
    return _sanitize_json_strings(result) if not result.get("error") else result


def run_cards_market_agent(home_team, away_team, news_summary="", bookmaker_odds=None):
    """Анализирует рынок карточек: тотал 3.5, красная карточка."""
    if bookmaker_odds is None:
        bookmaker_odds = {}
    prompt = f"""Ты — эксперт по рынку карточек в футболе.
Матч: {home_team} vs {away_team}
Новости: {news_summary[:300] if news_summary else 'нет'}

Ответь строго в JSON:
{{
  "summary": "краткий анализ дисциплины команд (2 предложения)",
  "total_cards_over_3_5": "Больше" или "Меньше",
  "total_cards_confidence": число 40-80,
  "total_cards_reason": "почему (1 предложение)",
  "red_card": "Вероятна" или "Маловероятна",
  "red_card_confidence": число 10-60,
  "more_cards_team": "{home_team}" или "{away_team}" или "Равно",
  "best_cards_bet": "лучшая ставка на карточки (1 предложение)"
}}"""
    result = call_ai(prompt, client, "gpt-4.1-mini")
    return _sanitize_json_strings(result) if not result.get("error") else result


def run_handicap_market_agent(home_team, away_team, prophet_data=None,
                              bookmaker_odds=None, gpt_result=None, llama_result=None):
    """Анализирует рынок гандикапов: азиатский гандикап и двойной шанс."""
    if bookmaker_odds is None:
        bookmaker_odds = {}
    _pd = prophet_data if prophet_data is not None else [0.33, 0.33, 0.34]
    gpt_summary  = (gpt_result  or {}).get("final_verdict_summary", "")
    llama_summary = (llama_result or {}).get("analysis_summary", "")
    prompt = f"""Ты — эксперт по рынку гандикапов в футболе.
Матч: {home_team} vs {away_team}
Нейросеть: П1={_pd[1]:.1%}, Х={_pd[0]:.1%}, П2={_pd[2]:.1%}
Коэффициенты: П1={bookmaker_odds.get('home_win',0)}, Х={bookmaker_odds.get('draw',0)}, П2={bookmaker_odds.get('away_win',0)}
Анализ агентов: {gpt_summary[:200] if gpt_summary else 'нет'} | {llama_summary[:200] if llama_summary else 'нет'}

Ответь строго в JSON:
{{
  "summary": "краткий анализ для гандикапов (2 предложения)",
  "asian_handicap_home": "Пройдёт" или "Не пройдёт",
  "asian_handicap_home_confidence": число 40-80,
  "asian_handicap_away": "Пройдёт" или "Не пройдёт",
  "asian_handicap_away_confidence": число 40-80,
  "double_chance": "1Х" или "Х2" или "12",
  "double_chance_reason": "почему (1 предложение)",
  "best_handicap_bet": "лучшая ставка на гандикап (1 предложение)"
}}"""
    result = call_ai(prompt, client, "gpt-4.1-mini")
    return _sanitize_json_strings(result) if not result.get("error") else result


def build_math_ensemble(prophet_data, poisson_probs=None, elo_probs=None,
                        gpt_result=None, llama_result=None, mixtral_result=None,
                        bookmaker_odds=None, dc_probs=None):
    """
    Взвешенный ансамбль всех моделей.
    Веса (если все модели доступны):
      Pinnacle no-vig 20% + ELO 20% + Dixon-Coles 20% + Poisson/xG 15%
      + Prophet 10% + AI 10% + Raw odds 5%
    Pinnacle no-vig — самая точная вероятность (62 букмекера, минимальная маржа).
    Недоступные модели исключаются, остальные перенормируются до 100%.
    """
    scores = {"home": 0.0, "draw": 0.0, "away": 0.0}
    total_weight = 0.0

    def _add(probs_h, probs_d, probs_a, weight):
        s = probs_h + probs_d + probs_a
        if s > 0:
            scores["home"] += (probs_h / s) * weight
            scores["draw"] += (probs_d / s) * weight
            scores["away"] += (probs_a / s) * weight
            return weight
        return 0.0

    # 1. Pinnacle no-vig вероятность — 20%
    # Самый точный источник: снята маржа с лучшего шарп-букмекера в мире
    try:
        if bookmaker_odds:
            nv_h = float(bookmaker_odds.get("no_vig_home", 0))
            nv_d = float(bookmaker_odds.get("no_vig_draw", 0))
            nv_a = float(bookmaker_odds.get("no_vig_away", 0))
            if nv_h > 0 and nv_d > 0 and nv_a > 0:
                total_weight += _add(nv_h, nv_d, nv_a, 0.20)
    except Exception:
        pass

    # 2. Dixon-Coles (обучен на 32k матчей) — 20%
    try:
        if dc_probs:
            total_weight += _add(
                float(dc_probs.get("home_win", 0)),
                float(dc_probs.get("draw", 0)),
                float(dc_probs.get("away_win", 0)),
                0.20,
            )
    except Exception:
        pass

    # 3. ELO + форма — 20%
    try:
        if elo_probs:
            total_weight += _add(
                float(elo_probs.get("home", 0)),
                float(elo_probs.get("draw", 0)),
                float(elo_probs.get("away", 0)),
                0.20,
            )
    except Exception:
        pass

    # 4. Пуассон/xG — 15%
    try:
        if poisson_probs:
            total_weight += _add(
                float(poisson_probs.get("home_win", 0)),
                float(poisson_probs.get("draw", 0)),
                float(poisson_probs.get("away_win", 0)),
                0.15,
            )
    except Exception:
        pass

    # 5. Prophet нейросеть (draw=0, home=1, away=2) — 10%
    try:
        if prophet_data is not None:
            _ph = float(prophet_data[1])
            _pd = float(prophet_data[0])
            _pa = float(prophet_data[2])
            _ps = _ph + _pd + _pa
            if _ps > 0:
                _ph, _pd, _pa = _ph / _ps, _pd / _ps, _pa / _ps
            total_weight += _add(_ph, _pd, _pa, 0.10)
    except Exception:
        pass

    # 6. AI вердикты (GPT + Llama + Mixtral) — 10%
    try:
        ai_votes = {"home": 0.0, "draw": 0.0, "away": 0.0}
        ai_count = 0.0
        verdict_map = {
            "home_win": "home", "победа хозяев": "home", "п1": "home",
            "draw": "draw", "ничья": "draw", "х": "draw",
            "away_win": "away", "победа гостей": "away", "п2": "away",
        }
        for ai_res in [gpt_result, llama_result, mixtral_result]:
            if not isinstance(ai_res, dict):
                continue
            v = str(ai_res.get("recommended_outcome", "")).lower().strip()
            conf = float(ai_res.get("final_confidence_percent", 50)) / 100.0
            key = verdict_map.get(v)
            if key:
                ai_votes[key] += conf
                ai_count += conf
        if ai_count > 0:
            total_weight += _add(
                ai_votes["home"] / ai_count,
                ai_votes["draw"] / ai_count,
                ai_votes["away"] / ai_count,
                0.10,
            )
    except Exception:
        pass

    # 7. Сырые подразумеваемые вероятности букмекеров — 5% (fallback если нет no-vig)
    try:
        if bookmaker_odds:
            nv_h = float(bookmaker_odds.get("no_vig_home", 0))
            if not nv_h:  # только если no-vig не доступен
                oh = float(bookmaker_odds.get("home_win", 0))
                od = float(bookmaker_odds.get("draw", 0))
                oa = float(bookmaker_odds.get("away_win", 0))
                if oh >= 1.02 and od >= 1.02 and oa >= 1.02:
                    total_weight += _add(1.0 / oh, 1.0 / od, 1.0 / oa, 0.05)
    except Exception:
        pass

    # Нормализация
    if total_weight > 0:
        for k in scores:
            scores[k] = scores[k] / total_weight
        s = sum(scores.values())
        if s > 0:
            for k in scores:
                scores[k] = round(scores[k] / s, 3)
    else:
        scores = {"home": 0.34, "draw": 0.33, "away": 0.33}

    # Калибровка по историческим данным Pinnacle (если таблица существует)
    try:
        from calibration import calibrate_odds
        h, d, a = calibrate_odds(scores["home"], scores["draw"], scores["away"])
        scores = {"home": h, "draw": d, "away": a}
    except Exception:
        pass

    return scores

def calculate_value_bets(predictions, odds):
    """Рассчитывает ценные ставки на основе ансамблевых вероятностей."""
    result = []
    if not isinstance(predictions, dict) or not isinstance(odds, dict):
        return result
    outcome_map = {
        "home": ("П1 (хозяева)", odds.get("home", 0)),
        "draw": ("Ничья",        odds.get("draw", 0)),
        "away": ("П2 (гости)",   odds.get("away", 0)),
    }
    for key, (label, odd) in outcome_map.items():
        our_prob = predictions.get(key, 0)
        if not odd or odd <= 1.0 or not our_prob:
            continue
        book_prob = 1.0 / odd
        ev = round((our_prob - book_prob) / book_prob * 100, 1)
        if ev >= 5.0:
            kelly = round(max(0, (our_prob * odd - 1) / (odd - 1)) * 100, 1)
            result.append({
                "outcome":   label,
                "odds":      round(odd, 2),
                "our_prob":  round(our_prob * 100, 1),
                "book_prob": round(book_prob * 100, 1),
                "ev":        ev,
                "kelly":     kelly,
            })
    result.sort(key=lambda x: x["ev"], reverse=True)
    return result
