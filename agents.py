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
            print(f"[{model} ОШИБКА попытка {attempt+1}] {error_type}: {str(e)[:100]}")
            
            # Специальная обработка для Groq 403 (Access Denied)
            if is_groq and "403" in str(e):
                print(f"[{model}] ⚠️ Groq API вернула 403 (Access Denied). Проверьте:")
                print(f"    1. API ключ в GROQ_API_KEY")
                print(f"    2. Региональные ограничения")
                print(f"    3. Лимит запросов")
                return {"error": f"Groq 403: Access Denied", "status": "blocked", "model": model}
            
            if attempt < retries - 1:
                time.sleep(2)
                
    # Если все попытки исчерпаны — возвращаем ошибку (БЕЗ подмены на другую модель!)
    error_msg = f"[{model}] ❌ Все попытки исчерпаны. Модель недоступна."
    print(error_msg)
    return {"error": error_msg, "status": "unavailable", "model": model}

# --- 3. Специализированные ИИ-агенты (основной анализ) ---

def run_statistician_agent(prophet_data, team_stats_text=None):
    """
    🐍 ЗМЕЯ (GPT-4o)
    Анализирует только цифры: вероятности, форму, статистику.
    """
    stats_block = f"""
    Дополнительная статистика сезона:
    {team_stats_text}
    """ if team_stats_text else ""
    
    prompt = f"""
    Ты — лучший в мире футбольный статистик. Анализируй только числовые данные.

    Данные нейросети Пророк (обучена на 10 сезонах АПЛ):
    - Вероятность победы хозяев (П1): {prophet_data[1]:.2%}
    - Вероятность ничьей (Х): {prophet_data[0]:.2%}
    - Вероятность победы гостей (П2): {prophet_data[2]:.2%}
    {stats_block}
    
    Задача: дай статистическую оценку с учётом ВСЕХ данных. Какой исход наиболее вероятен? Насколько равный матч?
    Если есть данные по форме и голам — обязательно используй их в анализе.

    Формат ответа (только JSON):
    {{
      "analysis_summary": "Краткое резюме статистической картины (2-3 предложения).",
      "home_win_prob": <число от 0.0 до 1.0>,
      "draw_prob": <число от 0.0 до 1.0>,
      "away_win_prob": <число от 0.0 до 1.0>,
      "match_balance": "равный" или "лёгкое преимущество хозяев" или "явный фаворит хозяева" или "лёгкое преимущество гостей" или "явный фаворит гости"
    }}
    """
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

def run_arbitrator_agent(stats_result, scout_result, bookmaker_odds):
    """
    🐐 КОЗЁЛ (GPT-4o)
    Синтезирует мнения Змеи и Льва, выносит финальный вердикт.
    """
    prompt = f"""
    Ты — финальный Арбитр, мастер-аналитик ставок с 20-летним опытом. Синтезируй отчёты и вынеси окончательное решение.

    ОТЧЁТ СТАТИСТИКА:
    - Резюме: {stats_result.get('analysis_summary', 'Нет данных')}
    - П1: {stats_result.get('home_win_prob', 0.33):.2%} | Х: {stats_result.get('draw_prob', 0.33):.2%} | П2: {stats_result.get('away_win_prob', 0.33):.2%}
    - Баланс матча: {stats_result.get('match_balance', 'неизвестно')}

    ОТЧЁТ РАЗВЕДЧИК:
    - Резюме: {scout_result.get('analysis_summary', 'Нет данных')}
    - Ключевой фактор: {scout_result.get('key_factor', 'Нет данных')}
    - Настроение хозяев: {scout_result.get('home_team_sentiment', 0.0):.2f} | Гостей: {scout_result.get('away_team_sentiment', 0.0):.2f}

    КОЭФФИЦИЕНТЫ БУКМЕКЕРОВ:
    - П1: {bookmaker_odds.get('home_win', 0)} | Х: {bookmaker_odds.get('draw', 0)} | П2: {bookmaker_odds.get('away_win', 0)}

    ТВОИ ЗАДАЧИ:
    1. Взвесь данные: статистика 60%, новостной фон 40%
    2. Рассчитай итоговые вероятности для трёх исходов
    3. Найди Value Bet: сравни свою вероятность с подразумеваемой букмекером (1/коэф). Value есть если твоя вероятность > вероятности букмекера на 5%+
    4. Критерий Келли: Ставка% = ((Вероятность × Коэффициент) - 1) / (Коэффициент - 1). Если нет ценности — ставка 0.
    5. ВАЖНО: Рекомендуй ставку ТОЛЬКО если уверенность >= 60% И есть реальная ценность. Иначе — "Пропустить матч"

    Формат ответа (только JSON):
    {{
      "final_verdict_summary": "Резюме финального решения (2-3 предложения).",
      "recommended_outcome": "Победа хозяев" или "Ничья" или "Победа гостей",
      "final_confidence_percent": <целое число от 0 до 100>,
      "bookmaker_odds": <коэффициент на рекомендуемый исход>,
      "expected_value_percent": <преимущество над букмекером в %>,
      "recommended_stake_percent": <результат критерия Келли, 0 если нет ценности>,
      "bet_signal": "СТАВИТЬ" или "ПРОПУСТИТЬ",
      "signal_reason": "Почему ставить или пропустить (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

# --- 4. Llama Агент (НЕЗАВИСИМОЕ МНЕНИЕ, БЕЗ FALLBACK) ---

def run_llama_agent(home_team, away_team, prophet_data, news_summary, bookmaker_odds, team_stats_text=None):
    """
    🌀 ТЕНЬ (Llama 3.3 70B через Groq)
    Даёт НЕЗАВИСИМОЕ второе мнение. Если недоступна — возвращаем ошибку, НЕ подменяем на GPT!
    """
    if not groq_client:
        error_msg = "[Llama] ❌ Groq клиент не инициализирован. Llama агент недоступен."
        print(error_msg)
        return {"error": error_msg, "status": "unavailable", "model": "llama-3.3-70b-versatile"}

    stats_block = f"""
    4. Статистика сезона (API-Football):
    {team_stats_text}
    """ if team_stats_text else ""

    prompt = f"""
    Ты — независимый футбольный аналитик на базе Llama. Дай СВОЙ прогноз, не копируй чужие выводы.
    Матч: {home_team} (хозяева) vs {away_team} (гости)

    Данные:
    1. Нейросеть (10 сезонов АПЛ): П1={prophet_data[1]:.2%}, Х={prophet_data[0]:.2%}, П2={prophet_data[2]:.2%}
    2. Новостной фон: {news_summary}
    3. Коэффициенты: П1={bookmaker_odds.get('home_win', 0)}, X={bookmaker_odds.get('draw', 0)}, П2={bookmaker_odds.get('away_win', 0)}
    {stats_block}
    
    Твои задачи:
    1. Дай НЕЗАВИСИМЫЙ прогноз на исход (П1/Х/П2) со своими вероятностями
    2. Прогноз тотала голов: Больше 2.5 или Меньше 2.5 — с обоснованием (используй среднее голов из статистики если есть)
    3. Прогноз "Обе забьют": Да или Нет — с обоснованием (учитывай сухие матчи из статистики если есть)
    4. Оцени уверенность в своём прогнозе от 0 до 100%
    5. Напиши краткое резюме своего анализа

    Формат ответа (только JSON):
    {{
      "analysis_summary": "Твой независимый анализ (2-3 предложения).",
      "recommended_outcome": "Победа хозяев" или "Ничья" или "Победа гостей",
      "home_win_prob": <число от 0.0 до 1.0>,
      "draw_prob": <число от 0.0 до 1.0>,
      "away_win_prob": <число от 0.0 до 1.0>,
      "final_confidence_percent": <целое число от 0 до 100>,
      "total_goals_prediction": "Больше 2.5" или "Меньше 2.5",
      "total_goals_reasoning": "Почему такой прогноз по голам (1 предложение)",
      "both_teams_to_score_prediction": "Да" или "Нет",
      "btts_reasoning": "Почему обе забьют или нет (1 предложение)"
    }}
    """
    return call_ai(prompt, groq_client, "llama-3.3-70b-versatile")

# --- 5. Дополнительные рыночные агенты (Маркет-мейкеры) ---

def run_goals_market_agent(home_team, away_team, stats_text, bookmaker_odds):
    """Агент по рынку голов: анализирует ТБ/ТМ."""
    prompt = f"""
    Ты — эксперт по ставкам на тоталы голов. Матч: {home_team} vs {away_team}.
    Статистика голов и xG: {stats_text}
    Коэффициенты на ТБ 2.5: {bookmaker_odds.get('over_2_5', 'Нет данных')}, ТМ 2.5: {bookmaker_odds.get('under_2_5', 'Нет данных')}.
    Задача: Дай прогноз на ТБ 2.5 или ТМ 2.5. Оцени вероятность и ценность.
    Формат ответа (JSON):
    {{
      "prediction": "ТБ 2.5" или "ТМ 2.5",
      "probability": <число от 0.0 до 1.0>,
      "reasoning": "Почему именно этот прогноз (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_corners_market_agent(home_team, away_team, stats_text):
    """Агент по рынку корнеров."""
    prompt = f"""
    Ты — эксперт по ставкам на корнеры. Матч: {home_team} vs {away_team}.
    Статистика: {stats_text}
    Дай прогноз на ТБ/ТМ 9.5 корнеров.
    Формат ответа (JSON):
    {{
      "prediction": "ТБ 9.5" или "ТМ 9.5",
      "reasoning": "Почему (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_cards_market_agent(home_team, away_team, stats_text):
    """Агент по рынку карточек."""
    prompt = f"""
    Ты — эксперт по ставкам на карточки. Матч: {home_team} vs {away_team}.
    Статистика: {stats_text}
    Дай прогноз на ТБ/ТМ 4.5 карточек.
    Формат ответа (JSON):
    {{
      "prediction": "ТБ 4.5" или "ТМ 4.5",
      "reasoning": "Почему (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_handicap_market_agent(home_team, away_team, stats_text, bookmaker_odds):
    """Агент по рынку гандикапов."""
    prompt = f"""
    Ты — эксперт по ставкам на гандикапы. Матч: {home_team} vs {away_team}.
    Статистика: {stats_text}
    Коэффициенты: {bookmaker_odds}
    Дай прогноз на гандикап -1 / +1.
    Формат ответа (JSON):
    {{
      "prediction": "Гандикап -1" или "Гандикап +1",
      "reasoning": "Почему (1 предложение)"
    }}
    """
    return call_ai(prompt, client, "gpt-4.1-mini")

def run_football_chimera_agents(
    home_team: str, away_team: str,
    math_probs: dict,
    bookmaker_odds: dict,
    news_summary: str = "",
    stats_text: str = "",
) -> dict:
    """
    Запускает CHIMERA Multi-Agent анализ для футбольного матча.
    math_probs: {"home": 0.45, "draw": 0.28, "away": 0.27}
    bookmaker_odds: {"home": 1.85, "draw": 3.40, "away": 4.20}
    """
    try:
        from chimera_multi_agent import run_all_agents, bayesian_combine, format_verdict_block
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

    agent_results = run_all_agents("football", match_info, probs, odds, favorite)
    final_probs = bayesian_combine(probs, agent_results.get("statistician", ""), agent_results.get("skeptic", ""))
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

def build_math_ensemble(prophet_data, poisson_probs=None, elo_probs=None,
                        gpt_result=None, llama_result=None, mixtral_result=None,
                        bookmaker_odds=None, dc_probs=None):
    """
    Взвешенный ансамбль всех моделей.
    Веса (если все модели доступны):
      ELO 25% + Dixon-Coles 25% + Poisson/xG 15% + Prophet 15% + AI 10% + Odds 10%
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

    # 1. Dixon-Coles (обучен на 32k матчей, самый точный) — 25%
    try:
        if dc_probs:
            total_weight += _add(
                float(dc_probs.get("home_win", 0)),
                float(dc_probs.get("draw", 0)),
                float(dc_probs.get("away_win", 0)),
                0.25,
            )
    except Exception:
        pass

    # 2. ELO + форма — 25%
    try:
        if elo_probs:
            total_weight += _add(
                float(elo_probs.get("home", 0)),
                float(elo_probs.get("draw", 0)),
                float(elo_probs.get("away", 0)),
                0.25,
            )
    except Exception:
        pass

    # 3. Пуассон/xG — 15%
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

    # 4. Prophet нейросеть (draw=0, home=1, away=2) — 15%
    try:
        if prophet_data is not None:
            total_weight += _add(
                float(prophet_data[1]),
                float(prophet_data[0]),
                float(prophet_data[2]),
                0.15,
            )
    except Exception:
        pass

    # 5. AI вердикты (Змея + Лев + Козёл + Тень) — 10%
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

    # 6. Подразумеваемые вероятности букмекеров — 10%
    try:
        if bookmaker_odds:
            oh = float(bookmaker_odds.get("home_win", 0))
            od = float(bookmaker_odds.get("draw", 0))
            oa = float(bookmaker_odds.get("away_win", 0))
            if oh >= 1.02 and od >= 1.02 and oa >= 1.02:
                total_weight += _add(1.0 / oh, 1.0 / od, 1.0 / oa, 0.10)
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
