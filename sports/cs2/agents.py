"""
sports/cs2/agents.py — AI агенты для анализа CS2 матчей
Использует GPT-4o и Llama 3.3 70B через те же клиенты что и футбол
"""
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from config import OPENAI_API_KEY, GROQ_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Инициализация клиентов
try:
    from openai import OpenAI
    _gpt_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
except Exception:
    _gpt_client = None
    _groq_client = None


def _call_ai(prompt, client, model, system_msg=None):
    """Вызов AI модели и возврат текстового ответа."""
    if not client:
        return f"❌ Клиент {model} не инициализирован"
    if system_msg is None:
        system_msg = "Ты — профессиональный аналитик CS2 с 10-летним опытом. Отвечай на русском языке, кратко и по делу."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Ошибка {model}: {str(e)[:100]}"


def run_cs2_analyst_agent(home_team, away_team, map_stats, bookmaker_odds, agent_type="gpt-4o"):
    """
    Запускает AI-агента для анализа матча CS2.
    agent_type: 'gpt-4o' (Стратег) или 'llama-3.3' (Тактик)
    """
    home_odds = bookmaker_odds.get("home_win", 1.90)
    away_odds = bookmaker_odds.get("away_win", 1.90)

    # Формируем блок статистики карт если есть
    maps_block = ""
    if map_stats:
        maps_block = f"\nСтатистика карт:\n{json.dumps(map_stats, indent=2, ensure_ascii=False)}"

    if agent_type == "gpt-4o":
        prompt = f"""Матч CS2: {home_team} vs {away_team}
Коэффициенты: {home_team}={home_odds} | {away_team}={away_odds}
{maps_block}

Как стратег — оцени:
1. Кто фаворит и почему (стиль игры, сильные карты)
2. Прогноз счёта серии (2:0, 2:1)
3. Есть ли value bet? Если да — на что?
4. Уверенность в прогнозе (%)

Ответ краткий, 4-5 предложений."""
        client = _gpt_client
        model = "gpt-4.1-mini"
    else:
        # llama-3.3 — тактический анализ
        prompt = f"""Матч CS2: {home_team} vs {away_team}
Коэффициенты: {home_team}={home_odds} | {away_team}={away_odds}
{maps_block}

Как тактик — оцени:
1. Мап-вето: какие карты выберут/забанят команды
2. Ключевые игроки которые решат исход
3. Твой прогноз на победителя
4. Тотал карт: больше или меньше 2.5

Ответ краткий, 4-5 предложений."""
        client = _groq_client
        model = "llama-3.3-70b-versatile"

    result = _call_ai(prompt, client, model)
    return result


def build_cs2_ensemble(math_prob, ai_probs, news_factor=1.0):
    """
    Собирает ансамбль из математики и AI.
    math_prob: (prob_home, prob_away)
    ai_probs: list of (prob_home, prob_away)
    """
    weights = {"math": 0.50, "ai": 0.50}

    final_home = math_prob[0] * weights["math"]
    avg_ai_home = sum([p[0] for p in ai_probs]) / len(ai_probs) if ai_probs else math_prob[0]
    final_home += avg_ai_home * weights["ai"]
    final_home *= news_factor

    final_home = max(0.05, min(0.95, final_home))
    final_away = 1.0 - final_home
    return (round(final_home, 3), round(final_away, 3))
