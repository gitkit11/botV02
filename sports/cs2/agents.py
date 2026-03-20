"""
sports/cs2/agents.py — AI агенты для анализа CS2 матчей
Использует GPT-4.1-mini и Llama 3.3 70B через те же клиенты что и футбол

Источники данных:
  - PandaScore (free): история матчей, WR, форма, H2H
  - HLTV stats (hltv_stats.py): winrate по картам, рейтинг игроков
"""
import json
import os
import time

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
    from groq import Groq
    _gpt_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    _groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    print(f"[CS2-Agents] Ошибка инициализации клиентов: {e}")
    _gpt_client = None
    _groq_client = None

# Подключаем HLTV статистику (карты + игроки)
try:
    from .hltv_stats import (
        format_map_stats_for_ai,
        format_players_for_ai,
        get_team_map_stats as get_map_stats,
        get_player_stats,
    )
    _HLTV_AVAILABLE = True
except ImportError:
    _HLTV_AVAILABLE = False


def _call_ai(prompt, client, model, system_msg=None, retries=2):
    """Вызов AI модели с retry при пустом ответе."""
    if not client:
        return f"❌ Клиент {model} не инициализирован"
    if system_msg is None:
        system_msg = "Ты — профессиональный аналитик CS2 с 10-летним опытом. Отвечай на русском языке, кратко и по делу."
    
    # Проверка типа клиента для Groq
    from groq import Groq
    is_groq = isinstance(client, Groq)
    
    for attempt in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 600
            }
            
            # Groq не поддерживает response_format в некоторых моделях или требует иного синтаксиса,
            # но здесь мы ожидаем текст, так что просто вызываем.
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            if text:
                try:
                    # Попытка распарсить JSON, если ожидается JSON
                    if 'json_object' in kwargs.get('response_format', {}).values():
                        return json.loads(text)
                    return text
                except json.JSONDecodeError:
                    print(f"[AI] Ошибка JSONDecodeError для модели {model}. Raw response: {text[:200]}...")
                    return f"❌ Ошибка парсинга JSON от {model}"
            if attempt < retries - 1:
                time.sleep(1)
        except Exception as e:
            err = str(e)[:120]
            # Если ошибка 403 (Access Denied) или другие критические ошибки Groq
            if "403" in err or "access denied" in err.lower():
                print(f"[AI] Критическая ошибка {model}: {err}. Требуется fallback.")
                raise e # Пробрасываем выше для обработки в run_cs2_analyst_agent
            
            if attempt < retries - 1 and ("rate" in err.lower() or "timeout" in err.lower()):
                time.sleep(3)
            else:
                return f"❌ Ошибка {model}: {err}"
    return "❌ Агент не дал ответа"


def run_cs2_analyst_agent(home_team, away_team, map_stats, bookmaker_odds,
                           agent_type="gpt-4o", home_stats=None, away_stats=None, h2h=None,
                           tournament_context: dict = None,
                           home_standin: dict = None, away_standin: dict = None,
                           data_confidence: float = 1.0):
    """
    Запускает AI-агента для анализа матча CS2.

    agent_type: 'gpt-4o' (Стратег) или 'llama-3.3' (Тактик)
    home_stats/away_stats: реальная статистика из PandaScore
    h2h: личные встречи
    """
    home_odds = bookmaker_odds.get("home_win", 1.90)
    away_odds = bookmaker_odds.get("away_win", 1.90)
    ctx = tournament_context or {"type": "online", "tier": "B", "label": "Online"}

    # ── Блок 0: Контекст турнира ─────────────────────────────────────────────
    ctx_block = f"\nФОРМАТ: {ctx.get('label','Online')} | Тир: {ctx.get('tier','B')}"
    is_lan = ctx.get("type") in ("major", "lan_s", "lan_a")
    ctx_block += f"\nMEDIUM: {'🎯 LAN — давление, публика, лучший интернет' if is_lan else '💻 Online — меньше нервов, привычная обстановка'}"

    # ── Блок доверия к данным ────────────────────────────────────────────────
    data_warn_block = ""
    if data_confidence < 0.5:
        missing = []
        if not (home_stats and home_stats.get("matches", 0) > 0):
            missing.append(f"нет матч. истории {home_team}")
        if not (away_stats and away_stats.get("matches", 0) > 0):
            missing.append(f"нет матч. истории {away_team}")
        if not _HLTV_AVAILABLE:
            missing.append("HLTV недоступен")
        data_warn_block = (
            f"\n⚠️ МАЛО ДАННЫХ (уверенность модели: {int(data_confidence*100)}%)"
            + (f" — {'; '.join(missing)}" if missing else "")
            + ". Вероятности близки к 50/50, прогноз ненадёжен."
        )

    # ── Stand-in блок ────────────────────────────────────────────────────────
    standin_block = ""
    if home_standin and home_standin.get("has_standin"):
        standin_block += f"\n⚠️ STAND-IN {home_team}: {home_standin['standin_player']} заменяет {home_standin['missing_player']} — учти в прогнозе!"
    if away_standin and away_standin.get("has_standin"):
        standin_block += f"\n⚠️ STAND-IN {away_team}: {away_standin['standin_player']} заменяет {away_standin['missing_player']} — учти в прогнозе!"

    # ── Блок 1: Статистика матчей (PandaScore) ──────────────────────────────
    stats_block = ""
    if home_stats and home_stats.get("matches", 0) > 0:
        wr5_h = int(home_stats.get("winrate_last5", home_stats["winrate"]) * 100)
        stats_block += f"\n📊 Статистика матчей:"
        stats_block += f"\n  {home_team}: WR={int(home_stats['winrate']*100)}%, last5={wr5_h}%, форма: {home_stats['form']}"
    if away_stats and away_stats.get("matches", 0) > 0:
        wr5_a = int(away_stats.get("winrate_last5", away_stats["winrate"]) * 100)
        if not stats_block:
            stats_block += f"\n📊 Статистика матчей:"
        stats_block += f"\n  {away_team}: WR={int(away_stats['winrate']*100)}%, last5={wr5_a}%, форма: {away_stats['form']}"
    if h2h and h2h.get("total", 0) >= 2:
        stats_block += f"\n  H2H: {home_team} {h2h['team1_wins']}:{h2h['team2_wins']} {away_team} (из {h2h['total']} матчей)"

    # ── Блок 2: Статистика карт и игроков (HLTV) ────────────────────────────
    hltv_block = ""
    if _HLTV_AVAILABLE:
        hltv_maps = format_map_stats_for_ai(home_team, away_team)
        hltv_players = format_players_for_ai(home_team, away_team)
        if "недоступна" not in hltv_maps:
            hltv_block = f"\n{hltv_maps}"
        if "недоступна" not in hltv_players:
            hltv_block += f"\n\n{hltv_players}"

    # Fallback: MIS карты если HLTV недоступен
    if not hltv_block and map_stats:
        hltv_block = f"\nСтатистика карт (MIS):\n{json.dumps(map_stats, indent=2, ensure_ascii=False)}"

    # ── Промпты для агентов ──────────────────────────────────────────────────
    if agent_type == "gpt-4o":
        prompt = f"""Ты — профессиональный аналитик CS2 для беттинга. Анализируй строго и честно.

МАТЧ: {home_team} vs {away_team}
КЭФЫ: {home_team}={home_odds} | {away_team}={away_odds}{ctx_block}{standin_block}{data_warn_block}
{stats_block}
{hltv_block}

Дай структурированный анализ:

**ФАВОРИТ:** [команда] — [главная причина в 1 предложении]

**КАРТЫ:** На каких картах каждая команда сильнее? Кто выиграет пике?

**КЛЮЧЕВЫЕ ИГРОКИ:** Чей стар-плеер в лучшей форме? Кто может стать MVP?

**ПРОГНОЗ:** [команда] победит [2:0 / 2:1] с вероятностью [X%]

**ТОТАЛ:** Тотал карт [Меньше 2.5 / Больше 2.5] — [почему]

**СТАВКА:** [СТАВИТЬ на X @ кэф Y / ПРОПУСТИТЬ] — причина в 1 предложении

Будь конкретным. Не пиши воду."""
        client = _gpt_client
        model = "gpt-4.1-mini"

    else:
        # llama-3.3 — независимый тактический анализ
        prompt = f"""Ты — тактический аналитик CS2, независимый от GPT. Смотри на данные критически.

МАТЧ: {home_team} vs {away_team}
КЭФЫ: {home_team}={home_odds} | {away_team}={away_odds}{ctx_block}{standin_block}{data_warn_block}
{stats_block}
{hltv_block}

Оцени независимо:

**ВЕТО:** Какие 2 карты заберут команды и почему? Кому выгоден децидер?

**ТАКТИКА:** Агрессивный vs пассивный стиль — у кого преимущество на этом пуле карт?

**РИСКИ:** Что может пойти не по плану для фаворита?

**ИТОГ:** [команда] победит | Тотал карт: [больше/меньше] 2.5 | Уверенность: [X%]

Если данных недостаточно для уверенного прогноза — скажи прямо."""
        client = _groq_client
        model = "llama-3.3-70b-versatile"

    try:
        result = _call_ai(prompt, client, model)
    except Exception as e:
        # Если Llama через Groq не сработала, пробуем еще раз с увеличенным таймаутом
        if agent_type == "llama-3.3":
            print(f"[CS2-Agents] Llama-3.3 агент упал с ошибкой: {e}.")
            result = f"❌ Ошибка Llama 3.3: {str(e)[:100]}"
        else:
            result = f"❌ Ошибка {model}: {str(e)[:100]}"
            
    return result


def run_cs2_chimera_agents(
    home_team: str, away_team: str,
    math_probs: dict,
    bookmaker_odds: dict,
    home_stats=None, away_stats=None, h2h=None,
    tournament_context: dict = None,
    home_standin: dict = None, away_standin: dict = None,
) -> dict:
    """
    Запускает CHIMERA Multi-Agent анализ для CS2 матча.
    """
    try:
        from chimera_multi_agent import run_all_agents, bayesian_combine, format_verdict_block
    except ImportError:
        return {}

    ctx = tournament_context or {"label": "Online", "tier": "B"}
    standin_note = ""
    if home_standin and home_standin.get("has_standin"):
        standin_note += f"\n⚠️ STAND-IN {home_team}: {home_standin['standin_player']}"
    if away_standin and away_standin.get("has_standin"):
        standin_note += f"\n⚠️ STAND-IN {away_team}: {away_standin['standin_player']}"

    stats_str = ""
    if home_stats and home_stats.get("matches", 0) > 0:
        stats_str += f"\n{home_team}: WR={int(home_stats['winrate']*100)}%, форма={home_stats['form']}"
    if away_stats and away_stats.get("matches", 0) > 0:
        stats_str += f"\n{away_team}: WR={int(away_stats['winrate']*100)}%, форма={away_stats['form']}"
    if h2h and h2h.get("total", 0) >= 2:
        stats_str += f"\nH2H: {home_team} {h2h['team1_wins']}:{h2h['team2_wins']} {away_team}"

    match_info = (
        f"МАТЧ CS2: {home_team} vs {away_team}\n"
        f"ФОРМАТ: {ctx.get('label','Online')} | Тир: {ctx.get('tier','B')}\n"
        f"КОЭФФИЦИЕНТЫ: {home_team}={bookmaker_odds.get('home_win',1.9)} | {away_team}={bookmaker_odds.get('away_win',1.9)}"
        f"{standin_note}{stats_str}"
    )

    bm_probs = {"home": bookmaker_odds.get("home_win", 1.9), "away": bookmaker_odds.get("away_win", 1.9)}
    favorite = home_team if math_probs.get("home", 0.5) >= math_probs.get("away", 0.5) else away_team

    agent_results = run_all_agents("cs2", match_info, math_probs, bm_probs, favorite)
    final_probs = bayesian_combine(math_probs, agent_results.get("statistician", ""), agent_results.get("skeptic", ""))
    verdict_block = format_verdict_block(agent_results, final_probs, bm_probs, favorite)

    return {
        "agent_results": agent_results,
        "final_probs": final_probs,
        "verdict_block": verdict_block,
    }


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
