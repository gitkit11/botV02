# -*- coding: utf-8 -*-
import asyncio
import logging
import requests
import os
import time
from datetime import datetime, timezone, timedelta
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import TELEGRAM_TOKEN, THE_ODDS_API_KEY
try:
    from config import API_FOOTBALL_KEY
except ImportError:
    API_FOOTBALL_KEY = None
from oracle_ai import oracle_analyze
from agents import (
    run_statistician_agent, run_scout_agent, run_arbitrator_agent,
    run_llama_agent, run_goals_market_agent,
    run_corners_market_agent, run_cards_market_agent, run_handicap_market_agent,
    run_mixtral_agent, build_math_ensemble, calculate_value_bets
)
from math_model import (
    load_elo_ratings, save_elo_ratings, update_elo, elo_win_probabilities,
    load_team_form, get_form_string, get_form_bonus,
    poisson_match_probabilities, calculate_expected_goals, format_math_report
)

# --- 0. Глобальные переменные и загрузка ---
_elo_ratings = load_elo_ratings()
_team_form = load_team_form()
print(f"[ELO] Загружено {len(_elo_ratings)} команд | Форма: {len(_team_form)} команд")

def update_elo_after_match(home_team: str, away_team: str, home_score: int, away_score: int):
    global _elo_ratings
    _elo_ratings = update_elo(home_team, away_team, home_score, away_score, _elo_ratings)
    save_elo_ratings(_elo_ratings)

from api_football import get_match_stats
try:
    from understat_stats import format_xg_stats, get_team_xg_stats
    UNDERSTAT_AVAILABLE = True
except ImportError:
    UNDERSTAT_AVAILABLE = False
    def format_xg_stats(h, a, s='2024'): return ""
    def get_team_xg_stats(t, s='2024'): return None
from database import init_db, save_prediction, get_statistics, get_pending_predictions, update_result, get_recent_predictions
try:
    from injuries import get_match_injuries, get_match_injuries_async
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False
    def get_match_injuries(h, a): return {}, {}, ""
    async def get_match_injuries_async(h, a): return {}, {}, ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

try:
    prophet_model = tf.keras.models.load_model("prophet_model.keras")
    print("[Загрузчик] Модель ИИ #1 'Пророк' загружена.")
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить модель: {e}")
    prophet_model = None

try:
    import json
    data = pd.read_csv("all_matches_featured.csv", index_col=0)
    feature_cols = [c for c in data.columns if c not in ('FTR','label','HomeTeam','AwayTeam')]
    scaler = MinMaxScaler()
    scaler.fit(data[feature_cols])
    with open('team_encoder.json', 'r', encoding='utf-8') as _f:
        team_encoder = json.load(_f)
except Exception as e:
    data, scaler, team_encoder = None, None, {}

init_db()
matches_cache = []
_current_league = "soccer_epl"
_last_matches_refresh = 0

TEAM_NAME_MAP = {
    "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves", "Manchester City": "Man City",
    "Manchester United": "Man United", "West Ham United": "West Ham", "Tottenham Hotspur": "Tottenham",
}

def normalize_team(name): return TEAM_NAME_MAP.get(name, name)

def get_prophet_prediction(home_team, away_team):
    if not prophet_model or data is None or scaler is None: return [0.33, 0.33, 0.34]
    try:
        home_norm, away_norm = normalize_team(home_team), normalize_team(away_team)
        home_id, away_id = team_encoder.get(home_norm), team_encoder.get(away_norm)
        if home_id is None or away_id is None: return [0.33, 0.33, 0.34]
        home_data = data[data['HomeTeam_encoded'] == home_id].tail(5)
        away_data = data[data['AwayTeam_encoded'] == away_id].tail(5)
        sample = pd.concat([home_data, away_data]).tail(10) if len(home_data) >= 3 and len(away_data) >= 3 else data.tail(10)
        scaled = scaler.transform(sample[feature_cols].tail(10))
        prediction = prophet_model.predict(np.array([scaled]), verbose=0)[0]
        return [float(prediction[0]), float(prediction[1]), float(prediction[2])]
    except: return [0.33, 0.33, 0.34]

FOOTBALL_LEAGUES = [
    ("soccer_epl", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 АПЛ"), ("soccer_spain_la_liga", "🇪🇸 Ла Лига"), ("soccer_germany_bundesliga", "🇩🇪 Бундеслига"),
    ("soccer_italy_serie_a", "🇮🇹 Серия А"), ("soccer_france_ligue_one", "🇫🇷 Лига 1"), ("soccer_uefa_champs_league", "🏆 Лига Чемпионов"),
]

def get_matches(league: str = None, force: bool = False):
    global matches_cache, _last_matches_refresh, _current_league
    if league: _current_league = league
    if not force and matches_cache and (time.time() - _last_matches_refresh) < 21600: return matches_cache
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{_current_league}/odds/"
        params = {"apiKey": THE_ODDS_API_KEY, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}
        response = requests.get(url, params=params, timeout=10)
        data_api = response.json()
        now = datetime.now(timezone.utc).isoformat()[:19]
        matches_cache = [m for m in data_api if m.get('commence_time', '') > now][:15]
        _last_matches_refresh = time.time()
        return matches_cache
    except: return matches_cache

def build_main_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="⚽ Футбол")
    builder.button(text="🎮 Киберспорт CS2")
    builder.button(text="📊 Статистика")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def build_football_keyboard():
    builder = InlineKeyboardBuilder()
    for code, name in FOOTBALL_LEAGUES: builder.button(text=name, callback_data=f"league_{code}")
    builder.adjust(2)
    return builder.as_markup()

# --- 8. Хендлеры Telegram ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    get_matches()
    name = message.from_user.first_name or "друг"
    await message.answer(
        f"🔮 *CHIMERA AI v4.4* — Искусственный Интеллект для ставок\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Привет, *{name}*! 👋\n\n"
        f"🧠 *5 независимых моделей анализа:*\n"
        f"🔮 Пророк — нейросеть (66 000+ матчей)\n"
        f"📰 Оракул — анализ новостей и травм\n"
        f"🧠 GPT-4o — стратегический анализ\n"
        f"🤖 Llama 3.3 70B — тактический анализ\n"
        f"📊 Пуассон + ELO — математическая модель\n\n"
        f"⬇️ Выбери спорт:",
        parse_mode="Markdown", reply_markup=build_main_keyboard()
    )

@dp.message(lambda m: m.text == "⚽ Футбол")
async def handle_football(message: types.Message):
    await message.answer("⚽ *Футбол* — выбери лигу:", parse_mode="Markdown", reply_markup=build_football_keyboard())

@dp.callback_query(lambda c: c.data.startswith("league_"))
async def select_league(call: types.CallbackQuery):
    league_code = call.data.replace("league_", "")
    get_matches(league_code, force=True)
    builder = InlineKeyboardBuilder()
    for i, m in enumerate(matches_cache):
        builder.button(text=f"{m['home_team']} vs {m['away_team']}", callback_data=f"match_{i}")
    builder.button(text="⬅️ Назад", callback_data="back_to_main")
    builder.adjust(1)
    await call.message.edit_text("📅 *Ближайшие матчи:*", reply_markup=builder.as_markup())

@dp.callback_query(lambda c: c.data == "back_to_main")
async def back_to_main(call: types.CallbackQuery):
    await call.message.edit_text("⚽ *Футбол* — выбери лигу:", reply_markup=build_football_keyboard())

# --- [ СЕКЦИЯ CS2 - ИЗОЛИРОВАННО ] ---

@dp.message(lambda m: m.text == "🎮 Киберспорт CS2")
async def handle_cs2(message: types.Message):
    await message.answer("⏳ Загружаю матчи CS2 (Tier-2/3)...")
    try:
        from cs2_pandascore import get_combined_cs2_matches
        matches = get_combined_cs2_matches()
        builder = InlineKeyboardBuilder()
        if not matches:
            builder.button(text="NaVi vs Vitality (Demo)", callback_data="cs2_demo")
        else:
            for i, m in enumerate(matches[:10]):
                builder.button(text=f"🎮 {m['home']} vs {m['away']}", callback_data=f"cs2_m_{i}")
        builder.button(text="⬅️ Назад", callback_data="back_to_start")
        builder.adjust(1)
        await message.answer("🎮 *Матчи CS2:*", parse_mode="Markdown", reply_markup=builder.as_markup())
    except Exception as e:
        await message.answer(f"❌ Ошибка CS2: {e}")

@dp.callback_query(lambda c: c.data.startswith("cs2_m_") or c.data == "cs2_demo")
async def analyze_cs2_handler(call: types.CallbackQuery):
    await call.message.edit_text("⏳ *Глубокий анализ CS2...*\n(MIS, Veto, AI Agents)", parse_mode="Markdown")
    try:
        from cs2_core import calculate_cs2_win_prob, get_golden_signal, format_cs2_full_report
        from cs2_pandascore import get_combined_cs2_matches
        from cs2_agents import run_cs2_gpt_agent, run_cs2_llama_agent
        
        if call.data == "cs2_demo":
            match_data = {"home": "NaVi", "away": "Vitality", "league": "BLAST", "odds": {"home_win": 1.85, "away_win": 1.95}}
        else:
            idx = int(call.data.replace("cs2_m_", ""))
            matches = get_combined_cs2_matches()
            match_data = matches[idx]
        
        h, a, l = match_data['home'], match_data['away'], match_data.get('league', 'Pro')
        analysis = calculate_cs2_win_prob(h, a)
        analysis.update({"home_team": h, "away_team": a})
        
        gpt_res = await run_cs2_gpt_agent(h, a, l)
        llama_res = await run_cs2_llama_agent(h, a, l)
        signals = get_golden_signal(analysis, match_data.get('odds', {}))
        
        report = format_cs2_full_report(h, a, analysis, gpt_res, llama_res, signals)
        await call.message.edit_text(report, parse_mode="Markdown", reply_markup=InlineKeyboardBuilder().button(text="⬅️ Назад", callback_data="back_to_cs2").as_markup())
    except Exception as e:
        await call.message.edit_text(f"❌ Ошибка: {e}")

@dp.callback_query(lambda c: c.data == "back_to_cs2")
async def back_to_cs2(call: types.CallbackQuery):
    await call.message.delete()
    await handle_cs2(call.message)

@dp.callback_query(lambda c: c.data == "back_to_start")
async def back_to_start(call: types.CallbackQuery):
    await call.message.delete()
    await send_welcome(call.message)

@dp.message(lambda m: m.text == "📊 Статистика")
async def handle_stats(message: types.Message):
    stats = get_statistics()
    await message.answer(f"📊 *Статистика:*\nВсего прогнозов: {stats['total']}\nТочность: {stats['winner_accuracy']}%", parse_mode="Markdown")

async def main():
    print("🚀 Chimera AI v4.4 (Stable Football) + CS2 (Isolated) запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
