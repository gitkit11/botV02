# -*- coding: utf-8 -*-
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Импортируем наши модули ИИ
from config import TELEGRAM_TOKEN, THE_ODDS_API_KEY
from oracle_ai import oracle_analyze, format_oracle_report
from maestro_ai import maestro_analyze, format_maestro_report

# --- 1. Настройка ---
logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- 2. Загрузка моделей и данных ---
PROPHET_MODEL_PATH = "prophet_model.keras"
DATA_PATH = "featured_football_data.csv"
prophet_model, featured_data, scaler, features_to_scale = None, None, None, []

try:
    prophet_model = tf.keras.models.load_model(PROPHET_MODEL_PATH)
    print("[Загрузчик] Модель ИИ #1 \"Пророк\" успешно загружена.")
    featured_data = pd.read_csv(DATA_PATH)
    features_to_scale = [col for col in featured_data.columns if col.startswith(("H_", "A_"))]
    scaler = StandardScaler()
    scaler.fit(featured_data[features_to_scale])
    print("[Загрузчик] Датасет и скалер для \"Пророка\" готовы.")
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить модель или данные: {e}")

# --- 3. Кэш матчей ---
matches_cache = []

# --- 4. Функции ---
def get_prophet_prediction(home_team, away_team):
    if prophet_model is None or featured_data is None:
        return [0.33, 0.33, 0.33]
    try:
        last_10_games = featured_data.tail(10)
        game_features = last_10_games[features_to_scale]
        scaled_features = scaler.transform(game_features)
        sequence = np.reshape(scaled_features, (1, 10, len(features_to_scale)))
        prediction = prophet_model.predict(sequence, verbose=0)[0]
        return [float(prediction[0]), float(prediction[1]), float(prediction[2])]
    except Exception as e:
        print(f"[Пророк] Ошибка: {e}")
        return [0.33, 0.33, 0.33]

def fetch_matches():
    """Получает список ближайших матчей из The Odds API."""
    global matches_cache
    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
        params = {
            "apiKey": THE_ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        matches_cache = data[:10]
        print(f"[API] Получено {len(matches_cache)} матчей.")
        return matches_cache
    except Exception as e:
        print(f"[API Ошибка] {e}")
        return matches_cache  # Возвращаем кэш если API недоступен

def build_main_keyboard():
    kb = [
        [types.KeyboardButton(text="⚽ Выбрать матч для анализа")],
        [types.KeyboardButton(text="🔄 Обновить матчи")],
        [types.KeyboardButton(text="💎 VIP-доступ"), types.KeyboardButton(text="📊 Статистика")]
    ]
    return types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def build_matches_keyboard(matches):
    builder = InlineKeyboardBuilder()
    for i, match in enumerate(matches):
        home = match["home_team"]
        away = match["away_team"]
        builder.button(
            text=f"⚽ {home} vs {away}",
            callback_data=f"m_{i}"  # Используем индекс, чтобы избежать длинных callback_data
        )
    builder.button(text="🔄 Обновить список", callback_data="refresh_matches")
    builder.adjust(1)
    return builder.as_markup()

# --- 5. Обработчики ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    welcome_text = (
        "🤖 *Добро пожаловать в Chimera AI!*\n\n"
        "Три независимых ИИ анализируют матч:\n"
        "🔮 *Пророк* — 30 лет исторических данных\n"
        "📰 *Оракул* — новости и настроения команд\n"
        "⚖️ *Маэстро* — итоговый прогноз и Value Bets\n\n"
        "Нажмите кнопку ниже, чтобы начать."
    )
    await message.answer(welcome_text, reply_markup=build_main_keyboard(), parse_mode="Markdown")

@dp.message(F.text.in_(["⚽ Выбрать матч для анализа", "🔄 Обновить матчи"]))
async def show_matches(message: types.Message):
    await message.answer("⏳ *Загружаю ближайшие матчи...*", parse_mode="Markdown")
    matches = fetch_matches()
    if not matches:
        await message.answer(
            "😔 Не удалось получить матчи. Проверьте API-ключ или попробуйте позже.",
            reply_markup=build_main_keyboard()
        )
        return
    await message.answer(
        f"📋 *Ближайшие матчи АПЛ ({len(matches)} шт.):*\n\nВыберите матч для анализа:",
        reply_markup=build_matches_keyboard(matches),
        parse_mode="Markdown"
    )

@dp.callback_query(F.data == "refresh_matches")
async def refresh_matches_callback(callback: types.CallbackQuery):
    await callback.answer("🔄 Обновляю...")
    matches = fetch_matches()
    if not matches:
        await callback.message.edit_text("😔 Не удалось обновить матчи. Попробуйте позже.")
        return
    await callback.message.edit_text(
        f"📋 *Ближайшие матчи АПЛ ({len(matches)} шт.):*\n\nВыберите матч для анализа:",
        reply_markup=build_matches_keyboard(matches),
        parse_mode="Markdown"
    )

@dp.callback_query(F.data.startswith("m_"))
async def analyze_match_callback(callback: types.CallbackQuery):
    idx = int(callback.data.split("_")[1])
    if idx >= len(matches_cache):
        await callback.answer("Матч не найден. Обновите список.", show_alert=True)
        return

    match = matches_cache[idx]
    home_team = match["home_team"]
    away_team = match["away_team"]

    await callback.message.edit_text(f"🔍 *Анализирую: {home_team} vs {away_team}*", parse_mode="Markdown")
    await callback.answer()

    # --- ЭТАП 1: ПРОРОК ---
    await callback.message.answer("🔮 *ИИ #1 «Пророк» анализирует историю...*", parse_mode="Markdown")
    prophet_probs = get_prophet_prediction(home_team, away_team)

    # --- ЭТАП 2: ОРАКУЛ ---
    await callback.message.answer("📰 *ИИ #2 «Оракул» сканирует новости...*", parse_mode="Markdown")
    oracle_results = oracle_analyze(home_team, away_team)
    oracle_report = format_oracle_report(home_team, away_team, oracle_results)
    await callback.message.answer(oracle_report)

    # --- ЭТАП 3: МАЭСТРО ---
    await callback.message.answer("⚖️ *ИИ #3 «Маэстро» ищет выгодные ставки...*", parse_mode="Markdown")
    maestro_result = maestro_analyze(home_team, away_team, prophet_probs, oracle_results)
    final_report = format_maestro_report(maestro_result)
    await callback.message.answer(final_report, parse_mode="Markdown")

    # Кнопка "Выбрать другой матч"
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Выбрать другой матч", callback_data="back_to_matches")
    await callback.message.answer("Хотите проанализировать другой матч?", reply_markup=builder.as_markup())

@dp.callback_query(F.data == "back_to_matches")
async def back_to_matches(callback: types.CallbackQuery):
    await callback.answer()
    matches = matches_cache if matches_cache else fetch_matches()
    if not matches:
        await callback.message.answer("😔 Матчи недоступны.")
        return
    await callback.message.answer(
        f"📋 *Ближайшие матчи ({len(matches)} шт.):*",
        reply_markup=build_matches_keyboard(matches),
        parse_mode="Markdown"
    )

@dp.message(F.text == "💎 VIP-доступ")
async def vip_access(message: types.Message):
    await message.answer("💎 *VIP-доступ*\n\nСкоро здесь появится информация о тарифах.", parse_mode="Markdown")

@dp.message(F.text == "📊 Статистика")
async def stats(message: types.Message):
    await message.answer("📊 *Статистика*\n\nРаздел в разработке.", parse_mode="Markdown")

# --- 6. Запуск ---
async def main():
    print("🚀 Chimera AI: Бот запущен!")
    fetch_matches()  # Загружаем матчи при старте
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
