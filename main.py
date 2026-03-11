# -*- coding: utf-8 -*-
import asyncio
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command

# Импортируем наши модули ИИ
from config import TELEGRAM_TOKEN, THE_ODDS_API_KEY, RAPID_API_KEY
from oracle_ai import oracle_analyze, format_oracle_report
from maestro_ai import maestro_analyze, format_maestro_report

# --- 1. Настройка ---
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- 2. Загрузка моделей и данных ---
PROPHET_MODEL_PATH = "prophet_model.keras"
DATA_PATH = "featured_football_data.csv"

# Загружаем модель "Пророк"
try:
    prophet_model = tf.keras.models.load_model(PROPHET_MODEL_PATH)
    print("[Загрузчик] Модель ИИ #1 \"Пророк\" успешно загружена.")
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить модель 'prophet_model.keras': {e}")
    prophet_model = None

# Загружаем датасет для подготовки данных и скалера
try:
    featured_data = pd.read_csv(DATA_PATH)
    features_to_scale = [col for col in featured_data.columns if col.startswith(("H_", "A_"))]
    scaler = StandardScaler()
    scaler.fit(featured_data[features_to_scale])
    print("[Загрузчик] Датасет и скалер для \"Пророка\" готовы.")
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось загрузить или обработать '{DATA_PATH}': {e}")
    featured_data = None

# --- 3. Функции для ИИ "Пророк" ---
def get_prophet_prediction(home_team, away_team):
    """
    Генерирует предсказание от ИИ #1 "Пророк".
    """
    if prophet_model is None or featured_data is None:
        return [0.33, 0.33, 0.33] # Возвращаем равные вероятности в случае ошибки

    try:
        # Находим последние 10 матчей с участием этих команд (упрощенно)
        # В реальной системе здесь будет более сложный поиск и подготовка данных
        last_10_games = featured_data.tail(10)
        
        # Подготовка данных для модели
        game_features = last_10_games[features_to_scale]
        scaled_features = scaler.transform(game_features)
        
        # LSTM ожидает 3D-массив (samples, timesteps, features)
        sequence = np.reshape(scaled_features, (1, 10, len(features_to_scale)))
        
        # Предсказание
        prediction = prophet_model.predict(sequence, verbose=0)[0]
        
        # [Ничья, Победа хозяев, Победа гостей]
        return [prediction[0], prediction[1], prediction[2]]

    except Exception as e:
        print(f"[Пророк] Ошибка при предсказании: {e}")
        return [0.33, 0.33, 0.33]

# --- 4. Главный обработчик прогнозов ---
@dp.message(F.text == "🎯 Получить прогноз ИИ")
async def get_full_prediction(message: types.Message):
    await message.answer("🤖 *Активация системы Chimera AI...*", parse_mode="Markdown")

    # Для примера, используем фиксированные команды. 
    # В будущем здесь будет логика выбора ближайшего матча.
    home_team = "Manchester City"
    away_team = "Arsenal"

    await message.answer(f"🔍 *Анализирую матч: {home_team} vs {away_team}*", parse_mode="Markdown")

    # --- ЭТАП 1: ИИ #1 "ПРОРОК" (Исторический анализ) ---
    await message.answer("🔮 *ИИ #1 \"Пророк\" анализирует 30 лет футбольной истории...*", parse_mode="Markdown")
    prophet_probs = get_prophet_prediction(home_team, away_team)
    
    # --- ЭТАП 2: ИИ #2 "ОРАКУЛ" (Новостной анализ) ---
    await message.answer("📰 *ИИ #2 \"Оракул\" сканирует мировые новости и соцсети...*", parse_mode="Markdown")
    oracle_results = oracle_analyze(home_team, away_team)
    oracle_report = format_oracle_report(home_team, away_team, oracle_results)
    await message.answer(oracle_report)

    # --- ЭТАП 3: ИИ #3 "МАЭСТРО" (Финальный вердикт) ---
    await message.answer("⚖️ *ИИ #3 \"Маэстро\" объединяет данные и ищет выгодные ставки...*", parse_mode="Markdown")
    maestro_result = maestro_analyze(home_team, away_team, prophet_probs, oracle_results)
    final_report = format_maestro_report(maestro_result)
    
    await message.answer(final_report, parse_mode="Markdown")

# --- 5. Стандартные обработчики ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    welcome_text = (
        "🤖 **Добро пожаловать в Chimera AI!**\n\n"
        "Я — передовая система для анализа футбольных матчей, использующая три независимых искусственных интеллекта:\n\n"
        "🔮 **Пророк** — анализирует исторические данные за 30 лет.\n"
        "📰 **Оракул** — изучает новостной фон и настроения в командах.\n"
        "⚖️ **Маэстро** — объединяет все данные и находит выгодные ставки (Value Bets).\n\n"
        "Нажмите кнопку ниже, чтобы получить комплексный анализ ближайшего топ-матча."
    )
    kb = [[types.KeyboardButton(text="🎯 Получить прогноз ИИ")],
          [types.KeyboardButton(text="💎 VIP-доступ"), types.KeyboardButton(text="📊 Статистика")]]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
    await message.answer(welcome_text, reply_markup=keyboard, parse_mode="Markdown")

@dp.message(F.text == "💎 VIP-доступ")
async def vip_access(message: types.Message):
    await message.answer("💎 **VIP-доступ**\n\nПодписка на VIP-канал откроет вам доступ к эксклюзивным прогнозам, углубленному анализу и лучшим ставкам от Chimera AI. Скоро здесь появится информация о тарифах.")

@dp.message(F.text == "📊 Статистика")
async def stats(message: types.Message):
    await message.answer("📊 **Статистика**\n\nРаздел статистики находится в разработке. Здесь будет отображаться история наших прогнозов, ROI и другая полезная информация.")

# --- 6. Запуск бота ---
async def main():
    print("🚀 Chimera AI: Бот запущен и готов к работе!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    if prophet_model is None or featured_data is None:
        print("\n[ОШИБКА] Не удалось запустить бота, так как критические компоненты (модель или данные) не загружены. Запустите train_prophet.py для создания необходимых файлов.")
    else:
        asyncio.run(main())
