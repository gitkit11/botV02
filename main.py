# -*- coding: utf-8 -*-
import asyncio
import os
import logging
import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from config import TELEGRAM_TOKEN

# --- 1. Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Инициализация бота ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- 3. Кэш и вспомогательные функции ---
cs2_matches_cache = []

def conf_icon(conf):
    if conf >= 75: return "🟢"
    if conf >= 60: return "🟡"
    return "🔴"

# --- 4. Клавиатуры ---

def build_main_keyboard():
    """Строит главную клавиатуру с секциями спорта."""
    builder = ReplyKeyboardBuilder()
    builder.button(text="⚽ Футбол")
    builder.button(text="🎮 Киберспорт CS2")
    builder.button(text="🎾 Теннис (В разработке)")
    builder.button(text="📊 Мой ROI / Статистика")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def build_cs2_matches_keyboard():
    """Клавиатура для выбора реальных матчей CS2 (Tier-1/2/3)."""
    global cs2_matches_cache
    
    # Изолированный импорт внутри функции (Lazy Loading)
    try:
        from cs2_pandascore import get_combined_cs2_matches
        cs2_matches_cache = get_combined_cs2_matches()
    except Exception as e:
        logger.error(f"Ошибка загрузки CS2: {e}")
        return None

    builder = InlineKeyboardBuilder()
    
    if not cs2_matches_cache:
        # Тестовые данные если API пустое
        cs2_matches_cache = [
            {"home": "Natus Vincere", "away": "Team Vitality", "time": "20:00", "odds": {"home_win": 1.95, "away_win": 1.85}},
            {"home": "FaZe Clan", "away": "G2 Esports", "time": "22:30", "odds": {"home_win": 2.10, "away_win": 1.70}}
        ]

    for i, m in enumerate(cs2_matches_cache[:10]):
        builder.button(
            text=f"🎮 {m['home']} vs {m['away']} [{m['time']}]",
            callback_data=f"cs2_m_{i}"
        )
    
    builder.button(text="🔄 Обновить список", callback_data="back_to_cs2")
    builder.button(text="⬅️ Назад в меню", callback_data="back_to_main")
    builder.adjust(1)
    return builder.as_markup()

# --- 5. Хендлеры ---

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    name = message.from_user.first_name or "друг"
    await message.answer(
        f"🔮 *CHIMERA AI v4.5.2* — Искусственный Интеллект для ставок\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Привет, *{name}*! 👋\n"
        f"Я использую ансамбль из 5 моделей ИИ для анализа спорта.\n\n"
        f"Выберите раздел в меню ниже:",
        parse_mode="Markdown",
        reply_markup=build_main_keyboard()
    )

@dp.message(lambda message: message.text == "🎮 Киберспорт CS2")
async def show_cs2_menu(message: types.Message):
    kb = build_cs2_matches_keyboard()
    if kb:
        await message.answer("🎮 *Выберите матч CS2 для глубокого анализа:*", parse_mode="Markdown", reply_markup=kb)
    else:
        await message.answer("❌ Ошибка загрузки модуля киберспорта. Проверьте файлы проекта.")

@dp.callback_query(lambda c: c.data == "back_to_cs2")
async def back_to_cs2_callback(call: types.CallbackQuery):
    kb = build_cs2_matches_keyboard()
    await call.message.edit_text("🎮 *Выберите матч CS2 для глубокого анализа:*", parse_mode="Markdown", reply_markup=kb)

@dp.callback_query(lambda c: c.data == "back_to_main")
async def back_to_main_callback(call: types.CallbackQuery):
    await call.message.delete()
    await call.message.answer(
        "🔮 *Главное меню Chimera AI*",
        reply_markup=build_main_keyboard()
    )

@dp.callback_query(lambda c: c.data.startswith('cs2_m_'))
async def handle_cs2_match_analysis(call: types.CallbackQuery):
    try:
        from cs2_core import calculate_cs2_win_prob, get_golden_signal, format_cs2_full_report
        from cs2_agents import run_cs2_analyst_agent
    except Exception as e:
        await call.answer(f"Ошибка модулей CS2: {e}", show_alert=True)
        return

    match_idx = int(call.data.split('_')[2])
    if not cs2_matches_cache or match_idx >= len(cs2_matches_cache):
        await call.answer("Матч не найден.")
        return

    m = cs2_matches_cache[match_idx]
    home, away, odds = m["home"], m["away"], m["odds"]
    
    await call.message.edit_text(f"⏳ *Анализирую {home} vs {away}...*", parse_mode="Markdown")
    
    # Анализ
    analysis = calculate_cs2_win_prob(home, away)
    gpt_res = run_cs2_analyst_agent(home, away, {}, {}, "gpt-4o")
    llama_res = run_cs2_analyst_agent(home, away, {}, {}, "llama-3.3")
    golden = get_golden_signal(analysis, odds)
    
    report = format_cs2_full_report(home, away, analysis, gpt_res, llama_res, golden)
    
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ К списку матчей", callback_data="back_to_cs2")
    await call.message.edit_text(report, parse_mode="Markdown", reply_markup=builder.as_markup())

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
