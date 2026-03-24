# -*- coding: utf-8 -*-
"""handlers/user.py — /start, онбординг, bankroll, mybet, noop, chimera_ask"""
import asyncio
import logging

from aiogram import Router, types
from aiogram.filters import Command

from state import (
    ADMIN_IDS, CHIMERA_DAILY_LIMIT,
    _chimera_waiting, _chimera_daily,
    _awaiting_bankroll,
)
from database import (
    upsert_user, log_action,
    set_user_language, get_user_language,
    set_user_bankroll, get_user_bankroll,
    get_pl_stats, mark_user_bet,
)
from i18n import t
from keyboards import build_main_keyboard
from handlers.common import get_matches

logger = logging.getLogger(__name__)
router = Router()


@router.message(Command("start"))
async def send_welcome(message: types.Message):
    upsert_user(message.from_user.id, message.from_user.username or "", message.from_user.first_name or "")
    log_action(message.from_user.id, "/start")
    kb = types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="🇷🇺 Русский", callback_data="set_lang_ru"),
        types.InlineKeyboardButton(text="🇬🇧 English", callback_data="set_lang_en"),
    ]])
    await message.answer("🐉", reply_markup=kb)


async def _run_onboarding(call: types.CallbackQuery, lang: str):
    m1 = await call.message.answer(t("appear_1", lang), parse_mode="HTML")
    await asyncio.sleep(1.2)
    await m1.edit_text(t("appear_2", lang), parse_mode="HTML")
    await asyncio.sleep(1.5)
    await m1.edit_text(t("appear_3", lang), parse_mode="HTML")
    await asyncio.sleep(1.0)
    await call.message.answer(t("legend", lang), parse_mode="HTML")
    await asyncio.sleep(0.5)
    await call.message.answer(t("features", lang), parse_mode="HTML")
    await asyncio.sleep(0.5)
    enter_kb = types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text=t("enter_btn", lang), callback_data=f"enter_chimera_{lang}"),
    ]])
    await call.message.answer(t("benefits", lang), parse_mode="HTML", reply_markup=enter_kb)


@router.callback_query(lambda c: c.data in ("set_lang_ru", "set_lang_en"))
async def cb_set_language(call: types.CallbackQuery):
    lang = call.data.replace("set_lang_", "")
    set_user_language(call.from_user.id, lang)
    await call.message.delete()
    await call.answer()
    await _run_onboarding(call, lang)


@router.callback_query(lambda c: c.data and c.data.startswith("enter_chimera_"))
async def cb_enter_chimera(call: types.CallbackQuery):
    lang = call.data.replace("enter_chimera_", "")
    await call.message.delete()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_matches)
    await call.message.answer(
        t("enter_msg", lang),
        parse_mode="HTML",
        reply_markup=build_main_keyboard(lang),
    )
    await call.answer()


@router.callback_query(lambda c: c.data == "set_bankroll")
async def set_bankroll_handler(call: types.CallbackQuery):
    await call.answer()
    current = get_user_bankroll(call.from_user.id)
    current_line = f"\nТекущий банк: <b>{current:.0f}</b>" if current else ""
    _awaiting_bankroll.add(call.from_user.id)
    await call.message.answer(
        f"💼 <b>Укажи размер своего банка</b>{current_line}\n\n"
        f"Введи сумму цифрами (например: <code>1000</code>)\n"
        f"Бот пересчитает прибыль в реальных деньгах на основе своего трекрекорда.\n\n"
        f"<i>Отправь 0 чтобы сбросить.</i>",
        parse_mode="HTML"
    )


@router.message(lambda m: m.from_user.id in _awaiting_bankroll)
async def bankroll_input_handler(message: types.Message):
    _awaiting_bankroll.discard(message.from_user.id)
    text = message.text.strip().replace(",", ".").replace(" ", "")
    try:
        amount = float(text)
        if amount < 0:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Неверный формат. Введи число, например: <code>1000</code>", parse_mode="HTML")
        return

    if amount == 0:
        set_user_bankroll(message.from_user.id, None)
        await message.answer("✅ Банк сброшен.", parse_mode="HTML")
        return

    set_user_bankroll(message.from_user.id, amount)
    pl = get_pl_stats(days=30)
    if pl["total"] > 0 and pl["profit_units"] != 0:
        profit_money = round(amount * pl["profit_units"] / 100, 2)
        sign = "+" if profit_money >= 0 else ""
        result_line = f"\nЕсли бы ты следовал боту 30 дней: <b>{sign}{profit_money:.0f}</b> ({sign}{pl['roi']}% ROI)"
    else:
        result_line = "\nДанные трекрекорда появятся после первых завершённых матчей."
    await message.answer(f"✅ Банк установлен: <b>{amount:.0f}</b>{result_line}", parse_mode="HTML")


@router.callback_query(lambda c: c.data == "noop")
async def cb_noop(call: types.CallbackQuery):
    await call.answer("Ставка уже записана ✅", show_alert=False)


@router.callback_query(lambda c: c.data == "chimera_ask")
async def chimera_ask_handler(call: types.CallbackQuery):
    from datetime import date as _date
    today = str(_date.today())
    _d = _chimera_daily.get(call.from_user.id, ("", 0))
    questions_used = _d[1] if _d[0] == today else 0

    if questions_used >= CHIMERA_DAILY_LIMIT and call.from_user.id not in ADMIN_IDS:
        await call.answer("Сегодня лимит исчерпан. Химера отдыхает до завтра.", show_alert=True)
        return

    if len(_chimera_waiting) > 500:
        _chimera_waiting.clear()
    _chimera_waiting.add(call.from_user.id)
    await call.answer()
    await call.message.answer(
        "🐉 <b>Химера слушает...</b>\n\nЗадай свой вопрос. Я отвечу.",
        parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data and c.data.startswith("mybet_"))
async def mybet_handler(call: types.CallbackQuery):
    parts = call.data.split("_")
    if len(parts) < 3:
        await call.answer("Ошибка: неверный формат.", show_alert=True)
        return
    sport = parts[1]
    try:
        pred_id = int(parts[2])
        odds  = int(parts[3]) / 100.0 if len(parts) >= 4 else 0.0
        units = int(parts[4]) if len(parts) >= 5 else 1
    except (ValueError, IndexError):
        await call.answer("Ошибка: неверный ID.", show_alert=True)
        return

    saved = mark_user_bet(call.from_user.id, sport, pred_id, odds, units)
    if saved:
        await call.answer("✅ Записано в твою статистику! Результат добавится автоматически после матча.", show_alert=True)
        try:
            orig_markup = call.message.reply_markup
            if orig_markup:
                new_rows = []
                for row in orig_markup.inline_keyboard:
                    new_row = [btn for btn in row if not btn.callback_data or not btn.callback_data.startswith("mybet_")]
                    if new_row:
                        new_rows.append(new_row)
                new_markup = types.InlineKeyboardMarkup(inline_keyboard=new_rows) if new_rows else None
                await call.message.edit_reply_markup(reply_markup=new_markup)
        except Exception as _e:
            logger.debug(f"[ignore] {_e}")
    else:
        await call.answer("Ты уже записал эту ставку.", show_alert=True)
