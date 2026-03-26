# -*- coding: utf-8 -*-
"""handlers/express.py — /express и express_ callbacks"""
import asyncio
import logging

from aiogram import Router, types

logger = logging.getLogger(__name__)
router = Router()

_EXPRESS_MENU_TEXT = (
    "🎯 <b>Chimera Express</b>\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Выбери тип экспресса:\n\n"
    "🟢 <b>Надёжный</b> — 2 события, высокая вероятность\n"
    "🟡 <b>Средний</b> — 3 события, баланс риска и кэфа\n"
    "🔴 <b>Рискованный</b> — 4-5 событий, высокий кэф"
)

_EXPRESS_MENU_KB = types.InlineKeyboardMarkup(inline_keyboard=[
    [types.InlineKeyboardButton(text="🟢 Надёжный (2 события)", callback_data="express_safe")],
    [types.InlineKeyboardButton(text="🟡 Средний (3 события)",  callback_data="express_medium")],
    [types.InlineKeyboardButton(text="🔴 Рискованный (4-5)",    callback_data="express_risky")],
])


async def cmd_express(message: types.Message):
    await message.answer(_EXPRESS_MENU_TEXT, parse_mode="HTML", reply_markup=_EXPRESS_MENU_KB)


@router.callback_query(lambda c: c.data and c.data.startswith("express_") and c.data != "express_back")
async def cb_express_variant(call: types.CallbackQuery):
    variant_key = call.data.replace("express_", "")
    titles = {
        "safe":   ("🟢 Надёжный экспресс",  "🟢"),
        "medium": ("🟡 Средний экспресс",    "🟡"),
        "risky":  ("🔴 Рискованный экспресс","🔴"),
    }
    title, emoji = titles.get(variant_key, ("🎯 Экспресс", "🎯"))

    await call.answer()
    status = await call.message.answer(
        f"{emoji} <b>{title}</b>\n\n⏳ Сканирую матчи...\n<i>10-20 секунд</i>",
        parse_mode="HTML"
    )
    loop = asyncio.get_running_loop()
    try:
        from express_builder import scan_all_matches, build_express_variants, format_express_card

        def _build():
            candidates = scan_all_matches()
            return build_express_variants(candidates)

        variants = await loop.run_in_executor(None, _build)
        variant  = variants.get(variant_key)

        if not variant:
            await status.edit_text(
                f"{emoji} <b>{title}</b>\n\n"
                "⚠️ Недостаточно качественных событий для этого варианта.\n"
                "<i>Попробуй другой тип или зайди позже.</i>",
                parse_mode="HTML"
            )
            return

        card = format_express_card(variant, title, emoji)
        back_kb = types.InlineKeyboardMarkup(inline_keyboard=[[
            types.InlineKeyboardButton(text="◀️ Другой вариант", callback_data="express_back")
        ]])
        await status.edit_text(card, parse_mode="HTML", reply_markup=back_kb)
    except Exception as e:
        logger.error(f"[Экспресс] Ошибка: {e}", exc_info=True)
        await status.edit_text("😔 Произошёл сбой. Напиши нам в поддержку.", parse_mode="HTML")


@router.callback_query(lambda c: c.data == "express_back")
async def cb_express_back(call: types.CallbackQuery):
    await call.answer()
    await call.message.edit_text(_EXPRESS_MENU_TEXT, parse_mode="HTML", reply_markup=_EXPRESS_MENU_KB)
