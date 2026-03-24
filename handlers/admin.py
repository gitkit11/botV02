# -*- coding: utf-8 -*-
"""handlers/admin.py — /ping, /resetmybets, /admin + callbacks"""
import asyncio
import html as _html_mod
import logging
import time

from aiogram import Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from state import ADMIN_IDS, _error_log, _bot_start_time
from circuit_breaker import all_statuses as cb_all_statuses
from database import (
    get_admin_stats, get_stavit_bets, set_manual_result,
    reset_user_bets, get_pending_stavit,
)

logger = logging.getLogger(__name__)
router = Router()


def _breakers_time(name: str) -> int:
    from circuit_breaker import _breakers
    b = _breakers.get(name)
    return b.time_until_retry() if b else 0


async def _send_admin_panel(target: types.Message, user_id: int):
    """Формирует и отправляет панель /admin. target — куда слать, user_id — кто запросил."""
    try:
        s = get_admin_stats()
        uptime_sec = int(time.time() - _bot_start_time)
        d, rem = divmod(uptime_sec, 86400)
        h, rem = divmod(rem, 3600)
        m = rem // 60
        uptime_str = (f"{d}д " if d else "") + f"{h}ч {m}м"

        lines = [
            "👁 <b>CHIMERA ADMIN</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"⏱ Аптайм: <b>{uptime_str}</b>",
            f"👥 Всего пользователей: <b>{s['total_users']}</b>",
            f"🟢 Активны сегодня: <b>{s['active_today']}</b>",
            f"📅 Активны за неделю: <b>{s['active_week']}</b>",
            f"🆕 Новых сегодня: <b>{s['new_today']}</b>  за неделю: <b>{s['new_week']}</b>",
            "",
            "🔥 <b>Популярные разделы (7 дней):</b>",
        ]
        for p in s["popular"]:
            lines.append(f"  • {p['action']} — {p['cnt']} раз")
        lines.append("")
        lines.append("🏆 <b>Топ пользователей:</b>")
        for u in s["top_users"]:
            name = u.get("username") and f"@{u['username']}" or u.get("first_name") or str(u["user_id"])
            lines.append(f"  • {name} — {u['analyses_total']} анализов")

        cb_stats = cb_all_statuses()
        if cb_stats:
            lines.append("")
            lines.append("🔌 <b>API Circuit Breakers:</b>")
            for name, (status, emoji) in cb_stats.items():
                lines.append(f"  {emoji} {name}: {status}")

        recent_errors = list(_error_log)[-5:]
        if recent_errors:
            lines.append("")
            lines.append(f"⚠️ <b>Последние ошибки ({len(_error_log)}):</b>")
            for err in recent_errors:
                icon = "🔴" if err["level"] in ("ERROR", "CRITICAL") else "🟡"
                lines.append(f"  {icon} {err['ts']} {err['msg'][:80]}")

        lines.append("")
        lines.append("⏱ <b>Последние действия:</b>")
        for a in s["last_actions"][:10]:
            ts = a["ts"][11:16]
            uname  = a.get("username") and f"@{a['username']}" or a.get("first_name") or str(a["user_id"])
            action = _html_mod.escape(str(a.get("action", "")))
            uname  = _html_mod.escape(str(uname))
            lines.append(f"  {ts} {uname}: {action}")

        lines.append("")
        lines.append("🎯 <b>СТАВИТЬ — все виды спорта:</b>")
        sport_labels = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀", "hockey": "🏒"}
        total_w = total_l = total_p = 0
        for sp, emoji in sport_labels.items():
            d = get_stavit_bets(sp, limit=0, offset=0)
            if d["total"] == 0:
                continue
            acc_e   = "🟢" if d["accuracy"] >= 60 else ("🟡" if d["accuracy"] >= 50 else "🔴")
            checked = d["wins"] + d["losses"]
            total_w += d["wins"]; total_l += d["losses"]; total_p += d["pending"]
            if checked > 0:
                lines.append(f"  {emoji} {acc_e} {d['accuracy']}% ({d['wins']}/{checked}) ⏳{d['pending']}")
            else:
                lines.append(f"  {emoji} ⏳ Ждём {d['pending']} результатов")
        total_checked = total_w + total_l
        if total_checked > 0:
            total_acc = round(total_w / total_checked * 100, 1)
            acc_e = "🟢" if total_acc >= 60 else ("🟡" if total_acc >= 50 else "🔴")
            lines.append(f"  {acc_e} Итого: <b>{total_acc}%</b> ({total_w}/{total_checked}) ⏳{total_p}")

        admin_kb = InlineKeyboardBuilder()
        admin_kb.button(text="📋 Ставить ⚽", callback_data="admin_bets_page_0")
        admin_kb.button(text="📋 Ставить 🎮", callback_data="admin_bets_cs2_0")
        admin_kb.button(text="📋 Ставить 🏀", callback_data="admin_bets_basketball_0")
        admin_kb.button(text="📋 Ставить 🎾", callback_data="admin_bets_tennis_0")
        admin_kb.button(text="✏️ Ввести результат", callback_data="admin_manual_result")
        admin_kb.button(text="🔄 Обновить", callback_data="admin_refresh")
        admin_kb.adjust(2)
        text_out = "\n".join(lines)
        if len(text_out) > 4000:
            text_out = text_out[:4000] + "\n..."
        await target.answer(text_out, parse_mode="HTML", reply_markup=admin_kb.as_markup())
    except Exception as e:
        logger.error(f"[admin] Ошибка: {e}", exc_info=True)
        admin_kb = InlineKeyboardBuilder()
        admin_kb.button(text="📋 Ставить ⚽", callback_data="admin_bets_page_0")
        admin_kb.button(text="📋 Ставить 🏀", callback_data="admin_bets_basketball_0")
        admin_kb.button(text="✏️ Ввести результат", callback_data="admin_manual_result")
        admin_kb.adjust(2)
        await target.answer(f"⚠️ Ошибка: {_html_mod.escape(str(e))}", parse_mode="HTML", reply_markup=admin_kb.as_markup())


@router.message(Command("ping"))
async def cmd_ping(message: types.Message):
    uptime_sec = int(time.time() - _bot_start_time)
    d, rem = divmod(uptime_sec, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    parts = []
    if d: parts.append(f"{d}д")
    if h: parts.append(f"{h}ч")
    parts.append(f"{m}м")
    uptime_str = " ".join(parts) or "< 1м"

    cb_stats = cb_all_statuses()
    api_lines = ""
    if cb_stats:
        api_lines = "\n\n<b>🔌 API статусы:</b>\n" + "\n".join(
            f"  {emoji} {name}: {status}" + (
                f" (retry через {_breakers_time(name)}с)" if status == "open" else ""
            )
            for name, (status, emoji) in cb_stats.items()
        )

    err_count = len(_error_log)
    await message.answer(
        f"🟢 <b>Chimera AI работает</b>\n"
        f"⏱ Аптайм: <b>{uptime_str}</b>\n"
        f"⚠️ Предупреждений в памяти: <b>{err_count}</b>"
        + api_lines,
        parse_mode="HTML"
    )


@router.message(Command("resetmybets"))
async def cmd_reset_my_bets(message: types.Message):
    uid = message.from_user.id
    deleted = reset_user_bets(uid)
    await message.answer(
        f"✅ Твои личные ставки сброшены ({deleted} записей удалено).\n"
        f"Банкролл и счётчик аналитик сохранены.",
        parse_mode="HTML"
    )


@router.message(Command("admin"))
async def cmd_admin(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    await _send_admin_panel(message, message.from_user.id)


@router.callback_query(lambda c: c.data and c.data.startswith("admin_bets_page_"))
async def cb_admin_bets(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    page     = int(call.data.split("_")[-1])
    per_page = 8
    offset   = page * per_page

    data    = get_stavit_bets("football", limit=per_page, offset=offset)
    total   = data["total"]
    checked = data["wins"] + data["losses"]
    acc     = data["accuracy"]
    roi     = data["roi"]

    acc_icon = "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")
    roi_icon = "🟢" if roi > 0 else "🔴"

    lines = [
        "📋 <b>СТАВИТЬ-сигналы (Футбол)</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Всего: <b>{total}</b> | Проверено: <b>{checked}</b> | ⏳ <b>{data['pending']}</b>",
    ]
    if checked > 0:
        lines.append(f"{acc_icon} Точность: <b>{acc}%</b> ({data['wins']}/{checked})")
        lines.append(f"{roi_icon} ROI: <b>{roi:+.2f}</b> ед.")
    lines.append("")

    for r in data["rows"]:
        icon    = "✅" if r["is_correct"] == 1 else ("❌" if r["is_correct"] == 0 else "⏳")
        outcome = r.get("recommended_outcome") or "?"
        date    = (r.get("match_date") or r.get("created_at") or "")[:10]
        hs  = r.get("real_home_score")
        as_ = r.get("real_away_score")
        score_str = f" {hs}:{as_}" if hs is not None else ""
        if "home" in str(outcome).lower() or outcome == "home_win":
            odds = r.get("bookmaker_odds_home") or "?"
        elif "away" in str(outcome).lower() or outcome == "away_win":
            odds = r.get("bookmaker_odds_away") or "?"
        else:
            odds = r.get("bookmaker_odds_draw") or "?"
        lines.append(
            f"{icon} <b>{r['home_team']}</b> vs <b>{r['away_team']}</b>"
            f"{score_str} @{odds} <i>({date})</i>"
        )

    text = "\n".join(lines)
    kb = InlineKeyboardBuilder()
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > 0:
        kb.button(text="◀️", callback_data=f"admin_bets_page_{page-1}")
    kb.button(text=f"{page+1}/{total_pages}", callback_data="noop")
    if offset + per_page < total:
        kb.button(text="▶️", callback_data=f"admin_bets_page_{page+1}")
    kb.adjust(3)
    kb.row()
    kb.button(text="🎮 CS2", callback_data="admin_bets_cs2_0")
    kb.button(text="🔙 Назад", callback_data="admin_back")
    kb.adjust(2)

    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[admin_bets] {_e}")
    await call.answer()


@router.callback_query(lambda c: c.data and c.data.startswith("admin_bets_cs2_"))
async def cb_admin_bets_cs2(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    page     = int(call.data.split("_")[-1])
    per_page = 8
    offset   = page * per_page

    data    = get_stavit_bets("cs2", limit=per_page, offset=offset)
    total   = data["total"]
    checked = data["wins"] + data["losses"]
    acc     = data["accuracy"]
    acc_icon = "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")

    lines = [
        "📋 <b>СТАВИТЬ-сигналы (CS2)</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Всего: <b>{total}</b> | Проверено: <b>{checked}</b> | ⏳ <b>{data['pending']}</b>",
    ]
    if checked > 0:
        lines.append(f"{acc_icon} Точность: <b>{acc}%</b> ({data['wins']}/{checked})")
    lines.append("")

    for r in data["rows"]:
        icon  = "✅" if r["is_correct"] == 1 else ("❌" if r["is_correct"] == 0 else "⏳")
        date  = (r.get("match_date") or r.get("created_at") or "")[:10]
        hs    = r.get("real_home_score")
        as_   = r.get("real_away_score")
        score_str = f" {hs}:{as_}" if hs is not None else ""
        odds  = r.get("bookmaker_odds_home") or "?"
        lines.append(f"{icon} <b>{r['home_team']}</b> vs <b>{r['away_team']}</b>{score_str} @{odds} <i>({date})</i>")

    text = "\n".join(lines)
    kb = InlineKeyboardBuilder()
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > 0:
        kb.button(text="◀️", callback_data=f"admin_bets_cs2_{page-1}")
    kb.button(text=f"{page+1}/{total_pages}", callback_data="noop")
    if offset + per_page < total:
        kb.button(text="▶️", callback_data=f"admin_bets_cs2_{page+1}")
    kb.adjust(3)
    kb.row()
    kb.button(text="⚽ Футбол", callback_data="admin_bets_page_0")
    kb.button(text="🔙 Назад", callback_data="admin_back")
    kb.adjust(2)

    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[admin_bets_cs2] {_e}")
    await call.answer()


@router.callback_query(lambda c: c.data and (
    c.data.startswith("admin_bets_basketball_") or
    c.data.startswith("admin_bets_tennis_") or
    c.data.startswith("admin_bets_hockey_")
))
async def cb_admin_bets_sport(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)

    # Определяем вид спорта и страницу из callback_data
    # Формат: admin_bets_{sport}_{page}
    parts  = call.data.split("_")
    sport  = parts[2]   # basketball / tennis / hockey
    page   = int(parts[3])
    labels = {"basketball": "🏀 Баскетбол", "tennis": "🎾 Теннис", "hockey": "🏒 Хоккей"}
    label  = labels.get(sport, sport)

    per_page = 8
    offset   = page * per_page

    data    = get_stavit_bets(sport, limit=per_page, offset=offset)
    total   = data["total"]
    checked = data["wins"] + data["losses"]
    acc     = data["accuracy"]
    roi     = data["roi"]
    acc_icon = "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")
    roi_icon = "🟢" if roi > 0 else "🔴"

    lines = [
        f"📋 <b>СТАВИТЬ-сигналы ({label})</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Всего: <b>{total}</b> | Проверено: <b>{checked}</b> | ⏳ <b>{data['pending']}</b>",
    ]
    if checked > 0:
        lines.append(f"{acc_icon} Точность: <b>{acc}%</b> ({data['wins']}/{checked})")
        lines.append(f"{roi_icon} ROI: <b>{roi:+.2f}</b> ед.")
    lines.append("")

    for r in data["rows"]:
        icon    = "✅" if r["is_correct"] == 1 else ("❌" if r["is_correct"] == 0 else "⏳")
        outcome = r.get("recommended_outcome") or "?"
        date    = (r.get("match_date") or r.get("created_at") or "")[:10]
        if "home" in str(outcome).lower():
            odds = r.get("bookmaker_odds_home") or "?"
        else:
            odds = r.get("bookmaker_odds_away") or "?"
        home = _html_mod.escape(str(r.get("home_team") or "?")[:18])
        away = _html_mod.escape(str(r.get("away_team") or "?")[:18])
        lines.append(f"{icon} <b>{home}</b> vs <b>{away}</b> @{odds} <i>({date})</i>")

    if not data["rows"] and total == 0:
        lines.append("<i>Пока нет сигналов СТАВИТЬ</i>")

    text = "\n".join(lines)
    kb = InlineKeyboardBuilder()
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > 0:
        kb.button(text="◀️", callback_data=f"admin_bets_{sport}_{page-1}")
    kb.button(text=f"{page+1}/{total_pages}", callback_data="noop")
    if offset + per_page < total:
        kb.button(text="▶️", callback_data=f"admin_bets_{sport}_{page+1}")
    kb.adjust(3)
    kb.row()
    kb.button(text="⚽ Футбол", callback_data="admin_bets_page_0")
    kb.button(text="🎮 CS2",   callback_data="admin_bets_cs2_0")
    kb.button(text="🔙 Назад", callback_data="admin_back")
    kb.adjust(3)

    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[admin_bets_{sport}] {_e}")
    await call.answer()


@router.callback_query(lambda c: c.data == "admin_refresh")
async def cb_admin_refresh(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    await call.answer("🔄 Обновлено", show_alert=False)
    await _send_admin_panel(call.message, call.from_user.id)


@router.callback_query(lambda c: c.data == "admin_back")
async def cb_admin_back(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    await call.answer()
    await _send_admin_panel(call.message, call.from_user.id)


@router.callback_query(lambda c: c.data == "admin_manual_result")
async def cb_admin_manual_result(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    await call.answer()
    loop = asyncio.get_running_loop()
    pending = await loop.run_in_executor(None, get_pending_stavit, 20)
    if not pending:
        await call.message.edit_text(
            "✅ Нет pending матчей — все результаты уже записаны.",
            reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[[
                types.InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")
            ]])
        )
        return

    sport_icons = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀", "hockey": "🏒"}
    kb = InlineKeyboardBuilder()
    lines = ["<b>✏️ Выбери матч для записи результата:</b>\n"]
    for i, m in enumerate(pending):
        sp   = m.get("sport", "football")
        icon = sport_icons.get(sp, "🏅")
        home = _html_mod.escape(m.get("home_team", "?")[:15])
        away = _html_mod.escape(m.get("away_team", "?")[:15])
        rec  = _html_mod.escape(m.get("recommended_outcome", "?")[:12])
        date = (m.get("match_date") or m.get("created_at") or "")[:10]
        lines.append(f"{i+1}. {icon} <b>{home} vs {away}</b> [{rec}] {date}")
        kb.button(
            text=f"{i+1}. {icon} {home[:10]} vs {away[:10]}",
            callback_data=f"mr_pick_{sp}_{m['match_id'][:30]}"
        )
    kb.button(text="◀️ Назад", callback_data="admin_back")
    kb.adjust(1)
    text = "\n".join(lines)
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[manual_result] {_e}")


@router.callback_query(lambda c: c.data and c.data.startswith("mr_pick_"))
async def cb_mr_pick(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    await call.answer()
    # mr_pick_{sport}_{match_id}  (match_id может содержать '_')
    parts    = call.data[len("mr_pick_"):].split("_", 1)
    sport    = parts[0]
    match_id = parts[1] if len(parts) > 1 else ""

    loop = asyncio.get_running_loop()
    pending    = await loop.run_in_executor(None, get_pending_stavit, 20)
    match_info = next((m for m in pending if m.get("match_id", "")[:30] == match_id and m.get("sport") == sport), None)

    home = _html_mod.escape((match_info.get("home_team", "?") if match_info else "?")[:25])
    away = _html_mod.escape((match_info.get("away_team", "?") if match_info else "?")[:25])
    rec  = _html_mod.escape((match_info.get("recommended_outcome", "") if match_info else "")[:20])

    sport_icons = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀", "hockey": "🏒"}
    icon = sport_icons.get(sport, "🏅")

    text = (
        f"{icon} <b>{home} vs {away}</b>\n"
        f"Рекомендация бота: <b>{rec}</b>\n\n"
        "Выбери результат:"
    )

    kb = InlineKeyboardBuilder()
    base = f"mr_set_{sport}_{match_id}"
    if sport == "football":
        kb.button(text="🏠 Победа хозяев", callback_data=f"{base}_home_win")
        kb.button(text="🤝 Ничья",          callback_data=f"{base}_draw")
        kb.button(text="✈️ Победа гостей",  callback_data=f"{base}_away_win")
    elif sport == "cs2":
        kb.button(text="🏠 Победа Team1",   callback_data=f"{base}_home_win")
        kb.button(text="✈️ Победа Team2",   callback_data=f"{base}_away_win")
    elif sport == "tennis":
        kb.button(text="🎾 Победа P1",      callback_data=f"{base}_home_win")
        kb.button(text="🎾 Победа P2",      callback_data=f"{base}_away_win")
    elif sport == "basketball":
        kb.button(text="🏀 Победа хозяев",  callback_data=f"{base}_home_win")
        kb.button(text="🏀 Победа гостей",  callback_data=f"{base}_away_win")
    elif sport == "hockey":
        kb.button(text="🏒 Победа хозяев",  callback_data=f"{base}_home_win")
        kb.button(text="🏒 Победа гостей",  callback_data=f"{base}_away_win")
    kb.button(text="◀️ Назад", callback_data="admin_manual_result")
    kb.adjust(1)
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[mr_pick] {_e}")


@router.callback_query(lambda c: c.data and c.data.startswith("mr_set_"))
async def cb_mr_set(call: types.CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return await call.answer("Нет доступа", show_alert=True)
    await call.answer()
    # mr_set_{sport}_{match_id}_{outcome}
    raw          = call.data[len("mr_set_"):]
    sport, rest  = raw.split("_", 1)
    # outcome — суффикс: home_win / draw / away_win
    outcome = "unknown"
    match_id = rest
    for o in ("home_win", "away_win", "draw"):
        if rest.endswith("_" + o):
            outcome  = o
            match_id = rest[:-(len(o) + 1)]
            break

    loop = asyncio.get_running_loop()
    pending    = await loop.run_in_executor(None, get_pending_stavit, 20)
    match_info = next((m for m in pending if m.get("match_id", "")[:30] == match_id and m.get("sport") == sport), None)
    rec        = (match_info.get("recommended_outcome", "") if match_info else "").lower().strip()

    outcome_aliases = {
        "home_win": ["home_win", "home", "п1", "1", "победа хозяев"],
        "draw":     ["draw", "ничья", "x"],
        "away_win": ["away_win", "away", "п2", "2", "победа гостей"],
    }
    is_correct = 0
    for key, aliases in outcome_aliases.items():
        if key == outcome and (rec in aliases or rec == outcome):
            is_correct = 1
            break

    await loop.run_in_executor(None, set_manual_result, sport, match_id, outcome, is_correct)

    sport_icons = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀", "hockey": "🏒"}
    icon    = sport_icons.get(sport, "🏅")
    verdict = "✅ Верно!" if is_correct else "❌ Неверно"
    home    = _html_mod.escape((match_info.get("home_team", "?") if match_info else "?")[:20])
    away    = _html_mod.escape((match_info.get("away_team", "?") if match_info else "?")[:20])
    outcome_labels = {"home_win": "Победа хозяев", "draw": "Ничья", "away_win": "Победа гостей"}

    text = (
        f"{icon} <b>{home} vs {away}</b>\n"
        f"Результат: <b>{outcome_labels.get(outcome, outcome)}</b>\n"
        f"Рекомендация бота: <b>{rec or '—'}</b>\n"
        f"Итог: {verdict}"
    )
    kb = InlineKeyboardBuilder()
    kb.button(text="✏️ Ещё матч", callback_data="admin_manual_result")
    kb.button(text="🏠 Меню",     callback_data="admin_back")
    kb.adjust(2)
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb.as_markup())
    except Exception as _e:
        logger.debug(f"[mr_set] {_e}")
