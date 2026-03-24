# -*- coding: utf-8 -*-
"""handlers/tennis.py — команда /tennis и tennis callbacks"""
import asyncio
import logging

from aiogram import Router, types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from state import tennis_matches_cache, _report_cache, _REPORT_CACHE_TTL
from handlers.common import show_ai_thinking
from formatters import _safe_truncate

logger = logging.getLogger(__name__)
router = Router()


async def cmd_tennis(message: types.Message):
    """Раздел тенниса — список турниров."""
    status = await message.answer("🎾 Загружаю теннисные турниры...", parse_mode="HTML")
    try:
        from sports.tennis.matches import get_tennis_matches
        from sports.tennis.rankings import detect_surface, detect_tour

        raw_matches = get_tennis_matches()
        all_matches = [m for m in raw_matches if m.get("bookmakers_count", 1) >= 2]
        logger.info(f"[Теннис] {len(raw_matches)} матчей получено, {len(all_matches)} с ≥2 букмекерами")

        if not all_matches:
            await status.edit_text(
                "🎾 <b>Теннис</b>\n\nСейчас нет матчей.\n"
                "<i>Попробуйте позже или в дни крупных турниров.</i>",
                parse_mode="HTML"
            )
            return

        tennis_matches_cache.clear()
        tennis_matches_cache.extend(all_matches)

        tournaments: dict = {}
        for m in all_matches:
            sk = m.get("sport_key", "tennis_atp_other")
            if sk not in tournaments:
                tournaments[sk] = []
            tournaments[sk].append(m)

        surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}

        builder = InlineKeyboardBuilder()
        for sk, t_matches in sorted(tournaments.items()):
            surface = t_matches[0].get("surface") or detect_surface(sk)
            tour    = t_matches[0].get("tour")    or detect_tour(sk)
            icon    = surf_icons.get(surface, "🎾")
            t_name  = t_matches[0].get("tournament") or sk.replace(f"tennis_{tour}_", "").replace("_", " ").title()
            label   = f"{icon} {tour.upper()} | {t_name} ({len(t_matches)})"
            builder.button(text=label, callback_data=f"tennis_tour_{sk}")
        builder.button(text="🏠 Главное меню", callback_data="back_to_main")
        builder.adjust(1)

        await status.edit_text(
            f"🎾 <b>ТЕННИС</b>\n"
            f"Турниров: <b>{len(tournaments)}</b> | Матчей: <b>{len(all_matches)}</b>\n\n"
            f"Выберите турнир:",
            parse_mode="HTML",
            reply_markup=builder.as_markup()
        )
    except Exception as e:
        logger.error(f"[Теннис] cmd_tennis: {e}", exc_info=True)
        await status.edit_text(f"🎾 <b>Теннис</b>\n\n⚠️ Ошибка: {str(e)[:120]}", parse_mode="HTML")


@router.callback_query(lambda c: c.data and c.data.startswith("tennis_tour_"))
async def tennis_select_tour(call: types.CallbackQuery):
    sport_key = call.data[len("tennis_tour_"):]
    tour_matches = [m for m in tennis_matches_cache if m.get("sport_key") == sport_key]
    if not tour_matches:
        await call.answer("Матчи не найдены. Обновите список.", show_alert=True)
        return
    from sports.tennis.rankings import detect_surface, detect_tour
    surface    = detect_surface(sport_key)
    tour       = detect_tour(sport_key)
    surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}
    icon       = surf_icons.get(surface, "🎾")
    name       = sport_key.replace(f"tennis_{tour}_", "").replace("_", " ").title()
    builder    = InlineKeyboardBuilder()
    for idx, m in enumerate(tour_matches):
        global_idx = tennis_matches_cache.index(m)
        label = f"{icon} {m['player1']} vs {m['player2']} | {m['odds_p1']} / {m['odds_p2']}"
        builder.button(text=label, callback_data=f"tennis_m_{global_idx}")
    builder.button(text="⬅️ Назад к турнирам", callback_data="back_to_tennis")
    builder.adjust(1)
    await call.answer()
    await call.message.edit_text(
        f"{icon} <b>{tour.upper()} {name}</b>\n\nВыберите матч для полного AI-анализа:",
        parse_mode="HTML", reply_markup=builder.as_markup()
    )


@router.callback_query(lambda c: c.data == "back_to_tennis")
async def back_to_tennis(call: types.CallbackQuery):
    await call.answer()
    await cmd_tennis(call.message)


@router.callback_query(lambda c: c.data and c.data.startswith("tennis_m_"))
async def tennis_analyze_match(call: types.CallbackQuery):
    try:
        match_idx = int(call.data.split("_")[2])
    except (IndexError, ValueError):
        await call.answer("⚠️ Некорректные данные.", show_alert=True)
        return
    if match_idx >= len(tennis_matches_cache):
        await call.answer("⚠️ Матч не найден. Список мог устареть — вернись назад и обнови.", show_alert=True)
        return

    m          = tennis_matches_cache[match_idx]
    p1, p2     = m["player1"], m["player2"]
    o1, o2     = m.get("odds_p1", 0.0), m.get("odds_p2", 0.0)
    sport_key  = m.get("sport_key", "")
    no_odds    = (o1 == 0.0 or o2 == 0.0)

    import time as _time_tn
    _tn_cache_key = f"tennis_{sport_key}_{match_idx}"
    _tn_cached    = _report_cache.get(_tn_cache_key)
    if _tn_cached and _time_tn.time() - _tn_cached.get("ts", 0) < _REPORT_CACHE_TTL:
        await call.answer()
        await call.message.edit_text(
            _tn_cached["text"], parse_mode=_tn_cached.get("parse_mode"),
            reply_markup=_tn_cached.get("kb"),
        )
        return

    if not no_odds:
        try:
            from line_movement import make_match_key, record_odds as _record_tn_odds
            _tn_lm_key = make_match_key(p1, p2, m.get("commence_time", ""))
            _record_tn_odds(_tn_lm_key, {"home_win": o1, "away_win": o2})
        except Exception as _e:
            logger.debug(f"[ignore] {_e}")

    no_odds_note = "\n<i>⚠️ Коэффициенты недоступны — только аналитика</i>" if no_odds else ""
    status_msg = await call.message.edit_text(
        f"⏳ <b>{p1} vs {p2}</b>{no_odds_note}", parse_mode="HTML"
    )
    await show_ai_thinking(status_msg, p1, p2, sport="tennis")

    try:
        from sports.tennis import analyze_tennis_match
        from sports.tennis.agents import run_tennis_gpt_agent, run_tennis_llama_agent, format_tennis_full_report
        from sports.tennis.rankings import detect_surface, detect_tour

        surface = detect_surface(sport_key)
        tour    = detect_tour(sport_key)

        result = analyze_tennis_match(p1, p2, o1, o2, sport_key=sport_key)
        probs  = result["probs"]
        cands  = result["candidates"]

        _bm_total_line  = m.get("bm_total_line", 0.0)
        _bm_total_over  = m.get("bm_total_over", 0.0)
        _bm_total_under = m.get("bm_total_under", 0.0)
        from sports.tennis.model import predict_tennis_game_totals
        from sports.tennis.rankings import detect_tour as _detect_tour
        _tour    = _detect_tour(sport_key)
        _best_of = 5 if "grand_slam" in sport_key.lower() else 3
        _game_totals = predict_tennis_game_totals(
            p1_win=probs["p1_win"], p2_win=probs["p2_win"],
            p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
            surface=surface, tour=_tour, best_of=_best_of,
            bm_total_line=_bm_total_line,
            bm_total_over=_bm_total_over,
            bm_total_under=_bm_total_under,
        )

        gpt_text = run_tennis_gpt_agent(
            p1, p2, probs, o1, o2, surface, tour,
            p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
            game_totals=_game_totals,
        )
        llama_text = run_tennis_llama_agent(
            p1, p2, probs, o1, o2, surface, tour,
            p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
            gpt_verdict=gpt_text,
            game_totals=_game_totals,
        )

        from sports.tennis.agents import run_tennis_chimera_agents
        chimera_data = run_tennis_chimera_agents(
            p1, p2, probs, o1, o2, surface, tour,
            p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
            gpt_text=gpt_text, llama_text=llama_text,
        )
        chimera_block = chimera_data.get("verdict_block", "")

        try:
            from expert_oracle import get_expert_consensus, format_expert_block
            _loop_tn = asyncio.get_running_loop()
            _exp_tn  = await _loop_tn.run_in_executor(None, get_expert_consensus, p1, p2, "tennis")
            _expert_block_tn = format_expert_block(_exp_tn, p1, p2)
            if _expert_block_tn:
                chimera_block = (chimera_block + "\n\n" + _expert_block_tn).strip()
        except Exception as _ee_tn:
            logger.debug(f"[ExpertOracle Tennis] {_ee_tn}")

        _chimera_probs = chimera_data.get("final_probs", {})
        if _chimera_probs.get("home_win") and _chimera_probs.get("away_win"):
            probs = dict(probs)
            probs["p1_win"] = round(_chimera_probs["home_win"], 4)
            probs["p2_win"] = round(_chimera_probs["away_win"], 4)

        report = format_tennis_full_report(
            p1, p2, probs, o1, o2, surface, tour,
            gpt_text, llama_text, cands, sport_key=sport_key,
            chimera_verdict_block=chimera_block,
            commence_time=m.get("commence_time"),
        )

        _tn_gpt_verdict   = "home_win" if probs["p1_win"] >= 0.5 else "away_win"
        _tn_llama_verdict = ""
        if llama_text and not llama_text.startswith("❌"):
            _p1_last = p1.split()[-1].lower()
            _p2_last = p2.split()[-1].lower()
            _ll_low  = llama_text.lower()
            if _p1_last in _ll_low and _p2_last not in _ll_low:
                _tn_llama_verdict = "home_win"
            elif _p2_last in _ll_low and _p1_last not in _ll_low:
                _tn_llama_verdict = "away_win"

        try:
            from database import save_prediction
            best = cands[0] if cands else None
            rec  = "home_win" if (best and best["outcome"] == "P1") else "away_win"
            if best:
                from signal_engine import get_bet_tier as _tn_get_tier
                _tennis_signal = _tn_get_tier(best["prob"] / 100, best["ev"], "tennis")
            else:
                _tennis_signal = "НЕ СТАВИТЬ"
            _tennis_pred_id = save_prediction(
                sport="tennis",
                match_id=m.get("event_id", f"{p1}_{p2}"),
                match_date=m.get("commence_time", ""),
                home_team=p1, away_team=p2,
                league=sport_key,
                gpt_verdict=_tn_gpt_verdict,
                llama_verdict=_tn_llama_verdict,
                recommended_outcome=rec,
                bet_signal=_tennis_signal,
                ensemble_home=probs["p1_win"],
                ensemble_away=probs["p2_win"],
                ensemble_best_outcome=rec,
                bookmaker_odds_home=o1,
                bookmaker_odds_away=o2,
            )
        except Exception as save_err:
            _tennis_pred_id = None
            logger.warning(f"[Tennis Save] {save_err}")

        try:
            from database import upsert_user, track_analysis, log_action
            upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
            track_analysis(call.from_user.id, "tennis")
            log_action(call.from_user.id, "анализ Теннис")
        except Exception as _e:
            logger.debug(f"[ignore] {_e}")

        tennis_kb = InlineKeyboardBuilder()
        tennis_kb.button(text="🎾 Победитель",       callback_data=f"tennis_mkt_winner_{match_idx}")
        tennis_kb.button(text="📊 Тотал геймов",     callback_data=f"tennis_mkt_games_{match_idx}")
        tennis_kb.button(text="🏅 Победа в 1-м сете", callback_data=f"tennis_mkt_set1_{match_idx}")
        tennis_kb.button(text="⬅️ Матчи",             callback_data=f"tennis_tour_{sport_key}")
        tennis_kb.button(text="🏠 Меню",              callback_data="back_to_main")
        if _tennis_signal.startswith("СТАВИТЬ") and _tennis_pred_id:
            _tn_bet_odds = o2 if rec == "away_win" else o1
            _tn_odds_enc = int(round((_tn_bet_odds or 0) * 100))
            _tn_kelly    = best["kelly"] if best else 2
            _tn_units    = 3 if _tn_kelly >= 4 else (2 if _tn_kelly >= 2 else 1)
            tennis_kb.button(
                text=f"✅ Я поставил {_tn_units}u — записать в статистику",
                callback_data=f"mybet_tennis_{_tennis_pred_id}_{_tn_odds_enc}_{_tn_units}"
            )
        tennis_kb.adjust(2)
        _tn_kb = tennis_kb.as_markup()
        report = _safe_truncate(report)
        await call.message.edit_text(report, parse_mode="HTML", reply_markup=_tn_kb)
        import time as _time
        _report_cache[f"tennis_{sport_key}_{match_idx}"] = {
            "text": report, "kb": _tn_kb,
            "parse_mode": "HTML", "ts": _time.time(),
        }

    except Exception as e:
        logger.error(f"[Tennis Match] Ошибка: {e}", exc_info=True)
        await call.message.edit_text(f"❌ Ошибка анализа тенниса: {str(e)[:150]}")


@router.callback_query(lambda c: c.data and c.data.startswith("tennis_mkt_"))
async def tennis_market(call: types.CallbackQuery):
    parts     = call.data.split("_")
    mkt_type  = parts[2]   # winner, games, set1
    match_idx = int(parts[3])
    if match_idx >= len(tennis_matches_cache):
        await call.answer("Матч не найден", show_alert=True)
        return

    m       = tennis_matches_cache[match_idx]
    home    = m.get("home_team", m.get("home", "Игрок A"))
    away    = m.get("away_team", m.get("away", "Игрок B"))
    odds    = m.get("bookmaker_odds", {}) or m.get("odds", {})
    h_odds  = float(odds.get("home_win", 1.75))
    a_odds  = float(odds.get("away_win", 2.1))
    h_pct   = round(100 / h_odds, 1) if h_odds > 1 else 57.1
    a_pct   = round(100 / a_odds, 1) if a_odds > 1 else 47.6
    sport_key = m.get("sport_key", "")

    if mkt_type == "winner":
        text = (
            f"🎾 <b>Победитель матча</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{home} vs {away}</b>\n\n"
            f"{'✅' if h_pct > a_pct else '▫️'} <b>{home}</b>\n"
            f"   Кэф: <b>{h_odds}</b> | Вероятность: <b>{h_pct}%</b>\n\n"
            f"{'✅' if a_pct > h_pct else '▫️'} <b>{away}</b>\n"
            f"   Кэф: <b>{a_odds}</b> | Вероятность: <b>{a_pct}%</b>\n\n"
            f"<i>💡 Рекомендация: {'<b>' + home + '</b>' if h_pct > a_pct else '<b>' + away + '</b>'}</i>"
        )
    elif mkt_type == "games":
        is_balanced = abs(h_pct - a_pct) < 10
        total_line  = 22.5 if is_balanced else 20.5
        text = (
            f"📊 <b>Тотал геймов в матче</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{home} vs {away}</b>\n\n"
            f"📏 Линия: <b>{total_line}</b> геймов\n\n"
            f"{'✅' if is_balanced else '▫️'} <b>Больше {total_line}</b>\n"
            f"   <i>{'Равные игроки — длинные розыгрыши' if is_balanced else ''}</i>\n"
            f"{'✅' if not is_balanced else '▫️'} <b>Меньше {total_line}</b>\n"
            f"   <i>{'Фаворит доминирует' if not is_balanced else ''}</i>\n\n"
            f"<i>Разрыв в классе: {abs(h_pct - a_pct):.0f}%</i>"
        )
    elif mkt_type == "set1":
        h_set      = round(min(max(h_pct * 0.9 + 5, 30), 70), 1)
        a_set      = round(100 - h_set, 1)
        h_set_odds = round(100 / h_set, 2) if h_set > 0 else 1.9
        a_set_odds = round(100 / a_set, 2) if a_set > 0 else 1.9
        text = (
            f"🏅 <b>Победа в 1-м сете</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{home} vs {away}</b>\n\n"
            f"{'✅' if h_set > a_set else '▫️'} <b>{home}</b>\n"
            f"   Вероятность: <b>{h_set}%</b> | Кэф: <b>{h_set_odds}</b>\n\n"
            f"{'✅' if a_set > h_set else '▫️'} <b>{away}</b>\n"
            f"   Вероятность: <b>{a_set}%</b> | Кэф: <b>{a_set_odds}</b>\n\n"
            f"<i>💡 Первый сет часто задаёт тон всему матчу</i>"
        )
    else:
        text = "⚠️ Неизвестный рынок"

    back_kb = InlineKeyboardBuilder()
    back_kb.button(
        text="🎾 Победитель" if mkt_type != "winner" else "📊 Тотал геймов",
        callback_data=f"tennis_mkt_{'winner' if mkt_type != 'winner' else 'games'}_{match_idx}"
    )
    back_kb.button(
        text="🏅 1-й сет" if mkt_type != "set1" else "🎾 Победитель",
        callback_data=f"tennis_mkt_{'set1' if mkt_type != 'set1' else 'winner'}_{match_idx}"
    )
    back_kb.button(text="↩️ К анализу", callback_data=f"back_to_report_tennis_{sport_key}_{match_idx}")
    back_kb.button(text="🏠 Меню", callback_data="back_to_main")
    back_kb.adjust(2)
    await call.answer()
    await call.message.edit_text(text, parse_mode="HTML", reply_markup=back_kb.as_markup())


@router.callback_query(lambda c: c.data and c.data.startswith("back_to_report_tennis_"))
async def back_to_tennis_report(call: types.CallbackQuery):
    """Возврат к кэшированному отчёту по матчу тенниса."""
    await call.answer()
    # format: back_to_report_tennis_{sport_key}_{match_idx}
    # sport_key may contain underscores, match_idx is always last
    parts     = call.data[len("back_to_report_tennis_"):].rsplit("_", 1)
    if len(parts) != 2:
        await call.message.edit_text("⚠️ Не удалось восстановить отчёт. Вернитесь к матчу.")
        return
    sport_key, idx_str = parts
    try:
        match_idx = int(idx_str)
    except ValueError:
        await call.message.edit_text("⚠️ Некорректный индекс матча.")
        return

    _tn_cache_key = f"tennis_{sport_key}_{match_idx}"
    _cached       = _report_cache.get(_tn_cache_key)
    if _cached:
        await call.message.edit_text(
            _cached["text"],
            parse_mode=_cached.get("parse_mode"),
            reply_markup=_cached.get("kb"),
        )
    else:
        # Кэш истёк — просто редиректим на повторный анализ
        await call.message.edit_text(
            "⏳ Кэш отчёта истёк. Открываю матч заново...", parse_mode="HTML"
        )
        fake_call = types.CallbackQuery(
            id=call.id, from_user=call.from_user,
            message=call.message, chat_instance=call.chat_instance,
            data=f"tennis_m_{match_idx}",
        )
        await tennis_analyze_match(fake_call)
