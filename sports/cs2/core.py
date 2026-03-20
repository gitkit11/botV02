# -*- coding: utf-8 -*-
from .veto_logic import simulate_bo3_veto, get_map_impact_score, get_team_player_stats
from .pandascore import (
    get_team_stats, get_head_to_head,
    get_team_weighted_form, classify_tournament, check_stand_in,
)
from .hltv_stats import get_team_map_stats
from .team_registry import CS2_ELO, DEFAULT_ELO, LAN_BONUS, normalize_team_name, get_elo
import json
import asyncio

# CS2_ELO, DEFAULT_ELO, LAN_BONUS, normalize_team_name — из team_registry
# _normalize_team_name — алиас для обратной совместимости внутри файла
_normalize_team_name = normalize_team_name
_LAN_BONUS = LAN_BONUS


def get_elo_prob(home_team, away_team):
    h_name = normalize_team_name(home_team)
    a_name = normalize_team_name(away_team)
    h_elo = CS2_ELO.get(h_name, DEFAULT_ELO)
    a_elo = CS2_ELO.get(a_name, DEFAULT_ELO)
    h_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    return round(h_prob, 3), round(1 - h_prob, 3)


def _apply_lan_bonus(base_prob: float, home_team: str, away_team: str,
                     tournament_type: str) -> float:
    """Корректирует вероятность с учётом LAN/Online исторической силы команд."""
    if tournament_type not in ("major", "lan_s", "lan_a"):
        return base_prob
    h_name = normalize_team_name(home_team)
    a_name = normalize_team_name(away_team)
    h_bonus = LAN_BONUS.get(h_name, LAN_BONUS.get(home_team, 0.0))
    a_bonus = LAN_BONUS.get(a_name, LAN_BONUS.get(away_team, 0.0))
    adj = (h_bonus - a_bonus) * 0.5
    return max(0.05, min(0.95, base_prob + adj))


def calculate_cs2_win_prob(home_team, away_team,
                            tournament_context: dict = None):
    """
    Расчёт вероятности победы — 5 источников данных:
    1. MIS (Map Impact Score) из вето-симуляции — 25%
    2. ELO рейтинг + LAN/Online корректировка — 30%
    3. Взвешенный винрейт из PandaScore (last5×60% + last20×40%) — 25%
    4. Рейтинг игроков с HLTV — 20%
    + бонус за H2H статистику
    + stand-in предупреждение
    """
    ctx = tournament_context or {"type": "online", "tier": "B", "label": "💻 Online"}

    # 1. Статистика карт и игроков
    home_map_stats = get_team_map_stats(home_team)
    away_map_stats = get_team_map_stats(away_team)
    team_map_stats_combined = {home_team: home_map_stats, away_team: away_map_stats}

    home_players = get_team_player_stats(home_team)
    away_players = get_team_player_stats(away_team)

    # Для неизвестных команд — используем общий винрейт как основу для карт
    from .veto_logic import TEAM_MAP_PREFERENCES, ACTIVE_DUTY_POOL
    from .hltv_stats import MAP_STATS

    def _build_map_bias(team_name: str, map_stats: dict) -> dict:
        """
        Если нет статистики по картам — строим приближение из общего винрейта.
        Пик-карты команды → +5% к винрейту, декайдер → нейтрально.
        """
        if map_stats:
            return {}  # есть реальные данные — не нужно
        if team_name in TEAM_MAP_PREFERENCES or team_name in MAP_STATS:
            return {}
        stats = get_team_stats(team_name, last_n=20)
        wr = stats.get("winrate", 0.5)
        if wr == 0.5:
            return {}  # нет данных вообще
        # Все карты получают базовый винрейт команды с небольшим разбросом
        import random
        rng = random.Random(hash(team_name))  # детерминировано по имени
        result = {}
        for mp in ACTIVE_DUTY_POOL:
            noise = rng.uniform(-0.05, 0.05)
            result[mp] = round((wr + noise) * 100, 1)
        return result

    h_live_maps = _build_map_bias(home_team, home_map_stats)
    a_live_maps = _build_map_bias(away_team, away_map_stats)

    # 2. Симуляция мап-вето
    maps, veto_log = simulate_bo3_veto(home_team, away_team, team_map_stats_combined)

    # 3. MIS для каждой карты
    map_scores = []
    for m in maps:
        h_mis = get_map_impact_score(home_team, m, team_map_stats_combined, live_stats=h_live_maps)
        a_mis = get_map_impact_score(away_team, m, team_map_stats_combined, live_stats=a_live_maps)
        total = h_mis + a_mis
        h_prob, a_prob = (h_mis / total, a_mis / total) if total > 0 else (0.5, 0.5)
        map_scores.append((m, h_prob, a_prob))

    weights = [0.35, 0.35, 0.30]
    mis_h = sum(map_scores[i][1] * weights[i] for i in range(3))

    # 4. ELO вероятность + LAN/Online корректировка
    elo_h_raw, _ = get_elo_prob(home_team, away_team)
    elo_h = _apply_lan_bonus(elo_h_raw, home_team, away_team, ctx["type"])

    # 5. Взвешенный винрейт из PandaScore (last5 60% + older 40%)
    h_form = get_team_weighted_form(home_team)
    a_form = get_team_weighted_form(away_team)
    total_wr = h_form["winrate"] + a_form["winrate"]
    h_wr_norm = h_form["winrate"] / total_wr if total_wr > 0 else 0.5

    # Обратная совместимость — h_stats для отчёта
    h_stats = {
        "winrate": h_form["winrate"], "wins": h_form["wins"],
        "losses": h_form["losses"], "form": h_form["form"],
        "matches": h_form["matches"],
        "winrate_last5": h_form["winrate_last5"],
    }
    a_stats = {
        "winrate": a_form["winrate"], "wins": a_form["wins"],
        "losses": a_form["losses"], "form": a_form["form"],
        "matches": a_form["matches"],
        "winrate_last5": a_form["winrate_last5"],
    }

    # 6. Средний рейтинг игроков (HLTV)
    h_rating = sum(p['rating'] for p in home_players) / len(home_players) if home_players else 1.0
    a_rating = sum(p['rating'] for p in away_players) / len(away_players) if away_players else 1.0
    total_rating = h_rating + a_rating
    h_rating_norm = h_rating / total_rating if total_rating > 0 else 0.5

    # 7. Личные встречи (бонус/штраф ±10%)
    h2h = get_head_to_head(home_team, away_team)
    h2h_bonus = (h2h["team1_wins"] / h2h["total"] - 0.5) * 0.1 if h2h["total"] >= 3 else 0.0

    # 8. Stand-in проверка (снижает уверенность)
    h_standin = check_stand_in(home_team)
    a_standin = check_stand_in(away_team)

    # Stand-in штраф: команда со стэнд-ином получает -4% к вероятности
    standin_adj = 0.0
    if h_standin["has_standin"]:
        standin_adj -= 0.04
    if a_standin["has_standin"]:
        standin_adj += 0.04  # это значит away слабее → home сильнее

    # 9. Momentum бонус: серия побед/поражений в последних матчах
    def _momentum(form_str: str) -> float:
        """±0.03 если команда в серии (3+ W или 3+ L подряд)."""
        if not form_str or len(form_str) < 3:
            return 0.0
        last3 = form_str[:3]  # form строится новые-первые
        if last3 == "WWW":
            return +0.03
        if last3 == "LLL":
            return -0.03
        return 0.0

    h_momentum = _momentum(h_form.get("form", ""))
    a_momentum = _momentum(a_form.get("form", ""))
    momentum_adj = (h_momentum - a_momentum)

    # Финальный ансамбль
    final_h = (
        mis_h         * 0.25 +
        elo_h         * 0.30 +
        h_wr_norm     * 0.25 +
        h_rating_norm * 0.20 +
        h2h_bonus     +
        standin_adj   +
        momentum_adj
    )
    final_h = max(0.05, min(0.95, final_h))

    # Оцениваем реальное наличие данных (0.0 = нет ничего, 1.0 = полные данные)
    h_name_norm = normalize_team_name(home_team)
    a_name_norm = normalize_team_name(away_team)
    h_elo_known = h_name_norm in CS2_ELO
    a_elo_known = a_name_norm in CS2_ELO
    h_ps_known  = h_stats.get("matches", 0) > 0
    a_ps_known  = a_stats.get("matches", 0) > 0
    h_hltv_known = len(home_players) > 0
    a_hltv_known = len(away_players) > 0
    data_confidence = round(
        (0.15 if h_elo_known  else 0) +
        (0.15 if a_elo_known  else 0) +
        (0.15 if h_ps_known   else 0) +
        (0.15 if a_ps_known   else 0) +
        (0.20 if h_hltv_known else 0) +
        (0.20 if a_hltv_known else 0),
        2
    )

    return {
        "home_prob":        round(final_h, 2),
        "away_prob":        round(1 - final_h, 2),
        "maps":             map_scores,
        "veto_log":         veto_log,
        "home_stats":       h_stats,
        "away_stats":       a_stats,
        "home_players":     home_players,
        "away_players":     away_players,
        "elo_home":         CS2_ELO.get(home_team, DEFAULT_ELO),
        "elo_away":         CS2_ELO.get(away_team, DEFAULT_ELO),
        "h2h":              h2h,
        "tournament_ctx":   ctx,
        "home_standin":     h_standin,
        "away_standin":     a_standin,
        "data_confidence":  data_confidence,
        "detail": {
            "mis":            round(mis_h, 2),
            "elo":            round(elo_h, 2),
            "elo_raw":        round(elo_h_raw, 2),
            "winrate":        round(h_wr_norm, 2),
            "winrate_last5":  round(h_form["winrate_last5"], 2),
            "player_rating":  round(h_rating_norm, 2),
            "h2h_bonus":      round(h2h_bonus, 3),
            "standin_adj":    round(standin_adj, 3),
            "lan_adj":        round(elo_h - elo_h_raw, 3),
            "momentum_adj":   round(momentum_adj, 3),
        }
    }


def get_golden_signal(analysis_data, bookmaker_odds):
    h_prob = analysis_data["home_prob"]
    a_prob = analysis_data["away_prob"]
    h_odds = bookmaker_odds.get("home_win", 0)
    a_odds = bookmaker_odds.get("away_win", 0)
    signals = []
    if h_odds > 1.0:
        h_ev = (h_prob * h_odds) - 1
        if h_prob >= 0.60 and h_odds >= 1.60 and h_ev >= 0.15:
            signals.append({
                "type": "GOLDEN",
                "team": analysis_data.get("home_team", ""),
                "outcome": "Победа (П1)",
                "odds": h_odds,
                "ev": round(h_ev * 100, 1),
                "confidence": int(h_prob * 100),
            })
    if a_odds > 1.0:
        a_ev = (a_prob * a_odds) - 1
        if a_prob >= 0.60 and a_odds >= 1.60 and a_ev >= 0.15:
            signals.append({
                "type": "GOLDEN",
                "team": analysis_data.get("away_team", ""),
                "outcome": "Победа (П2)",
                "odds": a_odds,
                "ev": round(a_ev * 100, 1),
                "confidence": int(a_prob * 100),
            })
    return signals


def format_cs2_full_report(
    home_team, away_team, analysis,
    gpt_analysis, llama_analysis,
    golden_signals, bookmaker_odds=None,
    signal_checks=None,       # результат check_cs2_signal из signal_engine
    ranked_bets=None,         # результат get_cs2_ranked_bets()
    totals_data=None,         # результат predict_cs2_totals()
    chimera_verdict_block="", # CHIMERA Multi-Agent вердикт
    commence_time=None,       # ISO UTC строка времени начала матча
):
    """
    Формирует полный отчёт CS2 для Telegram.
    signal_checks: список dict-ов из signal_engine.check_cs2_signal()
    """
    h_stats   = analysis.get("home_stats", {})
    a_stats   = analysis.get("away_stats", {})
    h_players = analysis.get("home_players", [])
    a_players = analysis.get("away_players", [])
    h2h       = analysis.get("h2h", {})
    detail    = analysis.get("detail", {})

    elo_h = analysis.get("elo_home", DEFAULT_ELO)
    elo_a = analysis.get("elo_away", DEFAULT_ELO)

    ctx        = analysis.get("tournament_ctx", {})
    h_standin  = analysis.get("home_standin", {})
    a_standin  = analysis.get("away_standin", {})

    report = f"🎮 *CHIMERA AI CS2 — АНАЛИЗ МАТЧА*\n"
    report += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    # Контекст турнира
    if ctx.get("label"):
        report += f"{ctx['label']}\n"
    report += f"⚔️ *{home_team}* vs *{away_team}*\n"
    # Дата и время матча
    if commence_time:
        try:
            from datetime import datetime, timezone, timedelta
            _dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            _dt_msk = _dt.astimezone(timezone(timedelta(hours=3)))
            _months = ["", "янв", "фев", "мар", "апр", "май", "июн",
                       "июл", "авг", "сен", "окт", "ноя", "дек"]
            _m = _months[_dt_msk.month]
            report += f"📅 *{_dt_msk.day} {_m} {_dt_msk.year}, {_dt_msk.strftime('%H:%M')} МСК*\n"
        except Exception:
            pass
    report += "\n"
    # Stand-in предупреждения
    if h_standin.get("has_standin"):
        report += f"⚠️ *STAND-IN {home_team}:* {h_standin['standin_player']} вместо {h_standin['missing_player']}\n"
    if a_standin.get("has_standin"):
        report += f"⚠️ *STAND-IN {away_team}:* {a_standin['standin_player']} вместо {a_standin['missing_player']}\n"
    if h_standin.get("has_standin") or a_standin.get("has_standin"):
        report += "\n"

    # ── РЕКОМЕНДУЕМЫЕ СТАВКИ (всегда вверху) ─────────────────────────────
    if ranked_bets:
        top = ranked_bets[0]
        report += f"🎯 *ЧТО СТАВИТЬ:*\n"
        for i, bet in enumerate(ranked_bets[:3], 1):
            star = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
            report += f"{star} *{bet['label']}*\n"
            _u = '3u' if bet['kelly'] >= 4 else ('2u' if bet['kelly'] >= 2 else '1u')
            report += f"   Вероятность: *{bet['prob']}%* | Кэф: *{bet['odds']}* | EV: *+{bet['ev']}%* | Банк: *{bet['kelly']}%* ({_u})\n"
            if bet.get("note"):
                report += f"   _{bet['note']}_\n"
        # Тотал карт после ranked_bets
        if totals_data:
            maps_pred = totals_data.get("prediction", "")
            maps_conf = totals_data.get("confidence", 0)
            maps_reason = totals_data.get("reason", "")
            round_pred = totals_data.get("round_prediction", "")
            round_conf = totals_data.get("round_confidence", 0)
            if maps_pred and maps_conf >= 62:
                # Проверяем: есть ли реальный кэф (не оценочный)
                if bookmaker_odds and (bookmaker_odds.get("under_2_5") or bookmaker_odds.get("over_2_5")):
                    report += f"📊 *Тотал карт:* {maps_pred} ({maps_conf}%)\n"
                else:
                    report += f"📊 *Тотал карт (оценка):* {maps_pred} ({maps_conf}%) _(нет букм. линии)_\n"
                if maps_reason:
                    report += f"   _{maps_reason}_\n"
            if round_pred and round_conf >= 62:
                report += f"📊 *Тотал раундов (1-я карта):* {round_pred} ({round_conf}%)\n"
        report += "\n"
    elif totals_data:
        # Есть только тоталы, без победителя
        report += f"🎯 *ПРОГНОЗ ТОТАЛА:* {totals_data['prediction']} ({totals_data['confidence']}%)\n"
        report += f"   _{totals_data['reason']}_\n\n"

    # ── Коэффициенты ──────────────────────────────────────────────────────
    if bookmaker_odds:
        h_odds = bookmaker_odds.get("home_win", 0)
        a_odds = bookmaker_odds.get("away_win", 0)
        if h_odds > 0 and a_odds > 0:
            report += f"💰 *КЭФЫ:* {home_team}: *{h_odds:.2f}* | {away_team}: *{a_odds:.2f}*\n\n"

    # ── ELO рейтинги ──────────────────────────────────────────────────────
    report += f"🏆 *ELO:* {home_team}: *{elo_h}* | {away_team}: *{elo_a}*\n\n"

    # ── Статистика (PandaScore) ────────────────────────────────────────────
    if h_stats.get("matches", 0) > 0 or a_stats.get("matches", 0) > 0:
        report += f"📊 *ФОРМА:*\n"
        if h_stats.get("matches", 0) > 0:
            wr5_h = int(h_stats.get("winrate_last5", h_stats["winrate"]) * 100)
            trend_h = "🔥" if wr5_h >= 70 else ("📉" if wr5_h <= 30 else "")
            report += (f" 🔹 {home_team}: `{h_stats.get('form','—')}` "
                       f"WR={int(h_stats['winrate']*100)}%  last5={wr5_h}% {trend_h}\n")
        if a_stats.get("matches", 0) > 0:
            wr5_a = int(a_stats.get("winrate_last5", a_stats["winrate"]) * 100)
            trend_a = "🔥" if wr5_a >= 70 else ("📉" if wr5_a <= 30 else "")
            report += (f" 🔸 {away_team}: `{a_stats.get('form','—')}` "
                       f"WR={int(a_stats['winrate']*100)}%  last5={wr5_a}% {trend_a}\n")
        report += "\n"

    # ── H2H ───────────────────────────────────────────────────────────────
    if h2h.get("total", 0) >= 2:
        report += f"🤝 *H2H:* {home_team} {h2h['team1_wins']}:{h2h['team2_wins']} {away_team} (из {h2h['total']} встреч)\n\n"

    # ── Составы (HLTV) ────────────────────────────────────────────────────
    report += f"👥 *СОСТАВЫ (HLTV Rating):*\n"
    h_p_str = ", ".join([f"{p['name']} ({p['rating']})" for p in h_players[:5]])
    a_p_str = ", ".join([f"{p['name']} ({p['rating']})" for p in a_players[:5]])
    if h_p_str:
        h_avg = sum(p['rating'] for p in h_players) / len(h_players)
        report += f" 🔹 {home_team} [ср. {h_avg:.2f}]: {h_p_str}\n"
    if a_p_str:
        a_avg = sum(p['rating'] for p in a_players) / len(a_players)
        report += f" 🔸 {away_team} [ср. {a_avg:.2f}]: {a_p_str}\n"
    report += "\n"

    # ── Вето ──────────────────────────────────────────────────────────────
    report += f"🗺 *СИМУЛЯЦИЯ VETO (BO3):*\n"
    for log in analysis["veto_log"]:
        report += f" {log}\n"
    report += "\n"

    # ── Вероятность по картам ─────────────────────────────────────────────
    report += f"📈 *ВЕРОЯТНОСТЬ ПО КАРТАМ (MIS):*\n"
    for m, hp, ap in analysis["maps"]:
        bar_h = "█" * int(hp * 10)
        bar_a = "█" * int(ap * 10)
        winner = "◀" if hp > ap else "▶"
        report += f" • {m}: {int(hp*100)}% {bar_h}{winner}{bar_a} {int(ap*100)}%\n"
    report += "\n"

    # ── Итоговый расчёт ───────────────────────────────────────────────────
    h_pct = int(analysis['home_prob'] * 100)
    a_pct = int(analysis['away_prob'] * 100)
    report += f"🔢 *ИТОГОВАЯ ВЕРОЯТНОСТЬ:*\n"
    report += f" {home_team}: *{h_pct}%* | {away_team}: *{a_pct}%*\n"
    # Детали формулы
    lan_adj   = detail.get('lan_adj', 0)
    si_adj    = detail.get('standin_adj', 0)
    adj_parts = []
    if abs(lan_adj) >= 0.01:
        adj_parts.append(f"LAN: {'+' if lan_adj>0 else ''}{int(lan_adj*100)}%")
    if abs(si_adj) >= 0.01:
        adj_parts.append(f"StandIn: {'+' if si_adj>0 else ''}{int(si_adj*100)}%")
    adj_str = f" · {' · '.join(adj_parts)}" if adj_parts else ""
    report += (f" _(MIS {int(detail.get('mis',0)*100)}% · ELO {int(detail.get('elo',0)*100)}% · "
               f"WR {int(detail.get('winrate',0)*100)}% [last5:{int(detail.get('winrate_last5',0)*100)}%] · "
               f"Players {int(detail.get('player_rating',0)*100)}%{adj_str})_\n\n")

    # ── AI анализ ─────────────────────────────────────────────────────────
    if gpt_analysis and gpt_analysis != "—" and not gpt_analysis.startswith("❌"):
        report += f"🧠 *GPT Стратег:*\n_{gpt_analysis}_\n\n"
    if llama_analysis and llama_analysis != "—" and not llama_analysis.startswith("❌"):
        report += f"🦙 *Llama Тактик:*\n_{llama_analysis}_\n\n"

    # ── Signal Engine ─────────────────────────────────────────────────────
    if signal_checks:
        report += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for sig in signal_checks:
            score = sig.get("score", 0)
            max_s = sig.get("max_score", 10)
            strength = sig.get("strength", "")
            ev = sig.get("ev", 0)
            kelly = sig.get("kelly", 0)
            team_name = sig.get("team", "")
            outcome = sig.get("outcome", "")
            odds_s = sig.get("odds", 0)
            prob_s = sig.get("prob", 0)

            report += f"\n✅ *СИГНАЛ: СТАВИТЬ* — {strength}\n"
            report += f"🎯 *{team_name}* ({outcome})\n"
            report += f"📊 Вероятность: *{prob_s}%* | Кэф: *{odds_s}* | EV: *+{ev}%*\n"
            _u = '3u' if kelly >= 4 else ('2u' if kelly >= 2 else '1u')
            report += f"⚖️ Ставка (Келли): *{kelly}% от банка* ({_u})\n"
            report += f"📋 Проверки ({score}/{max_s}):\n"
            for check in sig.get("checks", []):
                report += f" • {check}\n"
        report += "\n"
    elif golden_signals:
        report += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for sig in golden_signals:
            report += f"🌟 *ЗОЛОТОЙ СИГНАЛ:* {sig['outcome']} {sig['team']} @ {sig['odds']} (EV: +{sig['ev']}% | Уверенность: {sig['confidence']}%)\n"
    else:
        report += f"⏸ *СИГНАЛ: ПРОПУСТИТЬ* — недостаточно подтверждений\n"

    if chimera_verdict_block:
        # Конвертируем HTML-теги в Markdown (CS2 отчёт использует parse_mode="Markdown")
        md_block = (chimera_verdict_block
                    .replace("<b>", "*").replace("</b>", "*")
                    .replace("<i>", "_").replace("</i>", "_")
                    .replace("<code>", "`").replace("</code>", "`"))
        report += md_block + "\n"

    return report
