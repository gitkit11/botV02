# -*- coding: utf-8 -*-
"""
line_tracker.py — Трекер движения линий для Chimera AI
=======================================================

Два режима:
  A) get_line_movement()  — строка для сигнала: показывает как линия
     двигалась с момента открытия рынка до момента сигнала.

  B) get_closing_line_str() — строка для уведомления после матча:
     сравниваем наш вход с последними известными odds (≈ закрытие рынка).

Данные пишет odds_cache.py при каждом обновлении — 0 доп. запросов.
Хранятся в opening_lines.json, очищаются через 7 дней.
"""

import os
import json
import time
import threading
import logging

logger = logging.getLogger(__name__)

_FILE      = os.path.join(os.path.dirname(__file__), "opening_lines.json")
_lock      = threading.Lock()   # защищает _lines в памяти
_file_lock = threading.Lock()   # защищает запись на диск (один поток за раз)
_lines: dict = {}
_dirty = False
_save_scheduled = False         # debounce: не спавним 64 потока, только 1


# ─── Загрузка / сохранение ───────────────────────────────────────────────────

def _load():
    global _lines
    try:
        if os.path.exists(_FILE):
            with open(_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cutoff = time.time() - 7 * 86400
            _lines = {k: v for k, v in data.items()
                      if isinstance(v, dict) and v.get("last_updated", 0) > cutoff}
    except Exception as e:
        logger.warning(f"[LineTracker] Ошибка загрузки: {e}")
        _lines = {}


def _save():
    global _dirty, _save_scheduled
    try:
        with _file_lock:
            with _lock:
                data = dict(_lines)
                _dirty = False
            tmp = _FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _FILE)
    except Exception as e:
        logger.warning(f"[LineTracker] Ошибка сохранения: {e}")
    finally:
        with _lock:
            _save_scheduled = False
            # если пока писали на диск пришли новые данные — сохраняем ещё раз
            if _dirty:
                _save_scheduled = True
                threading.Thread(target=_save, daemon=True).start()


_load()


# ─── Обновление данных ────────────────────────────────────────────────────────

_MAX_SNAPSHOTS = 8   # храним последние 8 снимков (~80 мин при TTL=10 мин)
_STEAM_WINDOW  = 3600  # ищем движение за последний час


def update_match_odds(match_id: str, sport: str, home_team: str, away_team: str,
                      home_odds: float, away_odds: float, draw_odds: float = None):
    """
    Вызывается из odds_cache при каждом обновлении коэффициентов.
    Первый вызов = opening. Последующие = обновляют last_* и добавляют снимок.
    """
    if not match_id or not home_odds or not away_odds:
        return
    global _dirty
    now = time.time()
    snap = {"home": round(home_odds, 3), "away": round(away_odds, 3),
            "draw": round(draw_odds, 3) if draw_odds else None, "ts": now}
    with _lock:
        if match_id not in _lines:
            _lines[match_id] = {
                "sport":        sport,
                "home_team":    home_team,
                "away_team":    away_team,
                "opening_home": round(home_odds, 3),
                "opening_away": round(away_odds, 3),
                "opening_draw": round(draw_odds, 3) if draw_odds else None,
                "last_home":    round(home_odds, 3),
                "last_away":    round(away_odds, 3),
                "last_draw":    round(draw_odds, 3) if draw_odds else None,
                "first_seen":   now,
                "last_updated": now,
                "snapshots":    [snap],
            }
            logger.debug(f"[LineTracker] Новый матч: {home_team} vs {away_team} ({match_id})")
        else:
            entry = _lines[match_id]
            entry["last_home"]    = round(home_odds, 3)
            entry["last_away"]    = round(away_odds, 3)
            entry["last_draw"]    = round(draw_odds, 3) if draw_odds else entry.get("last_draw")
            entry["last_updated"] = now
            snaps = entry.setdefault("snapshots", [])
            snaps.append(snap)
            if len(snaps) > _MAX_SNAPSHOTS:
                entry["snapshots"] = snaps[-_MAX_SNAPSHOTS:]
        _dirty = True
        if not _save_scheduled:
            _save_scheduled = True
            threading.Thread(target=_save, daemon=True).start()


# ─── Охота Химеры: детектор паровых ударов ───────────────────────────────────

def get_steam_moves(hours_back: float = 2.0) -> list:
    """
    Возвращает список паровых ударов за последние hours_back часов.
    Паровой удар = коэффициент сдвинулся > 3% за ≤ 60 минут.
    Каждый элемент: dict с ключами home_team, away_team, sport, side,
    old_odds, new_odds, pct, minutes_ago, direction.
    """
    cutoff = time.time() - hours_back * 3600
    moves = []

    with _lock:
        snapshot = list(_lines.items())

    for match_id, entry in snapshot:
        snaps = entry.get("snapshots", [])
        if len(snaps) < 2:
            continue

        home_team = entry.get("home_team", "")
        away_team = entry.get("away_team", "")
        sport     = entry.get("sport", "")
        last_snap = snaps[-1]

        # Ищем снимок из 15–60 минут назад (не свежее 15 мин)
        ref_snap = None
        for s in reversed(snaps[:-1]):
            age = last_snap["ts"] - s["ts"]
            if 900 <= age <= _STEAM_WINDOW:  # от 15 мин до 1 часа
                ref_snap = s
                break

        if not ref_snap:
            continue

        # Проверяем только свежие движения (last_snap в окне hours_back)
        if last_snap["ts"] < cutoff:
            continue

        for side in ("home", "away"):
            old = ref_snap.get(side)
            new = last_snap.get(side)
            if not old or not new or old <= 1.01:
                continue
            pct = (old - new) / old * 100  # положит = кэф упал = деньги идут сюда
            if abs(pct) < 3:
                continue

            minutes_ago = round((time.time() - last_snap["ts"]) / 60)
            minutes_span = round((last_snap["ts"] - ref_snap["ts"]) / 60)
            team = home_team if side == "home" else away_team

            moves.append({
                "match_id":    match_id,
                "home_team":   home_team,
                "away_team":   away_team,
                "sport":       sport,
                "side":        side,
                "team":        team,
                "old_odds":    old,
                "new_odds":    new,
                "pct":         round(pct, 1),
                "minutes_ago": minutes_ago,
                "minutes_span": minutes_span,
                "direction":   "in" if pct > 0 else "out",  # деньги идут В ставку или ИЗ
            })

    # Сортируем по силе движения
    moves.sort(key=lambda x: abs(x["pct"]), reverse=True)
    return moves


PER_PAGE = 5


def format_steam_moves(moves: list, page: int = 0) -> str:
    """Форматирует список паровых ударов для Telegram с пагинацией (по 5)."""
    if not moves:
        return (
            "🔥 <b>ОХОТА ХИМЕРЫ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "👁️ <i>Сегодня мало сигналов.</i>\n\n"
            "Охота работает на основе активности пользователей бота — "
            "чем больше матчей запрашивают пользователи, тем больше линий "
            "Химера отслеживает. Это сделано специально, чтобы не перегружать "
            "систему и не сканировать весь рынок вслепую.\n\n"
            "<i>Как только пользователи начнут анализировать матчи и линии "
            "задвигаются — сигналы появятся здесь автоматически.</i>"
        )

    total_pages = max(1, (len(moves) + PER_PAGE - 1) // PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    slice_ = moves[page * PER_PAGE : (page + 1) * PER_PAGE]

    sport_icons = {"soccer": "⚽", "football": "⚽", "basketball": "🏀",
                   "tennis": "🎾", "cs2": "🎮", "esports": "🎮"}
    lines = [
        "🔥 <b>ОХОТА ХИМЕРЫ</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"<i>Движений острых денег: {len(moves)}  ·  стр. {page+1}/{total_pages}</i>\n",
    ]

    for m in slice_:
        icon  = sport_icons.get(m["sport"], "🎯")
        arrow = "🟢 ▲" if m["direction"] == "in" else "🔴 ▼"
        sign  = "+" if m["direction"] == "out" else "−"
        desc  = "острые деньги зашли" if m["direction"] == "in" else "деньги уходят"
        ago   = f"{m['minutes_ago']} мин назад" if m['minutes_ago'] > 0 else "только что"

        lines.append(
            f"{icon} <b>{m['home_team']} vs {m['away_team']}</b>\n"
            f"   {arrow} <b>{m['team']}</b>: {m['old_odds']} → {m['new_odds']} "
            f"({sign}{abs(m['pct'])}% за {m['minutes_span']} мин)\n"
            f"   <i>{desc} · {ago}</i>\n"
        )

    lines.append("<i>Данные Pinnacle · обновляются автоматически</i>")
    return "\n".join(lines)


# ─── Вариант A: движение линии для сигнала ────────────────────────────────────

def get_line_movement(match_id: str, rec_outcome: str) -> str:
    """
    Вариант A: строка движения линии для вставки в сигнал.
    rec_outcome: 'home_win' | 'away_win' | 'draw'
    Возвращает строку или '' если нет данных / движение < 3%.
    """
    with _lock:
        entry = dict(_lines.get(match_id) or {})
    if not entry:
        return ""

    if rec_outcome == "home_win":
        opening = entry.get("opening_home")
        current = entry.get("last_home")
    elif rec_outcome == "away_win":
        opening = entry.get("opening_away")
        current = entry.get("last_away")
    else:
        opening = entry.get("opening_draw")
        current = entry.get("last_draw")

    if not opening or not current or opening <= 1.01:
        return ""

    # Положительный % = кэф упал = деньги идут на этот исход
    pct = (opening - current) / opening * 100

    if abs(pct) < 3:
        return ""

    if pct >= 10:
        return f"📈 Линия: {opening} → {current} (−{pct:.0f}%, рынок активно поддерживает)"
    elif pct >= 5:
        return f"📈 Линия: {opening} → {current} (рынок соглашается с прогнозом)"
    elif pct >= 3:
        return f"📊 Линия: {opening} → {current} (слабое движение в нашу сторону)"
    elif pct <= -10:
        return f"⚠️ Линия: {opening} → {current} (+{abs(pct):.0f}%, рынок настроен иначе)"
    elif pct <= -5:
        return f"⚠️ Линия: {opening} → {current} (рынок движется против)"
    else:
        return f"📉 Линия: {opening} → {current} (небольшое движение против)"


# ─── Вариант B: закрытие рынка для уведомления ───────────────────────────────

def get_closing_line_str(match_id: str, rec_outcome: str, entry_odds: float) -> str:
    """
    Вариант B: сравниваем наш вход (entry_odds) с последними известными odds.
    Возвращает строку для уведомления или '' если нет данных.
    """
    with _lock:
        entry = dict(_lines.get(match_id) or {})
    if not entry or not entry_odds or entry_odds <= 1.01:
        return ""

    if rec_outcome == "home_win":
        closing = entry.get("last_home")
    elif rec_outcome == "away_win":
        closing = entry.get("last_away")
    elif rec_outcome == "draw":
        closing = entry.get("last_draw")
    else:
        return ""

    if not closing or closing <= 1.01:
        return ""

    # Если наш вход ВЫШЕ закрытия — мы взяли лучший кэф, рынок подтвердил
    # entry=2.10, closing=1.85 → diff_pct = +13.5% → "вошли раньше рынка"
    diff_pct = (entry_odds - closing) / closing * 100

    if abs(diff_pct) < 2:
        return f"📊 Рынок закрылся @ {closing} — линия не изменилась"
    elif diff_pct >= 10:
        return f"📈 Рынок закрылся @ {closing} — сильный ранний вход ✅"
    elif diff_pct >= 5:
        return f"📈 Рынок закрылся @ {closing} — вошли раньше рынка ✅"
    elif diff_pct >= 2:
        return f"📊 Рынок закрылся @ {closing} — чуть опередили рынок"
    elif diff_pct <= -10:
        return f"📉 Рынок закрылся @ {closing} — рынок не подтвердил вход"
    elif diff_pct <= -5:
        return f"📉 Рынок закрылся @ {closing} — линия пошла против"
    else:
        return f"📉 Рынок закрылся @ {closing} — линия слегка против"
