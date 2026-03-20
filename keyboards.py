# keyboards.py — клавиатуры и утилиты форматирования кнопок

from datetime import datetime, timezone, timedelta
from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from i18n import t

# Список лиг для загрузки матчей
FOOTBALL_LEAGUES = [
    ("soccer_epl",                    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 АПЛ"),
    ("soccer_spain_la_liga",           "🇪🇸 Ла Лига"),
    ("soccer_germany_bundesliga",      "🇩🇪 Бундеслига"),
    ("soccer_italy_serie_a",           "🇮🇹 Серия А"),
    ("soccer_france_ligue_one",        "🇫🇷 Лига 1"),
    ("soccer_uefa_champs_league",      "🏆 Лига Чемпионов"),
    ("soccer_uefa_europa_league",      "🥈 Лига Европы"),
    ("soccer_netherlands_eredivisie",  "🇳🇱 Эредивизи"),
    ("soccer_portugal_primeira_liga",  "🇵🇹 Примейра"),
    ("soccer_turkey_super_league",     "🇹🇷 Суперлига"),
]

PAGE_SIZE = 8  # матчей на страницу

_SHORT_NAMES = {
    "Manchester United": "Man Utd", "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves", "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Tottenham Hotspur": "Spurs",
    "Nottingham Forest": "Nott'm F", "Leicester City": "Leicester",
    "Leeds United": "Leeds", "Sheffield United": "Sheffield",
    "Atletico Madrid": "Atletico", "Paris Saint-Germain": "PSG",
    "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
}


def _short(name: str) -> str:
    return _SHORT_NAMES.get(name, name)


def _escape_md(text: str) -> str:
    """Экранирует спецсимволы для Markdown V1 в Telegram."""
    for ch in ('_', '*', '`', '['):
        text = text.replace(ch, "\\" + ch)
    return text


def _match_status_label(commence_time: str) -> str:
    """Возвращает статус-префикс для кнопки матча."""
    if not commence_time:
        return "📅"
    try:
        now = datetime.now(timezone.utc)
        dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        diff = (now - dt).total_seconds()
        moscow_tz = timezone(timedelta(hours=3))
        dt_m = dt.astimezone(moscow_tz)
        if 0 < diff < 10800:      # начался, прошло < 3ч — LIVE
            return "🟢 LIVE"
        elif diff >= 10800:        # скорее всего закончился
            return "🔴 Fin"
        else:                      # предстоящий
            return dt_m.strftime('%d.%m %H:%M')
    except Exception:
        return "📅"


def build_main_keyboard(lang: str = "ru"):
    """Строит главную клавиатуру с секциями спорта."""
    kb = [
        [types.KeyboardButton(text=t("btn_signals", lang)), types.KeyboardButton(text=t("btn_express", lang))],
        [types.KeyboardButton(text=t("btn_football", lang))],
        [types.KeyboardButton(text=t("btn_tennis", lang)), types.KeyboardButton(text=t("btn_cs2", lang))],
        [types.KeyboardButton(text=t("btn_basketball", lang))],
        [types.KeyboardButton(text=t("btn_hunt", lang))],
        [types.KeyboardButton(text=t("btn_stats", lang)), types.KeyboardButton(text=t("btn_cabinet", lang))],
        [types.KeyboardButton(text=t("btn_vip", lang)), types.KeyboardButton(text=t("btn_support", lang))],
    ]
    return types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def build_football_keyboard():
    """Клавиатура футбольного меню с выбором лиги."""
    builder = InlineKeyboardBuilder()
    for league_key, league_name in FOOTBALL_LEAGUES:
        builder.button(text=league_name, callback_data=f"league_{league_key}")
    builder.button(text="⬅️ Назад", callback_data="back_to_main")
    builder.adjust(2)
    return builder.as_markup()


def format_matches_list(matches) -> str:
    """Текстовый список матчей со статусом для показа над кнопками."""
    now = datetime.now(timezone.utc)
    lines = []
    for i, m in enumerate(matches, 1):
        ct = m.get('commence_time', '')
        status = "📅"
        time_label = ""
        if ct:
            try:
                dt = datetime.fromisoformat(ct.replace('Z', '+00:00'))
                diff = (now - dt).total_seconds()
                moscow_tz = timezone(timedelta(hours=3))
                dt_m = dt.astimezone(moscow_tz)
                if 0 < diff < 7200:       # начался, идёт < 2ч
                    status = "🟢"
                    time_label = "LIVE"
                elif 7200 <= diff < 10800: # идёт 2-3ч, скоро конец
                    status = "🟢"
                    time_label = "LIVE"
                elif diff >= 10800:        # скорее всего закончился
                    status = "🔴"
                    time_label = "Finished"
                else:                      # предстоящий
                    status = "🕐"
                    time_label = dt_m.strftime('%d.%m %H:%M')
            except Exception:
                time_label = ct[:10]
        h = _short(m.get('home_team', ''))
        a = _short(m.get('away_team', ''))
        lines.append(f"{i}. {status} {h} — {a}  {time_label}")
    return "\n".join(lines)


def build_matches_keyboard(matches, page: int = 0):
    """
    Строит клавиатуру со списком матчей со статусом и пагинацией.
    Показывает PAGE_SIZE матчей за раз с кнопками ← / →.
    """
    builder = InlineKeyboardBuilder()
    total = len(matches)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)

    for i in range(start, end):
        match = matches[i]
        h = _short(match.get('home_team', ''))
        a = _short(match.get('away_team', ''))
        label = _match_status_label(match.get('commence_time', ''))
        builder.button(
            text=f"⚽ {label}  {h} — {a}",
            callback_data=f"m_{i}"
        )

    # Служебные кнопки (каждая в свою строку)
    builder.button(text="🔄 Обновить", callback_data="refresh_matches")
    builder.button(text="🏆 Другая лига", callback_data="change_league")
    builder.button(text="🏠 Меню", callback_data="back_to_main")

    # Пагинация — 2 маленькие кнопки в одну строку
    has_prev = page > 0
    has_next = page < total_pages - 1
    if has_prev:
        builder.button(text=f"◀ {page}/{total_pages}", callback_data=f"matches_page_{page-1}")
    if has_next:
        builder.button(text=f"{page+2}/{total_pages} ▶", callback_data=f"matches_page_{page+1}")

    # adjust: матчи по 1, сервисные по 1, пагинация по 2
    nav_count = int(has_prev) + int(has_next)
    sizes = [1] * (end - start) + [1, 1, 1]  # матчи + 3 сервисные
    if nav_count == 2:
        sizes += [2]
    elif nav_count == 1:
        sizes += [1]
    builder.adjust(*sizes)
    return builder.as_markup()


def build_markets_keyboard(match_index):
    """Строит клавиатуру выбора рынка ставок."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🏆 Победитель матча", callback_data=f"mkt_winner_{match_index}")
    builder.button(text="⚽ Голы (тотал 2.5 / обе забьют)", callback_data=f"mkt_goals_{match_index}")
    builder.button(text="⚖️ Гандикапы / Двойной шанс", callback_data=f"mkt_handicap_{match_index}")
    builder.button(text="⬅️ Другой матч", callback_data="back_to_matches")
    builder.button(text="🏠 Главное меню", callback_data="back_to_main")
    builder.adjust(1)
    return builder.as_markup()


def build_back_to_markets_keyboard(match_index):
    """Кнопка возврата к выбору рынка."""
    builder = InlineKeyboardBuilder()
    builder.button(text="↩️ К анализу", callback_data=f"back_to_report_football_{match_index}")
    builder.button(text="🎯 Другой рынок", callback_data=f"show_markets_{match_index}")
    builder.button(text="⬅️ Матчи", callback_data="back_to_matches")
    builder.button(text="🏠 Меню", callback_data="back_to_main")
    builder.adjust(2)
    return builder.as_markup()


def _build_hunt_kb(page: int, total: int) -> types.InlineKeyboardMarkup:
    """Клавиатура Охоты: пагинация + кнопка обновить."""
    from line_tracker import PER_PAGE
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    nav = []
    if page > 0:
        nav.append(types.InlineKeyboardButton(text="◀️ Назад", callback_data=f"hunt_page_{page-1}"))
    if page < total_pages - 1:
        nav.append(types.InlineKeyboardButton(text="Вперёд ▶️", callback_data=f"hunt_page_{page+1}"))
    rows = []
    if nav:
        rows.append(nav)
    rows.append([types.InlineKeyboardButton(text="🔄 Обновить", callback_data="hunt_refresh")])
    return types.InlineKeyboardMarkup(inline_keyboard=rows)
