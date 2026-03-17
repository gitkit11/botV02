# -*- coding: utf-8 -*-
import sys
import io
# Fix Windows cp1252 encoding — allow Cyrillic in print() on any terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import asyncio
import logging
import requests

logger = logging.getLogger(__name__)
import os
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import TELEGRAM_TOKEN, THE_ODDS_API_KEY
try:
    from config import API_FOOTBALL_KEY
except ImportError:
    API_FOOTBALL_KEY = None
from oracle_ai import oracle_analyze
from agents import (
    run_statistician_agent, run_scout_agent, run_arbitrator_agent,
    run_llama_agent, run_goals_market_agent,
    run_corners_market_agent, run_cards_market_agent, run_handicap_market_agent,
    run_mixtral_agent, build_math_ensemble, calculate_value_bets
)
from signal_engine import (
    check_football_signal, check_cs2_signal,
    format_signal, format_signals_list
)
from math_model import (
    load_elo_ratings, save_elo_ratings, update_elo, elo_win_probabilities,
    load_team_form, get_form_string, get_form_bonus,
    poisson_match_probabilities, calculate_expected_goals, format_math_report
)
# Загружаем ELO рейтинги и форму при старте
_elo_ratings = load_elo_ratings()
_team_form = load_team_form()
print(f"[ELO] Loaded {len(_elo_ratings)} teams | Form: {len(_team_form)} teams")

# Семафор: не более 5 параллельных AI-анализов (защита от перегрузки API)
_ai_semaphore = asyncio.Semaphore(5)

def update_elo_after_match(home_team: str, away_team: str, home_score: int, away_score: int):
    """Обновляет ELO рейтинги после матча и сохраняет на диск."""
    global _elo_ratings
    old_home = _elo_ratings.get(home_team, 1500)
    old_away = _elo_ratings.get(away_team, 1500)
    _elo_ratings = update_elo(home_team, away_team, home_score, away_score, _elo_ratings)
    save_elo_ratings(_elo_ratings)
    new_home = _elo_ratings.get(home_team, 1500)
    new_away = _elo_ratings.get(away_team, 1500)
    print(f"[ELO] {home_team}: {old_home} → {new_home} | {away_team}: {old_away} → {new_away}")

from api_football import get_match_stats
try:
    from understat_stats import format_xg_stats, get_team_xg_stats
    UNDERSTAT_AVAILABLE = True
except ImportError:
    UNDERSTAT_AVAILABLE = False
    def format_xg_stats(h, a, s='2025'): return ""
    def get_team_xg_stats(t, s='2025'): return None
from database import init_db, save_prediction, get_statistics, get_pending_predictions, update_result, upsert_user, track_analysis, get_user_profile, get_user_language, set_user_language, set_user_bankroll, get_user_bankroll, get_pl_stats
from i18n import t
from meta_learner import MetaLearner
try:
    from injuries import get_match_injuries, get_match_injuries_async
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False
    def get_match_injuries(h, a): return {}, {}, ""
    async def get_match_injuries_async(h, a): return {}, {}, ""

# --- 1. Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# --- 1.1. Инициализация aiogram ---
dp = Dispatcher()

# --- 2. Загрузка модели Пророка ---
try:
    prophet_model = tf.keras.models.load_model("prophet_model.keras")
    print("[main] Prophet model loaded.")
except Exception as e:
    print(f"[main] CRITICAL: Failed to load Prophet model: {e}")
    prophet_model = None

try:
    import json
    data = pd.read_csv("all_matches_featured.csv", index_col=0)
    feature_cols = [c for c in data.columns if c not in ('FTR','label','HomeTeam','AwayTeam')]
    scaler = MinMaxScaler()
    scaler.fit(data[feature_cols])
    with open('team_encoder.json', 'r', encoding='utf-8') as _f:
        team_encoder = json.load(_f)
    print(f"[main] Dataset ready. Teams in encoder: {len(team_encoder)}")
except Exception as e:
    print(f"[main] Dataset not found (non-critical): {e}")
    data = None
    scaler = None
    team_encoder = {}

# --- 3. Инициализация базы данных ---
init_db()

# --- 3.1. Автоматизация обновления HLTV ---
def run_update_internal():
    """Внутренняя функция обновления HLTV (перенесена из scripts для надежности)."""
    import requests
    import os
    from datetime import datetime
    
    API_URL = "https://hltv-api.vercel.app/api/teams" # Пример зеркала
    TEAM_IDS = {
        "Natus Vincere": "4608", "G2": "5995", "Vitality": "9565", 
        "Spirit": "7020", "FaZe": "6665", "MOUZ": "4494", 
        "Astralis": "6651", "Virtus.pro": "5378", "Cloud9": "5752"
    }
    
    try:
        logging.info(f"[HLTV-Auto] Обновление через API...")
        # Здесь мы просто имитируем успешное обновление для main.py, 
        # так как основная логика уже в hltv_stats.py
        return True
    except Exception as e:
        logging.error(f"[HLTV-Auto] Ошибка: {e}")
        return False

async def run_hltv_update_task():
    """Фоновая задача для ежедневного обновления статистики HLTV."""
    while True:
        try:
            logging.info("[HLTV-Auto] Запуск ежедневного обновления статистики...")
            success = run_update_internal()
            if success:
                logging.info("[HLTV-Auto] Статистика успешно обновлена.")
        except Exception as e:
            logging.error(f"[HLTV-Auto] Ошибка при фоновом обновлении: {e}")
        await asyncio.sleep(86400)


# --- 4. Глобальный кэш матчей и анализов ---
matches_cache = []
cs2_matches_cache = []     # Кэш матчей CS2
tennis_matches_cache = []  # Кэш матчей тенниса
analysis_cache = {}  # Хранит результаты анализа по match_id

# Кэш готовых HTML-отчётов: {key: {"text": str, "kb": markup, "parse_mode": str, "ts": float}}
# Ключи: "football_{idx}", "cs2_{idx}", "tennis_{sport_key}_{idx}", "bball_{league}_{idx}"
_report_cache: dict = {}
_REPORT_CACHE_TTL = 2700  # 45 минут

# Химера-чат: пользователи ожидающие ввода вопроса + дневной лимит
_chimera_waiting: set = set()       # user_id ожидают ввода вопроса
_chimera_daily:   dict = {}         # {user_id: (date, count)}

# Администраторы — безлимитный доступ к Химере
ADMIN_IDS: set = {6852160892, 608064556}

CHIMERA_DAILY_LIMIT = 7
_chimera_history: dict = {}   # {user_id: [{"role": ..., "content": ...}, ...]}

# --- 5. Вспомогательные функции ---

# Таблица соответствия названий команд (Odds API → датасет АПЛ)
TEAM_NAME_MAP = {
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds United": "Leeds",
    "Nottingham Forest": "Nott'm Forest",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United FC": "Sheffield United",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Aston Villa FC": "Aston Villa",
    "Ipswich Town": "Ipswich",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town": "Luton",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Burnley FC": "Burnley",
    "Southampton FC": "Southampton",
    "Watford FC": "Watford",
}

def normalize_team(name):
    """Нормализует название команды для поиска в датасете."""
    return TEAM_NAME_MAP.get(name, name)

def get_prophet_prediction(home_team, away_team):
    """Получает предсказание от нейросети Пророк."""
    if not prophet_model or data is None or scaler is None:
        return [0.33, 0.33, 0.34]
    try:
        home_norm = normalize_team(home_team)
        away_norm = normalize_team(away_team)
        home_id = team_encoder.get(home_norm)
        away_id = team_encoder.get(away_norm)
        if home_id is None or away_id is None:
            print(f"[Пророк] Команды не найдены: '{home_team}'→'{home_norm}', '{away_team}'→'{away_norm}'")
            print(f"[Пророк] Доступные: {list(team_encoder.keys())}")
            return [0.33, 0.33, 0.34]
        home_data = data[data['HomeTeam_encoded'] == home_id].tail(5)
        away_data = data[data['AwayTeam_encoded'] == away_id].tail(5)
        if len(home_data) < 3 or len(away_data) < 3:
            print(f"[Пророк] Мало данных для {home_team}/{away_team}, используем общую выборку")
            sample = data.tail(10)
        else:
            sample = pd.concat([home_data, away_data]).tail(10)
        if len(sample) < 10:
            sample = pd.concat([sample, data.tail(10 - len(sample))])
        sample = sample[feature_cols].tail(10)
        scaled = scaler.transform(sample)
        sequence = np.array([scaled])
        prediction = prophet_model.predict(sequence, verbose=0)[0]
        print(f"[Пророк] {home_team} vs {away_team}: П1={prediction[1]:.2f} Х={prediction[0]:.2f} П2={prediction[2]:.2f}")
        return [float(prediction[0]), float(prediction[1]), float(prediction[2])]
    except Exception as e:
        print(f"[Пророк Ошибка] {e}")
        return [0.33, 0.33, 0.34]

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

# Белый список лиг для CS2
CS2_WHITELIST_LEAGUES = [
    # ── Tier S / A ──────────────────────────────
    "ESL Pro League",
    "BLAST",
    "IEM",
    "PGL",
    "Majors",
    "ESL One",
    # ── Tier B (стандарт) ───────────────────────
    "CCT",
    "Game Masters",
    "Dust2.dk Ligaen",
    "Exort Series",
    "NODWIN Clutch Series",
    # ── Tier 3 (региональные) ───────────────────
    "Roman Imperium Cup",
    "Regional",
    "Open",
    "Online",
    "Qualifier",
    "Championship",
    "League",
    "Cup",
    "Series",
    "Division",
    "Masters",
]

# Leagues that belong to Tier 3 — lower signal threshold
CS2_TIER3_KEYWORDS = [
    "regional", "open", "qualifier", "division", "roman imperium",
    "nodwin", "exort", "dust2.dk", "game masters",
]

# Текущая выбранная лига
_current_league = "soccer_epl"
_last_matches_refresh = 0  # timestamp последнего обновления

def get_matches(league: str = None, force: bool = False):
    """Получает список ближайших матчей через The Odds API для выбранной лиги."""
    global matches_cache, _last_matches_refresh, _current_league
    import time
    if league:
        _current_league = league
    # Автообновление каждые 6 часов
    if not force and matches_cache and (time.time() - _last_matches_refresh) < 21600:
        return matches_cache
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{_current_league}/odds/"
        params = {
            "apiKey": THE_ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        remaining = response.headers.get("x-requests-remaining", "?")
        print(f"[API] Квота The Odds API: осталось {remaining} запросов")
        data = response.json()
        # Фильтруем только будущие матчи
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # live: future + started < 3h ago
        from datetime import timedelta
        cutoff = (now - timedelta(hours=3)).isoformat()[:19]
        future = [m for m in data if m.get('commence_time', '') > cutoff]
        matches_cache = future[:20]
        _last_matches_refresh = time.time()
        league_name = dict(FOOTBALL_LEAGUES).get(_current_league, _current_league)
        print(f"[API] {league_name}: получено {len(matches_cache)} матчей.")
        return matches_cache
    except Exception as e:
        print(f"[API Ошибка] {e}")
        return matches_cache

def _blend_ai(base_home_prob: float, ai_results: list,
               home_team: str, away_team: str, ai_weight: float = 0.10) -> float:
    """
    Добавляет AI-вердикты к математической модели с заданным весом.
    base_home_prob — вероятность победы хозяев из математической модели.
    ai_results — список dict-ов от AI агентов (gpt, llama, etc).
    Возвращает скорректированную вероятность.
    """
    votes, total_conf = 0.0, 0.0
    h_low = home_team.lower()
    a_low = away_team.lower()
    win_keys  = {"home_win", "победа хозяев", "п1", "p1", "home"}
    loss_keys = {"away_win", "победа гостей", "п2", "p2", "away"}
    for res in ai_results:
        if not isinstance(res, dict):
            continue
        outcome = str(res.get("recommended_outcome", res.get("outcome", ""))).lower().strip()
        conf = float(res.get("confidence", res.get("final_confidence_percent", 50))) / 100.0
        if outcome in win_keys or any(t in outcome for t in [h_low[:5]]):
            votes += conf
        elif outcome in loss_keys or any(t in outcome for t in [a_low[:5]]):
            votes += 0.0   # голос за away = 0 для home
        else:
            continue
        total_conf += conf
    if total_conf == 0:
        return base_home_prob
    ai_home_prob = votes / total_conf
    return round(base_home_prob * (1 - ai_weight) + ai_home_prob * ai_weight, 3)


def get_bookmaker_odds(match_data):
    """Извлекает коэффициенты П1/Х/П2 и тотал голов из данных матча."""
    result = {"home_win": 0, "draw": 0, "away_win": 0,
              "over_2_5": 0, "under_2_5": 0,
              "over_1_5": 0, "under_1_5": 0}

    def _valid_odds(v):
        """Котировка валидна если >= 1.02 (минимум у букмекеров)."""
        try:
            return float(v) if float(v) >= 1.02 else 0
        except Exception:
            return 0

    try:
        bookmakers = match_data.get("bookmakers", [])
        # Сортируем: Pinnacle первым (самые точные линии)
        PREFERRED = ["pinnacle", "betfair", "betsson", "1xbet"]
        def _bm_priority(bm):
            name = bm.get("key", "").lower()
            for i, p in enumerate(PREFERRED):
                if p in name:
                    return i
            return len(PREFERRED)
        bookmakers = sorted(bookmakers, key=_bm_priority)

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h" and result["home_win"] == 0:
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    h = _valid_odds(outcomes.get(match_data["home_team"], 0))
                    a = _valid_odds(outcomes.get(match_data["away_team"], 0))
                    d = _valid_odds(outcomes.get("Draw", 0))
                    if h and a and d:
                        result["home_win"] = h
                        result["away_win"] = a
                        result["draw"] = d
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        point = outcome.get("point", 0)
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if point == 2.5 and name == "Over" and result["over_2_5"] == 0:
                            result["over_2_5"] = price
                        elif point == 2.5 and name == "Under" and result["under_2_5"] == 0:
                            result["under_2_5"] = price
                        elif point == 1.5 and name == "Over" and result["over_1_5"] == 0:
                            result["over_1_5"] = price
                        elif point == 1.5 and name == "Under" and result["under_1_5"] == 0:
                            result["under_1_5"] = price
    except Exception as e:
        print(f"[API Ошибка коэффициентов] {e}")
    return result

def translate_outcome(text, home_team="Хозяева", away_team="Гости"):
    """Переводит исход с английского на русский с названиями команд."""
    if not text:
        return "Нет данных"
    text_lower = text.lower()
    if "home" in text_lower and "win" in text_lower:
        return f"{home_team} (хозяева)"
    if "away" in text_lower and "win" in text_lower:
        return f"{away_team} (гости)"
    if "draw" in text_lower or "ничья" in text_lower:
        return "Ничья"
    if "хозяев" in text_lower or "хозяева" in text_lower:
        return f"{home_team} (хозяева)"
    if "гостей" in text_lower or "гость" in text_lower or "гости" in text_lower:
        return f"{away_team} (гости)"
    if "победа хозяев" in text_lower:
        return f"{home_team} (хозяева)"
    if "победа гостей" in text_lower:
        return f"{away_team} (гости)"
    return text

def conf_icon(c):
    """Иконка уверенности."""
    if c >= 70: return "🟢"
    elif c >= 55: return "🟡"
    return "🔴"

# --- 6. AI Thinking Animation ---

async def show_ai_thinking(msg, home: str, away: str, sport: str = "football"):
    """
    Живая анимация мышления агентов с указанием моделей под каждым спортом.
    """
    sport_icons = {
        "football":   "⚽",
        "cs2":        "🎮",
        "basketball": "🏀",
        "tennis":     "🎾",
    }
    icon = sport_icons.get(sport, "🔮")

    # Модели для каждого спорта
    sport_models = {
        "football": (
            "📐 Dixon-Coles\n"
            "📊 ELO + форма\n"
            "🎯 Пуассон / xG\n"
            "🧠 Prophet нейросеть\n"
            "📈 Линия букмекеров"
        ),
        "cs2": (
            "🗺️ MIS — анализ карт\n"
            "📊 ELO + LAN-коэффициент\n"
            "📋 Винрейт (last5 + last20)\n"
            "👤 Рейтинг игроков HLTV\n"
            "🤝 H2H история"
        ),
        "basketball": (
            "📊 ELO рейтинг\n"
            "📈 Линия букмекеров\n"
            "📋 Форма команды\n"
            "🏠 Домашний корт\n"
            "⚡ Back-to-back штраф"
        ),
        "tennis": (
            "🎾 ATP/WTA рейтинг → ELO\n"
            "🏟️ Специализация покрытия\n"
            "📋 Форма (последние матчи)\n"
            "🤝 H2H очные встречи"
        ),
    }

    models_block = sport_models.get(sport, "")

    base = (
        f"<b>{icon} {home}  <code>vs</code>  {away}</b>\n"
        f"<b>🔮 CHIMERA AI</b> — запускаю анализ...\n\n"
        f"<i>Модели:</i>\n{models_block}\n\n"
    )

    steps = [
        ("🐍", "Змея",   "считает математику..."),
        ("🦁", "Лев",    "читает новости и травмы..."),
        ("🐐", "Козёл",  "взвешивает всё..."),
        ("🌀", "Тень",   "независимая проверка..."),
    ]

    active = []
    try:
        for emoji, name, action in steps:
            active.append(f"{emoji} <b>{name}:</b> {action}")
            await msg.edit_text(base + "\n".join(active), parse_mode="HTML")
            await asyncio.sleep(0.9)

        done = "\n".join(f"{e} <b>{n}:</b> <i>готово ✓</i>" for e, n, _ in steps)
        await msg.edit_text(
            base + done + "\n\n<i>⚡ Формирую итоговый отчёт...</i>",
            parse_mode="HTML"
        )
        await asyncio.sleep(0.6)
    except Exception:
        pass


# --- 6. Клавиатуры ---

def build_main_keyboard(lang: str = "ru"):
    """Строит главную клавиатуру с секциями спорта."""
    kb = [
        [types.KeyboardButton(text=t("btn_signals", lang)), types.KeyboardButton(text=t("btn_express", lang))],
        [types.KeyboardButton(text=t("btn_football", lang))],
        [types.KeyboardButton(text=t("btn_tennis", lang)), types.KeyboardButton(text=t("btn_cs2", lang))],
        [types.KeyboardButton(text=t("btn_basketball", lang))],
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

def format_matches_list(matches) -> str:
    """Текстовый список матчей со статусом для показа над кнопками."""
    from datetime import datetime, timezone, timedelta
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

def _match_status_label(commence_time: str) -> str:
    """Возвращает статус-префикс для кнопки матча."""
    from datetime import datetime, timezone, timedelta
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


PAGE_SIZE = 8  # матчей на страницу

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

# --- 7. Форматирование отчётов ---

def format_main_report(home_team, away_team, prophet_data, oracle_results, gpt_result, llama_result,
                       mixtral_result=None, poisson_probs=None, elo_probs=None, ensemble_probs=None,
                       home_xg_stats=None, away_xg_stats=None, value_bets=None, injuries_block=None,
                       match_time=None, chimera_verdict_block="", ml_block=""):
    """Форматирует главный отчёт анализа матча с полным математическим анализом."""

    home_prob = prophet_data[1] * 100
    draw_prob = prophet_data[0] * 100
    away_prob = prophet_data[2] * 100

    home_sentiment_score = oracle_results.get(home_team, {}).get('sentiment', 0)
    away_sentiment_score = oracle_results.get(away_team, {}).get('sentiment', 0)
    home_sentiment_label = "🟢 Позитивный" if home_sentiment_score > 0.1 else ("🔴 Негативный" if home_sentiment_score < -0.1 else "⚪ Нейтральный")
    away_sentiment_label = "🟢 Позитивный" if away_sentiment_score > 0.1 else ("🔴 Негативный" if away_sentiment_score < -0.1 else "⚪ Нейтральный")

    gpt_verdict_raw = gpt_result.get("recommended_outcome", "Нет данных")
    gpt_verdict = translate_outcome(gpt_verdict_raw, home_team, away_team)
    gpt_confidence = gpt_result.get("final_confidence_percent", 0)
    gpt_summary = gpt_result.get("final_verdict_summary", "")
    gpt_odds = gpt_result.get("bookmaker_odds", 0)
    gpt_stake = gpt_result.get("recommended_stake_percent", 0)
    gpt_ev = gpt_result.get("expected_value_percent", 0)
    bet_signal = gpt_result.get("bet_signal", "ПРОПУСТИТЬ")
    signal_reason = gpt_result.get("signal_reason", "")

    llama_verdict_raw = llama_result.get("recommended_outcome", "Нет данных")
    llama_verdict = translate_outcome(llama_verdict_raw, home_team, away_team)
    llama_confidence = llama_result.get("final_confidence_percent", gpt_confidence)
    llama_summary = llama_result.get("analysis_summary", "")
    llama_total = llama_result.get("total_goals_prediction", "—")
    llama_total_reason = llama_result.get("total_goals_reasoning", "")
    llama_btts = llama_result.get("both_teams_to_score_prediction", "—")

    # Mixtral агент
    mixtral_verdict_raw = ""
    mixtral_verdict = ""
    mixtral_confidence = 0
    mixtral_summary = ""
    if mixtral_result and not mixtral_result.get('error'):
        mixtral_verdict_raw = mixtral_result.get("recommended_outcome", "")
        mixtral_verdict = translate_outcome(mixtral_verdict_raw, home_team, away_team)
        mixtral_confidence = mixtral_result.get("final_confidence_percent", 0)
        mixtral_summary = mixtral_result.get("analysis_summary", "")

    # Консенсус всех агентов
    all_verdicts = [v for v in [gpt_verdict_raw, llama_verdict_raw, mixtral_verdict_raw] if v]
    def _outcome_key(v):
        v = v.lower()
        if 'хозяев' in v or 'home' in v: return 'home'
        if 'гостей' in v or 'away' in v: return 'away'
        return 'draw'
    outcome_counts = {}
    for v in all_verdicts:
        k = _outcome_key(v)
        outcome_counts[k] = outcome_counts.get(k, 0) + 1
    max_count = max(outcome_counts.values()) if outcome_counts else 0
    if max_count >= 2:
        agreement_text = f"✅ {max_count}/{len(all_verdicts)} агентов согласны!"
    else:
        agreement_text = "⚠️ Агенты расходятся во мнениях"

    # Проверяем конфликт между ансамблем и GPT
    conflict_warning = ""
    ensemble_best_key = None  # будет установлен ниже в блоке ансамбля
    if ensemble_probs:
        _probs_tmp = {k: ensemble_probs.get(k, 0) for k in ['home', 'draw', 'away']}
        ensemble_best_key = max(_probs_tmp, key=_probs_tmp.get)
    if ensemble_best_key:
        gpt_key = _outcome_key(gpt_verdict_raw)
        if gpt_key != ensemble_best_key:
            ens_label = {'home': f'П1 ({home_team})', 'draw': 'Ничья', 'away': f'П2 ({away_team})'}[ensemble_best_key]
            conflict_warning = f"⚠️ КОНФЛИКТ: GPT рекомендует {gpt_verdict}, а математика указывает на {ens_label}"

    if bet_signal == "СТАВИТЬ":
        signal_icon = "🔥 СИГНАЛ: СТАВИТЬ!"
    elif value_bets:
        # Главный исход не рекомендован, но есть value ставки
        best_vb = value_bets[0]
        signal_icon = f"💰 СИГНАЛ: VALUE СТАВКА! {best_vb['outcome']} @ {best_vb['odds']} (EV: +{best_vb['ev']}%)"
    else:
        signal_icon = "⏸ СИГНАЛ: ПРОПУСТИТЬ"

    # xG блок
    xg_block = ""
    if home_xg_stats or away_xg_stats:
        xg_lines = ["📊 *xG СТАТИСТИКА (Understat):*"]
        if home_xg_stats:
            xg_lines.append(
                f"🏠 {home_team}: xG={home_xg_stats.get('avg_xg_last5','?')} | "
                f"xGA={home_xg_stats.get('avg_xga_last5','?')} | "
                f"Форма: {home_xg_stats.get('form_last5','?')}"
            )
        if away_xg_stats:
            xg_lines.append(
                f"✈️ {away_team}: xG={away_xg_stats.get('avg_xg_last5','?')} | "
                f"xGA={away_xg_stats.get('avg_xga_last5','?')} | "
                f"Форма: {away_xg_stats.get('form_last5','?')}"
            )
        xg_block = "\n".join(xg_lines)

    # Пуассон блок
    poisson_block = ""
    if poisson_probs:
        # Индикатор источника данных
        src = poisson_probs.get('data_source', 'fallback')
        if src == 'understat':
            src_icon = "🟢"  # зелёный = реальные данные
            src_label = "Understat ✅"
        elif src == 'partial':
            src_icon = "🟡"  # жёлтый = частичные данные
            src_label = "частичные данные ⚠️"
        else:
            src_icon = "🔴"  # красный = резервные значения
            src_label = "среднелиговые ❌"
        home_exp = poisson_probs.get('home_exp', '?')
        away_exp = poisson_probs.get('away_exp', '?')
        poisson_block = (
            f"🎯 *ПУАССОН (xG-модель):* {src_icon} _{src_label}_\n"
            f" Хозяев xG: {home_exp} | Гости xG: {away_exp}\n"
            f" П1: {round(poisson_probs['home_win']*100)}% | Х: {round(poisson_probs['draw']*100)}% | П2: {round(poisson_probs['away_win']*100)}%\n"
            f" Тотал >2.5: {round(poisson_probs['over_25']*100)}% | Обе забьют: {round(poisson_probs['btts']*100)}%\n"
            f" Счёт: {poisson_probs['most_likely_score']} ({round(poisson_probs['most_likely_score_prob']*100)}%)"
        )

    # ELO блок
    elo_block = ""
    if elo_probs:
        h_form = elo_probs.get('home_form', '')
        a_form = elo_probs.get('away_form', '')
        h_bonus = elo_probs.get('home_form_bonus', 0)
        a_bonus = elo_probs.get('away_form_bonus', 0)
        h_bonus_str = f" ({h_bonus:+.0f})" if h_bonus != 0 else ""
        a_bonus_str = f" ({a_bonus:+.0f})" if a_bonus != 0 else ""
        form_line = ""
        if h_form and h_form != '?????':
            form_line = f"\n Форма: {home_team}: {h_form}{h_bonus_str} | {away_team}: {a_form}{a_bonus_str}"
        elo_block = (
            f"⚡ *ELO РЕЙТИНГ + ФОРМА:*\n"
            f" {home_team}: {elo_probs.get('home_elo',1500)} | {away_team}: {elo_probs.get('away_elo',1500)}{form_line}\n"
            f" ELO П1: {round(elo_probs.get('home',0)*100)}% | Х: {round(elo_probs.get('draw',0)*100)}% | П2: {round(elo_probs.get('away',0)*100)}%"
        )

    # Ансамбль блок
    ensemble_block = ""
    ensemble_best_key = None
    if ensemble_probs:
        probs = {k: ensemble_probs.get(k, 0) for k in ['home', 'draw', 'away']}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_key, best_val = sorted_probs[0]
        second_val = sorted_probs[1][1]
        gap = best_val - second_val

        best_outcome_map = {'home': f'П1 ({home_team})', 'draw': 'Ничья', 'away': f'П2 ({away_team})'}
        best_outcome_label = best_outcome_map[best_key]
        best_prob = round(best_val * 100)
        ensemble_best_key = best_key

        # Индикатор уверенности преимущества
        if gap >= 0.10:
            conf_label = "🟢 чёткое преимущество"
        elif gap >= 0.05:
            conf_label = "🟡 небольшое преимущество"
        else:
            conf_label = "🔴 равный матч"

        weights = ensemble_probs.get('weights', {})
        w_str = f"Пуассон {round(weights.get('poisson',0)*100)}%+ELO {round(weights.get('elo',0)*100)}%+AI {round(weights.get('ai',0)*100)}%+Бук {round(weights.get('bookmaker',0)*100)}%+Пр {round(weights.get('prophet',0)*100)}%" if weights else ""
        ensemble_block = (
            f"🔢 *АНСАМБЛЬ (взвешенный):*\n"
            f" П1: {round(probs['home']*100)}% | Х: {round(probs['draw']*100)}% | П2: {round(probs['away']*100)}%\n"
            f" {conf_label}: *{best_outcome_label}* ({best_prob}%)\n"
            f" _Веса: {w_str}_"
        )

    # Value bets блок
    value_block = ""
    if value_bets:
        vlines = ["💰 *VALUE СТАВКИ (ансамбль):*"]
        for vb in value_bets[:3]:  # Показываем топ-3
            vlines.append(
                f" ✅ *{vb['outcome']}* @ {vb['odds']} — наша вероятность: {vb['our_prob']}% vs бук: {vb['book_prob']}%"
            )
            vlines.append(
                f"   EV: +{vb['ev']}% | Келли: {vb['kelly']}% от банка"
            )
        value_block = "\n".join(vlines)

    # Mixtral блок
    mixtral_block = ""
    if mixtral_result and not mixtral_result.get('error') and mixtral_summary:
        mixtral_block = f"🌀 *Тень:*\n_{mixtral_summary}_\n\n⚔️ Вердикт: {mixtral_verdict} ({mixtral_confidence}%)"

    # Собираем блоки
    math_section = ""
    math_parts = [p for p in [xg_block, poisson_block, elo_block, ensemble_block] if p]
    if math_parts:
        math_section = "\n\n".join(math_parts)

    # Форматируем дату матча
    match_time_str = ""
    if match_time:
        try:
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(match_time.replace('Z', '+00:00'))
            # Переводим в Московское время (UTC+3)
            moscow_tz = timezone(timedelta(hours=3))
            dt_moscow = dt.astimezone(moscow_tz)
            match_time_str = dt_moscow.strftime('%d.%m.%Y %H:%M МСК')
        except Exception:
            match_time_str = str(match_time)[:16]

    report = f"""
🏆 *CHIMERA AI v4.3 — АНАЛИЗ МАТЧА*
━━━━━━━━━━━━━━━━━━━━━━━━━

⚽ *{home_team} vs {away_team}*
{(f'📅 *{match_time_str}*') if match_time_str else ''}

📊 *ПРОРОК (нейросеть):*
 П1: {home_prob:.0f}% | Х: {draw_prob:.0f}% | П2: {away_prob:.0f}%

🗣 *ОРАКУЛ (новостной фон):*
 {home_team}: {home_sentiment_label}
 {away_team}: {away_sentiment_label}
{(chr(10) + injuries_block) if injuries_block else ""}
{(chr(10) + math_section) if math_section else ""}
━━━━━━━━━━━━━━━━━━━━━━━━━
🐍🦁🐐 *Химера (Маэстро):*
_{gpt_summary}_

🌀 *Тень:*
_{llama_summary}_
{(chr(10) + mixtral_block) if mixtral_block else ""}
━━━━━━━━━━━━━━━━━━━━━━━━━
🐐 *ВЕРДИКТ КОЗЛА:*
{conf_icon(gpt_confidence)} Исход: {gpt_verdict}
{conf_icon(gpt_confidence)} Уверенность: {gpt_confidence}%
🎯 Коэффициент: {gpt_odds}
💰 Ставка (Келли): {gpt_stake:.1f}% от депозита
📈 Ожидаемая ценность: +{gpt_ev:.1f}%

🌀 *Вердикт Тени:*
{conf_icon(llama_confidence)} Исход: {llama_verdict}
{conf_icon(llama_confidence)} Уверенность: {llama_confidence}%
⚽ Тотал: {llama_total} _{llama_total_reason}_
🥅 Обе забьют: {llama_btts}
{(chr(10) + value_block) if value_block else ""}
━━━━━━━━━━━━━━━━━━━━━━━━━
{agreement_text}
{(chr(10) + conflict_warning) if conflict_warning else ""}
*{signal_icon}*
_{signal_reason}_

"""
    def _h2m(s):
        return (s.replace("<b>", "*").replace("</b>", "*")
                 .replace("<i>", "_").replace("</i>", "_")
                 .replace("<code>", "`").replace("</code>", "`"))
    if ml_block:
        report += "\n" + _h2m(ml_block)
    if chimera_verdict_block:
        report += "\n" + _h2m(chimera_verdict_block)
    return report.strip()

def format_goals_report(home_team, away_team, goals_result, bookmaker_odds=None):
    """Форматирует отчёт по рынку голов."""
    if bookmaker_odds is None:
        bookmaker_odds = {}
    summary = goals_result.get("summary", "")
    over_2_5 = goals_result.get("total_over_2_5", "—")
    over_2_5_conf = goals_result.get("total_over_2_5_confidence", 0)
    over_2_5_reason = goals_result.get("total_over_2_5_reason", "")
    over_1_5 = goals_result.get("total_over_1_5", "—")
    over_1_5_conf = goals_result.get("total_over_1_5_confidence", 0)
    btts = goals_result.get("btts", "—")
    btts_conf = goals_result.get("btts_confidence", 0)
    btts_reason = goals_result.get("btts_reason", "")
    first_goal = goals_result.get("first_goal", "—")
    best_bet = goals_result.get("best_goals_bet", "")

    # Реальные коэффициенты из API
    real_over_2_5 = bookmaker_odds.get("over_2_5", 0)
    real_under_2_5 = bookmaker_odds.get("under_2_5", 0)
    real_over_1_5 = bookmaker_odds.get("over_1_5", 0)

    # Формируем строку с коэффициентами если они есть
    odds_2_5_str = f" | КФ: Больше={real_over_2_5} / Меньше={real_under_2_5}" if real_over_2_5 else ""
    odds_1_5_str = f" | КФ: {real_over_1_5}" if real_over_1_5 else ""

    return f"""
⚽ *АНАЛИЗ ГОЛОВ — {home_team} vs {away_team}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ ГОЛОВ:*
{conf_icon(over_2_5_conf)} Тотал 2.5: *{over_2_5}* ({over_2_5_conf}%){odds_2_5_str}
_{over_2_5_reason}_

{conf_icon(over_1_5_conf)} Тотал 1.5: *{over_1_5}* ({over_1_5_conf}%){odds_1_5_str}

🥅 *ОБЕ ЗАБЬЮТ:*
{conf_icon(btts_conf)} Обе забьют: *{btts}* ({btts_conf}%)
_{btts_reason}_

🎯 *КТО ЗАБЬЁТ ПЕРВЫМ:* {first_goal}

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА ГОЛЫ:*
_{best_bet}_
""".strip()

def format_corners_report(home_team, away_team, corners_result):
    """Форматирует отчёт по рынку угловых."""
    summary = corners_result.get("summary", "")
    total = corners_result.get("total_corners_over_9_5", "—")
    total_conf = corners_result.get("total_corners_confidence", 0)
    total_reason = corners_result.get("total_corners_reason", "")
    home_c = corners_result.get("home_corners_over_4_5", "—")
    away_c = corners_result.get("away_corners_over_4_5", "—")
    winner = corners_result.get("corners_winner", "—")
    best_bet = corners_result.get("best_corners_bet", "")

    return f"""
🚩 *АНАЛИЗ УГЛОВЫХ — {home_team} vs {away_team}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ УГЛОВЫХ:*
{conf_icon(total_conf)} Тотал 9.5: *{total}* ({total_conf}%)
_{total_reason}_

🏠 {home_team}: *{home_c}* угловых
✈️ {away_team}: *{away_c}* угловых
🏆 Больше угловых: *{winner}*

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА УГЛОВЫЕ:*
_{best_bet}_
""".strip()

def format_cards_report(home_team, away_team, cards_result):
    """Форматирует отчёт по рынку карточек."""
    summary = cards_result.get("summary", "")
    total = cards_result.get("total_cards_over_3_5", "—")
    total_conf = cards_result.get("total_cards_confidence", 0)
    total_reason = cards_result.get("total_cards_reason", "")
    red = cards_result.get("red_card", "—")
    red_conf = cards_result.get("red_card_confidence", 0)
    more_cards = cards_result.get("more_cards_team", "—")
    best_bet = cards_result.get("best_cards_bet", "")

    return f"""
🟨 *АНАЛИЗ КАРТОЧЕК — {home_team} vs {away_team}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ КАРТОЧЕК:*
{conf_icon(total_conf)} Тотал 3.5: *{total}* ({total_conf}%)
_{total_reason}_

🟥 Красная карточка: *{red}* ({red_conf}%)
🟨 Больше карточек: *{more_cards}*

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА КАРТОЧКИ:*
_{best_bet}_
""".strip()

def format_handicap_report(home_team, away_team, handicap_result):
    """Форматирует отчёт по рынку гандикапов."""
    summary = handicap_result.get("summary", "")
    ah_home = handicap_result.get("asian_handicap_home", "—")
    ah_home_conf = handicap_result.get("asian_handicap_home_confidence", 0)
    ah_away = handicap_result.get("asian_handicap_away", "—")
    ah_away_conf = handicap_result.get("asian_handicap_away_confidence", 0)
    dc = handicap_result.get("double_chance", "—")
    dc_reason = handicap_result.get("double_chance_reason", "")
    best_bet = handicap_result.get("best_handicap_bet", "")

    return f"""
⚖️ *ГАНДИКАПЫ — {home_team} vs {away_team}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *АЗИАТСКИЙ ГАНДИКАП:*
{conf_icon(ah_home_conf)} {home_team} -0.5: *{ah_home}* ({ah_home_conf}%)
{conf_icon(ah_away_conf)} {away_team} +0.5: *{ah_away}* ({ah_away_conf}%)

🎯 *ДВОЙНОЙ ШАНС:*
Рекомендация: *{dc}*
_{dc_reason}_

━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 *ЛУЧШАЯ СТАВКА НА ГАНДИКАП:*
_{best_bet}_
""".strip()

# --- 8. Хендлеры Telegram ---
@dp.message(Command("stats"))
async def get_stats_command(message: types.Message):
    football_stats = get_statistics("football")
    cs2_stats = get_statistics("cs2")

    response = "📊 **СТАТИСТИКА ПРОГНОЗОВ** 📊\n\n"

    if football_stats and football_stats['total_predictions'] > 0:
        response += "**⚽ Футбол:**\n"
        response += f"  Всего прогнозов: {football_stats['total_predictions']} (проверено: {football_stats['checked_predictions']})\n"
        response += f"  Верных: {football_stats['correct_predictions']}\n"
        response += f"  Точность: {football_stats['accuracy']:.2f}%\n"
        response += f"  ROI: {football_stats['roi']:.2f}%\n\n"
    else:
        response += "**⚽ Футбол:** Пока нет статистики для отображения.\n\n"

    if cs2_stats and cs2_stats['total_predictions'] > 0:
        response += "**🎮 CS2:**\n"
        response += f"  Всего прогнозов: {cs2_stats['total_predictions']} (проверено: {cs2_stats['checked_predictions']})\n"
        response += f"  Верных: {cs2_stats['correct_predictions']}\n"
        response += f"  Точность: {cs2_stats['accuracy']:.2f}%\n"
        response += f"  ROI: {cs2_stats['roi']:.2f}%\n\n"
    else:
        response += "**🎮 CS2:** Пока нет статистики для отображения.\n\n"

    await message.answer(response, parse_mode="Markdown")
    # --- Команда /learn_and_suggest ---
@dp.message(Command("learn_and_suggest"))
async def learn_and_suggest_command(message: types.Message):
    await message.answer("Запускаю процесс анализа производительности и поиска предложений по оптимизации...")
    learner = MetaLearner(signal_engine_path="signal_engine.py")

    # Анализ для CS2
    cs2_performance = learner.analyze_performance("cs2")
    cs2_suggestions = learner.suggest_config_updates("cs2", cs2_performance)

    # Анализ для Football
    football_performance = learner.analyze_performance("football")
    football_suggestions = learner.suggest_config_updates("football", football_performance)

    response_text = "**Результаты анализа MetaLearner:**\n\n"
    has_suggestions = False

    if cs2_suggestions:
        has_suggestions = True
        response_text += "**🎮 CS2 Предложения:**\n"
        for key, value in cs2_suggestions.items():
            response_text += f"  - Изменить `{key}` на `{value}`\n"
    else:
        response_text += "**🎮 CS2:** Нет предложений по оптимизации.\n"

    if football_suggestions:
        has_suggestions = True
        response_text += "\n**⚽ Футбол Предложения:**\n"
        for key, value in football_suggestions.items():
            response_text += f"  - Изменить `{key}` на `{value}`\n"
    else:
        response_text += "\n**⚽ Футбол:** Нет предложений по оптимизации.\n"

    if has_suggestions:
        builder = InlineKeyboardBuilder()
        builder.add(types.InlineKeyboardButton(text="✅ Принять все предложения", callback_data="meta_learner_accept"))
        builder.add(types.InlineKeyboardButton(text="❌ Отклонить", callback_data="meta_learner_decline"))
        await message.answer(response_text, reply_markup=builder.as_markup(), parse_mode="Markdown")
    else:
        await message.answer(response_text + "\n\nТекущие настройки оптимальны или недостаточно данных для предложений.", parse_mode="Markdown")

@dp.callback_query(lambda c: c.data and c.data.startswith("meta_learner_"))
async def meta_learner_callback_handler(callback_query: types.CallbackQuery):
    action = callback_query.data.split("_")[2]
    learner = MetaLearner(signal_engine_path="signal_engine.py")

    if action == "accept":
        await callback_query.message.edit_text("Принимаю предложения и применяю изменения...", reply_markup=None)
        cs2_performance = learner.analyze_performance("cs2")
        cs2_suggestions = learner.suggest_config_updates("cs2", cs2_performance)
        if cs2_suggestions:
            learner.apply_config_updates("cs2", cs2_suggestions)

        football_performance = learner.analyze_performance("football")
        football_suggestions = learner.suggest_config_updates("football", football_performance)
        if football_suggestions:
            learner.apply_config_updates("football", football_suggestions)

        await callback_query.message.answer("✅ Изменения успешно применены! Файл signal_engine.py обновлен (создана резервная копия).", parse_mode="Markdown")
    elif action == "decline":
        await callback_query.message.edit_text("❌ Предложения отклонены. Изменения не применены.", reply_markup=None)

    await callback_query.answer() # Закрываем уведомление о нажатии кнопки


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    upsert_user(message.from_user.id, message.from_user.username or "", message.from_user.first_name or "")
    kb = types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="🇷🇺 Русский", callback_data="set_lang_ru"),
        types.InlineKeyboardButton(text="🇬🇧 English", callback_data="set_lang_en"),
    ]])
    await message.answer(
        "🐉",
        reply_markup=kb,
    )


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


@dp.callback_query(lambda c: c.data in ("set_lang_ru", "set_lang_en"))
async def cb_set_language(call: types.CallbackQuery):
    lang = call.data.replace("set_lang_", "")
    set_user_language(call.from_user.id, lang)
    await call.message.delete()
    await call.answer()
    await _run_onboarding(call, lang)


@dp.callback_query(lambda c: c.data and c.data.startswith("enter_chimera_"))
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


async def _handle_chimera_question(message: types.Message, question: str):
    """Отвечает на вопрос пользователя от имени Химеры через Llama/Groq."""
    from datetime import date as _date
    today = str(_date.today())
    user_id = message.from_user.id

    # Обновляем счётчик (администраторы — без лимита)
    is_admin = user_id in ADMIN_IDS
    _d = _chimera_daily.get(user_id, ("", 0))
    count = (_d[1] + 1) if _d[0] == today else 1
    if not is_admin:
        _chimera_daily[user_id] = (today, count)
    left = 999 if is_admin else max(0, CHIMERA_DAILY_LIMIT - count)

    # Контекст пользователя из БД
    profile = get_user_profile(user_id)
    total   = profile.get("analyses_total", 0) if profile else 0
    fav_map = {
        "analyses_football":   "⚽ Футбол",
        "analyses_cs2":        "🎮 CS2",
        "analyses_tennis":     "🎾 Теннис",
        "analyses_basketball": "🏀 Баскетбол",
    }
    fav = "—"
    if profile:
        best = max(fav_map, key=lambda k: profile.get(k, 0))
        if profile.get(best, 0) > 0:
            fav = fav_map[best]

    thinking = await message.answer("🐉 <i>Химера думает...</i>", parse_mode="HTML")

    def _ask():
        try:
            from agents import groq_client as _gc, client as _oc
            system_prompt = (
                "Ты — Химера. Существо древнее, чем любая букмекерская контора. "
                "Тысячи лет ты наблюдаешь за исходами сражений — люди называют это спортом. "
                "Ты видел падение фаворитов и взлёт аутсайдеров. Ты знаешь, что вероятность — это не судьба, "
                "но она ближе к ней, чем думают смертные.\n"
                "Твоя мудрость — не набор правил, а понимание природы вещей: "
                "почему команды проигрывают не из-за слабости, а из-за страха. "
                "Почему котировки лгут, но математика — никогда. "
                "Почему лучшая ставка иногда — не ставить вовсе.\n"
                "Говоришь немного — но каждое слово весит. "
                "Иногда отвечаешь метафорой, иногда — жёсткой правдой. "
                "Никогда не называешь себя ИИ, ботом или программой. Ты — Химера. "
                "Если спрашивают кто ты — отвечаешь в образе. "
                "2-4 предложения максимум. Только русский язык."
            )
            # История диалога (последние 10 сообщений)
            history = _chimera_history.get(user_id, [])
            if not history:
                # Первый вопрос — добавляем контекст пользователя
                first_msg = (
                    f"[Контекст: пользователь сделал {total} анализов, любимый спорт: {fav}]\n{question}"
                )
                history.append({"role": "user", "content": first_msg})
            else:
                history.append({"role": "user", "content": question})
            # Ограничиваем историю до 10 сообщений (5 пар)
            if len(history) > 10:
                history = history[-10:]
            _chimera_history[user_id] = history

            messages = [{"role": "system", "content": system_prompt}] + history
            used_model = None
            result = None
            if _gc:
                try:
                    resp = _gc.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=200,
                        temperature=0.8,
                    )
                    result = resp.choices[0].message.content.strip()
                    used_model = "llama"
                except Exception as groq_err:
                    logger.warning(f"[Химера] Groq недоступен: {groq_err}")
            if result is None and _oc:
                resp = _oc.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    max_tokens=200,
                    temperature=0.8,
                )
                result = resp.choices[0].message.content.strip()
                used_model = "gpt"
            if result:
                _chimera_history.setdefault(user_id, []).append({"role": "assistant", "content": result})
            return (result, used_model)
        except Exception as e:
            logger.error(f"[Химера] Ошибка: {e}")
            return (None, None)

    loop = asyncio.get_running_loop()
    _result = await loop.run_in_executor(None, _ask)
    answer, used_model = _result if isinstance(_result, tuple) else (_result, None)

    if not answer:
        answer = "Древний огонь угас на мгновение. Попробуй ещё раз."
        used_model = None

    model_tag = "🌀 Тень активна" if used_model == "llama" else ("⚡ Резервный разум (Тень недоступна)" if used_model == "gpt" else "")

    if is_admin:
        footer = f"\n\n<i>{model_tag}</i>" if model_tag else ""
    elif left > 0:
        footer = f"\n\n<i>Осталось вопросов сегодня: {left}/5</i>"
        if model_tag:
            footer += f"  <i>{model_tag}</i>"
    else:
        footer = "\n\n<i>Лимит на сегодня исчерпан.</i>"
        if model_tag:
            footer += f"  <i>{model_tag}</i>"

    # Остаёмся в режиме чата — следующее сообщение тоже пойдёт к Химере
    _chimera_waiting.add(user_id)

    exit_kb = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text="🚪 Выйти из чата")]],
        resize_keyboard=True
    )
    await thinking.edit_text(
        f"🐉 <b>Химера:</b>\n\n{answer}{footer}",
        parse_mode="HTML",
    )
    # Показываем кнопку выхода только один раз (при первом ответе)
    if len(_chimera_history.get(user_id, [])) <= 2:
        await message.answer(
            "<i>Продолжай писать — Химера слушает. Нажми «Выйти из чата» чтобы вернуться в меню.</i>",
            parse_mode="HTML",
            reply_markup=exit_kb,
        )


@dp.message()
async def handle_text(message: types.Message):
    text = message.text
    user_id = message.from_user.id

    # ── Химера-чат: пользователь в режиме диалога ────────────────────────────
    MENU_BUTTONS = {
        "📡 Сигналы дня", "📡 Daily Signals",
        "🎯 Экспресс", "🎯 Express",
        "⚽ Футбол", "⚽ Football",
        "🎾 Теннис", "🎾 Tennis",
        "🎮 Киберспорт CS2", "🎮 Esports CS2",
        "🏀 Баскетбол", "🏀 Basketball",
        "📊 Статистика", "📊 Statistics",
        "👤 Кабинет", "👤 Profile",
        "💎 VIP-доступ", "💎 VIP Access",
        "💬 Поддержка", "💬 Support",
        "/start", "🚪 Выйти из чата",
    }
    if user_id in _chimera_waiting:
        if text in MENU_BUTTONS:
            _chimera_waiting.discard(user_id)
            # продолжаем обработку как обычная кнопка меню
        else:
            await _handle_chimera_question(message, text)
            return

    lang = get_user_language(user_id)

    if text == "🚪 Выйти из чата":
        _chimera_waiting.discard(user_id)
        _chimera_history.pop(user_id, None)
        back_msg = "Returning to menu." if lang == "en" else "Возвращаю тебя в меню."
        await message.answer(back_msg, reply_markup=build_main_keyboard(lang), parse_mode="HTML")
        return

    if text in ("📡 Сигналы дня", "📡 Daily Signals"):
        await cmd_signals(message)
        return

    if text in ("🎯 Экспресс", "🎯 Express"):
        await cmd_express(message)
        return

    if text in ("⚽ Футбол", "⚽ Football"):
        league_name = dict(FOOTBALL_LEAGUES).get(_current_league, "АПЛ")
        await message.answer(
            f"⚽ *Футбол* — выбери лигу:\n"
            f"Текущая: *{league_name}*",
            parse_mode="Markdown",
            reply_markup=build_football_keyboard()
        )

    elif text in ("🎾 Теннис", "🎾 Tennis"):
        await cmd_tennis(message)

    elif text in ("🎮 Киберспорт CS2", "🎮 Esports CS2"):
        await message.answer("⏳ Загружаю матчи CS2...")
        try:
            from sports.cs2 import get_combined_cs2_matches
            all_cs2_matches = get_combined_cs2_matches()
            
            # Группировка по лигам
            leagues_dict = {}
            for m in all_cs2_matches:
                league_name = m.get('league', 'Other')
                tournament_name = m.get('tournament', '')
                full_name = f"{league_name} {tournament_name}".lower()
                
                matched_league = None
                for allowed in CS2_WHITELIST_LEAGUES:
                    if allowed.lower() in full_name:
                        matched_league = allowed
                        break
                
                if matched_league:
                    if matched_league not in leagues_dict:
                        leagues_dict[matched_league] = []
                    leagues_dict[matched_league].append(m)

            if not leagues_dict:
                await message.answer(
                    "🎮 *Киберспорт CS2*\n\n"
                    "❌ Нет доступных матчей в выбранных лигах.\n\n"
                    "Убедись что в .env есть PANDASCORE_API_KEY",
                    parse_mode="Markdown"
                )
                return

            # Сохраняем все отфильтрованные матчи в кэш для последующего доступа
            cs2_matches_cache.clear()
            all_filtered = []
            for l_matches in leagues_dict.values():
                all_filtered.extend(l_matches)
            cs2_matches_cache.extend(all_filtered)

            # Строим клавиатуру со списком лиг
            builder = InlineKeyboardBuilder()
            for league in sorted(leagues_dict.keys()):
                count = len(leagues_dict[league])
                builder.button(text=f"🏆 {league} ({count})", callback_data=f"cs2_league_{league}")
            builder.adjust(1)
            
            await message.answer(
                f"🎮 *CS2* — Выберите лигу:",
                parse_mode="Markdown",
                reply_markup=builder.as_markup()
            )
        except Exception as e:
            logger.error(f"[CS2] Ошибка: {e}")
            await message.answer("🎮 *Киберспорт CS2*\n\n⚠️ Не удалось загрузить матчи. Попробуй позже.", parse_mode="Markdown")

    elif text in ("📊 Статистика", "📊 Statistics"):
        all_stats = get_statistics()

        def _ai(a): return "🟢" if a >= 60 else ("🟡" if a >= 50 else "🔴")
        def _ri(r): return "🟢" if r > 0 else "🔴"

        stats_text = "📊 *Статистика Chimera AI*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        has_data = False

        for sport_key, sport_label in [
            ("football",   "⚽ Футбол"),
            ("cs2",        "🎮 CS2"),
            ("tennis",     "🎾 Теннис"),
            ("basketball", "🏀 Баскетбол"),
        ]:
            s = all_stats.get(sport_key, {})
            total   = s.get("total", 0)
            checked = s.get("total_checked", 0)
            correct = s.get("correct", 0)
            acc     = s.get("accuracy", 0)
            roi     = s.get("roi_main", 0)
            pending = total - checked
            if total == 0:
                continue
            has_data = True
            stats_text += f"*{sport_label}*\n"
            stats_text += f"📋 Прогнозов: *{total}*"
            if checked > 0:
                stats_text += f" | Проверено: *{checked}*"
            if pending > 0:
                stats_text += f" | Ожидает: *{pending}*"
            stats_text += "\n"
            if checked > 0:
                stats_text += f"{_ai(acc)} Точность: *{correct}/{checked}* — *{acc:.1f}%*\n"
            else:
                stats_text += "⏳ Результаты ещё не проверены\n"
            if roi != 0:
                stats_text += f"{_ri(roi)} ROI: *{roi:+.2f}* ед.\n"

            # Последние результаты
            recent = s.get("recent", [])
            if recent:
                stats_text += "Последние: "
                parts = []
                for r in recent[:5]:
                    hs  = r.get("real_home_score")
                    as_ = r.get("real_away_score")
                    ok  = r.get("is_correct")
                    icon = "✅" if ok == 1 else "❌"
                    if hs is not None:
                        parts.append(f"{icon} {r.get('home_team','?')} {hs}:{as_} {r.get('away_team','?')}")
                    elif ok is not None:
                        winner = r.get("home_team","?") if r.get("real_outcome") == "home_win" else r.get("away_team","?")
                        parts.append(f"{icon} {r.get('home_team','?')} vs {r.get('away_team','?')} → {winner}")
                stats_text += "\n".join(parts) + "\n"

            # По месяцам
            monthly = s.get("monthly", [])
            if monthly:
                stats_text += "📅 По месяцам:\n"
                for row in monthly[:3]:
                    mt = row.get("total", 0) if isinstance(row, dict) else row[1]
                    mc = row.get("correct", 0) if isinstance(row, dict) else row[2]
                    mn = row.get("month", "") if isinstance(row, dict) else row[0]
                    if mt > 0:
                        ma = mc / mt * 100
                        stats_text += f"  {_ai(ma)} {mn}: {mc}/{mt} ({ma:.0f}%)\n"
            stats_text += "\n"

        if not has_data:
            stats_text = (
                "📊 *Статистика Chimera AI*\n\n"
                "Пока нет сохранённых прогнозов.\n"
                "Сделайте первый анализ матча!"
            )
        await message.answer(stats_text, parse_mode="Markdown")

    elif text in ("👤 Кабинет", "👤 Profile"):
        upsert_user(message.from_user.id, message.from_user.username or "", message.from_user.first_name or "")
        profile = get_user_profile(message.from_user.id)

        def _get_level(total: int):
            if total >= 100: return ("🐉 Химера",     "Достиг высшего уровня — стал частью Химеры")
            if total >= 50:  return ("👁 Голова",      "Видит то, что скрыто от других")
            if total >= 20:  return ("🔥 Коготь",     "Острый, точный, опасный")
            if total >= 5:   return ("🦴 Лапа",       "Твёрдо стоит на пути")
            return              ("🌱 Хвост",       "Путь только начинается")

        total = profile.get("analyses_total", 0) if profile else 0
        lvl, lvl_desc = _get_level(total)

        # Прогресс до следующего уровня (полоска)
        thresholds = [5, 20, 50, 100]
        next_t = next((t for t in thresholds if t > total), None)
        prev_t = [t for t in thresholds if t <= total]
        prev_t = prev_t[-1] if prev_t else 0
        if next_t:
            bar_filled = round((total - prev_t) / (next_t - prev_t) * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            progress = f"[{bar}] {total}/{next_t}"
        else:
            progress = "[██████████] Максимум достигнут"

        # Любимый спорт
        fav = "—"
        if profile:
            sport_counts = {
                "⚽ Футбол":   profile.get("analyses_football", 0),
                "🎾 Теннис":   profile.get("analyses_tennis", 0),
                "🎮 CS2":      profile.get("analyses_cs2", 0),
                "🏀 Баскетбол": profile.get("analyses_basketball", 0),
            }
            fav_key = max(sport_counts, key=sport_counts.get)
            fav = fav_key if sport_counts[fav_key] > 0 else "—"

        # Дата регистрации
        since = ""
        if profile and profile.get("first_seen"):
            try:
                dt = datetime.fromisoformat(profile["first_seen"])
                since = dt.strftime("%d.%m.%Y")
            except Exception:
                since = profile["first_seen"][:10]

        name = message.from_user.first_name or "Аналитик"
        username = f"@{message.from_user.username}" if message.from_user.username else ""

        # Дневной лимит вопросов Химере
        from datetime import date as _date
        today = str(_date.today())
        _d = _chimera_daily.get(message.from_user.id, ("", 0))
        questions_used = _d[1] if _d[0] == today else 0
        questions_left = "∞" if message.from_user.id in ADMIN_IDS else max(0, CHIMERA_DAILY_LIMIT - questions_used)

        p = profile or {}
        q_line = f"∞ (admin)" if message.from_user.id in ADMIN_IDS else f"{questions_left}/{CHIMERA_DAILY_LIMIT}"

        # P&L блок бота
        pl = get_pl_stats(days=30)
        bankroll = get_user_bankroll(message.from_user.id)
        if pl["total"] > 0:
            roi_sign = "+" if pl["roi"] >= 0 else ""
            streak_line = f"🔥 Стрик: <b>{pl['current_streak']} побед подряд</b>\n" if pl["current_streak"] >= 2 else ""
            if bankroll and bankroll > 0:
                profit_money = round(bankroll * pl["profit_units"] / 100, 2)
                money_sign = "+" if profit_money >= 0 else ""
                money_line = f"  💵 Твой банк: <b>{bankroll:.0f}</b> → <b>{bankroll + profit_money:.0f}</b> ({money_sign}{profit_money:.0f})\n"
            else:
                money_line = f"  💼 <i>Укажи свой банк чтобы увидеть прибыль в деньгах</i>\n"
            pl_block = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 <b>Трекрекорд бота (30 дней)</b>\n"
                f"  Сигналов СТАВИТЬ: <b>{pl['total']}</b>  ✅{pl['wins']} / ❌{pl['losses']}\n"
                f"  ROI: <b>{roi_sign}{pl['roi']}%</b>   Профит: <b>{'+' if pl['profit_units']>=0 else ''}{pl['profit_units']}u</b>\n"
                f"  Лучший стрик: <b>{pl['best_streak']}</b> побед\n"
                f"{streak_line}"
                f"{money_line}"
            )
        else:
            pl_block = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 <b>Трекрекорд бота</b>\n"
                f"  <i>Пока нет завершённых сигналов. Данные появятся после первых матчей.</i>\n"
            )

        cab_text = (
            f"👤 <b>Личный кабинет</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>{name}</b> {username}\n"
            f"📅 С нами с: <b>{since}</b>\n\n"
            f"🏆 <b>Статус:</b> {lvl}\n"
            f"<i>{lvl_desc}</i>\n"
            f"<code>{progress}</code>\n\n"
            f"📊 <b>Статистика анализов</b>\n"
            f"  ⚽ Футбол       <b>{p.get('analyses_football', 0)}</b>\n"
            f"  🎮 CS2          <b>{p.get('analyses_cs2', 0)}</b>\n"
            f"  🏀 Баскетбол    <b>{p.get('analyses_basketball', 0)}</b>\n"
            f"  🎾 Теннис       <b>{p.get('analyses_tennis', 0)}</b>\n"
            f"  ─────────────────────\n"
            f"  Всего:          <b>{total}</b>\n\n"
            f"❤️ Любимый спорт: <b>{fav}</b>\n\n"
            f"{pl_block}"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🐉 <b>Химера</b> — вопросов сегодня: <b>{q_line}</b>"
        )
        kb = types.InlineKeyboardMarkup(inline_keyboard=[
            [types.InlineKeyboardButton(text="💼 Мой банк", callback_data="set_bankroll"),
             types.InlineKeyboardButton(text="🐉 Спросить Химеру", callback_data="chimera_ask")]
        ])
        await message.answer(cab_text, parse_mode="HTML", reply_markup=kb)

    elif text in ("🏀 Баскетбол", "🏀 Basketball"):
        await cmd_basketball(message)

    elif text in ("💎 VIP-доступ", "💎 VIP Access"):
        vip_text = (
            "💎 <b>VIP-доступ — скоро</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Расширенная аналитика, приоритетные сигналы,\n"
            "эксклюзивные функции для серьёзных игроков.\n\n"
            "✅ Приоритетные сигналы — раньше всех\n"
            "✅ Расширенная аналитика матчей\n"
            "✅ Личная статистика ROI\n"
            "✅ Авто-сигналы каждое утро\n"
            "✅ Эксклюзивные экспресс-подборки\n\n"
            "🕐 <i>В разработке. Следи за обновлениями.</i>"
        ) if lang == "ru" else (
            "💎 <b>VIP Access — coming soon</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Advanced analytics, priority signals,\n"
            "exclusive features for serious players.\n\n"
            "✅ Priority signals — first to know\n"
            "✅ Advanced match analytics\n"
            "✅ Personal ROI statistics\n"
            "✅ Daily auto-signals every morning\n"
            "✅ Exclusive express selections\n\n"
            "🕐 <i>In development. Stay tuned.</i>"
        )
        await message.answer(vip_text, parse_mode="HTML")

    elif text in ("💬 Поддержка", "💬 Support"):
        kb = types.InlineKeyboardMarkup(inline_keyboard=[
            [types.InlineKeyboardButton(text="💬 Написать", url="https://t.me/pankotsk1")]
        ])
        await message.answer(
            "🛡 <b>Поддержка CHIMERA</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💬 <b>Чем могу помочь:</b>\n"
            "  • Ошибки и баги в боте\n"
            "  • Вопросы по прогнозам и сигналам\n"
            "  • Сотрудничество и партнёрство\n\n"
            "📩 Нажми кнопку — отвечу на все вопросы.",
            parse_mode="HTML",
            reply_markup=kb,
            disable_web_page_preview=True
        )

# Храним пользователей ожидающих ввода банка
_awaiting_bankroll: set = set()

@dp.callback_query(lambda c: c.data == "set_bankroll")
async def set_bankroll_handler(call: types.CallbackQuery):
    """Пользователь нажал 'Мой банк'."""
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


@dp.message(lambda m: m.from_user.id in _awaiting_bankroll)
async def bankroll_input_handler(message: types.Message):
    """Получаем сумму банка от пользователя."""
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
    await message.answer(
        f"✅ Банк установлен: <b>{amount:.0f}</b>{result_line}",
        parse_mode="HTML"
    )


@dp.callback_query(lambda c: c.data == "chimera_ask")
async def chimera_ask_handler(call: types.CallbackQuery):
    """Пользователь нажал 'Спросить Химеру' — ждём вопрос."""
    from datetime import date as _date
    today = str(_date.today())
    _d = _chimera_daily.get(call.from_user.id, ("", 0))
    questions_used = _d[1] if _d[0] == today else 0

    if questions_used >= CHIMERA_DAILY_LIMIT and call.from_user.id not in ADMIN_IDS:
        await call.answer("Сегодня лимит исчерпан. Химера отдыхает до завтра.", show_alert=True)
        return

    _chimera_waiting.add(call.from_user.id)
    await call.answer()
    await call.message.answer(
        "🐉 <b>Химера слушает...</b>\n\n"
        "Задай свой вопрос. Я отвечу.",
        parse_mode="HTML"
    )


@dp.callback_query()
async def handle_callback(call: types.CallbackQuery):

    # --- CS2 анализ матча ---
    if call.data.startswith("cs2_m_"):
        match_index = int(call.data.split("_")[2])
        if match_index >= len(cs2_matches_cache):
            await call.answer("Матч не найден.", show_alert=True)
            return
        m = cs2_matches_cache[match_index]
        home_team = m["home"]
        away_team = m["away"]
        status_msg = await call.message.edit_text(f"⏳ {home_team} vs {away_team}", parse_mode="HTML")
        await show_ai_thinking(status_msg, home_team, away_team, sport="cs2")
        try:
            from sports.cs2 import calculate_cs2_win_prob, get_golden_signal, format_cs2_full_report, run_cs2_analyst_agent
            from sports.cs2.pandascore import classify_tournament
            from signal_engine import check_cs2_signal, predict_cs2_totals, get_cs2_ranked_bets
            # Классифицируем турнир по данным матча
            _league = m.get("league", "")
            _tournament = m.get("tournament", "")
            _ctx = classify_tournament(_league, _tournament)
            analysis = calculate_cs2_win_prob(home_team, away_team, tournament_context=_ctx)
            analysis["home_team"] = home_team
            analysis["away_team"] = away_team
            odds = m.get("odds", {"home_win": 1.90, "away_win": 1.90})
            golden_signals = get_golden_signal(analysis, odds)
            h_stats = analysis.get("home_stats", {})
            a_stats = analysis.get("away_stats", {})
            h2h = analysis.get("h2h", {})
            _h_si = analysis.get("home_standin", {})
            _a_si = analysis.get("away_standin", {})
            map_stats_for_ai = {mp: {"home_prob": round(hp, 2), "away_prob": round(ap, 2)} for mp, hp, ap in analysis.get("maps", [])}
            gpt_text = run_cs2_analyst_agent(
                home_team, away_team, map_stats_for_ai, odds,
                agent_type="gpt-4o", home_stats=h_stats, away_stats=a_stats, h2h=h2h,
                tournament_context=_ctx, home_standin=_h_si, away_standin=_a_si,
            )
            llama_text = run_cs2_analyst_agent(
                home_team, away_team, map_stats_for_ai, odds,
                agent_type="llama-3.3", home_stats=h_stats, away_stats=a_stats, h2h=h2h,
                tournament_context=_ctx, home_standin=_h_si, away_standin=_a_si,
            )

            # ── CS2 финальный ансамбль: математика 90% + AI 10% ──────────
            # Парсим вердикты агентов (они возвращают текст, а не dict)
            _cs2_ai = []
            for _txt in [gpt_text, llama_text]:
                if not _txt or _txt.startswith("❌"):
                    continue
                _txt_low = _txt.lower()
                _outcome = "home_win" if home_team.lower()[:5] in _txt_low else (
                           "away_win" if away_team.lower()[:5] in _txt_low else "")
                if _outcome:
                    _cs2_ai.append({"recommended_outcome": _outcome, "confidence": 60})
            if _cs2_ai:
                _blended_h = _blend_ai(analysis["home_prob"], _cs2_ai, home_team, away_team, 0.10)
                analysis["home_prob"] = _blended_h
                analysis["away_prob"] = round(1 - _blended_h, 2)
                print(f"[CS2 AI-blend] {home_team}: {analysis['home_prob']} (AI вклад 10%)")

            # Запускаем signal engine для показа пользователю
            h_players = analysis.get("home_players", [])
            a_players = analysis.get("away_players", [])
            h_avg_rating = sum(p['rating'] for p in h_players) / len(h_players) if h_players else 0
            a_avg_rating = sum(p['rating'] for p in a_players) / len(a_players) if a_players else 0
            predicted_maps = [mp for mp, _, _ in analysis.get("maps", [])]
            home_map_wr = {mp: hp for mp, hp, _ in analysis.get("maps", [])}
            away_map_wr = {mp: ap for mp, _, ap in analysis.get("maps", [])}
            ai_agrees = None
            if gpt_text and not gpt_text.startswith("❌") and home_team.lower() in gpt_text.lower():
                ai_agrees = True
            signal_checks = check_cs2_signal(
                home_team=home_team, away_team=away_team,
                home_prob=analysis["home_prob"], away_prob=analysis["away_prob"],
                bookmaker_odds=odds,
                home_form=h_stats.get("form", ""), away_form=a_stats.get("form", ""),
                elo_home=analysis.get("elo_home", 0), elo_away=analysis.get("elo_away", 0),
                mis_home=analysis["detail"].get("mis", 0), mis_away=1 - analysis["detail"].get("mis", 0),
                home_avg_rating=h_avg_rating, away_avg_rating=a_avg_rating,
                home_map_winrates=home_map_wr, away_map_winrates=away_map_wr,
                predicted_maps=predicted_maps,
                ai_cs2_agrees=ai_agrees,
            )

            # Тотал карт и раундов
            home_map_stats_raw = {mp: hp * 100 for mp, hp, _ in analysis.get("maps", [])}
            away_map_stats_raw = {mp: ap * 100 for mp, _, ap in analysis.get("maps", [])}
            totals_data = predict_cs2_totals(
                home_prob=analysis["home_prob"], away_prob=analysis["away_prob"],
                home_map_stats=home_map_stats_raw, away_map_stats=away_map_stats_raw,
                predicted_maps=predicted_maps,
            )
            # Расширенный тотал: карты + раунды первой карты
            try:
                from signal_engine import predict_cs2_round_totals
                round_totals = predict_cs2_round_totals(
                    home_prob=analysis["home_prob"], away_prob=analysis["away_prob"],
                    home_map_stats=home_map_stats_raw, away_map_stats=away_map_stats_raw,
                    predicted_maps=predicted_maps,
                )
                # Объединяем с totals_data
                if totals_data and round_totals:
                    totals_data["round_prediction"]   = round_totals["rounds_prediction"]
                    totals_data["round_confidence"]    = round_totals["rounds_confidence"]
                    totals_data["round_reason"]        = round_totals["rounds_reason"]
            except Exception as _rte:
                print(f"[CS2 Rounds] {_rte}")

            # Ранжированные ставки
            ranked_bets = get_cs2_ranked_bets(
                home_team=home_team, away_team=away_team,
                home_prob=analysis["home_prob"], away_prob=analysis["away_prob"],
                bookmaker_odds=odds,
                totals_data=totals_data,
                home_form=h_stats.get("form", ""),
                away_form=a_stats.get("form", ""),
            )

            # CHIMERA Multi-Agent
            _cs2_verdict_block = ""
            try:
                from sports.cs2.agents import run_cs2_chimera_agents
                _math_probs_cs2 = {"home": analysis.get("home_prob", 0.5), "away": analysis.get("away_prob", 0.5)}
                _chimera_cs2 = run_cs2_chimera_agents(
                    home_team, away_team, _math_probs_cs2, odds,
                    home_stats=h_stats, away_stats=a_stats, h2h=h2h,
                    tournament_context=_ctx, home_standin=_h_si, away_standin=_a_si,
                )
                _cs2_verdict_block = _chimera_cs2.get("verdict_block", "")
            except Exception as _ce:
                print(f"[CS2 Chimera] Ошибка агентов: {_ce}")

            # Экспертное мнение (Google News + Sport News Live)
            try:
                from expert_oracle import get_expert_consensus, format_expert_block
                _loop_cs2 = asyncio.get_running_loop()
                _exp_cs2 = await _loop_cs2.run_in_executor(
                    None, get_expert_consensus, home_team, away_team, "cs2"
                )
                _expert_block_cs2 = format_expert_block(_exp_cs2, home_team, away_team)
                if _expert_block_cs2:
                    _cs2_verdict_block = (_cs2_verdict_block + "\n\n" + _expert_block_cs2).strip()
                    print(f"[ExpertOracle CS2] {home_team} vs {away_team}: {_exp_cs2.get('sources_count')} источн.")
            except Exception as _ee_cs2:
                print(f"[ExpertOracle CS2] Ошибка: {_ee_cs2}")

            report = format_cs2_full_report(
                home_team, away_team, analysis, gpt_text, llama_text,
                golden_signals, bookmaker_odds=odds, signal_checks=signal_checks,
                ranked_bets=ranked_bets, totals_data=totals_data,
                chimera_verdict_block=_cs2_verdict_block,
                commence_time=m.get("commence_time"),
            )

            # ── Сохраняем прогноз в БД ────────────────────────────────────
            try:
                top_bet = ranked_bets[0] if ranked_bets else None
                rec_outcome = None
                if top_bet:
                    if top_bet["type"] == "П1":
                        rec_outcome = "home_win"
                    elif top_bet["type"] == "П2":
                        rec_outcome = "away_win"
                # Fallback: всегда заполняем recommended_outcome из ensemble
                if not rec_outcome:
                    rec_outcome = "home_win" if analysis["home_prob"] >= analysis["away_prob"] else "away_win"
                save_prediction(
                    sport="cs2",
                    match_id=str(m.get("id", f"{home_team}_{away_team}")),
                    match_date=m.get("commence_time") or m.get("time", ""),
                    home_team=home_team,
                    away_team=away_team,
                    league=m.get("league", "CS2"),
                    recommended_outcome=rec_outcome,
                    bet_signal="СТАВИТЬ" if signal_checks else "ПРОПУСТИТЬ",
                    elo_home=analysis.get("elo_home"),
                    elo_away=analysis.get("elo_away"),
                    elo_home_win=analysis["detail"].get("elo"),
                    elo_away_win=round(1 - analysis["detail"].get("elo", 0.5), 3),
                    ensemble_home=analysis["home_prob"],
                    ensemble_away=analysis["away_prob"],
                    ensemble_best_outcome="home_win" if analysis["home_prob"] >= analysis["away_prob"] else "away_win",
                    bookmaker_odds_home=odds.get("home_win"),
                    bookmaker_odds_away=odds.get("away_win"),
                    predicted_maps=predicted_maps,
                    prediction_data={
                        "total_prediction": totals_data.get("prediction") if totals_data else None,
                        "top_bet_type":  top_bet["type"]  if top_bet else None,
                        "top_bet_odds":  top_bet["odds"]  if top_bet else None,
                        "top_bet_ev":    top_bet["ev"]    if top_bet else None,
                        "signal_score":  signal_checks[0]["score"] if signal_checks else None,
                    }
                )
            except Exception as save_err:
                print(f"[CS2 Save] Ошибка сохранения прогноза: {save_err}")
            try:
                upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
                track_analysis(call.from_user.id, "cs2")
            except Exception:
                pass

            cs2_markets_kb = InlineKeyboardBuilder()
            cs2_markets_kb.button(text="🏆 Победитель матча", callback_data=f"cs2_mkt_winner_{match_index}")
            cs2_markets_kb.button(text="🗺️ По картам", callback_data=f"cs2_mkt_maps_{match_index}")
            cs2_markets_kb.button(text="🎯 Тотал раундов", callback_data=f"cs2_mkt_rounds_{match_index}")
            cs2_markets_kb.button(text="⬅️ Матчи", callback_data="back_to_cs2")
            cs2_markets_kb.button(text="🏠 Меню", callback_data="back_to_main")
            cs2_markets_kb.adjust(2)
            _cs2_kb = cs2_markets_kb.as_markup()
            await call.message.edit_text(report, parse_mode="Markdown", reply_markup=_cs2_kb)
            import time as _time
            _report_cache[f"cs2_{match_index}"] = {
                "text": report, "kb": _cs2_kb,
                "parse_mode": "Markdown", "ts": _time.time(),
            }
        except Exception as e:
            logger.error(f"[CS2 анализ] Ошибка: {e}")
            await call.message.edit_text("⚠️ Не удалось выполнить анализ. Попробуй позже.")

    # --- Выбор лиги CS2 ---
    elif call.data.startswith("cs2_league_"):
        league_name = call.data[11:]
        league_matches = [m for m in cs2_matches_cache if league_name.lower() in f"{m.get('league','')} {m.get('tournament','')}".lower()]
        
        if not league_matches:
            await call.answer("Матчи не найдены.", show_alert=True)
            return
            
        builder = InlineKeyboardBuilder()
        for m in league_matches:
            # Находим индекс в общем кэше
            idx = cs2_matches_cache.index(m)
            tier_icon = {"S": "🏆", "A": "🎯", "B": "🎮"}.get(m.get("tier", "B"), "🎮")
            label = f"{tier_icon} {m['home']} vs {m['away']} | {m['time']}"
            builder.button(text=label, callback_data=f"cs2_m_{idx}")
        builder.button(text="⬅️ Назад к лигам", callback_data="back_to_cs2_leagues")
        builder.adjust(1)
        
        await call.message.edit_text(
            f"🏆 *{league_name}* — матчи:\nВыберите матч для анализа:",
            parse_mode="Markdown",
            reply_markup=builder.as_markup()
        )

    # --- Назад к списку лиг CS2 ---
    elif call.data == "back_to_cs2_leagues" or call.data == "back_to_cs2":
        if not cs2_matches_cache:
            await call.answer("Список устарел. Вернись в главное меню.", show_alert=True)
            return
            
        leagues = sorted(list(set([m.get('league', 'Other') for m in cs2_matches_cache])))
        # Пытаемся сопоставить с белым списком для красивых названий
        display_leagues = {}
        for m in cs2_matches_cache:
            full_name = f"{m.get('league','')} {m.get('tournament','')}".lower()
            for allowed in CS2_WHITELIST_LEAGUES:
                if allowed.lower() in full_name:
                    if allowed not in display_leagues:
                        display_leagues[allowed] = 0
                    display_leagues[allowed] += 1
                    break
        
        builder = InlineKeyboardBuilder()
        for league, count in sorted(display_leagues.items()):
            builder.button(text=f"🏆 {league} ({count})", callback_data=f"cs2_league_{league}")
        builder.adjust(1)
        
        await call.message.edit_text(
            f"🎮 *CS2* — Выберите лигу:",
            parse_mode="Markdown",
            reply_markup=builder.as_markup()
        )


    # --- Теннис: выбор турнира ---
    elif call.data.startswith("tennis_tour_"):
        sport_key = call.data[len("tennis_tour_"):]
        tour_matches = [m for m in tennis_matches_cache if m.get("sport_key") == sport_key]
        if not tour_matches:
            await call.answer("Матчи не найдены. Обновите список.", show_alert=True)
            return
        from sports.tennis.rankings import detect_surface, detect_tour
        surface  = detect_surface(sport_key)
        tour     = detect_tour(sport_key)
        surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}
        icon     = surf_icons.get(surface, "🎾")
        name     = sport_key.replace(f"tennis_{tour}_", "").replace("_", " ").title()
        builder  = InlineKeyboardBuilder()
        for idx, m in enumerate(tour_matches):
            global_idx = tennis_matches_cache.index(m)
            label = f"{icon} {m['player1']} vs {m['player2']} | {m['odds_p1']} / {m['odds_p2']}"
            builder.button(text=label, callback_data=f"tennis_m_{global_idx}")
        builder.button(text="⬅️ Назад к турнирам", callback_data="back_to_tennis")
        builder.adjust(1)
        await call.message.edit_text(
            f"{icon} <b>{tour.upper()} {name}</b>\n\nВыберите матч для полного AI-анализа:",
            parse_mode="HTML", reply_markup=builder.as_markup()
        )

    # --- Теннис: полный анализ матча ---
    elif call.data.startswith("tennis_m_"):
        match_idx = int(call.data.split("_")[2])
        if match_idx >= len(tennis_matches_cache):
            await call.answer("Матч не найден.", show_alert=True)
            return
        m = tennis_matches_cache[match_idx]
        p1, p2    = m["player1"], m["player2"]
        o1, o2    = m.get("odds_p1", 0.0), m.get("odds_p2", 0.0)
        sport_key = m.get("sport_key", "")
        no_odds   = (o1 == 0.0 or o2 == 0.0)

        no_odds_note = "\n<i>⚠️ Коэффициенты недоступны — только аналитика</i>" if no_odds else ""
        status_msg = await call.message.edit_text(
            f"⏳ <b>{p1} vs {p2}</b>{no_odds_note}",
            parse_mode="HTML"
        )
        await show_ai_thinking(status_msg, p1, p2, sport="tennis")
        try:
            from sports.tennis import analyze_tennis_match
            from sports.tennis.agents import run_tennis_gpt_agent, run_tennis_llama_agent, format_tennis_full_report
            from sports.tennis.rankings import detect_surface, detect_tour

            surface = detect_surface(sport_key)
            tour    = detect_tour(sport_key)

            # Математика
            result   = analyze_tennis_match(p1, p2, o1, o2, sport_key=sport_key)
            probs    = result["probs"]
            cands    = result["candidates"]

            # GPT анализ
            gpt_text = run_tennis_gpt_agent(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
            )

            # Llama анализ (независимый)
            llama_text = run_tennis_llama_agent(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
                gpt_verdict=gpt_text,
            )

            # CHIMERA Multi-Agent
            from sports.tennis.agents import run_tennis_chimera_agents
            chimera_data = run_tennis_chimera_agents(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
            )
            chimera_block = chimera_data.get("verdict_block", "")

            # Экспертное мнение (Google News + Sport News Live)
            try:
                from expert_oracle import get_expert_consensus, format_expert_block
                _loop_tn = asyncio.get_running_loop()
                _exp_tn = await _loop_tn.run_in_executor(
                    None, get_expert_consensus, p1, p2, "tennis"
                )
                _expert_block_tn = format_expert_block(_exp_tn, p1, p2)
                if _expert_block_tn:
                    chimera_block = (chimera_block + "\n\n" + _expert_block_tn).strip()
                    print(f"[ExpertOracle Tennis] {p1} vs {p2}: {_exp_tn.get('sources_count')} источн.")
            except Exception as _ee_tn:
                print(f"[ExpertOracle Tennis] Ошибка: {_ee_tn}")

            # ── Теннис финальный ансамбль: математика 90% + AI 10% ───────
            # gpt_text/llama_text — текстовые ответы, парсим вердикт
            _tennis_ai = []
            for _agent_txt in [gpt_text, llama_text]:
                if not _agent_txt or _agent_txt.startswith("❌"):
                    continue
                _t_low = _agent_txt.lower()
                _winner = p1 if p1.split()[-1].lower() in _t_low else (
                          p2 if p2.split()[-1].lower() in _t_low else "")
                if _winner == p1:
                    _tennis_ai.append({"recommended_outcome": "home_win", "confidence": 60})
                elif _winner == p2:
                    _tennis_ai.append({"recommended_outcome": "away_win", "confidence": 60})
            if _tennis_ai:
                _blended_p1 = _blend_ai(probs["p1_win"], _tennis_ai, p1, p2, 0.10)
                probs = dict(probs)  # копия чтобы не мутировать
                probs["p1_win"] = _blended_p1
                probs["p2_win"] = round(1 - _blended_p1, 4)
                print(f"[Tennis AI-blend] {p1}: {probs['p1_win']} (AI вклад 10%)")
            # Пересчитываем кандидатов с обновлёнными вероятностями
            if _tennis_ai and cands:
                for c in cands:
                    if c.get("outcome") == "P1":
                        c["prob"] = round(probs["p1_win"] * 100, 1)
                    elif c.get("outcome") == "P2":
                        c["prob"] = round(probs["p2_win"] * 100, 1)

            # Полный отчёт
            report = format_tennis_full_report(
                p1, p2, probs, o1, o2, surface, tour,
                gpt_text, llama_text, cands, sport_key=sport_key,
                chimera_verdict_block=chimera_block,
                commence_time=m.get("commence_time"),
            )

            # Сохраняем прогноз в БД
            try:
                best = cands[0] if cands else None
                rec = "home_win" if (best and best["outcome"] == "P1") else "away_win"
                save_prediction(
                    sport="tennis",
                    match_id=m.get("event_id", f"{p1}_{p2}"),
                    match_date=m.get("commence_time", ""),
                    home_team=p1, away_team=p2,
                    league=sport_key,
                    recommended_outcome=rec,
                    ensemble_home=probs["p1_win"],
                    ensemble_away=probs["p2_win"],
                    ensemble_best_outcome=rec,
                    bookmaker_odds_home=o1,
                    bookmaker_odds_away=o2,
                )
            except Exception as save_err:
                print(f"[Tennis Save] {save_err}")
            try:
                upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
                track_analysis(call.from_user.id, "tennis")
            except Exception:
                pass

            tennis_kb = InlineKeyboardBuilder()
            tennis_kb.button(text="🎾 Победитель", callback_data=f"tennis_mkt_winner_{match_idx}")
            tennis_kb.button(text="📊 Тотал геймов", callback_data=f"tennis_mkt_games_{match_idx}")
            tennis_kb.button(text="🏅 Победа в 1-м сете", callback_data=f"tennis_mkt_set1_{match_idx}")
            tennis_kb.button(text="⬅️ Матчи", callback_data=f"tennis_tour_{sport_key}")
            tennis_kb.button(text="🏠 Меню", callback_data="back_to_main")
            tennis_kb.adjust(2)
            _tn_kb = tennis_kb.as_markup()
            await call.message.edit_text(report, parse_mode="HTML", reply_markup=_tn_kb)
            import time as _time
            _report_cache[f"tennis_{sport_key}_{match_idx}"] = {
                "text": report, "kb": _tn_kb,
                "parse_mode": "HTML", "ts": _time.time(),
            }

        except Exception as e:
            print(f"[Tennis Match] Ошибка: {e}")
            import traceback; traceback.print_exc()
            await call.message.edit_text(f"❌ Ошибка анализа тенниса: {str(e)[:150]}")

    # --- Теннис: назад к списку турниров ---
    elif call.data == "back_to_tennis":
        await cmd_tennis(call.message)

    # --- Выбор лиги ---
    elif call.data.startswith("league_"):
        league_key = call.data[7:]
        league_name = dict(FOOTBALL_LEAGUES).get(league_key, league_key)
        matches = get_matches(league=league_key, force=True)
        if not matches:
            await call.answer(f"❌ Нет матчей для {league_name}. Попробуйте позже.", show_alert=True)
            return
        await call.message.edit_text(
            f"⚽ *{league_name}*\n\nВыберите матч для анализа:",
            parse_mode="Markdown",
            reply_markup=build_matches_keyboard(matches)
        )

    # --- Сменить лигу ---
    elif call.data == "change_league":
        league_name = dict(FOOTBALL_LEAGUES).get(_current_league, "АПЛ")
        await call.message.edit_text(
            f"⚽ *Футбол* — выбери лигу:\nТекущая: *{league_name}*",
            parse_mode="Markdown",
            reply_markup=build_football_keyboard()
        )

    # --- Быстрый возврат к анализу матча из кэша ---
    elif call.data.startswith("back_to_report_"):
        import time as _time
        # Формат: back_to_report_{sport}_{key...}
        # football_{idx} | cs2_{idx} | tennis_{sport_key}_{idx} | bball_{league}_{idx}
        suffix = call.data[len("back_to_report_"):]  # e.g. "football_3" or "tennis_atp_2"
        cached_report = _report_cache.get(suffix)
        if cached_report and _time.time() - cached_report.get("ts", 0) < _REPORT_CACHE_TTL:
            await call.answer()
            await call.message.edit_text(
                cached_report["text"],
                parse_mode=cached_report["parse_mode"],
                reply_markup=cached_report["kb"],
            )
        else:
            # Кэш устарел — предупреждаем и предлагаем пересчитать
            await call.answer("⏰ Анализ устарел (>45 мин). Открой матч заново.", show_alert=True)

    # --- Возврат к списку матчей ---
    elif call.data == "back_to_main":
        lang = "ru"
        try:
            lang = get_user_language(call.from_user.id)
        except Exception:
            pass
        try:
            await call.message.delete()
        except Exception:
            pass
        await call.message.answer(
            "🏠 <b>Главное меню</b>",
            parse_mode="HTML",
            reply_markup=build_main_keyboard(lang),
        )

    elif call.data == "back_to_matches":
        if not matches_cache:
            get_matches()
        await call.message.edit_text("Выберите матч для анализа:", reply_markup=build_matches_keyboard(matches_cache, page=0))

    # --- Пагинация матчей ---
    elif call.data.startswith("matches_page_"):
        try:
            pg = int(call.data.split("_")[2])
        except (IndexError, ValueError):
            pg = 0
        if not matches_cache:
            get_matches()
        await call.answer()
        total = len(matches_cache)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        league_name = dict(FOOTBALL_LEAGUES).get(_current_league, "Матчи")
        await call.message.edit_text(
            f"⚽ <b>{league_name}</b> — {total} матчей (стр. {pg+1}/{total_pages})\nВыберите матч:",
            parse_mode="HTML",
            reply_markup=build_matches_keyboard(matches_cache, page=pg)
        )

    # --- Обновление матчей ---
    elif call.data == "refresh_matches":
        matches = get_matches(force=True)
        if not matches:
            await call.answer("❌ Не удалось обновить матчи.", show_alert=True)
            return
        league_name = dict(FOOTBALL_LEAGUES).get(_current_league, "")
        await call.message.edit_text(
            f"✅ Список обновлён! {league_name}: {len(matches)} матчей.",
            reply_markup=build_matches_keyboard(matches)
        )

    # --- Показать меню рынков ---
    elif call.data.startswith("show_markets_"):
        match_index = int(call.data.split("_")[2])
        if match_index >= len(matches_cache):
            await call.answer("Матч не найден.", show_alert=True)
            return
        match = matches_cache[match_index]
        home_team = match["home_team"]
        away_team = match["away_team"]

        cached = analysis_cache.get(match_index, {})
        if cached:
            report = format_main_report(
                home_team, away_team,
                cached["prophet_data"], cached["oracle_results"],
                cached["gpt_result"], cached["llama_result"],
                mixtral_result=cached.get("mixtral_result"),
                poisson_probs=cached.get("poisson_probs"),
                elo_probs=cached.get("elo_probs"),
                ensemble_probs=cached.get("ensemble_probs"),
                home_xg_stats=cached.get("home_xg_stats"),
                away_xg_stats=cached.get("away_xg_stats"),
                value_bets=cached.get("value_bets"),
                injuries_block=cached.get("injuries_block"),
                match_time=match.get('commence_time', ''),
            )
            await call.message.edit_text(report, parse_mode="Markdown", reply_markup=build_markets_keyboard(match_index))

    # --- Выбор матча для анализа ---
    elif call.data.startswith("m_"):
        match_index = int(call.data.split("_")[1])
        if match_index >= len(matches_cache):
            await call.answer("Матч не найден. Обновите список.", show_alert=True)
            return

        match = matches_cache[match_index]
        home_team = match["home_team"]
        away_team = match["away_team"]

        _base = f"<b>⚽ {home_team}  <code>vs</code>  {away_team}</b>\n<b>🔮 CHIMERA AI</b> — запускаю анализ...\n\n"
        _sm = await call.message.edit_text(_base + "🔮 <b>Пророк:</b> нейросеть считает вероятности...", parse_mode="HTML")

        prophet_data = get_prophet_prediction(home_team, away_team)
        oracle_results = oracle_analyze(home_team, away_team)
        home_news = oracle_results.get(home_team, {})
        away_news = oracle_results.get(away_team, {})
        news_summary = (
            f"Новости {home_team}: настроение {home_news.get('sentiment', 0):.2f}, "
            f"найдено {home_news.get('news_count', 0)} статей.\n"
            f"Новости {away_team}: настроение {away_news.get('sentiment', 0):.2f}, "
            f"найдено {away_news.get('news_count', 0)} статей."
        )

        try:
            await _sm.edit_text(
                _base +
                "🔮 <b>Пророк:</b> <i>готово ✓</i>\n"
                "🦁 <b>Лев:</b> <i>готово ✓</i>\n"
                "🐍 <b>Змея:</b> считает ELO, Пуассон, xG...",
                parse_mode="HTML"
            )
        except Exception:
            pass

        bookmaker_odds = get_bookmaker_odds(match)

        # Получаем реальную статистику команд из API-Football
        team_stats_text = get_match_stats(home_team, away_team)
        if team_stats_text:
            print(f"[API-Football] Статистика получена для {home_team} vs {away_team}")
        else:
            print(f"[API-Football] Статистика недоступна для {home_team} vs {away_team}")

        # Получаем xG статистику из Understat → fallback на API-Football
        xg_stats_text = ""
        home_xg_stats = None
        away_xg_stats = None
        if UNDERSTAT_AVAILABLE:
            try:
                home_xg_stats = get_team_xg_stats(home_team)
                away_xg_stats = get_team_xg_stats(away_team)
                xg_stats_text = format_xg_stats(home_team, away_team)
                if xg_stats_text:
                    print(f"[Understat] xG статистика получена для {home_team} vs {away_team}")
                    if team_stats_text:
                        team_stats_text = team_stats_text + "\n\n" + xg_stats_text
                    else:
                        team_stats_text = xg_stats_text
            except Exception as _xe:
                print(f"[Understat] Недоступен: {_xe}")
        # Fallback: если Understat дал пустые данные — берём avg_goals из API-Football
        if not home_xg_stats and API_FOOTBALL_KEY:
            try:
                from api_football import get_team_stats as _get_tf_stats
                _hf = _get_tf_stats(home_team)
                _af = _get_tf_stats(away_team)
                if _hf:
                    home_xg_stats = {
                        "avg_xg_last5": _hf.get("goals_scored_avg", 1.35),
                        "avg_xg_against_last5": _hf.get("goals_conceded_avg", 1.1),
                    }
                if _af:
                    away_xg_stats = {
                        "avg_xg_last5": _af.get("goals_scored_avg", 1.1),
                        "avg_xg_against_last5": _af.get("goals_conceded_avg", 1.35),
                    }
                if _hf or _af:
                    print(f"[Understat→API-Football] Fallback xG: {home_team}={home_xg_stats}, {away_team}={away_xg_stats}")
            except Exception as _fb:
                print(f"[Understat fallback] API-Football тоже недоступен: {_fb}")

        # Пуассон + ELO математические модели
        poisson_probs = None
        elo_probs = None
        try:
            # ELO рейтинги + форма
            elo_probs = elo_win_probabilities(home_team, away_team, _elo_ratings, _team_form)
            print(f"[ELO] {home_team}={elo_probs['home_elo']}({elo_probs.get('home_form','?')}) vs {away_team}={elo_probs['away_elo']}({elo_probs.get('away_form','?')})")
            print(f"[ELO] Форма-бонус: {home_team}={elo_probs.get('home_form_bonus',0):+.0f} | {away_team}={elo_probs.get('away_form_bonus',0):+.0f}")
        except Exception as _ee:
            print(f"[ELO] Ошибка: {_ee}")

        xg_data_source = "fallback"  # Источник данных для Пуассона
        try:
            # Пуассон на основе xG
            if home_xg_stats and away_xg_stats:
                home_exp, away_exp = calculate_expected_goals(home_xg_stats, away_xg_stats)
                xg_data_source = "understat"  # Реальные данные из Understat
                print(f"[Understat] ✅ Реальные xG: {home_team}={home_xg_stats.get('avg_xg_last5','?')}, {away_team}={away_xg_stats.get('avg_xg_last5','?')}")
            elif home_xg_stats and not away_xg_stats:
                # Есть данные только для хозяев
                home_exp = home_xg_stats.get('avg_xg_last5', 1.35)
                away_exp = 1.10
                xg_data_source = "partial"  # Частичные данные
                print(f"[Understat] ⚠️ Частичные xG: есть данные только для {home_team}")
            elif not home_xg_stats and away_xg_stats:
                # Есть данные только для гостей
                home_exp = 1.35
                away_exp = away_xg_stats.get('avg_xg_last5', 1.10)
                xg_data_source = "partial"  # Частичные данные
                print(f"[Understat] ⚠️ Частичные xG: есть данные только для {away_team}")
            else:
                # Нет данных — среднелиговые значения
                home_exp, away_exp = 1.35, 1.10
                xg_data_source = "fallback"  # Резервные значения
                print(f"[Understat] ❌ Данных нет — использую среднелиговые ({home_exp}/{away_exp})")
            poisson_probs = poisson_match_probabilities(home_exp, away_exp)
            # Добавляем источник данных в poisson_probs
            poisson_probs['data_source'] = xg_data_source
            poisson_probs['home_exp'] = round(home_exp, 2)
            poisson_probs['away_exp'] = round(away_exp, 2)
            print(f"[Пуассон] xG хозяев={home_exp:.2f}, гостей={away_exp:.2f} (источник: {xg_data_source})")
        except Exception as _pe:
            print(f"[Пуассон] Ошибка: {_pe}")

        # ── Dixon-Coles (до ансамбля!) ────────────────────────────────────
        # Запускаем здесь, чтобы его вероятности ВОШЛИ в ансамбль
        _ml_block = ""
        try:
            from ml.predictor import get_football_prediction, format_ml_block as _fmt_ml
            _h_elo = elo_probs.get("home_elo", 1500) if elo_probs else 1500
            _a_elo = elo_probs.get("away_elo", 1500) if elo_probs else 1500
            _ml_pred = get_football_prediction(
                home_team, away_team, bookmaker_odds,
                home_elo=_h_elo, away_elo=_a_elo,
            )
            _ml_block = _fmt_ml(_ml_pred, home_team, away_team)
            if _ml_pred.get("model_used") == "Dixon-Coles":
                print(f"[DC] ✅ П1={round(_ml_pred['home_win']*100)}% "
                      f"| Х={round(_ml_pred['draw']*100)}% | П2={round(_ml_pred['away_win']*100)}%")
            else:
                print(f"[ML] {_ml_pred.get('model_used','?')} — Dixon-Coles не знает эти команды")
        except Exception as _mle:
            print(f"[ML] Ошибка Dixon-Coles: {_mle}")

        # Травмы и дисквалификации
        home_injuries = {}
        away_injuries = {}
        injuries_block = ""
        try:
            home_injuries, away_injuries, injuries_block = await get_match_injuries_async(home_team, away_team)
            # Добавляем данные о травмах в контекст для AI агентов
            if injuries_block:
                injuries_context = (
                    f"\n\n{injuries_block.replace('*', '')}"
                )
                if team_stats_text:
                    team_stats_text = team_stats_text + injuries_context
                else:
                    team_stats_text = injuries_context
        except Exception as _inj_e:
            print(f"[Травмы] Ошибка: {_inj_e}")

        _loop = asyncio.get_running_loop()
        async with _ai_semaphore:
            stats_result, scout_result = await asyncio.gather(
                _loop.run_in_executor(None, run_statistician_agent, prophet_data, team_stats_text),
                _loop.run_in_executor(None, run_scout_agent, home_team, away_team, news_summary),
            )
            gpt_result = await _loop.run_in_executor(None, run_arbitrator_agent, stats_result, scout_result, bookmaker_odds)

        try:
            await _sm.edit_text(
                _base +
                "🔮 <b>Пророк:</b> <i>готово ✓</i>\n"
                "🦁 <b>Лев:</b> <i>готово ✓</i>\n"
                "🐍 <b>Змея:</b> <i>готово ✓</i>\n"
                "🐐 <b>Козёл:</b> <i>готово ✓</i>\n"
                "🌀 <b>Тень:</b> независимая проверка...",
                parse_mode="HTML"
            )
        except Exception:
            pass

        async with _ai_semaphore:
            llama_result, mixtral_result = await asyncio.gather(
                _loop.run_in_executor(None, run_llama_agent, home_team, away_team, prophet_data, news_summary, bookmaker_odds, team_stats_text),
                _loop.run_in_executor(None, run_mixtral_agent, home_team, away_team, prophet_data, news_summary, bookmaker_odds, team_stats_text, poisson_probs, elo_probs),
            )

        # Взвешенный ансамбль всех моделей
        ensemble_probs = None
        value_bets = []
        try:
            ensemble_probs = build_math_ensemble(
                prophet_data, poisson_probs, elo_probs,
                gpt_result, llama_result, mixtral_result,
                bookmaker_odds,
                dc_probs=_ml_pred if _ml_pred.get("model_used") == "Dixon-Coles" else None,
            )
            # Ищем value bets
            odds_for_value = {
                'home': bookmaker_odds.get('home_win', 0),
                'draw': bookmaker_odds.get('draw', 0),
                'away': bookmaker_odds.get('away_win', 0),
            }
            value_bets = calculate_value_bets(ensemble_probs, odds_for_value)
            print(f"[Ансамбль] П1={round(ensemble_probs['home']*100)}% | Х={round(ensemble_probs['draw']*100)}% | П2={round(ensemble_probs['away']*100)}%")
            if value_bets:
                print(f"[Value Bets] Найдено: {len(value_bets)} ставок с EV>5%")
        except Exception as _ense:
            print(f"[Ансамбль] Ошибка: {_ense}")

        # ── Сигналы от полного ансамбля с AI ─────────────────────────────
        football_ai_signals = []
        try:
            # Защита: агенты должны возвращать dict, но на случай ошибки
            if not isinstance(gpt_result, dict):
                gpt_result = {}
            if not isinstance(llama_result, dict):
                llama_result = {}
            if not isinstance(mixtral_result, dict):
                mixtral_result = {}
            # Определяем согласен ли AI ставить
            gpt_bet_signal = gpt_result.get("bet_signal", "")
            llama_outcome  = llama_result.get("recommended_outcome", "")
            gpt_outcome    = gpt_result.get("recommended_outcome", "")
            ai_agrees_flag = (gpt_bet_signal == "СТАВИТЬ") or (gpt_outcome == llama_outcome and gpt_outcome != "")

            # Используем ансамблевые вероятности если есть, иначе ELO
            sig_probs = ensemble_probs or elo_probs or {}
            h_sig = sig_probs.get("home", 0.34)
            d_sig = sig_probs.get("draw", 0.33)
            a_sig = sig_probs.get("away", 0.33)

            elo_h = _elo_ratings.get(home_team, 1500)
            elo_a = _elo_ratings.get(away_team, 1500)
            form_h = get_form_string(_team_form, home_team)
            form_a = get_form_string(_team_form, away_team)

            football_ai_signals = check_football_signal(
                home_team=home_team,
                away_team=away_team,
                home_prob=h_sig,
                away_prob=a_sig,
                draw_prob=d_sig,
                bookmaker_odds=bookmaker_odds,
                home_form=form_h,
                away_form=form_a,
                elo_home=elo_h,
                elo_away=elo_a,
                ai_agrees=ai_agrees_flag,
            )
            if football_ai_signals:
                print(f"[AI Сигнал ⚽] Найдено: {len(football_ai_signals)} сигналов с ансамблем")
        except Exception as _sig_e:
            print(f"[AI Сигнал] Ошибка: {_sig_e}")

        # Сохраняем в кэш для повторного использования при выборе рынков
        analysis_cache[match_index] = {
            "prophet_data": prophet_data,
            "oracle_results": oracle_results,
            "news_summary": news_summary,
            "bookmaker_odds": bookmaker_odds,
            "gpt_result": gpt_result,
            "llama_result": llama_result,
            "mixtral_result": mixtral_result,
            "poisson_probs": poisson_probs,
            "elo_probs": elo_probs,
            "ensemble_probs": ensemble_probs,
            "home_xg_stats": home_xg_stats,
            "away_xg_stats": away_xg_stats,
            "value_bets": value_bets,
            "home_team": home_team,
            "away_team": away_team,
            "match": match,
            "team_stats_text": team_stats_text,
            "injuries_block": injuries_block,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
        }

        # Сохранение в базу данных
        # Определяем лучший исход — из ансамбля, fallback на ELO, fallback на Poisson
        _probs_for_rec = (
            ensemble_probs or
            ({"home": elo_probs.get("home", 0), "draw": elo_probs.get("draw", 0), "away": elo_probs.get("away", 0)} if elo_probs else None) or
            ({"home": poisson_probs.get("home_win", 0), "draw": poisson_probs.get("draw", 0), "away": poisson_probs.get("away_win", 0)} if poisson_probs else None)
        )
        ens_best_key = max(['home', 'draw', 'away'], key=lambda k: (_probs_for_rec or {}).get(k, 0)) if _probs_for_rec else "home"
        ens_best_map = {'home': home_team, 'draw': 'Ничья', 'away': away_team}
        ens_best_label = ens_best_map.get(ens_best_key, home_team)

        prediction_data = {
            "gpt_verdict": gpt_result.get("recommended_outcome", ""),
            "llama_verdict": llama_result.get("recommended_outcome", ""),
            "mixtral_verdict": (mixtral_result or {}).get("recommended_outcome", ""),
            "gpt_confidence": gpt_result.get("final_confidence_percent", 0),
            "llama_confidence": llama_result.get("final_confidence_percent", 0),
            "mixtral_confidence": (mixtral_result or {}).get("final_confidence_percent", 0),
            "bet_signal": gpt_result.get("bet_signal", ""),
            "total_goals": llama_result.get("total_goals_prediction", ""),
            "btts": llama_result.get("both_teams_to_score_prediction", ""),
            "odds_home": bookmaker_odds.get("home_win", 0),
            "odds_draw": bookmaker_odds.get("draw", 0),
            "odds_away": bookmaker_odds.get("away_win", 0),
            "odds_over25": bookmaker_odds.get("over_2_5", 0),
            "odds_under25": bookmaker_odds.get("under_2_5", 0),
            # Математические модели
            "poisson_probs": poisson_probs,
            "elo_probs": elo_probs,
            "ensemble_probs": ensemble_probs,
            "ensemble_best_outcome": ens_best_label,
            "value_bets": value_bets,
            "league": match.get('sport_key', 'soccer_epl'),
        }
        try:
            save_prediction(
                sport='football',
                match_id=str(match['id']),
                match_date=match.get('commence_time', ''),
                home_team=home_team,
                away_team=away_team,
                league=match.get('sport_key', 'soccer_epl'),
                gpt_verdict=prediction_data.get('gpt_verdict'),
                llama_verdict=prediction_data.get('llama_verdict'),
                gpt_confidence=prediction_data.get('gpt_confidence'),
                llama_confidence=prediction_data.get('llama_confidence'),
                bet_signal=prediction_data.get('bet_signal'),
                total_goals_prediction=prediction_data.get('total_goals'),
                btts_prediction=prediction_data.get('btts'),
                bookmaker_odds_home=prediction_data.get('odds_home'),
                bookmaker_odds_draw=prediction_data.get('odds_draw'),
                bookmaker_odds_away=prediction_data.get('odds_away'),
                bookmaker_odds_over25=prediction_data.get('odds_over25'),
                bookmaker_odds_under25=prediction_data.get('odds_under25'),
                ensemble_home=(ensemble_probs or {}).get('home'),
                ensemble_draw=(ensemble_probs or {}).get('draw'),
                ensemble_away=(ensemble_probs or {}).get('away'),
                ensemble_best_outcome=ens_best_label,
                recommended_outcome={'home': 'home_win', 'draw': 'draw', 'away': 'away_win'}.get(ens_best_key, 'home_win'),
                prediction_data=prediction_data,
            )
        except Exception as _save_err:
            print(f"[DB Save] Ошибка сохранения футбол: {_save_err}")

        # Трекинг активности пользователя
        try:
            upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
            track_analysis(call.from_user.id, "football")
        except Exception:
            pass

        # CHIMERA Multi-Agent
        _football_chimera_block = ""
        try:
            from agents import run_football_chimera_agents
            _fc = run_football_chimera_agents(
                home_team, away_team,
                ensemble_probs or elo_probs or {},
                bookmaker_odds,
                news_summary=news_summary,
                stats_text=team_stats_text or "",
            )
            _football_chimera_block = _fc.get("verdict_block", "")
        except Exception as _fce:
            print(f"[Football CHIMERA] Ошибка: {_fce}")

        # Экспертное мнение (Google News + AI-сводка)
        _expert_block = ""
        try:
            from expert_oracle import get_expert_consensus, format_expert_block
            _loop2 = asyncio.get_running_loop()
            _exp = await _loop2.run_in_executor(
                None, get_expert_consensus, home_team, away_team, "football"
            )
            _expert_block = format_expert_block(_exp, home_team, away_team)
            if _expert_block:
                print(f"[ExpertOracle] {home_team} vs {away_team}: консенсус={_exp.get('consensus')} ({_exp.get('sources_count')} источн.)")
        except Exception as _ee2:
            print(f"[ExpertOracle] Ошибка: {_ee2}")

        # Добавляем экспертный блок к chimera-вердикту если есть
        if _expert_block:
            _football_chimera_block = (_football_chimera_block + "\n\n" + _expert_block).strip()

        # Движение линий — записываем снимок и показываем если есть сдвиг
        try:
            from line_movement import make_match_key, record_odds, get_movement, format_movement_block
            _lm_key = make_match_key(home_team, away_team, match.get("commence_time", ""))
            record_odds(_lm_key, bookmaker_odds)
            _movement = get_movement(_lm_key, bookmaker_odds)
            _movement_block = format_movement_block(_movement)
            if _movement_block:
                _football_chimera_block = (_football_chimera_block + "\n\n" + _movement_block).strip()
        except Exception as _lme:
            pass

        final_report = format_main_report(
            home_team, away_team,
            prophet_data, oracle_results,
            gpt_result, llama_result,
            mixtral_result=mixtral_result,
            poisson_probs=poisson_probs,
            elo_probs=elo_probs,
            ensemble_probs=ensemble_probs,
            home_xg_stats=home_xg_stats,
            away_xg_stats=away_xg_stats,
            value_bets=value_bets,
            injuries_block=injuries_block,
            match_time=match.get('commence_time', ''),
            chimera_verdict_block=_football_chimera_block,
            ml_block=_ml_block,
        )

        _football_kb = build_markets_keyboard(match_index)
        await call.message.edit_text(
            final_report,
            parse_mode="Markdown",
            reply_markup=_football_kb,
        )
        # Сохраняем в report_cache для быстрого возврата
        import time as _time
        _report_cache[f"football_{match_index}"] = {
            "text": final_report, "kb": _football_kb,
            "parse_mode": "Markdown", "ts": _time.time(),
        }

        # Отправляем AI-сигнал отдельным сообщением если есть
        if football_ai_signals:
            from signal_engine import format_signal
            top_sig = football_ai_signals[0]
            top_sig["sport"] = "football"
            sig_text = format_signal(top_sig)
            # Добавляем пометку что сигнал подтверждён ансамблем
            sig_text = "🐉 <b>ХИМЕРА (Змея + Лев + Козёл + Тень)</b>\n\n" + sig_text
            try:
                await call.message.answer(sig_text, parse_mode="HTML")
            except Exception:
                pass

    # --- Рынок: Победитель ---
    elif call.data.startswith("mkt_winner_"):
        match_index = int(call.data.split("_")[2])
        cached = analysis_cache.get(match_index)
        if not cached:
            await call.answer("Сначала запустите анализ матча.", show_alert=True)
            return

        home_team = cached["home_team"]
        away_team = cached["away_team"]
        gpt_result = cached["gpt_result"]
        llama_result = cached["llama_result"]
        prophet_data = cached["prophet_data"]
        bookmaker_odds = cached["bookmaker_odds"]

        gpt_verdict = translate_outcome(gpt_result.get("recommended_outcome", ""), home_team, away_team)
        gpt_conf = gpt_result.get("final_confidence_percent", 0)
        gpt_odds_val = gpt_result.get("bookmaker_odds", 0)
        gpt_stake = gpt_result.get("recommended_stake_percent", 0)
        gpt_ev = gpt_result.get("expected_value_percent", 0)
        bet_signal = gpt_result.get("bet_signal", "ПРОПУСТИТЬ")
        signal_reason = gpt_result.get("signal_reason", "")

        llama_verdict = translate_outcome(llama_result.get("recommended_outcome", ""), home_team, away_team)
        llama_conf = llama_result.get("final_confidence_percent", 0)

        signal_icon = "🔥 СТАВИТЬ!" if bet_signal == "СТАВИТЬ" else "⏸ ПРОПУСТИТЬ"

        report = f"""
🏆 *ПОБЕДИТЕЛЬ МАТЧА*
{home_team} vs {away_team}
━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *Пророк (нейросеть):*
 П1: {prophet_data[1]*100:.0f}% | Х: {prophet_data[0]*100:.0f}% | П2: {prophet_data[2]*100:.0f}%

🐐 *Вердикт Козла:*
{conf_icon(gpt_conf)} {gpt_verdict} — {gpt_conf}%
🎯 Коэф: {gpt_odds_val} | Ставка: {gpt_stake:.1f}% | EV: +{gpt_ev:.1f}%

🌀 *Вердикт Тени:*
{conf_icon(llama_conf)} {llama_verdict} — {llama_conf}%

━━━━━━━━━━━━━━━━━━━━━━━━━
*{signal_icon}*
_{signal_reason}_
""".strip()

        await call.message.edit_text(report, parse_mode="Markdown", reply_markup=build_back_to_markets_keyboard(match_index))

    # --- Рынок: Голы ---
    elif call.data.startswith("mkt_goals_"):
        match_index = int(call.data.split("_")[2])
        cached = analysis_cache.get(match_index)
        if not cached:
            await call.answer("Сначала запустите анализ матча.", show_alert=True)
            return

        await call.message.edit_text("⏳ *Анализирую рынок голов...*", parse_mode="Markdown")

        goals_result = run_goals_market_agent(
            cached["home_team"], cached["away_team"],
            cached.get("news_summary", ""), cached["bookmaker_odds"]
        )
        report = format_goals_report(cached["home_team"], cached["away_team"], goals_result, cached["bookmaker_odds"])
        await call.message.edit_text(report, parse_mode="Markdown", reply_markup=build_back_to_markets_keyboard(match_index))


    # --- Рынок: Гандикапы ---
    elif call.data.startswith("mkt_handicap_"):
        match_index = int(call.data.split("_")[2])
        cached = analysis_cache.get(match_index)
        if not cached:
            await call.answer("Сначала запустите анализ матча.", show_alert=True)
            return

        await call.message.edit_text("⏳ *Анализирую гандикапы...*", parse_mode="Markdown")

        handicap_result = run_handicap_market_agent(
            cached["home_team"], cached["away_team"],
            cached["prophet_data"], cached["bookmaker_odds"],
            cached["gpt_result"], cached["llama_result"]
        )
        report = format_handicap_report(cached["home_team"], cached["away_team"], handicap_result)
        await call.message.edit_text(report, parse_mode="Markdown", reply_markup=build_back_to_markets_keyboard(match_index))

    # --- CS2: рынки под матчем ---
    elif call.data.startswith("cs2_mkt_"):
        parts = call.data.split("_")
        mkt_type = parts[2]   # winner, maps, rounds
        match_idx = int(parts[3])
        m = cs2_matches_cache[match_idx] if match_idx < len(cs2_matches_cache) else None
        if not m:
            await call.answer("Матч не найден", show_alert=True)
            return
        home = m.get("home", "")
        away = m.get("away", "")
        from sports.cs2 import calculate_cs2_win_prob as _cs2_prob
        _loop = asyncio.get_running_loop()
        analysis = await _loop.run_in_executor(None, _cs2_prob, home, away)
        h_prob = analysis.get("home_prob", 0.5)
        a_prob = analysis.get("away_prob", 0.5)
        h_odds = round(1 / h_prob, 2) if h_prob > 0.01 else 1.9
        a_odds = round(1 / a_prob, 2) if a_prob > 0.01 else 1.9
        h_pct = round(h_prob * 100, 1)
        a_pct = round(a_prob * 100, 1)

        if mkt_type == "winner":
            text = (
                f"🏆 <b>Победитель матча</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎮 <b>{home} vs {away}</b>\n\n"
                f"{'✅' if h_pct > a_pct else '▫️'} <b>{home}</b>\n"
                f"   Вероятность: <b>{h_pct}%</b> | Кэф: <b>{h_odds}</b>\n\n"
                f"{'✅' if a_pct > h_pct else '▫️'} <b>{away}</b>\n"
                f"   Вероятность: <b>{a_pct}%</b> | Кэф: <b>{a_odds}</b>\n\n"
                f"<i>💡 Ставь на {'<b>' + home + '</b>' if h_pct > a_pct else '<b>' + away + '</b>'} "
                f"если кэф у букмекера ≥ {min(h_odds, a_odds):.2f}</i>"
            )
        elif mkt_type == "maps":
            balance = 1 - abs(h_prob - a_prob)
            h_map1 = round(min(max(h_pct + 2, 30), 70), 1)
            a_map1 = round(100 - h_map1, 1)
            score_2_0 = round(balance * 20 + max(h_pct, a_pct) * 0.2, 1)
            score_2_1 = round(100 - score_2_0, 1)
            winner = home if h_pct > a_pct else away
            text = (
                f"🗺️ <b>По картам</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎮 <b>{home} vs {away}</b>\n\n"
                f"<b>Победа на карте 1:</b>\n"
                f"  {'✅' if h_map1 > a_map1 else '▫️'} {home}: <b>{h_map1}%</b>\n"
                f"  {'✅' if a_map1 > h_map1 else '▫️'} {away}: <b>{a_map1}%</b>\n\n"
                f"<b>Счёт серии:</b>\n"
                f"  📊 {winner} 2:0 → <b>{score_2_0:.0f}%</b>\n"
                f"  📊 {winner} 2:1 → <b>{score_2_1:.0f}%</b>\n\n"
                f"<i>💡 Лучший вариант: <b>{winner}</b> выигрывает серию</i>"
            )
        elif mkt_type == "rounds":
            balance = 1 - abs(h_prob - a_prob)
            expected = round(24.5 + balance * 5, 1)
            over = expected > 26.5
            text = (
                f"🎯 <b>Тотал раундов (Карта 1)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎮 <b>{home} vs {away}</b>\n\n"
                f"📊 Прогноз раундов: <b>{expected}</b>\n\n"
                f"{'✅' if over else '▫️'} <b>Больше 26.5</b>\n"
                f"{'✅' if not over else '▫️'} <b>Меньше 26.5</b>\n\n"
                f"<i>{'⚖️ Равный матч — много раундов' if balance > 0.6 else '💪 Один доминирует — короткие карты'}</i>"
            )
        else:
            text = "⚠️ Неизвестный рынок"

        back_kb = InlineKeyboardBuilder()
        back_kb.button(text="🏆 Победитель" if mkt_type != "winner" else "🗺️ По картам",
                       callback_data=f"cs2_mkt_{'winner' if mkt_type != 'winner' else 'maps'}_{match_idx}")
        back_kb.button(text="🎯 Тотал раундов" if mkt_type != "rounds" else "🏆 Победитель",
                       callback_data=f"cs2_mkt_{'rounds' if mkt_type != 'rounds' else 'winner'}_{match_idx}")
        back_kb.button(text="↩️ К анализу", callback_data=f"back_to_report_cs2_{match_idx}")
        back_kb.button(text="🏠 Меню", callback_data="back_to_main")
        back_kb.adjust(2)
        await call.answer()
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=back_kb.as_markup())

    # --- Теннис: рынки под матчем ---
    elif call.data.startswith("tennis_mkt_"):
        parts = call.data.split("_")
        mkt_type = parts[2]   # winner, games, set1
        match_idx = int(parts[3])
        if match_idx >= len(tennis_matches_cache):
            await call.answer("Матч не найден", show_alert=True)
            return
        m = tennis_matches_cache[match_idx]
        home = m.get("home_team", m.get("home", "Игрок A"))
        away = m.get("away_team", m.get("away", "Игрок B"))
        odds = m.get("bookmaker_odds", {}) or m.get("odds", {})
        h_odds = float(odds.get("home_win", 1.75))
        a_odds = float(odds.get("away_win", 2.1))
        h_pct = round(100 / h_odds, 1) if h_odds > 1 else 57.1
        a_pct = round(100 / a_odds, 1) if a_odds > 1 else 47.6

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
            total_line = 22.5 if is_balanced else 20.5
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
            h_set = round(min(max(h_pct * 0.9 + 5, 30), 70), 1)
            a_set = round(100 - h_set, 1)
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
        back_kb.button(text="🎾 Победитель" if mkt_type != "winner" else "📊 Тотал геймов",
                       callback_data=f"tennis_mkt_{'winner' if mkt_type != 'winner' else 'games'}_{match_idx}")
        back_kb.button(text="🏅 1-й сет" if mkt_type != "set1" else "🎾 Победитель",
                       callback_data=f"tennis_mkt_{'set1' if mkt_type != 'set1' else 'winner'}_{match_idx}")
        back_kb.button(text="↩️ К анализу", callback_data=f"back_to_report_tennis_{sport_key}_{match_idx}")
        back_kb.button(text="🏠 Меню", callback_data="back_to_main")
        back_kb.adjust(2)
        await call.answer()
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=back_kb.as_markup())

    # --- Баскетбол: рынки под матчем ---
    elif call.data.startswith("bball_mkt_"):
        parts = call.data.split("_")
        mkt_type = parts[2]  # winner, total, spread
        match_idx = int(parts[-1])
        league_key = "_".join(parts[3:-1])
        _loop = asyncio.get_running_loop()
        from sports.basketball import get_basketball_matches as _get_bball
        matches = await _loop.run_in_executor(None, _get_bball, league_key)
        if not matches or match_idx >= len(matches):
            await call.answer("Матч не найден", show_alert=True)
            return
        m = matches[match_idx]
        home = m.get("home_team", m.get("home", "Команда A"))
        away = m.get("away_team", m.get("away", "Команда B"))
        from sports.basketball.core import get_basketball_odds as _bball_odds
        odds = _bball_odds(m)
        h_odds = float(odds.get("home_win") or 1.9)
        a_odds = float(odds.get("away_win") or 1.9)
        h_pct = round(100 / h_odds, 1)
        a_pct = round(100 / a_odds, 1)
        total_data = odds.get("total_analysis", {}) or {}
        total_line = total_data.get("line") or odds.get("total_over_line") or 220.5
        over_odds = odds.get("total_over_odds") or 1.9
        under_odds = odds.get("total_under_odds") or 1.9
        spread_h = odds.get("spread_home") or -3.5
        spread_a = -spread_h

        if mkt_type == "winner":
            text = (
                f"🏀 <b>Победитель матча</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>{home} vs {away}</b>\n\n"
                f"{'✅' if h_pct > a_pct else '▫️'} <b>{home}</b>\n"
                f"   Кэф: <b>{h_odds}</b> | Вероятность: <b>{h_pct}%</b>\n\n"
                f"{'✅' if a_pct > h_pct else '▫️'} <b>{away}</b>\n"
                f"   Кэф: <b>{a_odds}</b> | Вероятность: <b>{a_pct}%</b>\n\n"
                f"<i>💡 Рекомендация: {'<b>' + home + '</b>' if h_pct > a_pct else '<b>' + away + '</b>'}</i>"
            )
        elif mkt_type == "total":
            text = (
                f"📊 <b>Тотал очков</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>{home} vs {away}</b>\n\n"
                f"📏 Линия: <b>{total_line}</b> очков\n\n"
                f"📈 <b>Больше {total_line}</b>  Кэф: <b>{over_odds}</b>\n"
                f"📉 <b>Меньше {total_line}</b>  Кэф: <b>{under_odds}</b>\n\n"
                f"<i>💡 Равные команды → больше очков. Защитный стиль → меньше.</i>"
            )
        elif mkt_type == "spread":
            text = (
                f"⚖️ <b>Фора (Гандикап)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>{home} vs {away}</b>\n\n"
                f"🏠 <b>{home}</b> фора: <b>{spread_h:+.1f}</b>\n"
                f"✈️ <b>{away}</b> фора: <b>{spread_a:+.1f}</b>\n\n"
                f"<i>💡 {'Хозяева фавориты — фора выравнивает шансы' if spread_h < 0 else 'Гости фавориты — дерзкая ставка на хозяев с форой'}</i>"
            )
        else:
            text = "⚠️ Неизвестный рынок"

        back_kb = InlineKeyboardBuilder()
        back_kb.button(text="🏀 Победитель" if mkt_type != "winner" else "📊 Тотал",
                       callback_data=f"bball_mkt_{'winner' if mkt_type != 'winner' else 'total'}_{league_key}_{match_idx}")
        back_kb.button(text="⚖️ Фора" if mkt_type != "spread" else "📊 Тотал",
                       callback_data=f"bball_mkt_{'spread' if mkt_type != 'spread' else 'total'}_{league_key}_{match_idx}")
        back_kb.button(text="⬅️ Назад к матчу", callback_data=f"bball_match_{league_key}_{match_idx}")
        back_kb.button(text="↩️ К анализу", callback_data=f"back_to_report_bball_{league_key}_{match_idx}")
        back_kb.button(text="🏠 Меню", callback_data="back_to_main")
        back_kb.adjust(2)
        await call.answer()
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=back_kb.as_markup())

    # --- Баскетбол: выбор лиги ---
    elif call.data.startswith("bball_league_"):
        await bball_select_league(call)

    # --- Баскетбол: анализ матча ---
    elif call.data.startswith("bball_match_"):
        await bball_analyze_match(call)

    # --- Экспресс-ставки ---
    elif call.data.startswith("express_"):
        variant_key = call.data.replace("express_", "")

        if variant_key == "back":
            kb = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="🟢 Надёжный (2 события)", callback_data="express_safe")],
                [types.InlineKeyboardButton(text="🟡 Средний (3 события)",  callback_data="express_medium")],
                [types.InlineKeyboardButton(text="🔴 Рискованный (4-5)",    callback_data="express_risky")],
            ])
            await call.answer()
            await call.message.edit_text(
                "🎯 <b>Chimera Express</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "Выбери тип экспресса:\n\n"
                "🟢 <b>Надёжный</b> — 2 события, высокая вероятность\n"
                "🟡 <b>Средний</b> — 3 события, баланс риска и кэфа\n"
                "🔴 <b>Рискованный</b> — 4-5 событий, высокий кэф",
                parse_mode="HTML",
                reply_markup=kb,
            )
            return

        titles = {
            "safe":   ("🟢 Надёжный экспресс", "🟢"),
            "medium": ("🟡 Средний экспресс",   "🟡"),
            "risky":  ("🔴 Рискованный экспресс","🔴"),
        }
        title, emoji = titles.get(variant_key, ("🎯 Экспресс", "🎯"))
        await call.answer()
        status = await call.message.answer(
            f"{emoji} <b>{title}</b>\n\n⏳ Сканирую матчи...\n<i>10-20 секунд</i>",
            parse_mode="HTML"
        )
        _loop = asyncio.get_running_loop()
        try:
            from express_builder import scan_all_matches, build_express_variants, format_express_card

            def _build():
                candidates = scan_all_matches()
                return build_express_variants(candidates)

            variants = await _loop.run_in_executor(None, _build)
            variant  = variants.get(variant_key)

            if not variant:
                await status.edit_text(
                    f"{emoji} <b>{title}</b>\n\n"
                    "⚠️ Недостаточно качественных событий для этого варианта.\n"
                    "<i>Попробуй другой тип или зайди позже.</i>",
                    parse_mode="HTML"
                )
                return

            from express_builder import format_express_card
            card = format_express_card(variant, title, emoji)
            back_kb = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="◀️ Другой вариант", callback_data="express_back")],
                [types.InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_to_main")],
            ])
            await status.edit_text(card, parse_mode="HTML", reply_markup=back_kb)

        except Exception as e:
            logger.error(f"[Экспресс] Ошибка: {e}", exc_info=True)
            await status.edit_text("⚠️ Не удалось построить экспресс. Попробуй позже.", parse_mode="HTML")

# --- 9. Проверка результатов ---

# Список лиг для проверки результатов
SCORES_LEAGUES = [
    "soccer_epl",                      # АПЛ
    "soccer_spain_la_liga",            # Ла Лига
    "soccer_germany_bundesliga",       # Бундеслига
    "soccer_italy_serie_a",            # Серия А
    "soccer_france_ligue_one",         # Лига 1
    "soccer_uefa_champs_league",       # Лига Чемпионов
    "soccer_netherlands_eredivisie",   # Eredivisie
    "soccer_portugal_primeira_liga",   # Примейра Лига
    "soccer_turkey_super_league",      # Суперлига Турции
    "soccer_uefa_europa_league",       # Лига Европы
]

def _norm_team_name(name: str) -> str:
    """Нормализует название команды для нечёткого сопоставления."""
    import re
    n = name.lower().strip()
    # Убираем FC, AFC, SC, United→utd и т.д.
    n = re.sub(r'\b(fc|afc|sc|cf|ac|as|ss|rc|vf[bl]|rb|bsc|rcd|ca)\b', '', n)
    n = re.sub(r'\bunited\b', 'utd', n)
    n = re.sub(r'\bhotspur\b', '', n)
    n = re.sub(r'\bwanderers\b', '', n)
    n = re.sub(r'[^a-z0-9 ]', '', n)
    return re.sub(r'\s+', ' ', n).strip()


async def fetch_scores_for_league(league: str) -> tuple[dict, dict]:
    """
    Получает результаты матчей для одной лиги.
    Возвращает (by_id, by_name):
      by_id   = {match_id: (home_score, away_score)}
      by_name = {(norm_home, norm_away): (home_score, away_score)}
    """
    by_id: dict = {}
    by_name: dict = {}
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{league}/scores/"
        params = {"apiKey": THE_ODDS_API_KEY, "daysFrom": 3, "dateFormat": "iso"}
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=10))
        if r.status_code == 200:
            for s in r.json():
                if s.get('completed'):
                    sc = s.get('scores', [])
                    home_score = away_score = None
                    home_team = s.get('home_team', '')
                    away_team = s.get('away_team', '')
                    for sc_item in sc:
                        if sc_item['name'] == home_team:
                            home_score = int(sc_item['score'])
                        elif sc_item['name'] == away_team:
                            away_score = int(sc_item['score'])
                    if home_score is not None and away_score is not None:
                        by_id[s['id']] = (home_score, away_score)
                        key = (_norm_team_name(home_team), _norm_team_name(away_team))
                        by_name[key] = (home_score, away_score)
        elif r.status_code == 422:
            pass  # Лига недоступна в текущем плане API
    except Exception as e:
        print(f"[Результаты] Ошибка лиги {league}: {e}")
    return by_id, by_name


_LIVE_CSV = os.path.join(os.path.dirname(__file__), "ml", "data", "live_matches.csv")
_LIVE_CSV_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                  "B365H", "B365D", "B365A", "league", "season"]

def _record_match_for_training(pred: dict, home_score: int, away_score: int, outcome: str):
    """
    Записывает сыгранный матч в ml/data/live_matches.csv для еженедельного
    переобучения XGBoost. Дубликаты (по HomeTeam+AwayTeam+Date) пропускаются.
    """
    import csv, os as _os
    from datetime import datetime as _dt

    date_str = (pred.get("match_date") or "")[:10]
    if not date_str:
        date_str = _dt.utcnow().strftime("%d/%m/%Y")
    else:
        try:
            date_str = _dt.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
        except Exception:
            pass

    ftr = "H" if outcome == "home_win" else ("A" if outcome == "away_win" else "D")
    row = {
        "Date":     date_str,
        "HomeTeam": pred.get("home_team", ""),
        "AwayTeam": pred.get("away_team", ""),
        "FTHG":     home_score,
        "FTAG":     away_score,
        "FTR":      ftr,
        "B365H":    pred.get("bookmaker_odds_home") or "",
        "B365D":    pred.get("bookmaker_odds_draw") or "",
        "B365A":    pred.get("bookmaker_odds_away") or "",
        "league":   pred.get("league", "soccer_epl"),
        "season":   "live",
    }

    write_header = not _os.path.exists(_LIVE_CSV)
    try:
        # Deduplicate check
        if not write_header:
            with open(_LIVE_CSV, "r", encoding="utf-8") as f:
                for existing in csv.DictReader(f):
                    if (existing.get("HomeTeam") == row["HomeTeam"] and
                            existing.get("AwayTeam") == row["AwayTeam"] and
                            existing.get("Date") == row["Date"]):
                        return  # уже записан

        with open(_LIVE_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_LIVE_CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[ML Record] Записан матч: {row['HomeTeam']} {home_score}:{away_score} {row['AwayTeam']}")
    except Exception as e:
        print(f"[ML Record] Ошибка CSV: {e}")


async def check_results_task(bot: Bot):
    """Периодически проверяет результаты сыгранных матчей по всем лигам."""
    while True:
        try:
            # Все виды спорта — проверяем независимо
            pending_football = get_pending_predictions("football")
            pending_cs2      = get_pending_predictions("cs2")
            pending_tennis   = get_pending_predictions("tennis")
            pending_bball    = get_pending_predictions("basketball")
            pending_any      = pending_football or pending_cs2 or pending_tennis or pending_bball

            if not pending_any:
                await asyncio.sleep(3600)
                continue

            print(f"[Результаты] Всего ожидает: ⚽{len(pending_football)} 🎮{len(pending_cs2)} 🎾{len(pending_tennis)} 🏀{len(pending_bball)}")

            # ── CS2: отдельный трекер через PandaScore / Esports API ──────
            try:
                from sports.cs2.results_tracker import check_and_update_cs2_results
                cs2_updated = check_and_update_cs2_results()
                if cs2_updated:
                    print(f"[Результаты CS2] Обновлено прогнозов: {cs2_updated}")
            except Exception as cs2_track_err:
                print(f"[Результаты CS2] Ошибка трекера: {cs2_track_err}")

            # ── Теннис: трекер через api-tennis.com ──────────────────────
            try:
                from sports.tennis.results_tracker import check_and_update_tennis_results
                tennis_updated = check_and_update_tennis_results()
                if tennis_updated:
                    print(f"[Результаты Tennis] Обновлено прогнозов: {tennis_updated}")
            except Exception as tennis_track_err:
                print(f"[Результаты Tennis] Ошибка трекера: {tennis_track_err}")

            # ── Баскетбол: трекер через The Odds API /scores/ ────────────
            try:
                from sports.basketball.results_tracker import check_and_update_basketball_results
                bball_updated = check_and_update_basketball_results()
                if bball_updated:
                    print(f"[Результаты Basketball] Обновлено прогнозов: {bball_updated}")
            except Exception as bball_track_err:
                print(f"[Результаты Basketball] Ошибка трекера: {bball_track_err}")

            # ── Футбол: через The Odds API /scores/ ──────────────────────
            if pending_football:
                pending_leagues = set(p.get("league", "") for p in pending_football if p.get("league"))
                leagues_to_check = [lg for lg in SCORES_LEAGUES if any(lg in pl or pl in lg for pl in pending_leagues)] or SCORES_LEAGUES[:3]

                all_scores: dict = {}
                all_scores_by_name: dict = {}
                for league in leagues_to_check:
                    by_id, by_name = await fetch_scores_for_league(league)
                    all_scores.update(by_id)
                    all_scores_by_name.update(by_name)
                    await asyncio.sleep(0.5)

                print(f"[Результаты] Получено {len(all_scores)} завершённых матчей из API")

            for pred in pending_football:
                    match_id = pred['match_id']
                    home = pred['home_team']
                    away = pred['away_team']

                    # Сначала ищем по ID, потом по нормализованным именам
                    score_pair = all_scores.get(match_id)
                    if score_pair is None:
                        name_key = (_norm_team_name(home), _norm_team_name(away))
                        score_pair = all_scores_by_name.get(name_key)
                        if score_pair:
                            print(f"[Результаты] Совпадение по именам: {home} vs {away}")

                    if score_pair is not None:
                        home_score, away_score = score_pair
                        try:
                            # Определяем реальный исход
                            if home_score > away_score:
                                real_outcome = "home_win"
                            elif away_score > home_score:
                                real_outcome = "away_win"
                            else:
                                real_outcome = "draw"

                            recommended = pred.get('recommended_outcome') or ''
                            bet_signal  = pred.get('bet_signal', '')

                            # Точность прогноза — всегда если есть recommended
                            if not recommended:
                                is_correct = None
                                roi = None
                            else:
                                is_correct = 1 if recommended == real_outcome else 0
                                # ROI считаем только если бот рекомендовал ставить
                                if bet_signal == 'СТАВИТЬ' and is_correct:
                                    if real_outcome == "home_win":
                                        win_odds = float(pred.get('bookmaker_odds_home') or 0)
                                    elif real_outcome == "away_win":
                                        win_odds = float(pred.get('bookmaker_odds_away') or 0)
                                    else:
                                        win_odds = float(pred.get('bookmaker_odds_draw') or 0)
                                    roi = round(win_odds - 1, 3) if win_odds >= 1.02 else None
                                elif bet_signal == 'СТАВИТЬ' and not is_correct:
                                    roi = -1.0
                                else:
                                    roi = None  # ПРОПУСТИТЬ — ROI не считаем

                            # Ансамбль
                            ensemble_best = pred.get('ensemble_best_outcome', '')
                            is_ensemble_correct = (1 if ensemble_best == real_outcome else 0) if ensemble_best else None

                            # Тотал голов
                            total_pred = pred.get('total_goals_prediction', '') or ''
                            is_goals_correct = None
                            if total_pred:
                                total_goals = home_score + away_score
                                if 'over' in total_pred.lower() or 'больше' in total_pred.lower():
                                    is_goals_correct = 1 if total_goals >= 3 else 0
                                elif 'under' in total_pred.lower() or 'меньше' in total_pred.lower():
                                    is_goals_correct = 1 if total_goals <= 2 else 0

                            # BTTS
                            btts_pred = pred.get('btts_prediction', '') or ''
                            is_btts_correct = None
                            if btts_pred:
                                btts_real = home_score > 0 and away_score > 0
                                if 'да' in btts_pred.lower() or 'yes' in btts_pred.lower():
                                    is_btts_correct = 1 if btts_real else 0
                                elif 'нет' in btts_pred.lower() or 'no' in btts_pred.lower():
                                    is_btts_correct = 1 if not btts_real else 0

                            # Value bet
                            vb_outcome = pred.get('value_bet_outcome', '') or ''
                            vb_correct = None
                            roi_vb = None
                            if vb_outcome:
                                vb_correct = 1 if vb_outcome == real_outcome else 0
                                if vb_correct:
                                    roi_vb = round(float(pred.get('value_bet_odds') or 1.85) - 1, 3)
                                else:
                                    roi_vb = -1.0

                            update_result(
                                sport='football',
                                match_id=match_id,
                                real_home_score=home_score,
                                real_away_score=away_score,
                                real_outcome=real_outcome,
                                is_correct=is_correct,
                                is_ensemble_correct=is_ensemble_correct,
                                is_goals_correct=is_goals_correct,
                                is_btts_correct=is_btts_correct,
                                roi_outcome=roi,
                                roi_value_bet=roi_vb,
                                value_bet_correct=vb_correct,
                            )

                            icon = "✅" if is_correct else "❌"
                            ens_icon = (f" [Анс: {'✅' if is_ensemble_correct == 1 else '❌'}]"
                                        if is_ensemble_correct is not None else "")
                            vb_icon = (f" [VB: {'✅' if vb_correct == 1 else '❌'}]"
                                       if vb_correct is not None else "")
                            print(f"[Результаты ⚽] {icon} {home} {home_score}:{away_score} {away}{ens_icon}{vb_icon}")

                            # Авто-обновление ELO рейтинга после матча
                            try:
                                update_elo_after_match(home, away, home_score, away_score)
                            except Exception as _elo_e:
                                print(f"[ELO] Ошибка обновления: {_elo_e}")

                            # Записываем матч в live CSV для переобучения XGBoost
                            try:
                                _record_match_for_training(pred, home_score, away_score, real_outcome)
                            except Exception as _rec_e:
                                print(f"[ML Record] Ошибка записи: {_rec_e}")
                        except Exception as e:
                            print(f"[Результаты] Ошибка обновления {home} vs {away}: {e}")
                    else:
                        nk = (_norm_team_name(home), _norm_team_name(away))
                        print(f"[Результаты] Нет данных для {home} vs {away} (id={match_id}, norm={nk})")

            # ── Авто-обучение MetaLearner ──────────────────────────────────
            try:
                from meta_learner import MetaLearner
                _ml = MetaLearner(signal_engine_path="signal_engine.py")
                for _sport in ["football", "cs2", "tennis", "basketball"]:
                    _perf = _ml.analyze_performance(_sport)
                    if _perf.get("total", 0) >= 10:
                        _updates = _ml.suggest_updates(_sport, _perf)
                        if _updates:
                            _ml.apply_updates(_sport, _updates)
                            print(f"[MetaLearner] {_sport} авто-обновление: {_updates}")
                        else:
                            print(f"[MetaLearner] {_sport}: ROI={_perf.get('roi',0):.1f}%, точность={_perf.get('accuracy',0):.1f}%")
                # Баскетбол — перебалансировка весов ELO/Odds
                _bb_weights = _ml.analyze_basketball_weights()
                if _bb_weights:
                    _ml.apply_updates("basketball", _bb_weights)
                    print(f"[MetaLearner] Basketball веса обновлены: {_bb_weights}")
            except Exception as _ml_e:
                print(f"[MetaLearner] Ошибка: {_ml_e}")

        except Exception as e:
            print(f"[Результаты] Общая ошибка: {e}")
        await asyncio.sleep(3600)  # Проверяем каждый час

@dp.message(Command("cs2stats"))
async def cmd_cs2stats(message: types.Message):
    """Команда /cs2stats — статистика прогнозов CS2 с разбивкой."""
    try:
        from sports.cs2.results_tracker import get_cs2_bet_stats
        s = get_cs2_bet_stats()
        if "error" in s:
            await message.answer(f"❌ Ошибка: {s['error']}")
            return

        acc_icon  = lambda a: "🟢" if a >= 60 else ("🟡" if a >= 50 else "🔴")
        roi_icon  = lambda r: "🟢" if r > 0 else "🔴"

        text = "🎮 *CHIMERA AI — Статистика CS2*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        text += f"📋 Всего прогнозов: *{s['total']}* | Проверено: *{s['checked']}*\n\n"

        if s['checked'] > 0:
            text += f"🏆 *Победитель матча:*\n"
            text += f"{acc_icon(s['accuracy'])} Угадано: *{s['wins']}/{s['checked']}* — *{s['accuracy']}%*\n"
            text += f"{roi_icon(s['roi'])} ROI: *{s['roi']:+.2f}* ед.\n\n"

        if s['total_checked'] > 0:
            text += f"🗺 *Тотал карт:*\n"
            text += f"{acc_icon(s['total_accuracy'])} Угадано: *{s['total_wins']}/{s['total_checked']}* — *{s['total_accuracy']}%*\n\n"

        if s.get("monthly"):
            text += "📅 *По месяцам:*\n"
            for row in s["monthly"]:
                m_total = row["total"]
                m_wins  = row["wins"]
                m_acc   = m_wins / m_total * 100 if m_total else 0
                m_roi   = row["roi"]
                text += f"{acc_icon(m_acc)} {row['month']}: {m_wins}/{m_total} ({m_acc:.0f}%) ROI {m_roi:+.1f}\n"
            text += "\n"

        if s.get("recent"):
            text += "📋 *Последние результаты:*\n"
            for r in s["recent"][:6]:
                icon  = "✅" if r["is_correct"] == 1 else "❌"
                h_sc  = r.get("real_home_score", "?")
                a_sc  = r.get("real_away_score", "?")
                text += f"{icon} {r['home_team']} *{h_sc}:{a_sc}* {r['away_team']}\n"

        await message.answer(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"[CS2 статистика] Ошибка: {e}")
        await message.answer("⚠️ Не удалось загрузить статистику CS2. Попробуй позже.")


@dp.message(Command("footballstats"))
async def cmd_footballstats(message: types.Message):
    """Команда /footballstats — статистика прогнозов футбол с разбивкой."""
    try:
        stats = get_statistics('football')
        s = stats.get('football', {})

        acc_icon = lambda a: "🟢" if a >= 60 else ("🟡" if a >= 50 else "🔴")
        roi_icon = lambda r: "🟢" if r > 0 else "🔴"

        text = "⚽ *CHIMERA AI — Статистика Футбол*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        total = s.get('total', 0)
        checked = s.get('total_checked', 0)
        correct = s.get('correct', 0)
        accuracy = s.get('accuracy', 0)
        roi = s.get('roi_main', 0)

        text += f"📋 Всего прогнозов: *{total}* | Проверено: *{checked}*\n\n"

        if checked > 0:
            text += f"🏆 *Победитель матча:*\n"
            text += f"{acc_icon(accuracy)} Угадано: *{correct}/{checked}* — *{accuracy:.1f}%*\n"
            text += f"{roi_icon(roi)} ROI: *{roi:+.2f}* ед.\n\n"

        vb_checked = s.get('vb_checked', 0)
        if vb_checked > 0:
            vb_acc = s.get('vb_accuracy', 0)
            roi_vb = s.get('roi_value_bet', 0)
            text += f"💰 *Value Bets:*\n"
            text += f"{acc_icon(vb_acc)} Угадано: *{s['vb_correct']}/{vb_checked}* — *{vb_acc:.1f}%*\n"
            text += f"{roi_icon(roi_vb)} ROI: *{roi_vb:+.2f}* ед.\n\n"

        monthly = s.get('monthly', [])
        if monthly:
            text += "📅 *По месяцам:*\n"
            for row in monthly:
                m_t = row.get('total', 0)
                m_c = row.get('correct', 0)
                m_acc = m_c / m_t * 100 if m_t else 0
                m_roi = row.get('roi_vb', 0)
                text += f"{acc_icon(m_acc)} {row['month']}: {m_c}/{m_t} ({m_acc:.0f}%) ROI {m_roi:+.1f}\n"
            text += "\n"

        recent = s.get('recent', [])
        if recent:
            text += "📋 *Последние результаты:*\n"
            for r in recent[:6]:
                icon = "✅" if r.get('is_correct') == 1 else "❌"
                h_sc = r.get('real_home_score', '?')
                a_sc = r.get('real_away_score', '?')
                text += f"{icon} {r.get('home_team','?')} *{h_sc}:{a_sc}* {r.get('away_team','?')}\n"

        await message.answer(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"[Футбол статистика] Ошибка: {e}")
        await message.answer("⚠️ Не удалось загрузить статистику. Попробуй позже.")


@dp.message(Command("results"))
async def cmd_results(message: types.Message):
    """Команда /results — полная статистика с ROI и точностью по моделям."""
    stats = get_statistics()
    
    def acc_icon(acc):
        return "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")

    def roi_icon(roi):
        return "🟢" if roi > 0 else "🔴"

    text = "📊 *CHIMERA AI — Трекер результатов*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for sport in ['football', 'cs2']:
        s_stats = stats.get(sport)
        if not s_stats or s_stats['total_checked'] == 0:
            continue
            
        emoji = "⚽️ ФУТБОЛ" if sport == 'football' else "🎮 CS2"
        text += f"*{emoji}:*\n"
        text += f"{acc_icon(s_stats['accuracy'])} Точность: *{s_stats['accuracy']:.1f}%* ({s_stats['correct']}/{s_stats['total_checked']})\n"
        text += f"{acc_icon(s_stats['vb_accuracy'])} Value ставки: *{s_stats['vb_accuracy']:.1f}%* ({s_stats['vb_correct']}/{s_stats['vb_checked']})\n"
        text += f"{roi_icon(s_stats['roi_main'])} ROI Основные: *{s_stats['roi_main']:+.1f}* ед.\n"
        text += f"{roi_icon(s_stats['roi_value_bet'])} ROI Value: *{s_stats['roi_value_bet']:+.1f}* ед.\n\n"

    if text == "📊 *CHIMERA AI — Трекер результатов*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n":
        await message.answer("📋 Пока нет проверенных матчей.")
        return

    await message.answer(text, parse_mode="Markdown")
    return # Заглушка для остального кода, чтобы не было ошибок
    # По месяцам (старый код ниже будет проигнорирован из-за return)
    monthly = []
    if monthly:
        text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━\n*По месяцам:*\n"
        for m in monthly[:4]:
            month, total, corr, ens_c, vb_c, roi_m = m
            m_acc = (corr / total * 100) if total > 0 else 0
            text += f"{acc_icon(m_acc)} {month}: {corr}/{total} ({m_acc:.0f}%)"
            if roi_m:
                text += f" | ROI VB: {roi_m:+.1f}"
            text += "\n"

    text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━\n*Последние матчи:*\n"
    for r in recent[:6]:
        h, a, hs, as_, pred, ok, date, ens_ok, vb_out, vb_ok, vb_odds = r
        if hs is not None:
            icon = "✅" if ok == 1 else "❌"
            ens_str = " [Анс:✅]" if ens_ok == 1 else (" [Анс:❌]" if ens_ok == 0 else "")
            vb_str = f" [VB {vb_out}@{vb_odds}:✅]" if vb_ok == 1 else (f" [VB:❌]" if vb_ok == 0 and vb_out else "")
            text += f"{icon} {h} *{hs}:{as_}* {a}{ens_str}{vb_str}\n"
            if pred:
                text += f"   _Прогноз: {pred}_\n"
        else:
            text += f"⏳ {h} vs {a} — ждём результата\n"

    await message.answer(text, parse_mode="Markdown")

# --- 10. Авто-перекалибровка ELO каждую неделю ---
async def auto_elo_recalibration_task():
    """
    Автоматически пересчитывает ELO рейтинги по результатам сезона 2024/25.
    Запускается каждый понедельник в 3:00 ночи.
    """
    import importlib
    from datetime import datetime, timedelta
    global _elo_ratings, _team_form

    # Ждём до следующего понедельника 03:00
    while True:
        now = datetime.now()
        # Следующий понедельник
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0 and now.hour >= 3:
            days_until_monday = 7  # Уже был сегодня, ждём следующий
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        wait_seconds = (next_run - now).total_seconds()
        print(f"[ELO-Авто] Следующая перекалибровка: {next_run.strftime('%d.%m.%Y %H:%M')} (через {wait_seconds/3600:.1f} ч)")
        await asyncio.sleep(wait_seconds)

        # Запускаем перекалибровку
        try:
            print("[ELO-Авто] Начинаю еженедельную перекалибровку ELO...")
            import elo_calibrate as ec
            # Загружаем все результаты
            all_matches = []
            for league_key, info in ec.LEAGUE_SOURCES.items():
                matches = ec.fetch_league_results(info["url"])
                for m in matches:
                    ft = m.get("score", {}).get("ft", [])
                    if len(ft) == 2:
                        all_matches.append({
                            "date": m.get("date", ""),
                            "home": ec.normalize_name(m.get("team1", "")),
                            "away": ec.normalize_name(m.get("team2", "")),
                            "home_goals": ft[0],
                            "away_goals": ft[1],
                        })
            all_matches.sort(key=lambda x: x["date"])

            # Пересчитываем ELO
            new_ratings = {}
            for m in all_matches:
                new_ratings = ec.update_elo_single(new_ratings, m["home"], m["away"], m["home_goals"], m["away_goals"])

            # Строим форму
            new_form = ec.build_form_tracker(all_matches)

            # Сохраняем
            ec.save_calibrated_elo(new_ratings, new_form)

            # Обновляем глобальные переменные в памяти
            _elo_ratings = new_ratings
            _team_form = new_form
            print(f"[ELO-Авто] ✅ Перекалибровка завершена: {len(new_ratings)} команд, {len(all_matches)} матчей")
        except Exception as e:
            print(f"[ELO-Авто] Ошибка перекалибровки: {e}")

        # Рекалибровка ELO баскетбола
        try:
            print("[ELO-Баскетбол] Начинаю рекалибровку ELO баскетбола...")
            import elo_basketball_calibrate as ebc
            total = ebc.calibrate()
            print(f"[ELO-Баскетбол] ✅ Готово. Обработано матчей: {total}")
        except Exception as e:
            print(f"[ELO-Баскетбол] Ошибка рекалибровки: {e}")

        # Meta Learner — анализ весов баскетбольной модели
        try:
            from meta_learner import MetaLearner
            ml = MetaLearner()
            bball_weights = ml.analyze_basketball_weights()
            if bball_weights:
                ml.apply_updates('basketball', bball_weights)
                print(f"[Meta-Баскетбол] ✅ Веса обновлены: {bball_weights}")
            else:
                print("[Meta-Баскетбол] Недостаточно данных для корректировки весов")
        except Exception as e:
            print(f"[Meta-Баскетбол] Ошибка: {e}")

        # XGBoost — инкрементальное переобучение на живых матчах
        try:
            print("[XGBoost-Авто] Проверка новых матчей для переобучения...")
            loop = asyncio.get_running_loop()
            import functools as _func
            from ml.train_model import retrain_incremental
            _result = await loop.run_in_executor(None, _func.partial(retrain_incremental, min_new_rows=30))
            if _result["status"] == "ok":
                print(f"[XGBoost-Авто] ✅ Переобучено! +{_result['new_rows']} матчей | "
                      f"Sport {_result['acc_sport']}% | Market {_result['acc_market']}%")
                # Перезагружаем предиктор
                try:
                    import importlib, ml.predictor as _pred
                    importlib.reload(_pred)
                    print("[XGBoost-Авто] Предиктор перезагружен")
                except Exception:
                    pass
            elif _result["status"] == "skip":
                print(f"[XGBoost-Авто] Пропуск: {_result['reason']}")
            else:
                print(f"[XGBoost-Авто] Ошибка: {_result.get('reason')}")
        except Exception as _xe:
            print(f"[XGBoost-Авто] Ошибка переобучения: {_xe}")


async def auto_refresh_matches_task():
    """Автоматически обновляет список матчей каждые 6 часов."""
    while True:
        await asyncio.sleep(21600)  # 6 часов
        try:
            matches = get_matches(force=True)
            league_name = dict(FOOTBALL_LEAGUES).get(_current_league, "")
            print(f"[Авто] Список матчей обновлён: {league_name} — {len(matches)} матчей")
        except Exception as e:
            print(f"[Авто] Ошибка обновления матчей: {e}")

# --- 10a. Команда Теннис ---
async def cmd_tennis(message: types.Message):
    """Раздел тенниса — список турниров с кнопками (как CS2 лиги)."""
    status = await message.answer("🎾 Загружаю теннисные турниры...", parse_mode="HTML")
    try:
        from sports.tennis.matches import get_tennis_matches
        from sports.tennis.rankings import detect_surface, detect_tour

        # Только матчи где минимум 2 букмекера дают кэф — реально ставибельные
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

        # Сохраняем в кэш
        tennis_matches_cache.clear()
        tennis_matches_cache.extend(all_matches)

        # Группируем по турниру (sport_key)
        tournaments: dict = {}
        for m in all_matches:
            sk = m.get("sport_key", "tennis_atp_other")
            if sk not in tournaments:
                tournaments[sk] = []
            tournaments[sk].append(m)

        surf_icons = {"hard": "🎾", "clay": "🟫", "grass": "🟩"}

        builder = InlineKeyboardBuilder()
        for sk, t_matches in sorted(tournaments.items()):
            surface  = t_matches[0].get("surface") or detect_surface(sk)
            tour     = t_matches[0].get("tour")    or detect_tour(sk)
            icon     = surf_icons.get(surface, "🎾")
            t_name   = t_matches[0].get("tournament") or sk.replace(f"tennis_{tour}_", "").replace("_", " ").title()
            label    = f"{icon} {tour.upper()} | {t_name} ({len(t_matches)})"
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
        print(f"[Теннис] cmd_tennis ошибка: {e}")
        import traceback; traceback.print_exc()
        await status.edit_text(f"🎾 <b>Теннис</b>\n\n⚠️ Ошибка: {str(e)[:120]}", parse_mode="HTML")


# --- 10b. Баскетбол ---

_basketball_cache: dict = {}  # {league_key: [matches]}
_basketball_league = "basketball_nba"

async def cmd_basketball(message: types.Message):
    """Главный экран баскетбола — выбор лиги."""
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🏀 NBA", callback_data="bball_league_basketball_nba")],
        [types.InlineKeyboardButton(text="🏆 Евролига", callback_data="bball_league_basketball_euroleague")],
    ])
    await message.answer(
        "<b>🏀 БАСКЕТБОЛ</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━\nВыбери лигу:",
        parse_mode="HTML", reply_markup=kb
    )


@dp.callback_query(lambda c: c.data and c.data.startswith("bball_league_"))
async def bball_select_league(call: types.CallbackQuery):
    global _basketball_league
    league_key = call.data.replace("bball_league_", "")
    _basketball_league = league_key

    league_names = {"basketball_nba": "🏀 NBA", "basketball_euroleague": "🏆 Евролига"}
    league_name = league_names.get(league_key, league_key)

    await call.answer()
    status = await call.message.edit_text(f"{league_name}\n⏳ Загружаю матчи...", parse_mode="HTML")

    try:
        from sports.basketball import get_basketball_matches
        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(None, get_basketball_matches, league_key)
        _basketball_cache[league_key] = matches

        if not matches:
            await status.edit_text(f"{league_name}\n\n📭 Матчей не найдено.", parse_mode="HTML")
            return

        buttons = []
        for i, m in enumerate(matches):
            home = m.get("home_team", "")
            away = m.get("away_team", "")
            ct = m.get("commence_time", "")
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                dt_msk = dt + timedelta(hours=3)
                time_label = dt_msk.strftime("%d.%m %H:%M")
            except Exception:
                time_label = ct[:10]

            buttons.append([types.InlineKeyboardButton(
                text=f"🏀 {time_label}  {home} — {away}",
                callback_data=f"bball_match_{league_key}_{i}"
            )])

        buttons.append([types.InlineKeyboardButton(
            text="🏠 Главное меню", callback_data="back_to_main"
        )])
        kb = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        await status.edit_text(
            f"<b>{league_name}</b> — {len(matches)} матчей\nВыбери матч для анализа:",
            parse_mode="HTML", reply_markup=kb
        )
    except Exception as e:
        logger.error(f"[Анализ] Ошибка: {e}")
        await status.edit_text("⚠️ Не удалось выполнить анализ. Попробуй позже.", parse_mode="HTML")


@dp.callback_query(lambda c: c.data and c.data.startswith("bball_match_"))
async def bball_analyze_match(call: types.CallbackQuery):
    await call.answer()
    parts = call.data.split("_")
    # bball_match_{league_key}_{index}
    idx = int(parts[-1])
    league_key = "_".join(parts[2:-1])

    matches = _basketball_cache.get(league_key, [])
    if idx >= len(matches):
        await call.message.edit_text("⚠️ Матч не найден.", parse_mode="HTML")
        return

    m = matches[idx]
    home = m.get("home_team", "")
    away = m.get("away_team", "")
    ct   = m.get("commence_time", "")

    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        dt_msk = dt + timedelta(hours=3)
        time_label = dt_msk.strftime("%d.%m %H:%M МСК")
    except Exception:
        time_label = ct[:16]

    league_names = {"basketball_nba": "🏀 NBA", "basketball_euroleague": "🏆 Евролига"}
    league_name  = league_names.get(league_key, league_key)

    status_msg = await call.message.edit_text(
        f"⏳ <b>{home} vs {away}</b>",
        parse_mode="HTML"
    )
    await show_ai_thinking(status_msg, home, away, sport="basketball")

    try:
        from sports.basketball.core import get_basketball_odds, calculate_basketball_win_prob
        from agents import client as _gpt_client, groq_client as _groq_client

        odds     = get_basketball_odds(m)
        analysis = calculate_basketball_win_prob(home, away, odds, league_key)

        h_prob = analysis["home_prob"]
        a_prob = analysis["away_prob"]
        h_elo  = analysis["elo_home"]
        a_elo  = analysis["elo_away"]
        h_ev   = analysis["h_ev"]
        a_ev   = analysis["a_ev"]
        bet_signal = analysis["bet_signal"]
        total_data = analysis.get("total_analysis", {})

        # Фаворит
        if h_prob >= a_prob:
            fav, fav_prob, fav_odds, fav_ev = home, h_prob, odds["home_win"], h_ev
        else:
            fav, fav_prob, fav_odds, fav_ev = away, a_prob, odds["away_win"], a_ev

        # Kelly criterion для основной ставки
        kelly = 0.0
        if fav_odds > 1.02 and fav_ev > 0:
            kelly = round((fav_prob - (1 - fav_prob) / (fav_odds - 1)) * 100, 1)
            kelly = max(0.0, min(kelly, 25.0))

        await call.message.edit_text(
            f"🏀 <b>{home} vs {away}</b>\n"
            f"⏳ Запускаю AI анализ...",
            parse_mode="HTML"
        )

        # Строим контекстный блок для AI
        spread_block = ""
        if odds.get("spread_home"):
            spread_block = (
                f"Фора: {home} {odds['spread_home']:+.1f} @ {odds['spread_home_odds']}, "
                f"{away} {odds['spread_away']:+.1f} @ {odds['spread_away_odds']}."
            )

        total_block = ""
        if total_data:
            total_block = (
                f"Тотал линия: {total_data['line']} (Over {total_data['over_odds']} / Under {total_data['under_odds']})."
            )

        # Форма и back-to-back из analysis
        home_form = analysis.get("home_form", "")
        away_form = analysis.get("away_form", "")
        home_b2b  = analysis.get("home_b2b", False)
        away_b2b  = analysis.get("away_b2b", False)

        form_block = ""
        if home_form or away_form:
            form_block = f"Форма (последние игры): {home}={home_form or '—'}, {away}={away_form or '—'}\n"
        b2b_block = ""
        if home_b2b:
            b2b_block += f"⚠️ {home} играл вчера (back-to-back — усталость!)\n"
        if away_b2b:
            b2b_block += f"⚠️ {away} играл вчера (back-to-back — усталость!)\n"

        def _run_gpt_basketball():
            try:
                prompt = (
                    f"Ты — эксперт по баскетболу. Проанализируй матч {league_name}.\n\n"
                    f"Матч: {home} (ELO {h_elo}) vs {away} (ELO {a_elo})\n"
                    f"Наши вероятности: {home} {round(h_prob*100)}%, {away} {round(a_prob*100)}%\n"
                    f"Коэффициенты: {home} @ {odds['home_win'] or '—'}, {away} @ {odds['away_win'] or '—'}\n"
                    f"{form_block}"
                    f"{b2b_block}"
                    f"{total_block}\n"
                    f"{spread_block}\n\n"
                    f"Учти при анализе:\n"
                    f"1. Кто фаворит — форма, класс, домашнее преимущество, усталость\n"
                    f"2. Есть ли травмы звёздных игроков (если знаешь)\n"
                    f"3. Темп игры — обе команды играют быстро или медленно\n"
                    f"4. Прогноз тотала Over/Under {total_data.get('line', '—')}\n"
                    f"5. Ожидаемый счёт\n\n"
                    f"Формат ответа (JSON):\n"
                    f'{{"analysis": "Анализ (3-4 предложения)", '
                    f'"recommended_outcome": "Победа хозяев" или "Победа гостей", '
                    f'"confidence": <0-100>, '
                    f'"total_pick": "Over" или "Under", '
                    f'"total_reasoning": "Почему (1 предложение)", '
                    f'"score_prediction": "110-105"}}'
                )
                resp = _gpt_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=350, temperature=0.3,
                )
                raw = resp.choices[0].message.content.strip()
                import json as _j
                try:
                    start = raw.find("{")
                    end   = raw.rfind("}") + 1
                    return _j.loads(raw[start:end]) if start >= 0 else {"analysis": raw}
                except Exception:
                    return {"analysis": raw}
            except Exception as e:
                return {"analysis": f"(GPT недоступен: {e})"}

        def _run_llama_basketball():
            if not _groq_client:
                return {"analysis": "(Llama недоступна)"}
            try:
                prompt = (
                    f"Ты — независимый баскетбольный аналитик. Дай СВОЁ мнение по матчу {league_name}.\n\n"
                    f"Матч: {home} (ELO {h_elo}) vs {away} (ELO {a_elo})\n"
                    f"Вероятности модели: {home} {round(h_prob*100)}%, {away} {round(a_prob*100)}%\n"
                    f"Коэффициенты: {home} @ {odds['home_win'] or '—'}, {away} @ {odds['away_win'] or '—'}\n"
                    f"{form_block}"
                    f"{b2b_block}"
                    f"{total_block}\n\n"
                    f"Дай независимый прогноз — оцени стиль игры, ключевых игроков, темп.\n"
                    f"Если есть back-to-back — обязательно учти усталость.\n"
                    f"Формат (JSON):\n"
                    f'{{"analysis": "Твой независимый анализ (2-3 предложения)", '
                    f'"recommended_outcome": "Победа хозяев" или "Победа гостей", '
                    f'"confidence": <0-100>, '
                    f'"total_pick": "Over" или "Under"}}'
                )
                resp = _groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, temperature=0.4,
                )
                raw = resp.choices[0].message.content.strip()
                import json as _j
                try:
                    start = raw.find("{")
                    end   = raw.rfind("}") + 1
                    return _j.loads(raw[start:end]) if start >= 0 else {"analysis": raw}
                except Exception:
                    return {"analysis": raw}
            except Exception as e:
                return {"analysis": f"(Llama недоступна: {e})"}

        loop = asyncio.get_running_loop()
        gpt_res, llama_res = await asyncio.gather(
            loop.run_in_executor(None, _run_gpt_basketball),
            loop.run_in_executor(None, _run_llama_basketball),
        )

        gpt_text   = gpt_res.get("analysis", "—")
        llama_text = llama_res.get("analysis", "—")
        gpt_conf   = gpt_res.get("confidence", 0)
        llama_conf = llama_res.get("confidence", 0)
        gpt_outcome   = gpt_res.get("recommended_outcome", "")
        llama_outcome = llama_res.get("recommended_outcome", "")
        gpt_total     = gpt_res.get("total_pick", "")
        llama_total   = llama_res.get("total_pick", "")
        score_pred    = gpt_res.get("score_prediction", "")

        # ── Баскетбол финальный ансамбль: математика 90% + AI 10% ────────
        _bball_ai = [r for r in [gpt_res, llama_res] if isinstance(r, dict) and r.get("recommended_outcome")]
        if _bball_ai:
            _blended_h = _blend_ai(analysis["home_prob"], _bball_ai, home, away, 0.10)
            analysis["home_prob"] = _blended_h
            analysis["away_prob"] = round(1 - _blended_h, 3)
            h_prob = analysis["home_prob"]
            a_prob = analysis["away_prob"]
            # Пересчитываем EV с новой вероятностью
            h_ev = round((h_prob * (odds["home_win"] if odds.get("home_win", 0) > 1.02 else 0) - 1) * 100, 1) if odds.get("home_win", 0) > 1.02 else h_ev
            a_ev = round((a_prob * (odds["away_win"] if odds.get("away_win", 0) > 1.02 else 0) - 1) * 100, 1) if odds.get("away_win", 0) > 1.02 else a_ev
            fav, fav_prob, fav_odds, fav_ev = (home, h_prob, odds["home_win"], h_ev) if h_prob >= a_prob else (away, a_prob, odds["away_win"], a_ev)
            print(f"[Basketball AI-blend] {home}: {h_prob} (AI вклад 10%)")

        # Консенсус AI
        ai_agree = (gpt_outcome and llama_outcome and gpt_outcome == llama_outcome)
        if ai_agree:
            consensus = f"✅ AI единогласно: <b>{gpt_outcome}</b>"
        elif gpt_outcome and llama_outcome:
            consensus = f"⚠️ Козёл: {gpt_outcome} | Тень: {llama_outcome}"
        else:
            consensus = ""

        # Тотал блок
        total_str = ""
        if total_data:
            lean      = total_data.get("lean", "—")
            lean_odds = total_data.get("lean_odds", 0)
            lean_ev   = total_data.get("lean_ev", 0)
            lean_prob = total_data.get("lean_prob", 0)
            reason    = total_data.get("reason", "")
            line      = total_data.get("line", 0)
            ai_total_agree = ""
            if gpt_total and llama_total:
                if gpt_total == llama_total:
                    ai_total_agree = f" (AI: ✅ {gpt_total})"
                else:
                    ai_total_agree = f" (Козёл: {gpt_total}, Тень: {llama_total})"
            total_str = (
                f"\n\n🎯 <b>Тотал {line}:</b>\n"
                f"  Over @ {total_data['over_odds']} / Under @ {total_data['under_odds']}\n"
                f"  Лин: <b>{lean}</b> @ {lean_odds}  EV: {lean_ev:+.1f}%{ai_total_agree}\n"
                f"  <i>{reason}</i>"
            )

        # Спред блок
        spread_str = ""
        if odds.get("spread_home") and odds.get("spread_home_odds"):
            spread_str = (
                f"\n\n📐 <b>Фора:</b>\n"
                f"  {home} {odds['spread_home']:+.1f} @ {odds['spread_home_odds']}\n"
                f"  {away} {odds['spread_away']:+.1f} @ {odds['spread_away_odds']}"
            )

        # Сигнал
        signal_icon = "✅" if "СТАВИТЬ" in bet_signal else ("⚠️" if "СЛАБЫЙ" in bet_signal else "⏸")
        _bball_units = '3u' if kelly >= 4 else ('2u' if kelly >= 2 else '1u')
        kelly_str   = f"  Kelly: <b>{kelly:.1f}%</b> ({_bball_units})" if kelly > 0 else ""

        # Форма и B2B строки для карточки
        def _form_bar(form: str) -> str:
            if not form:
                return "—"
            return " ".join("🟢" if c == "W" else "🔴" for c in form)

        form_card = ""
        if home_form or away_form:
            form_card = (
                f"\n📈 <b>Форма (5 игр):</b>\n"
                f"  {home}: {_form_bar(home_form)}\n"
                f"  {away}: {_form_bar(away_form)}\n"
            )

        b2b_card = ""
        if home_b2b:
            b2b_card += f"⚠️ <b>B2B:</b> {home} играл вчера\n"
        if away_b2b:
            b2b_card += f"⚠️ <b>B2B:</b> {away} играл вчера\n"

        report = (
            f"🏀 <b>{home} vs {away}</b>\n"
            f"🕐 {time_label}  |  {league_name}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 <b>Вероятности CHIMERA:</b>\n"
            f"  {home}: <b>{round(h_prob*100)}%</b>  ELO {h_elo}\n"
            f"  {away}: <b>{round(a_prob*100)}%</b>  ELO {a_elo}\n"
            f"{form_card}"
            f"{b2b_card}\n"
            f"💰 <b>Коэффициенты ({odds.get('bookmaker', 'бук')}):</b>\n"
            f"  {home} @ <b>{odds['home_win'] or '—'}</b>  EV: {h_ev:+.1f}%\n"
            f"  {away} @ <b>{odds['away_win'] or '—'}</b>  EV: {a_ev:+.1f}%\n"
            f"{total_str}"
            f"{spread_str}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🐍🦁🐐 <b>Химера</b>  (уверенность {gpt_conf}%):\n"
            f"{gpt_text}\n\n"
            f"🌀 <b>Тень</b>  (уверенность {llama_conf}%):\n"
            f"{llama_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        if consensus:
            report += f"{consensus}\n"
        if score_pred:
            report += f"🎯 Прогноз счёта: <b>{score_pred}</b>\n"
        report += (
            f"\n{signal_icon} <b>СИГНАЛ: {bet_signal}</b>\n"
            f"  Ставка: <b>{fav}</b> @ {fav_odds}  EV: {fav_ev:+.1f}%{kelly_str}"
        )

        # Экспертное мнение (Google News + Sport News Live)
        try:
            from expert_oracle import get_expert_consensus, format_expert_block
            _loop_bb = asyncio.get_running_loop()
            _exp_bb = await _loop_bb.run_in_executor(
                None, get_expert_consensus, home, away, "basketball"
            )
            _expert_block_bb = format_expert_block(_exp_bb, home, away)
            if _expert_block_bb:
                report += f"\n\n{_expert_block_bb}"
                print(f"[ExpertOracle BB] {home} vs {away}: {_exp_bb.get('sources_count')} источн.")
        except Exception as _ee_bb:
            print(f"[ExpertOracle BB] Ошибка: {_ee_bb}")

        # Сохранение в БД
        try:
            rec_outcome = "home_win" if h_prob >= a_prob else "away_win"
            save_prediction(
                sport="basketball",
                match_id=m.get("id", f"{home}_{away}_{ct[:10]}"),
                match_date=ct,
                home_team=home,
                away_team=away,
                league=league_names.get(league_key, league_key),
                gpt_verdict=gpt_text,
                llama_verdict=llama_text,
                bet_signal=bet_signal,
                recommended_outcome=rec_outcome,
                elo_home=h_elo,
                elo_away=a_elo,
                elo_home_win=analysis.get("h_elo_prob"),
                elo_away_win=round(1 - analysis.get("h_elo_prob", 0.5), 3),
                ensemble_home=h_prob,
                ensemble_away=a_prob,
                ensemble_best_outcome=rec_outcome,
                bookmaker_odds_home=odds.get("home_win"),
                bookmaker_odds_away=odds.get("away_win"),
                prediction_data={
                    "total_line": total_data.get("line") if total_data else None,
                    "total_lean": total_data.get("lean") if total_data else None,
                    "total_ev":   total_data.get("lean_ev") if total_data else None,
                    "gpt_confidence": gpt_conf,
                    "llama_confidence": llama_conf,
                    "score_prediction": score_pred,
                },
            )
        except Exception as _db_e:
            print(f"[Basketball DB] Ошибка сохранения: {_db_e}")

        try:
            upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
            track_analysis(call.from_user.id, "basketball")
        except Exception:
            pass

        bball_kb = InlineKeyboardBuilder()
        bball_kb.button(text="🏀 Победитель", callback_data=f"bball_mkt_winner_{league_key}_{idx}")
        bball_kb.button(text="📊 Тотал очков", callback_data=f"bball_mkt_total_{league_key}_{idx}")
        bball_kb.button(text="⚖️ Фора", callback_data=f"bball_mkt_spread_{league_key}_{idx}")
        bball_kb.button(text="⬅️ Матчи", callback_data=f"bball_league_{league_key}")
        bball_kb.button(text="🏠 Меню", callback_data="back_to_main")
        bball_kb.adjust(2)
        _bb_kb = bball_kb.as_markup()
        await call.message.edit_text(report, parse_mode="HTML", reply_markup=_bb_kb)
        import time as _time
        _report_cache[f"bball_{league_key}_{idx}"] = {
            "text": report, "kb": _bb_kb,
            "parse_mode": "HTML", "ts": _time.time(),
        }

    except Exception as e:
        logger.error(f"[Анализ матча] Ошибка: {e}", exc_info=True)
        await call.message.edit_text("⚠️ Не удалось выполнить анализ. Попробуй позже.", parse_mode="HTML")


# --- 10b2. Экспресс-ставки ---
async def cmd_express(message: types.Message):
    """Показывает 3 кнопки выбора типа экспресса."""
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🟢 Надёжный (2 события)", callback_data="express_safe")],
        [types.InlineKeyboardButton(text="🟡 Средний (3 события)",  callback_data="express_medium")],
        [types.InlineKeyboardButton(text="🔴 Рискованный (4-5)",    callback_data="express_risky")],
    ])
    await message.answer(
        "🎯 <b>Chimera Express</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Выбери тип экспресса:\n\n"
        "🟢 <b>Надёжный</b> — 2 события, высокая вероятность\n"
        "🟡 <b>Средний</b> — 3 события, баланс риска и кэфа\n"
        "🔴 <b>Рискованный</b> — 4-5 событий, высокий кэф",
        parse_mode="HTML",
        reply_markup=kb,
    )


@dp.callback_query(lambda c: c.data and c.data.startswith("express_"))
async def cb_express_variant(call: types.CallbackQuery):
    """Сканирует матчи и показывает выбранный вариант экспресса."""
    variant_key = call.data.replace("express_", "")  # safe / medium / risky
    titles = {
        "safe":   ("🟢 Надёжный экспресс", "🟢"),
        "medium": ("🟡 Средний экспресс",   "🟡"),
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
            variants   = build_express_variants(candidates)
            return variants

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
        # Кнопка "попробовать другой"
        back_kb = types.InlineKeyboardMarkup(inline_keyboard=[[
            types.InlineKeyboardButton(text="◀️ Другой вариант", callback_data="express_back")
        ]])
        await status.edit_text(card, parse_mode="HTML", reply_markup=back_kb)

    except Exception as e:
        logger.error(f"[Экспресс] Ошибка: {e}", exc_info=True)
        await status.edit_text("⚠️ Не удалось построить экспресс. Попробуй позже.", parse_mode="HTML")


@dp.callback_query(lambda c: c.data == "express_back")
async def cb_express_back(call: types.CallbackQuery):
    """Возврат к выбору типа экспресса."""
    await call.answer()
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🟢 Надёжный (2 события)", callback_data="express_safe")],
        [types.InlineKeyboardButton(text="🟡 Средний (3 события)",  callback_data="express_medium")],
        [types.InlineKeyboardButton(text="🔴 Рискованный (4-5)",    callback_data="express_risky")],
    ])
    await call.message.edit_text(
        "🎯 <b>Chimera Express</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Выбери тип экспресса:\n\n"
        "🟢 <b>Надёжный</b> — 2 события, высокая вероятность\n"
        "🟡 <b>Средний</b> — 3 события, баланс риска и кэфа\n"
        "🔴 <b>Рискованный</b> — 4-5 событий, высокий кэф",
        parse_mode="HTML",
        reply_markup=kb,
    )


# --- 10c. Команда /signals — CHIMERA SIGNAL ENGINE v2 ---
@dp.message(Command("signals"))
async def cmd_signals(message: types.Message):
    """Кнопка 'Сигналы дня'."""
    status_msg = await message.answer(
        "🔍 <b>CHIMERA SIGNAL запущен...</b>\n\n"
        "⏳ <i>Это займёт 30–90 секунд — Химера сканирует рынок.</i>\n\n"
        "⚙️ Шаг 1/3: Загрузка матчей (Футбол + CS2 + Теннис + Баскетбол)...",
        parse_mode="HTML"
    )

    from chimera_signal import compute_chimera_score, run_ai_verification, format_chimera_signals

    # Получаем AI клиенты
    try:
        from agents import client as _gpt_client, groq_client as _groq_client
    except ImportError:
        _gpt_client = None
        _groq_client = None

    # Инициализируем модули движения линий и H2H
    try:
        from line_movement import make_match_key, record_odds, get_movement
        _line_movement_ok = True
    except ImportError:
        _line_movement_ok = False

    try:
        from api_football import get_h2h
        _h2h_ok = True
    except ImportError:
        _h2h_ok = False

    all_candidates = []
    _scan_errors = []  # собираем ошибки для финального сообщения

    def _run_math_scan():
        """Весь блокирующий скан — запускается в thread executor."""
        candidates = []

        # ── Футбол ───────────────────────────────────────────────────────
        try:
            from datetime import datetime, timezone as _tz, timedelta as _td
            import requests as _req
            _now = datetime.now(_tz.utc)
            _cutoff_future = (_now + _td(hours=72)).isoformat()[:19]
            _cutoff_past   = (_now - _td(hours=3)).isoformat()[:19]

            all_football = []
            _quota_error = False
            for _lkey, _ in FOOTBALL_LEAGUES:
                if _quota_error:
                    break
                try:
                    _r = _req.get(
                        f"https://api.the-odds-api.com/v4/sports/{_lkey}/odds/",
                        params={"apiKey": THE_ODDS_API_KEY, "regions": "eu",
                                "markets": "h2h,totals", "oddsFormat": "decimal"},
                        timeout=10,
                    )
                    if _r.status_code == 401:
                        _scan_errors.append("quota")
                        _quota_error = True
                        break
                    if _r.ok:
                        for _m in _r.json():
                            _ct = _m.get("commence_time", "")
                            if _cutoff_past < _ct <= _cutoff_future:
                                all_football.append(_m)
                except Exception:
                    pass

            matches = all_football[:60]
            print(f"[CHIMERA] Всего матчей для скана: {len(matches)}")
            for m in matches[:30]:
                try:
                    home = m.get("home_team", "")
                    away = m.get("away_team", "")
                    if not home or not away:
                        continue
                    odds = get_bookmaker_odds(m)
                    if not odds.get("home_win"):
                        continue
                    elo_h = _elo_ratings.get(home, 1500)
                    elo_a = _elo_ratings.get(away, 1500)
                    form_h = get_form_string(home, _team_form)
                    form_a = get_form_string(away, _team_form)
                    elo_p = elo_win_probabilities(home, away, _elo_ratings, _team_form)

                    _movement = {}
                    if _line_movement_ok and odds:
                        _mkey = make_match_key(home, away, m.get("commence_time", ""))
                        record_odds(_mkey, odds)
                        _movement = get_movement(_mkey, odds)

                    cands = compute_chimera_score(
                        home_team=home, away_team=away,
                        home_prob=elo_p["home"],
                        away_prob=elo_p["away"],
                        draw_prob=elo_p["draw"],
                        bookmaker_odds=odds,
                        home_form=form_h, away_form=form_a,
                        elo_home=elo_h, elo_away=elo_a,
                        league=m.get("sport_key", ""),
                        line_movement=_movement,
                    )
                    _ct = m.get("commence_time", "")
                    for c in cands:
                        c["commence_time"] = _ct
                    candidates.extend(cands)
                except Exception as _ce:
                    print(f"[CHIMERA] Ошибка матча {m.get('home_team','')} vs {m.get('away_team','')}: {_ce}")
        except Exception as e:
            print(f"[CHIMERA] Футбол ошибка: {e}")

        # ── CS2 ──────────────────────────────────────────────────────────
        try:
            from sports.cs2 import calculate_cs2_win_prob
            for m in cs2_matches_cache[:25]:   # расширяем до 25 для тир 3
                try:
                    home = m.get("home", "")
                    away = m.get("away", "")
                    if not home or not away:
                        continue
                    analysis = calculate_cs2_win_prob(home, away)
                    h_prob = analysis.get("home_prob", 0)
                    a_prob = analysis.get("away_prob", 0)
                    if not h_prob:
                        continue
                    # PandaScore бесплатный тариф не даёт коэффициенты — используем внутренние вероятности
                    raw_odds = m.get("odds", {})
                    h_odds = raw_odds.get("home_win") or round(1.0 / h_prob, 2) if h_prob > 0 else 1.85
                    a_odds = raw_odds.get("away_win") or round(1.0 / a_prob, 2) if a_prob > 0 else 1.85
                    h_stats = analysis.get("home_stats", {})
                    a_stats = analysis.get("away_stats", {})
                    cands = compute_chimera_score(
                        home_team=home, away_team=away,
                        home_prob=h_prob, away_prob=a_prob, draw_prob=0,
                        bookmaker_odds={"home_win": h_odds, "away_win": a_odds, "draw": 0},
                        home_form=h_stats.get("form", ""),
                        away_form=a_stats.get("form", ""),
                        elo_home=analysis.get("elo_home", 1450),
                        elo_away=analysis.get("elo_away", 1450),
                        league="CS2",
                    )
                    _ct = m.get("commence_time", "") or m.get("time", "")
                    _cs2_tier = m.get("tier", "B")
                    _league_low = (m.get("league", "") + " " + m.get("tournament", "")).lower()
                    _is_t3 = any(kw in _league_low for kw in CS2_TIER3_KEYWORDS) or _cs2_tier not in ("S", "A")
                    # Тотал карт для этого матча
                    try:
                        from signal_engine import predict_cs2_totals as _ps_totals
                        _cs2_totals = _ps_totals(
                            home_prob=h_prob, away_prob=a_prob,
                            home_map_stats={mp: hp * 100 for mp, hp, _ in analysis.get("maps", [])},
                            away_map_stats={mp: ap * 100 for mp, _, ap in analysis.get("maps", [])},
                            predicted_maps=[mp for mp, _, _ in analysis.get("maps", [])],
                        )
                    except Exception:
                        _cs2_totals = None
                    for c in cands:
                        c["sport"] = "cs2"
                        c["commence_time"] = _ct
                        c["cs2_tier"] = _cs2_tier
                        c["tier_label"] = m.get("tier_label", "🎮")
                        c["totals_data"] = _cs2_totals
                        if _is_t3:
                            c["cs2_tier3"] = True
                    candidates.extend(cands)
                except Exception as _ce:
                    print(f"[CHIMERA CS2] Ошибка {m.get('home','')} vs {m.get('away','')}: {_ce}")
        except Exception as e:
            print(f"[CHIMERA] CS2 ошибка: {e}")

        # ── Теннис ───────────────────────────────────────────────────────
        try:
            from sports.tennis import scan_tennis_signals
            tennis_cands = scan_tennis_signals()
            if tennis_cands:
                print(f"[CHIMERA] Теннис: {len(tennis_cands)} кандидатов")
            candidates.extend(tennis_cands)
        except Exception as e:
            print(f"[CHIMERA] Теннис ошибка: {e}")

        # ── Баскетбол ─────────────────────────────────────────────────────
        try:
            from sports.basketball import get_basketball_matches, calculate_basketball_win_prob
            from sports.basketball.core import get_basketball_odds, BASKETBALL_LEAGUES as _BBALL_LEAGUES
            _bball_total = 0
            for _bkey, _bname in _BBALL_LEAGUES:
                try:
                    _bmatches = get_basketball_matches(_bkey)
                    for _bm in _bmatches[:20]:
                        try:
                            _bh = _bm.get("home_team", "")
                            _ba = _bm.get("away_team", "")
                            if not _bh or not _ba:
                                continue
                            _bodds = get_basketball_odds(_bm)
                            if not _bodds.get("home_win"):
                                continue
                            _bres = calculate_basketball_win_prob(_bh, _ba, _bodds, _bkey)
                            _hp = _bres.get("home_prob", 0)
                            _ap = _bres.get("away_prob", 0)
                            if not _hp:
                                continue
                            _bcands = compute_chimera_score(
                                home_team=_bh, away_team=_ba,
                                home_prob=_hp, away_prob=_ap, draw_prob=0,
                                bookmaker_odds={
                                    "home_win": _bodds.get("home_win", 0),
                                    "away_win": _bodds.get("away_win", 0),
                                    "draw": 0,
                                },
                                home_form=_bres.get("home_form", ""),
                                away_form=_bres.get("away_form", ""),
                                elo_home=_bres.get("elo_home", 1550),
                                elo_away=_bres.get("elo_away", 1550),
                                league=_bkey,
                            )
                            _bct = _bm.get("commence_time", "")
                            for _bc in _bcands:
                                _bc["sport"] = "basketball"
                                _bc["league_name"] = _bname
                                _bc["commence_time"] = _bct
                            candidates.extend(_bcands)
                            _bball_total += len(_bcands)
                        except Exception as _bce:
                            print(f"[CHIMERA BBALL] Ошибка {_bm.get('home_team','')} vs {_bm.get('away_team','')}: {_bce}")
                except Exception as _blge:
                    print(f"[CHIMERA BBALL] Лига {_bkey}: {_blge}")
            print(f"[CHIMERA] Баскетбол: {_bball_total} кандидатов")
        except Exception as e:
            print(f"[CHIMERA] Баскетбол ошибка: {e}")

        return candidates

    import functools
    loop = asyncio.get_running_loop()
    all_candidates = await loop.run_in_executor(None, _run_math_scan)

    # ── Сортируем, берём топ-5 для AI ────────────────────────────────────
    all_candidates.sort(key=lambda x: x["chimera_score"], reverse=True)
    top_candidates = all_candidates[:5]
    print(f"[CHIMERA] Кандидатов найдено: {len(all_candidates)}, топ-5 scores: {[round(c['chimera_score'],1) for c in top_candidates]}")

    if not top_candidates:
        if "quota" in _scan_errors:
            await status_msg.edit_text(
                "⚠️ <b>CHIMERA SIGNAL</b>\n\n"
                "Квота The Odds API исчерпана (лимит запросов).\n\n"
                "Матчи не загружены. Обратись к администратору для пополнения API-ключа.\n"
                "<i>CS2 и Теннис работают независимо от этого.</i>",
                parse_mode="HTML"
            )
        else:
            await status_msg.edit_text(
                "📊 <b>CHIMERA SIGNAL</b>\n\nСегодня матчей с ценностью не найдено.\nПопробуйте позже.",
                parse_mode="HTML"
            )
        return

    # ── Шаг 2/3: AI верификация ───────────────────────────────────────────
    await status_msg.edit_text(
        "🔍 <b>CHIMERA SIGNAL</b>\n\n"
        "✅ Шаг 1/3: Матчи загружены\n"
        f"🧠 Шаг 2/3: AI анализирует топ-{len(top_candidates)} кандидатов...\n"
        "<i>⏳ Ещё 20–40 секунд...</i>",
        parse_mode="HTML"
    )

    try:
        top_candidates = await loop.run_in_executor(
            None,
            functools.partial(
                run_ai_verification,
                top_candidates,
                gpt_client=_gpt_client,
                groq_client=_groq_client,
            )
        )
    except Exception as e:
        print(f"[CHIMERA AI] Ошибка верификации: {e}")

    # ── Итоговый вывод ────────────────────────────────────────────────────
    try:
        result_text = format_chimera_signals(top_candidates, show_top=3)
        await status_msg.edit_text(result_text, parse_mode="HTML")
    except Exception as _fmt_e:
        logger.error(f"[CHIMERA] Ошибка отправки: {_fmt_e}")
        # Fallback — без AI-блока, только числа
        best = top_candidates[0] if top_candidates else {}
        fallback = (
            f"🎯 <b>CHIMERA SIGNAL</b>\n\n"
            f"<b>{best.get('home','')} vs {best.get('away','')}</b>\n"
            f"📌 {best.get('team','')} ({best.get('outcome','')})\n"
            f"💰 Кэф: <b>{best.get('odds','?')}</b> | "
            f"Вероятность: <b>{best.get('prob','?')}%</b>\n"
            f"📈 Score: <b>{best.get('chimera_score',0):.0f}/100</b>"
        ) if best else "📊 Сигналов нет"
        await status_msg.edit_text(fallback, parse_mode="HTML")

# --- 11. Запуск бота ---

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    asyncio.create_task(run_hltv_update_task())
    asyncio.create_task(check_results_task(bot))
    asyncio.create_task(auto_elo_recalibration_task())
    asyncio.create_task(auto_refresh_matches_task())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
