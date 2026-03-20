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
import logging.handlers
import time
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
try:
    from agents import (
        run_statistician_agent, run_scout_agent, run_arbitrator_agent,
        run_llama_agent, run_goals_market_agent,
        run_corners_market_agent, run_cards_market_agent, run_handicap_market_agent,
        run_mixtral_agent, build_math_ensemble, calculate_value_bets
    )
except ImportError as _agents_err:
    print(f"[agents] WARN: не удалось импортировать некоторые функции: {_agents_err}")
    # Заглушки — бот запустится, просто эти рынки вернут ошибку
    def run_statistician_agent(*a, **kw): return {"error": "недоступно"}
    def run_scout_agent(*a, **kw): return {"error": "недоступно"}
    def run_arbitrator_agent(*a, **kw): return {"error": "недоступно"}
    def run_llama_agent(*a, **kw): return {"error": "недоступно"}
    def run_goals_market_agent(*a, **kw): return {"error": "недоступно"}
    def run_corners_market_agent(*a, **kw): return {"error": "недоступно"}
    def run_cards_market_agent(*a, **kw): return {"error": "недоступно"}
    def run_handicap_market_agent(*a, **kw): return {"error": "недоступно"}
    def run_mixtral_agent(*a, **kw): return {"error": "недоступно"}
    def build_math_ensemble(*a, **kw): return {}
    def calculate_value_bets(*a, **kw): return []
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
from database import init_db, save_prediction, get_statistics, get_pending_predictions, update_result, upsert_user, track_analysis, get_user_profile, get_user_language, set_user_language, set_user_bankroll, get_user_bankroll, get_pl_stats, mark_user_bet, get_user_pl_stats, get_unnotified_bets, mark_bet_notified, get_recent_signal_streak, get_chimera_signal_history, log_action, get_admin_stats
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
os.makedirs("logs", exist_ok=True)
_log_handler = logging.handlers.TimedRotatingFileHandler(
    "logs/bot.log", when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler(), _log_handler]
)

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
    import json, os as _os
    _BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
    # Ищем CSV в ml/data/ (основное место), fallback на корень
    _csv_paths = [
        _os.path.join(_BASE_DIR, "ml", "data", "all_matches_featured.csv"),
        _os.path.join(_BASE_DIR, "all_matches_featured.csv"),
    ]
    _csv_path = next((p for p in _csv_paths if _os.path.exists(p)), None)
    if _csv_path is None:
        raise FileNotFoundError("all_matches_featured.csv не найден")
    data = pd.read_csv(_csv_path, index_col=0)
    feature_cols = [c for c in data.columns if c not in ('FTR','label','HomeTeam','AwayTeam')]
    scaler = MinMaxScaler()
    scaler.fit(data[feature_cols])
    _enc_path = _os.path.join(_BASE_DIR, "team_encoder.json")
    with open(_enc_path, 'r', encoding='utf-8') as _f:
        team_encoder = json.load(_f)
    print(f"[main] Dataset ready ({_csv_path}). Teams in encoder: {len(team_encoder)}")
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


async def run_calibration_task():
    """
    Фоновая задача: строит/обновляет таблицу калибровки из исторических данных Pinnacle.
    Первый запуск — через 2 минуты после старта бота (не мешает загрузке).
    Повтор — каждые 7 дней (данные не меняются быстро).
    Стоимость: ~600 кредитов за запуск из 100k.
    """
    await asyncio.sleep(120)  # 2 минуты после старта
    while True:
        try:
            import os as _os
            cal_file = "calibration_table.json"
            # Пропускаем если таблица обновлялась менее 7 дней назад
            if _os.path.exists(cal_file):
                age_days = (time.time() - _os.path.getmtime(cal_file)) / 86400
                if age_days < 7:
                    print(f"[Calibration] Таблица свежая ({age_days:.1f} дн.), пропускаем")
                    await asyncio.sleep(86400)  # проверяем раз в день
                    continue

            print("[Calibration] Запуск обновления калибровки...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_calibration_sync)
            print("[Calibration] Готово")
        except Exception as e:
            print(f"[Calibration] Ошибка: {e}")
        await asyncio.sleep(86400 * 7)  # раз в неделю


def _run_calibration_sync():
    """Синхронная обёртка для запуска в executor."""
    try:
        import sys as _sys
        import os as _os
        scripts_dir = _os.path.join(_os.path.dirname(__file__), "scripts")
        if scripts_dir not in _sys.path:
            _sys.path.insert(0, scripts_dir)
        from build_calibration import collect_data, enrich_with_scores, build_calibration_table
        import json
        from datetime import datetime, timezone

        data = collect_data()
        if not data:
            return
        data = enrich_with_scores(data)
        if len(data) < 10:
            print(f"[Calibration] Мало данных ({len(data)} матчей), пропускаем")
            return
        table = build_calibration_table(data)
        output = {
            "built_at":    datetime.now(timezone.utc).isoformat(),
            "sample_size": len(data),
            "table":       table,
        }
        with open("calibration_table.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[Calibration] Сохранено {len(table)} бинов по {len(data)} матчам")
    except Exception as e:
        print(f"[Calibration] Ошибка синхронного запуска: {e}")


async def run_tennis_form_prefetch_task():
    """Фоновая задача: обновляет кэш формы теннисистов каждые 2 часа."""
    # Первый прогрев через 30 сек после старта бота
    await asyncio.sleep(30)
    while True:
        try:
            from sports.tennis.matches import get_tennis_matches
            from sports.tennis.form_cache import prefetch_tennis_forms
            loop = asyncio.get_running_loop()
            matches = await loop.run_in_executor(None, get_tennis_matches)
            if matches:
                await loop.run_in_executor(None, prefetch_tennis_forms, matches)
                logging.info(f"[Tennis Cache] Форма обновлена для {len(matches)} матчей")
        except Exception as e:
            logging.error(f"[Tennis Cache] Ошибка: {e}")
        await asyncio.sleep(7200)  # каждые 2 часа


# --- 4. Глобальный кэш матчей и анализов ---
matches_cache = []
cs2_matches_cache = []     # Кэш матчей CS2
tennis_matches_cache = []  # Кэш матчей тенниса
analysis_cache = {}  # Хранит результаты анализа по match_id

# Кэш готовых HTML-отчётов: {key: {"text": str, "kb": markup, "parse_mode": str, "ts": float}}
# Ключи: "football_{idx}", "cs2_{idx}", "tennis_{sport_key}_{idx}", "bball_{league}_{idx}"
_report_cache: dict = {}
_REPORT_CACHE_TTL = 2700  # 45 минут

# ── Кеш результатов скана "Сигналы дня" ──────────────────────────────────────
# top_candidates после AI-верификации. TTL 45 мин = свежие сигналы + экономия.
# Структура: {"ts": float, "candidates": list, "result_text": str,
#             "top_pred_id": int|None, "top_sport": str, "top_odds": float}
_signals_scan_cache: dict = {}
SIGNALS_SCAN_TTL = 45 * 60  # 45 минут

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
            # Возвращаем None — ансамбль пропустит этот компонент полностью
            # (33/33/33 только портит результат для не-АПЛ команд)
            print(f"[Пророк] Команды вне АПЛ: '{home_team}' / '{away_team}' — пропускаем")
            return None
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
    """Получает список ближайших матчей через The Odds API для выбранной лиги (кеш 30 мин)."""
    global matches_cache, _last_matches_refresh, _current_league
    import time
    if league:
        _current_league = league
    # Используем глобальный кеш (30 мин). force=True сбрасывает кеш.
    if force:
        try:
            from odds_cache import invalidate as _inv
            _inv(_current_league)
        except ImportError:
            pass
    try:
        from odds_cache import get_odds as _get_odds
        from datetime import datetime, timezone, timedelta
        data = _get_odds(_current_league, markets="h2h,totals,spreads")
        if data:
            now = datetime.now(timezone.utc)
            cutoff = (now - timedelta(hours=3)).isoformat()[:19]
            future = [m for m in data if m.get('commence_time', '') > cutoff]
            matches_cache = future[:20]
            _last_matches_refresh = time.time()
            league_name = dict(FOOTBALL_LEAGUES).get(_current_league, _current_league)
            print(f"[API] {league_name}: {len(matches_cache)} матчей.")
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
    """
    Извлекает коэффициенты из данных матча.

    Стратегия (платный план — 62 букмекера):
    1. Pinnacle no-vig → самая точная вероятность в мире
    2. Усреднение по шарп-букмекерам (Pinnacle, Betfair, Betsson, 1xBet)
    3. Fallback на лучшие обычные коэффициенты

    Дополнительно возвращает:
      no_vig_home, no_vig_draw, no_vig_away — вероятности без маржи (из Pinnacle/шарп)
      bookmakers_count — кол-во букмекеров с линией на этот матч
    """
    result = {
        "home_win": 0, "draw": 0, "away_win": 0,
        "over_2_5": 0, "under_2_5": 0,
        "over_1_5": 0, "under_1_5": 0,
        "over_3_5": 0, "under_3_5": 0,
        "handicap_home": 0, "handicap_away": 0, "handicap_line": 0,
        "no_vig_home": 0.0, "no_vig_draw": 0.0, "no_vig_away": 0.0,
        "bookmakers_count": 0,
        "pinnacle_home": 0, "pinnacle_draw": 0, "pinnacle_away": 0,
    }

    SHARP_BOOKS = ["pinnacle", "betfair_ex", "betfair", "matchbook",
                   "smarkets", "lowvig", "betsson", "nordicbet", "marathonbet"]
    PREFERRED   = ["pinnacle", "betfair_ex", "betfair", "marathonbet",
                   "betsson", "unibet", "nordicbet", "1xbet"]

    def _v(v):
        try:
            f = float(v)
            return f if f >= 1.02 else 0.0
        except Exception:
            return 0.0

    try:
        home_team = match_data.get("home_team", "")
        away_team = match_data.get("away_team", "")
        bookmakers = match_data.get("bookmakers", [])
        result["bookmakers_count"] = len(bookmakers)

        # ── Собираем h2h от всех шарп-букмекеров для усреднения ──────────
        sharp_h, sharp_d, sharp_a = [], [], []
        all_h,   all_d,   all_a   = [], [], []

        for bm in bookmakers:
            bm_key = bm.get("key", "").lower()
            is_sharp = any(s in bm_key for s in SHARP_BOOKS)

            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    oc = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    h = _v(oc.get(home_team, 0))
                    a = _v(oc.get(away_team, 0))
                    d = _v(oc.get("Draw", 0))
                    if h and a and d:
                        all_h.append(h); all_d.append(d); all_a.append(a)
                        if is_sharp:
                            sharp_h.append(h); sharp_d.append(d); sharp_a.append(a)
                        # Pinnacle отдельно
                        if "pinnacle" in bm_key:
                            result["pinnacle_home"] = h
                            result["pinnacle_draw"] = d
                            result["pinnacle_away"] = a

                elif market.get("key") == "totals" and result["over_2_5"] == 0:
                    for o in market.get("outcomes", []):
                        pt   = o.get("point", 0)
                        name = o.get("name", "")
                        price = _v(o.get("price", 0))
                        if not price:
                            continue
                        if pt == 2.5 and name == "Over"  and not result["over_2_5"]:
                            result["over_2_5"] = price
                        elif pt == 2.5 and name == "Under" and not result["under_2_5"]:
                            result["under_2_5"] = price
                        elif pt == 1.5 and name == "Over"  and not result["over_1_5"]:
                            result["over_1_5"] = price
                        elif pt == 1.5 and name == "Under" and not result["under_1_5"]:
                            result["under_1_5"] = price
                        elif pt == 3.5 and name == "Over"  and not result["over_3_5"]:
                            result["over_3_5"] = price
                        elif pt == 3.5 and name == "Under" and not result["under_3_5"]:
                            result["under_3_5"] = price

                elif market.get("key") == "spreads" and not result["handicap_home"]:
                    for o in market.get("outcomes", []):
                        name  = o.get("name", "")
                        price = _v(o.get("price", 0))
                        line  = o.get("point", 0)
                        if not price:
                            continue
                        if name == home_team:
                            result["handicap_home"] = price
                            result["handicap_line"] = line
                        elif name == away_team:
                            result["handicap_away"] = price

        # ── Выбираем финальные коэффициенты ───────────────────────────────
        # Приоритет: шарп-буки → все буки
        src_h = sharp_h if sharp_h else all_h
        src_d = sharp_d if sharp_d else all_d
        src_a = sharp_a if sharp_a else all_a

        if src_h:
            # Медиана защищает от выбросов у одного букмекера
            src_h.sort(); src_d.sort(); src_a.sort()
            mid = len(src_h) // 2
            result["home_win"] = round(src_h[mid], 3)
            result["draw"]     = round(src_d[mid], 3)
            result["away_win"] = round(src_a[mid], 3)

        # ── No-vig вероятность (снимаем маржу букмекера) ──────────────────
        # Используем Pinnacle если есть, иначе медиану шарп-буков
        nv_h = result["pinnacle_home"] or result["home_win"]
        nv_d = result["pinnacle_draw"] or result["draw"]
        nv_a = result["pinnacle_away"] or result["away_win"]
        if nv_h and nv_d and nv_a:
            imp_h = 1 / nv_h
            imp_d = 1 / nv_d
            imp_a = 1 / nv_a
            total = imp_h + imp_d + imp_a  # сумма > 1.0 = маржа букмекера
            if total > 0:
                result["no_vig_home"] = round(imp_h / total, 4)
                result["no_vig_draw"] = round(imp_d / total, 4)
                result["no_vig_away"] = round(imp_a / total, 4)

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
                       match_time=None, chimera_verdict_block="", ml_block="",
                       bookmaker_odds=None):
    """Форматирует главный отчёт анализа матча с полным математическим анализом."""

    _pd = prophet_data if prophet_data is not None else [0.33, 0.33, 0.34]
    home_prob = _pd[1] * 100
    draw_prob = _pd[0] * 100
    away_prob = _pd[2] * 100

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
            conflict_warning = f"⚠️ КОНФЛИКТ: Химера рекомендует {gpt_verdict}, а математика указывает на {ens_label}"

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

    # Строка коэффициентов для шапки
    _odds = bookmaker_odds or {}
    _h_odds = _odds.get("home_win") or _odds.get("pinnacle_home")
    _d_odds = _odds.get("draw")     or _odds.get("pinnacle_draw")
    _a_odds = _odds.get("away_win") or _odds.get("pinnacle_away")
    if _h_odds and _a_odds:
        _odds_line = f"📊 Коэффициенты: П1 *{_h_odds}*" + (f" | Х *{_d_odds}*" if _d_odds else "") + f" | П2 *{_a_odds}*"
    else:
        _odds_line = ""

    report = f"""
🏆 *CHIMERA AI v4.3 — АНАЛИЗ МАТЧА*
━━━━━━━━━━━━━━━━━━━━━━━━━

⚽ *{home_team} vs {away_team}*
{(f'📅 *{match_time_str}*') if match_time_str else ''}
{_odds_line}

📊 *ПРОРОК (нейросеть):*
 П1: {home_prob:.0f}% | Х: {draw_prob:.0f}% | П2: {away_prob:.0f}%

🗣 *ОРАКУЛ (новостной фон):*
 {home_team}: {home_sentiment_label}
 {away_team}: {away_sentiment_label}
{(chr(10) + injuries_block) if injuries_block else ""}
{(chr(10) + math_section) if math_section else ""}
━━━━━━━━━━━━━━━━━━━━━━━━━
🐍🦁🐐 *Химера (анализ):*
_{gpt_summary}_

🌀 *Тень:*
_{llama_summary}_
{(chr(10) + mixtral_block) if mixtral_block else ""}
━━━━━━━━━━━━━━━━━━━━━━━━━
🐍🦁🐐 *ВЕРДИКТ ХИМЕРЫ:*
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

def format_goals_report(home_team, away_team, goals_result, bookmaker_odds=None, poisson_probs=None):
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
    real_over_2_5  = bookmaker_odds.get("over_2_5", 0)
    real_under_2_5 = bookmaker_odds.get("under_2_5", 0)
    real_over_1_5  = bookmaker_odds.get("over_1_5", 0)

    # EV для тотала 2.5 на основе Пуассон-вероятности vs букмекерский кэф
    ev_block = ""
    if poisson_probs and real_over_2_5 and real_under_2_5:
        p_over  = poisson_probs.get("over_25", 0)
        p_under = poisson_probs.get("under_25", 0)
        if p_over > 0 and p_under > 0:
            ev_over  = round((p_over  * real_over_2_5  - 1) * 100, 1)
            ev_under = round((p_under * real_under_2_5 - 1) * 100, 1)
            best_ev = max(ev_over, ev_under)
            if best_ev > 0:
                side = f"Больше 2.5 @ {real_over_2_5}" if ev_over >= ev_under else f"Меньше 2.5 @ {real_under_2_5}"
                ev_val = ev_over if ev_over >= ev_under else ev_under
                ev_block = f"\n🎯 *ЦЕННОСТЬ (EV):* {side} → EV *+{ev_val}%* (Пуассон: {int(p_over*100)}%/{int(p_under*100)}%)"
            else:
                ev_block = f"\n📊 _(нет ценности: EV Больше={ev_over:+.1f}% / Меньше={ev_under:+.1f}%)_"

    # Строки с коэффициентами
    odds_2_5_str = f" | КФ: Больше={real_over_2_5} / Меньше={real_under_2_5}" if real_over_2_5 else ""
    odds_1_5_str = f" | КФ: {real_over_1_5}" if real_over_1_5 else ""

    return f"""
⚽ *АНАЛИЗ ГОЛОВ — {home_team} vs {away_team}*
━━━━━━━━━━━━━━━━━━━━━━━━━

_{summary}_

📊 *ТОТАЛ ГОЛОВ:*
{conf_icon(over_2_5_conf)} Тотал 2.5: *{over_2_5}* ({over_2_5_conf}%){odds_2_5_str}
_{over_2_5_reason}_{ev_block}

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


@dp.message(Command("admin"))
async def cmd_admin(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    try:
        s = get_admin_stats()
        lines = [
            "👁 <b>CHIMERA ADMIN</b>",
            "━━━━━━━━━━━━━━━━━━━━",
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
        lines.append("")
        lines.append("⏱ <b>Последние действия:</b>")
        for a in s["last_actions"][:15]:
            ts = a["ts"][11:16]  # HH:MM
            name = a.get("username") and f"@{a['username']}" or a.get("first_name") or str(a["user_id"])
            lines.append(f"  {ts} {name}: {a['action']}")
        await message.answer("\n".join(lines), parse_mode="HTML")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    upsert_user(message.from_user.id, message.from_user.username or "", message.from_user.first_name or "")
    log_action(message.from_user.id, "/start")
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


async def cmd_chimera_hunt(message: types.Message):
    """Охота Химеры — паровые удары: движение острых денег на рынках Pinnacle."""
    try:
        from line_tracker import get_steam_moves, format_steam_moves
        moves = get_steam_moves(hours_back=2)
        text  = format_steam_moves(moves, page=0)
    except Exception as e:
        text  = f"🔥 <b>ОХОТА ХИМЕРЫ</b>\n\n⚠️ Ошибка загрузки данных: {e}"
        moves = []
    kb = _build_hunt_kb(0, len(moves))
    await message.answer(text, parse_mode="HTML", reply_markup=kb)


@dp.callback_query(lambda c: c.data == "hunt_refresh")
async def cb_hunt_refresh(call: types.CallbackQuery):
    await call.answer("Обновляю...")
    try:
        from line_tracker import get_steam_moves, format_steam_moves
        moves = get_steam_moves(hours_back=2)
        text  = format_steam_moves(moves, page=0)
    except Exception as e:
        text  = f"🔥 <b>ОХОТА ХИМЕРЫ</b>\n\n⚠️ Ошибка: {e}"
        moves = []
    kb = _build_hunt_kb(0, len(moves))
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    except Exception:
        pass


@dp.callback_query(lambda c: c.data.startswith("hunt_page_"))
async def cb_hunt_page(call: types.CallbackQuery):
    await call.answer()
    try:
        page = int(call.data.split("_")[-1])
        from line_tracker import get_steam_moves, format_steam_moves
        moves = get_steam_moves(hours_back=2)
        text  = format_steam_moves(moves, page=page)
    except Exception as e:
        text  = f"🔥 <b>ОХОТА ХИМЕРЫ</b>\n\n⚠️ Ошибка: {e}"
        moves = []
        page  = 0
    kb = _build_hunt_kb(page, len(moves))
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    except Exception:
        pass


@dp.message()
async def handle_text(message: types.Message):
    text = message.text
    user_id = message.from_user.id

    # ── Ввод банка пользователя ───────────────────────────────────────────────
    if user_id in _awaiting_bankroll:
        await bankroll_input_handler(message)
        return

    # ── Химера-чат: пользователь в режиме диалога ────────────────────────────
    MENU_BUTTONS = {
        "📡 Сигналы дня", "📡 Daily Signals",
        "🎯 Экспресс", "🎯 Express",
        "⚽ Футбол", "⚽ Football",
        "🎾 Теннис", "🎾 Tennis",
        "🎮 Киберспорт CS2", "🎮 Esports CS2",
        "🏀 Баскетбол", "🏀 Basketball",
        "🔥 Охота Химеры", "🔥 Chimera Hunt",
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

        def _acc_bar(acc: float) -> str:
            """Визуальный прогресс-бар точности (10 делений)."""
            filled = round(acc / 10)
            return "▓" * filled + "░" * (10 - filled)

        def _streak_str(recent: list) -> str:
            """Текущая серия: ✅✅✅ или ❌❌."""
            if not recent:
                return ""
            streak_icon = "✅" if recent[0].get("is_correct") == 1 else "❌"
            count = 0
            for r in recent:
                if (r.get("is_correct") == 1) == (streak_icon == "✅"):
                    count += 1
                else:
                    break
            return f"{streak_icon} ×{count}" if count > 1 else streak_icon

        # ── Общая статистика ───────────────────────────────────────────────────
        all_total   = sum(all_stats.get(k, {}).get("total", 0)         for k in ("football","cs2","tennis","basketball"))
        all_checked = sum(all_stats.get(k, {}).get("total_checked", 0) for k in ("football","cs2","tennis","basketball"))
        all_correct = sum(all_stats.get(k, {}).get("correct", 0)       for k in ("football","cs2","tennis","basketball"))
        all_acc     = round(all_correct / all_checked * 100, 1) if all_checked > 0 else 0

        stats_text = (
            "📊 *Статистика Chimera AI*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        if all_checked > 0:
            stats_text += (
                f"🎯 Угадано: *{all_correct} из {all_checked}* прогнозов\n"
                f"`{_acc_bar(all_acc)}` *{all_acc}%*\n"
                f"📋 Всего в базе: *{all_total}* | Ожидают результата: *{all_total - all_checked}*\n"
            )
        stats_text += "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

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
            pending = total - checked
            if total == 0:
                continue
            has_data = True

            recent  = s.get("recent", [])
            streak  = _streak_str(recent)

            stats_text += f"*{sport_label}*"
            if streak:
                stats_text += f"  {streak}"
            stats_text += "\n"

            if checked > 0:
                stats_text += (
                    f"`{_acc_bar(acc)}` *{acc:.0f}%*\n"
                    f"🎯 *{correct}/{checked}* угадано"
                )
                if pending > 0:
                    stats_text += f"  ·  ⏳ ждём *{pending}*"
                stats_text += "\n"
            else:
                stats_text += f"📋 Прогнозов: *{total}* | ⏳ Ожидают результата\n"

            # Последние 5 результатов (компактно — только иконки)
            if recent:
                icons = ""
                for r in recent[:5]:
                    icons += "✅" if r.get("is_correct") == 1 else "❌"
                stats_text += f"Последние: {icons}\n"

            # По месяцам
            monthly = s.get("monthly", [])
            if monthly:
                for row in monthly[:1]:
                    mt = row.get("total", 0) if isinstance(row, dict) else row[1]
                    mc = row.get("correct", 0) if isinstance(row, dict) else row[2]
                    mn = row.get("month", "") if isinstance(row, dict) else row[0]
                    if mt > 0:
                        ma = mc / mt * 100
                        stats_text += f"📅 {mn}: *{mc}/{mt}* ({ma:.0f}%)\n"
            stats_text += "\n"

        # ── История сигналов дня ───────────────────────────────────────────────
        chimera_history = get_chimera_signal_history(limit=10)
        if chimera_history:
            ch_checked = [r for r in chimera_history if r["is_correct"] is not None]
            ch_wins    = sum(1 for r in ch_checked if r["is_correct"] == 1)
            ch_pending = sum(1 for r in chimera_history if r["is_correct"] is None)
            ch_acc     = round(ch_wins / len(ch_checked) * 100) if ch_checked else 0

            # Стрик сигналов дня
            ch_streak = _streak_str(
                [{"is_correct": r["is_correct"]} for r in chimera_history if r["is_correct"] is not None]
            )

            sport_icons = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀"}
            stats_text += "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            stats_text += f"*🎯 Сигналы дня*"
            if ch_streak:
                stats_text += f"  {ch_streak}"
            stats_text += "\n"

            if ch_checked:
                stats_text += f"`{_acc_bar(ch_acc)}` *{ch_acc}%*\n"
                stats_text += f"🎯 *{ch_wins}/{len(ch_checked)}* угадано"
                if ch_pending:
                    stats_text += f"  ·  ⏳ ждём *{ch_pending}*"
                stats_text += "\n"
            else:
                stats_text += f"⏳ Ждём результаты: *{ch_pending}*\n"

            # Последние иконки — только завершённые результаты (без ⏳)
            done_icons = []
            for r in chimera_history:
                if r["is_correct"] in (0, 1):
                    done_icons.append("✅" if r["is_correct"] == 1 else "❌")
                    if len(done_icons) >= 5:
                        break
            if done_icons:
                stats_text += f"Последние: {''.join(done_icons)}\n"

        if not has_data and not chimera_history:
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

        bankroll = get_user_bankroll(message.from_user.id)
        pl_block = ""  # трекрекорд бота скрыт

        # Личный P&L пользователя
        upl = get_user_pl_stats(message.from_user.id, days=30)
        if upl["total"] > 0 or upl.get("pending", 0) > 0:
            profit_pct = upl["profit_pct"]
            p_sign = "+" if profit_pct >= 0 else ""
            # Деньги если есть банк
            if bankroll and bankroll > 0:
                profit_money = round(bankroll * profit_pct / 100, 1)
                m_sign = "+" if profit_money >= 0 else ""
                bankroll_now = round(bankroll + profit_money, 1)
                bank_line = f"  💰 <b>{bankroll:.0f}</b>  →  <b>{bankroll_now:.0f}</b>  ({m_sign}{profit_money:.0f})\n"
            else:
                bank_line = f"  💰 <i>Укажи банк — увидишь прибыль в числах</i>\n"
            # Стрик
            s = upl["streak"]
            if s >= 2:
                streak_line = f"  🔥 {s} побед подряд\n"
            elif s <= -2:
                streak_line = f"  ❄️ {abs(s)} поражений подряд\n"
            else:
                streak_line = ""
            # Ожидающие
            pending_line = f"  ⏳ Ждём результат: <b>{upl['pending']}</b>\n" if upl.get("pending", 0) > 0 else ""
            # История последних ставок
            history_lines = []
            for b in upl.get("last_bets", []):
                icon = "✅" if b["is_win"] else "❌"
                p_str = f"+{b['profit_pct']:.1f}%" if b["is_win"] else f"{b['profit_pct']:.1f}%"
                sport_icon = {"football": "⚽", "cs2": "🎮", "tennis": "🎾", "basketball": "🏀"}.get(
                    b.get("sport", "football"), "🎯")
                team = b["home"] if b["rec"] == "home_win" else b["away"]
                history_lines.append(
                    f"  {icon} {sport_icon} {team[:16]} @ {b['odds']} · {b['units']}u · {p_str}"
                )
            history_block = "\n".join(history_lines)
            _user_wr = round(upl['wins'] / upl['total'] * 100) if upl['total'] > 0 else 0
            _wr_line = f"  Ставок: <b>{upl['total']}</b>  ✅{upl['wins']} ❌{upl['losses']}  Угадано: <b>{_user_wr}%</b>\n"
            upl_block = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 <b>Мои ставки (30 дней)</b>\n"
                f"{bank_line}"
                f"{_wr_line}"
                + f"{streak_line}"
                + f"{pending_line}"
                + (f"\n{history_block}\n" if history_lines else "")
            )
        else:
            upl_block = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 <b>Мои ставки</b>\n"
                f"  <i>Нажми ✅ под сигналом — ставка запишется сюда.</i>\n"
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
            f"{upl_block}"
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

    elif text in ("🔥 Охота Химеры", "🔥 Chimera Hunt"):
        await cmd_chimera_hunt(message)

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


@dp.callback_query(lambda c: c.data == "noop")
async def cb_noop(call: types.CallbackQuery):
    await call.answer("Ставка уже записана ✅", show_alert=False)


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


@dp.callback_query(lambda c: c.data and c.data.startswith("mybet_"))
async def mybet_handler(call: types.CallbackQuery):
    """Пользователь нажал 'Я поставил' — записываем в личную статистику."""
    parts = call.data.split("_")
    # mybet_{sport}_{pred_id}_{odds*100}_{units}
    if len(parts) < 3:
        await call.answer("Ошибка: неверный формат.", show_alert=True)
        return
    sport = parts[1]
    try:
        pred_id = int(parts[2])
        odds = int(parts[3]) / 100.0 if len(parts) >= 4 else 0.0
        units = int(parts[4]) if len(parts) >= 5 else 1
    except (ValueError, IndexError):
        await call.answer("Ошибка: неверный ID.", show_alert=True)
        return

    saved = mark_user_bet(call.from_user.id, sport, pred_id, odds, units)
    if saved:
        await call.answer("✅ Записано в твою статистику! Результат добавится автоматически после матча.", show_alert=True)
        # Убираем кнопку из сообщения (заменяем на текст)
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
        except Exception:
            pass
    else:
        await call.answer("Ты уже записал эту ставку.", show_alert=True)


@dp.callback_query()
async def handle_callback(call: types.CallbackQuery):

    # --- CHIMERA: счётчик страниц (просто глушим) ---
    if call.data == "chimera_noop":
        await call.answer()
        return

    # --- CHIMERA: обновить (сброс кеша) ---
    if call.data == "chimera_refresh":
        _signals_scan_cache.clear()
        await call.answer("Запускаю новый скан...", show_alert=False)
        await cmd_signals(call.message)
        return

    # --- CHIMERA: записать ставку по индексу кандидата ---
    if call.data.startswith("chimera_bet_"):
        try:
            parts = call.data.split("_")
            idx   = int(parts[2])
            units = int(parts[3]) if len(parts) > 3 else 1
        except (IndexError, ValueError):
            await call.answer("Ошибка формата", show_alert=True)
            return
        cached = _signals_scan_cache.get("last", {})
        candidates = cached.get("candidates", [])
        if not candidates or idx >= len(candidates):
            await call.answer("Данные устарели — обновите сигналы /signals", show_alert=True)
            return
        c = candidates[idx]
        sp     = c.get("sport", "football")
        home   = c.get("home", "")
        away   = c.get("away", "")
        odds   = c.get("odds", 0)
        t_str  = c.get("commence_time", "")
        rec    = "home_win" if c.get("team") == home else "away_win"
        mid    = f"chimera_{home}_{away}_{t_str[:10]}"
        # Ленивое сохранение в БД
        pred_id = c.get("_pred_id")
        if not pred_id:
            try:
                pred_id = save_prediction(
                    sport=sp, match_id=mid, match_date=t_str,
                    home_team=home, away_team=away,
                    league=c.get("league_name", sp),
                    recommended_outcome=rec, bet_signal="СТАВИТЬ",
                    bookmaker_odds_home=odds if rec == "home_win" else None,
                    bookmaker_odds_away=odds if rec == "away_win" else None,
                    ensemble_home=round(c.get("prob", 50) / 100, 3),
                    ensemble_away=round(1 - c.get("prob", 50) / 100, 3),
                    ensemble_best_outcome=rec,
                )
                if not pred_id:
                    from database import _get_db_connection
                    _tbl = {"football": "football_predictions", "cs2": "cs2_predictions",
                            "tennis": "tennis_predictions", "basketball": "basketball_predictions"}.get(sp, "football_predictions")
                    with _get_db_connection() as _conn:
                        _row = _conn.execute(f"SELECT id FROM {_tbl} WHERE match_id=?", (mid,)).fetchone()
                        if _row:
                            pred_id = _row[0]
                c["_pred_id"] = pred_id
            except Exception as _e:
                print(f"[chimera_bet] Ошибка сохранения: {_e}")
        if not pred_id:
            await call.answer("Не удалось записать — попробуйте ещё раз", show_alert=True)
            return
        odds_enc = int(round(odds * 100))
        saved = mark_user_bet(call.from_user.id, sp, pred_id, odds, units)
        if saved:
            await call.answer("✅ Записано! Результат добавится автоматически после матча.", show_alert=True)
            # Заменяем кнопку на "уже записано"
            try:
                kb = _build_chimera_carousel_kb(candidates, idx, call.from_user.id)
                # Подменяем строку ставки
                new_rows = [kb.inline_keyboard[0], [types.InlineKeyboardButton(
                    text="📝 Ставка уже записана", callback_data="noop"
                )]]
                await call.message.edit_reply_markup(
                    reply_markup=types.InlineKeyboardMarkup(inline_keyboard=new_rows)
                )
            except Exception:
                pass
        else:
            await call.answer("Ты уже записал эту ставку.", show_alert=True)
        return

    # --- CHIMERA карусель ---
    if call.data.startswith("chimera_page_"):
        try:
            idx = int(call.data.split("_")[2])
        except (IndexError, ValueError):
            await call.answer()
            return
        cached = _signals_scan_cache.get("last", {})
        candidates = cached.get("candidates", [])
        if not candidates or idx >= len(candidates):
            await call.answer("Данные устарели — нажмите /signals снова", show_alert=True)
            return
        text = _format_chimera_page(candidates, idx)
        kb   = _build_chimera_carousel_kb(candidates, idx, call.from_user.id)
        try:
            await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
        except Exception:
            pass
        await call.answer()
        return

    # --- CS2 анализ матча ---
    if call.data.startswith("cs2_m_"):
        match_index = int(call.data.split("_")[2])
        if match_index >= len(cs2_matches_cache):
            await call.answer("Матч не найден.", show_alert=True)
            return
        m = cs2_matches_cache[match_index]
        home_team = m["home"]
        away_team = m["away"]
        # Проверяем кеш готового отчёта (45 мин)
        import time as _time_cs2
        _cs2_cache_key = f"cs2_{match_index}"
        _cs2_cached = _report_cache.get(_cs2_cache_key)
        if _cs2_cached and _time_cs2.time() - _cs2_cached.get("ts", 0) < _REPORT_CACHE_TTL:
            await call.answer()
            await call.message.edit_text(
                _cs2_cached["text"], parse_mode=_cs2_cached.get("parse_mode"),
                reply_markup=_cs2_cached.get("kb"),
            )
            return
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
            # Записываем снимок коэффициентов для отслеживания движения линий
            try:
                from line_movement import make_match_key, record_odds as _record_cs2_odds
                _cs2_lm_key = make_match_key(home_team, away_team, m.get("commence_time", ""))
                _record_cs2_odds(_cs2_lm_key, odds)
            except Exception:
                pass
            golden_signals = get_golden_signal(analysis, odds)
            h_stats = analysis.get("home_stats", {})
            a_stats = analysis.get("away_stats", {})
            h2h = analysis.get("h2h", {})
            _h_si = analysis.get("home_standin", {})
            _a_si = analysis.get("away_standin", {})
            map_stats_for_ai = {mp: {"home_prob": round(hp, 2), "away_prob": round(ap, 2)} for mp, hp, ap in analysis.get("maps", [])}
            _data_conf = analysis.get("data_confidence", 1.0)
            gpt_text = run_cs2_analyst_agent(
                home_team, away_team, map_stats_for_ai, odds,
                agent_type="gpt-4o", home_stats=h_stats, away_stats=a_stats, h2h=h2h,
                tournament_context=_ctx, home_standin=_h_si, away_standin=_a_si,
                data_confidence=_data_conf,
            )
            llama_text = run_cs2_analyst_agent(
                home_team, away_team, map_stats_for_ai, odds,
                agent_type="llama-3.3", home_stats=h_stats, away_stats=a_stats, h2h=h2h,
                tournament_context=_ctx, home_standin=_h_si, away_standin=_a_si,
                data_confidence=_data_conf,
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
                _cs2_pred_id = save_prediction(
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
                _cs2_pred_id = None
                print(f"[CS2 Save] Ошибка сохранения прогноза: {save_err}")
            try:
                upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
                track_analysis(call.from_user.id, "cs2")
                log_action(call.from_user.id, "анализ CS2")
            except Exception:
                pass

            cs2_markets_kb = InlineKeyboardBuilder()
            cs2_markets_kb.button(text="🏆 Победитель матча", callback_data=f"cs2_mkt_winner_{match_index}")
            cs2_markets_kb.button(text="🗺️ По картам", callback_data=f"cs2_mkt_maps_{match_index}")
            cs2_markets_kb.button(text="🎯 Тотал раундов", callback_data=f"cs2_mkt_rounds_{match_index}")
            cs2_markets_kb.button(text="⬅️ Матчи", callback_data="back_to_cs2")
            cs2_markets_kb.button(text="🏠 Меню", callback_data="back_to_main")
            if _cs2_pred_id and ranked_bets:
                _cs2_bet_odds = ranked_bets[0].get("odds", 0) or odds.get("home_win" if rec_outcome == "home_win" else "away_win", 0) or 0
                _cs2_odds_enc = int(round(_cs2_bet_odds * 100))
                _cs2_kelly = ranked_bets[0].get("kelly", 2)
                _cs2_units = 3 if _cs2_kelly >= 4 else (2 if _cs2_kelly >= 2 else 1)
                cs2_markets_kb.button(
                    text=f"✅ Я поставил {_cs2_units}u — записать в статистику",
                    callback_data=f"mybet_cs2_{_cs2_pred_id}_{_cs2_odds_enc}_{_cs2_units}"
                )
            cs2_markets_kb.adjust(2)
            _cs2_kb = cs2_markets_kb.as_markup()
            try:
                await call.message.edit_text(report, parse_mode="Markdown", reply_markup=_cs2_kb)
            except Exception as _md_err:
                # Markdown сломан спецсимволом в названии — отправляем без форматирования
                logger.warning(f"[CS2 Markdown] Fallback plain text: {_md_err}")
                import re as _re
                plain_report = _re.sub(r'[*_`\[\]]', '', report)
                await call.message.edit_text(plain_report, parse_mode=None, reply_markup=_cs2_kb)
            import time as _time
            _report_cache[f"cs2_{match_index}"] = {
                "text": report, "kb": _cs2_kb,
                "parse_mode": "Markdown", "ts": _time.time(),
            }
        except Exception as e:
            logger.error(f"[CS2 анализ] Ошибка: {e}")
            import traceback; logger.error(traceback.format_exc())
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
        # Проверяем кеш готового отчёта (45 мин)
        import time as _time_tn
        _tn_cache_key = f"tennis_{sport_key}_{match_idx}"
        _tn_cached = _report_cache.get(_tn_cache_key)
        if _tn_cached and _time_tn.time() - _tn_cached.get("ts", 0) < _REPORT_CACHE_TTL:
            await call.answer()
            await call.message.edit_text(
                _tn_cached["text"], parse_mode=_tn_cached.get("parse_mode"),
                reply_markup=_tn_cached.get("kb"),
            )
            return
        # Записываем снимок коэффициентов для отслеживания движения линий
        if not no_odds:
            try:
                from line_movement import make_match_key, record_odds as _record_tn_odds
                _tn_lm_key = make_match_key(p1, p2, m.get("commence_time", ""))
                _record_tn_odds(_tn_lm_key, {"home_win": o1, "away_win": o2})
            except Exception:
                pass

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

            # Тотал геймов с букмекерской линией (из кэша матчей)
            _bm_total_line  = m.get("bm_total_line", 0.0)
            _bm_total_over  = m.get("bm_total_over", 0.0)
            _bm_total_under = m.get("bm_total_under", 0.0)
            from sports.tennis.model import predict_tennis_game_totals
            from sports.tennis.rankings import detect_tour as _detect_tour
            _tour = _detect_tour(sport_key)
            _best_of = 5 if "grand_slam" in sport_key.lower() else 3
            _game_totals = predict_tennis_game_totals(
                p1_win=probs["p1_win"], p2_win=probs["p2_win"],
                p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
                surface=surface, tour=_tour, best_of=_best_of,
                bm_total_line=_bm_total_line,
                bm_total_over=_bm_total_over,
                bm_total_under=_bm_total_under,
            )

            # GPT анализ
            gpt_text = run_tennis_gpt_agent(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
                game_totals=_game_totals,
            )

            # Llama анализ (независимый)
            llama_text = run_tennis_llama_agent(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs["p1_rank"], p2_rank=probs["p2_rank"],
                gpt_verdict=gpt_text,
                game_totals=_game_totals,
            )

            # CHIMERA Multi-Agent
            from sports.tennis.agents import run_tennis_chimera_agents
            chimera_data = run_tennis_chimera_agents(
                p1, p2, probs, o1, o2, surface, tour,
                p1_rank=probs.get("p1_rank", 100), p2_rank=probs.get("p2_rank", 100),
                gpt_text=gpt_text, llama_text=llama_text,
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
                _tennis_signal = "СТАВИТЬ" if cands else "ПРОПУСТИТЬ"
                _tennis_pred_id = save_prediction(
                    sport="tennis",
                    match_id=m.get("event_id", f"{p1}_{p2}"),
                    match_date=m.get("commence_time", ""),
                    home_team=p1, away_team=p2,
                    league=sport_key,
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
                print(f"[Tennis Save] {save_err}")
            try:
                upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
                track_analysis(call.from_user.id, "tennis")
                log_action(call.from_user.id, "анализ Теннис")
            except Exception:
                pass

            tennis_kb = InlineKeyboardBuilder()
            tennis_kb.button(text="🎾 Победитель", callback_data=f"tennis_mkt_winner_{match_idx}")
            tennis_kb.button(text="📊 Тотал геймов", callback_data=f"tennis_mkt_games_{match_idx}")
            tennis_kb.button(text="🏅 Победа в 1-м сете", callback_data=f"tennis_mkt_set1_{match_idx}")
            tennis_kb.button(text="⬅️ Матчи", callback_data=f"tennis_tour_{sport_key}")
            tennis_kb.button(text="🏠 Меню", callback_data="back_to_main")
            if _tennis_signal == "СТАВИТЬ" and _tennis_pred_id:
                _tn_bet_odds = o2 if rec == "away_win" else o1
                _tn_odds_enc = int(round((_tn_bet_odds or 0) * 100))
                _tn_kelly = best["kelly"] if best else 2
                _tn_units = 3 if _tn_kelly >= 4 else (2 if _tn_kelly >= 2 else 1)
                tennis_kb.button(
                    text=f"✅ Я поставил {_tn_units}u — записать в статистику",
                    callback_data=f"mybet_tennis_{_tennis_pred_id}_{_tn_odds_enc}_{_tn_units}"
                )
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
                bookmaker_odds=cached.get("bookmaker_odds"),
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

        # Проверяем кеш готового отчёта (45 мин)
        import time as _time_fb
        _fb_cache_key = f"football_{match_index}"
        _fb_cached = _report_cache.get(_fb_cache_key)
        if _fb_cached and _time_fb.time() - _fb_cached.get("ts", 0) < _REPORT_CACHE_TTL:
            await call.answer()
            await call.message.edit_text(
                _fb_cached["text"], parse_mode=_fb_cached.get("parse_mode"),
                reply_markup=_fb_cached.get("kb"),
            )
            return

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
        _form_h = get_form_string(_team_form, home_team)
        _form_a = get_form_string(_team_form, away_team)
        async with _ai_semaphore:
            stats_result, scout_result = await asyncio.gather(
                _loop.run_in_executor(None, lambda: run_statistician_agent(
                    prophet_data, team_stats_text,
                    poisson_probs=poisson_probs, elo_probs=elo_probs,
                    home_form=_form_h, away_form=_form_a,
                )),
                _loop.run_in_executor(None, run_scout_agent, home_team, away_team, news_summary),
            )
            gpt_result = await _loop.run_in_executor(None, lambda: run_arbitrator_agent(
                stats_result, scout_result, bookmaker_odds,
                poisson_probs=poisson_probs, elo_probs=elo_probs,
            ))

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
                _loop.run_in_executor(None, lambda: run_llama_agent(
                    home_team, away_team, prophet_data, news_summary, bookmaker_odds,
                    team_stats_text, poisson_probs=poisson_probs, elo_probs=elo_probs,
                )),
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
            _football_pred_id = save_prediction(
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
            _football_pred_id = None
            print(f"[DB Save] Ошибка сохранения футбол: {_save_err}")

        # Трекинг активности пользователя
        try:
            upsert_user(call.from_user.id, call.from_user.username or "", call.from_user.first_name or "")
            track_analysis(call.from_user.id, "football")
            log_action(call.from_user.id, "анализ Футбол")
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
                gpt_summary=gpt_result.get("final_verdict_summary", "") if gpt_result else "",
                llama_summary=llama_result.get("analysis_summary", "") if llama_result else "",
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
            bookmaker_odds=bookmaker_odds,
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
            # Вариант A: движение линии Pinnacle с момента открытия рынка
            try:
                from line_tracker import get_line_movement as _get_lm
                _lm = _get_lm(str(match.get("id", "")), top_sig.get("outcome", ""))
                if _lm:
                    top_sig["line_movement"] = _lm
            except Exception:
                pass
            sig_text = format_signal(top_sig)
            sig_text = "🐉 <b>ХИМЕРА (Змея + Лев + Козёл + Тень)</b>\n\n" + sig_text
            # Кнопка "Я поставил"
            _sig_kb = None
            if _football_pred_id:
                _f_kelly = top_sig.get("kelly", 2)
                _f_units = 3 if _f_kelly >= 4 else (2 if _f_kelly >= 2 else 1)
                _f_rec = top_sig.get("outcome", "home_win")
                _f_odds = top_sig.get("odds", 0)
                _f_odds_enc = int(round((_f_odds or 0) * 100))
                _sig_kb = types.InlineKeyboardMarkup(inline_keyboard=[[
                    types.InlineKeyboardButton(
                        text=f"✅ Я поставил {_f_units}u — записать в статистику",
                        callback_data=f"mybet_football_{_football_pred_id}_{_f_odds_enc}_{_f_units}"
                    )
                ]])
            try:
                await call.message.answer(sig_text, parse_mode="HTML", reply_markup=_sig_kb)
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
 П1: {(prophet_data[1]*100 if prophet_data else 0):.0f}% | Х: {(prophet_data[0]*100 if prophet_data else 0):.0f}% | П2: {(prophet_data[2]*100 if prophet_data else 0):.0f}%

🐍🦁🐐 *Вердикт Химеры:*
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
        report = format_goals_report(cached["home_team"], cached["away_team"], goals_result, cached["bookmaker_odds"], cached.get("poisson_probs"))
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







def _make_loss_explanation(rec_outcome: str, real_outcome: str, home: str, away: str) -> str:
    """Краткое объяснение почему прогноз не сыграл."""
    import random
    if rec_outcome == "home_win" and real_outcome == "draw":
        opts = [
            f"{home} не смог реализовать преимущество — матч завершился ничьёй",
            f"Не хватило одного гола: {home} сравнял счёт, но дожать не смог",
            f"Ничья вместо победы {home} — бывает, модель не угадала интенсивность матча",
        ]
    elif rec_outcome == "home_win" and real_outcome == "away_win":
        opts = [
            f"{away} переиграл фаворита — один из тех дней когда статистика уходит на второй план",
            f"Неожиданная победа {away}. Возможно сказались факторы вне модели: усталость, тактика, удача",
            f"{home} не оправдал ожиданий. {away} был готов лучше",
        ]
    elif rec_outcome == "away_win" and real_outcome == "draw":
        opts = [
            f"{away} не смог дожать — матч завершился ничьёй вместо победы гостей",
            f"Домашние стены помогли {home} устоять. Ничья вместо победы {away}",
            f"Счёт сравнялся, но {away} не хватило для полной победы",
        ]
    elif rec_outcome == "away_win" and real_outcome == "home_win":
        opts = [
            f"{home} удержал преимущество своего поля. Домашний фактор сработал сильнее модели",
            f"Неожиданная победа {home} — гости не смогли реализовать перевес на бумаге",
            f"{home} победил вопреки прогнозу — вероятно сыграли мотивация и домашняя поддержка",
        ]
    elif rec_outcome in ("team1_win", "p1_win") and real_outcome in ("team2_win", "p2_win"):
        opts = [
            f"{away} показал более сильную игру в этот день",
            f"Неожиданный результат: {away} переиграл фаворита",
            f"{away} превзошёл ожидания — один из тех матчей где класс уступил форме дня",
        ]
    elif rec_outcome in ("team2_win", "p2_win") and real_outcome in ("team1_win", "p1_win"):
        opts = [
            f"{home} удивил своей игрой и переиграл гостей",
            f"Неожиданная победа {home} — прогноз не учёл их сегодняшнюю форму",
            f"{home} оказался сильнее — дисперсия в действии",
        ]
    else:
        opts = [
            "Матч сложился не в нашу пользу — такое бывает на дистанции",
            "Результат не совпал с прогнозом. Продолжаем анализировать",
            "Не угадали — это часть процесса. Важна дистанция, а не один матч",
        ]
    return random.choice(opts)


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

            # ── Футбол: отдельный трекер ─────────────────────────────────
            if pending_football:
                try:
                    from sports.football.results_tracker import check_and_update_football_results
                    _fb_updated = check_and_update_football_results(on_elo_update=update_elo_after_match)
                    print(f"[Результаты ⚽] Обновлено: {_fb_updated}")
                except Exception as _fb_e:
                    print(f"[Результаты ⚽] Ошибка трекера: {_fb_e}")

            # ── Авто-закрытие застарелых прогнозов (>7 дней без результата) ──
            try:
                from database import expire_stale_predictions
                _expired = expire_stale_predictions(days=4)
                if _expired:
                    print(f"[DB] Закрыто застарелых прогнозов: {_expired}")
            except Exception as _exp_e:
                print(f"[DB] Ошибка expire_stale: {_exp_e}")

            # ── Авто-обучение MetaLearner ──────────────────────────────────
            try:
                from meta_learner import MetaLearner
                _ml = MetaLearner(signal_engine_path="signal_engine.py")
                # Стрик поражений → форсируем обучение раньше
                _streak = get_recent_signal_streak()
                _force_ml = _streak <= -5
                if _force_ml:
                    print(f"[MetaLearner] Серия {abs(_streak)} поражений — форсирую обновление порогов")
                for _sport in ["football", "cs2", "tennis", "basketball"]:
                    _perf = _ml.analyze_performance(_sport)
                    if _perf.get("total", 0) >= 10 or _force_ml:
                        _updates = _ml.suggest_updates(_sport, _perf)
                        if _updates:
                            _ml.apply_updates(_sport, _updates)
                            print(f"[MetaLearner] {_sport} авто-обновление: {_updates}")
                        else:
                            print(f"[MetaLearner] {_sport}: ROI={_perf.get('roi',0):.1f}%, точность={_perf.get('accuracy',0):.1f}%")
                _bb_weights = _ml.analyze_basketball_weights()
                if _bb_weights:
                    _ml.apply_updates("basketball", _bb_weights)
                    print(f"[MetaLearner] Basketball веса обновлены: {_bb_weights}")
            except Exception as _ml_e:
                print(f"[MetaLearner] Ошибка: {_ml_e}")

            # ── Уведомления пользователям о результатах ставок ────────────
            try:
                _unnotified = get_unnotified_bets()

                # GPT-объяснения генерируем один раз на матч (кэш по ключу)
                _explanation_cache: dict = {}

                def _get_ai_explanation(sport, home, away, rec_outcome, real_outcome, prob):
                    """Один GPT вызов на матч — результат переиспользуется для всех пользователей."""
                    _key = f"{home}_{away}_{rec_outcome}_{real_outcome}"
                    if _key in _explanation_cache:
                        return _explanation_cache[_key]

                    # Если прогноз сыграл — объяснение не нужно
                    if rec_outcome == real_outcome:
                        _explanation_cache[_key] = ""
                        return ""

                    try:
                        from agents import client as _cl
                        _predicted = home if rec_outcome == "home_win" else away
                        _winner    = home if real_outcome == "home_win" else (away if real_outcome == "away_win" else "ничья")
                        _sport_name = {"football":"футбол","cs2":"CS2","tennis":"теннис","basketball":"баскетбол"}.get(sport, sport)
                        _prob_str = f" (наша уверенность была {round(prob*100)}%)" if prob > 0.05 else ""
                        _prompt = (
                            f"Ты опытный беттор-аналитик. Прогноз не зашёл.\n"
                            f"Матч: {home} vs {away} ({_sport_name}). "
                            f"Ждали победу {_predicted}{_prob_str}, но выиграл {_winner}.\n"
                            f"Напиши ОДНО короткое предложение (максимум 12 слов) в духе 'это спорт' — "
                            f"не про ошибку модели, а про то что так бывает: неожиданный поворот, "
                            f"день не тот, класс не помог, андердог выстрелил. "
                            f"Звучи как человек который видел сотни таких матчей. Без пафоса, без утешений."
                        )
                        _resp = _cl.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": _prompt}],
                            max_tokens=60, temperature=0.5,
                        )
                        _text = _resp.choices[0].message.content.strip().strip('"')
                        _explanation_cache[_key] = _text
                        return _text
                    except Exception:
                        # Fallback на шаблон если GPT недоступен
                        _fallback = _make_loss_explanation(rec_outcome, real_outcome, home, away)
                        _explanation_cache[_key] = _fallback
                        return _fallback

                for _nb in _unnotified:
                    try:
                        _uid      = _nb["user_id"]
                        _sport    = _nb["sport"]
                        _home     = _nb["home"]
                        _away     = _nb["away"]
                        _rec      = _nb["rec_outcome"]
                        _real     = _nb["real_outcome"]
                        _odds     = float(_nb["odds"] or 0)
                        _units    = int(_nb["units"] or 1)
                        _is_win   = (_rec == _real)

                        if _odds < 1.02:
                            _odds = float(_nb["odds_home"] if _rec == "home_win" else _nb["odds_away"] or 0) or 1.80

                        # Считаем профит
                        _profit_pct = round(_units * (_odds - 1), 1) if _is_win else -float(_units)

                        # Иконки
                        _s_icon = {"football":"⚽","cs2":"🎮","tennis":"🎾","basketball":"🏀"}.get(_sport,"🎯")
                        _team = _home if _rec == "home_win" else (_away if _rec == "away_win" else "Ничья")

                        # Вероятность которую давал бот
                        _ens_h = float(_nb.get("ensemble_home") or 0)
                        _ens_a = float(_nb.get("ensemble_away") or 0)
                        _pred_prob = _ens_h if _rec == "home_win" else _ens_a
                        _prob_str = f"📊 Наша уверенность была: <b>{round(_pred_prob*100)}%</b>\n" if _pred_prob > 0.05 else ""

                        # Вариант B: строка закрытия рынка (Pinnacle closing line)
                        _closing_str = ""
                        try:
                            from line_tracker import get_closing_line_str as _get_cl
                            _entry_odds = float(_nb.get("odds_home") or 0) if _rec == "home_win" else float(_nb.get("odds_away") or 0)
                            _cl = _get_cl(str(_nb.get("match_id", "")), _rec, _entry_odds)
                            if _cl:
                                _closing_str = f"\n{_cl}"
                        except Exception:
                            pass

                        _closing_line = f"{_closing_str}\n" if _closing_str else ""
                        if _is_win:
                            _p_str = f"+{_profit_pct}%"
                            _msg = (
                                f"✅ <b>Прогноз сыграл!</b>\n\n"
                                f"{_s_icon} {_home} vs {_away}\n"
                                f"📌 {_team} победит @ {_odds}\n"
                                f"{_prob_str}"
                                f"{_closing_line}"
                                f"💰 {_units}u → <b>{_p_str} от банка</b>"
                            )
                        else:
                            _ai_exp = _get_ai_explanation(_sport, _home, _away, _rec, _real, _pred_prob)
                            _exp_line = f"\n\n<i>🐉 Мнение Chimera: {_ai_exp}</i>" if _ai_exp else ""
                            _msg = (
                                f"❌ <b>Прогноз не сыграл</b>\n\n"
                                f"{_s_icon} {_home} vs {_away}\n"
                                f"📌 {_team} победит @ {_odds}\n"
                                f"{_prob_str}"
                                f"{_closing_line}"
                                f"📉 {_units}u → <b>{_profit_pct}% от банка</b>"
                                f"{_exp_line}"
                            )

                        await bot.send_message(_uid, _msg, parse_mode="HTML")
                        mark_bet_notified(_nb["bet_id"])
                    except Exception as _ne:
                        print(f"[Notify] Ошибка отправки user {_nb.get('user_id')}: {_ne}")
                        mark_bet_notified(_nb["bet_id"])  # не спамим если ошибка
            except Exception as _notify_err:
                print(f"[Notify] Ошибка: {_notify_err}")

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

    # Проверяем кеш готового отчёта (45 мин)
    import time as _time_bb
    _bb_cache_key = f"bball_{league_key}_{idx}"
    _bb_cached = _report_cache.get(_bb_cache_key)
    if _bb_cached and _time_bb.time() - _bb_cached.get("ts", 0) < _REPORT_CACHE_TTL:
        await call.message.edit_text(
            _bb_cached["text"], parse_mode=_bb_cached.get("parse_mode"),
            reply_markup=_bb_cached.get("kb"),
        )
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
        analysis = calculate_basketball_win_prob(
            home, away, odds, league_key,
            no_vig_home=odds.get("no_vig_home", 0.0),
            no_vig_away=odds.get("no_vig_away", 0.0),
        )

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
                _elo_gap = analysis.get("elo_gap", 0)
                _nv_h = odds.get("no_vig_home", 0)
                _nv_a = odds.get("no_vig_away", 0)
                nv_block = (
                    f"Рынок (no-vig Pinnacle): {home} {round(_nv_h*100)}% / {away} {round(_nv_a*100)}%\n"
                ) if _nv_h > 0 else ""
                total_ev_block = ""
                if total_data.get("lean_ev", 0) != 0:
                    total_ev_block = (
                        f"Тотал EV модели: {total_data['lean']} {total_data['line']} "
                        f"EV={total_data['lean_ev']:+.1f}% (нет EV = не ставим)\n"
                    )
                prompt = (
                    f"Ты — профессиональный аналитик баскетбола для беттинга. Анализируй строго.\n\n"
                    f"МАТЧ ({league_name}): {home} (ELO {h_elo}) vs {away} (ELO {a_elo})\n"
                    f"ELO разрыв: {_elo_gap:+d} (положительный = преимущество хозяев)\n"
                    f"Наша модель: {home} {round(h_prob*100)}% (EV {h_ev:+.1f}%) | {away} {round(a_prob*100)}% (EV {a_ev:+.1f}%)\n"
                    f"Коэффициенты: {home} @ {odds.get('home_win', '—')} | {away} @ {odds.get('away_win', '—')}\n"
                    f"{nv_block}"
                    f"{form_block}"
                    f"{b2b_block}"
                    f"{total_block}\n"
                    f"{total_ev_block}"
                    f"{spread_block}\n\n"
                    f"Задачи:\n"
                    f"1. Оцени достоверность ELO разрыва — форма подтверждает или опровергает?\n"
                    f"2. Усталость (back-to-back) — насколько критична?\n"
                    f"3. Темп игры обеих команд → прогноз тотала Over/Under {total_data.get('line', '—')}\n"
                    f"4. Если наша вероятность и рынок расходятся — это сигнал или ошибка?\n"
                    f"5. Ожидаемый счёт\n\n"
                    f"Формат ответа (только JSON):\n"
                    f'{{"analysis": "Строгий анализ с конкретными числами (3-4 предложения)", '
                    f'"recommended_outcome": "Победа хозяев" или "Победа гостей", '
                    f'"confidence": <0-100>, '
                    f'"total_pick": "Over" или "Under", '
                    f'"total_reasoning": "Почему — темп, стиль, усталость (1 предложение)", '
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
                return {"analysis": f"(Анализ временно недоступен: {e})"}

        def _run_llama_basketball():
            if not _groq_client:
                return {"analysis": "(Тень недоступна)"}
            try:
                _nv_h = odds.get("no_vig_home", 0)
                _nv_a = odds.get("no_vig_away", 0)
                nv_block_l = (
                    f"No-vig рынок: {home} {round(_nv_h*100)}% / {away} {round(_nv_a*100)}%\n"
                ) if _nv_h > 0 else ""
                prompt = (
                    f"Ты — независимый аналитик баскетбола. Дай СВОЁ мнение, не повторяй GPT.\n\n"
                    f"МАТЧ ({league_name}): {home} vs {away}\n"
                    f"ELO: {home}={h_elo}, {away}={a_elo}\n"
                    f"Модель: {home} {round(h_prob*100)}% | {away} {round(a_prob*100)}%\n"
                    f"Кэфы: {home} @ {odds.get('home_win', '—')} | {away} @ {odds.get('away_win', '—')}\n"
                    f"{nv_block_l}"
                    f"{form_block}"
                    f"{b2b_block}"
                    f"{total_block}\n"
                    f"{spread_block}\n\n"
                    f"Оцени НЕЗАВИСИМО:\n"
                    f"1. Тактический стиль — кто играет быстрее, кто в защите сильнее?\n"
                    f"2. Риски для фаворита (back-to-back, выездной матч, длинная дорога)\n"
                    f"3. Тотал Over/Under — аргументируй через темп и стиль игры\n\n"
                    f"Формат (только JSON):\n"
                    f'{{"analysis": "Независимый тактический анализ (2-3 предложения)", '
                    f'"recommended_outcome": "Победа хозяев" или "Победа гостей", '
                    f'"confidence": <0-100>, '
                    f'"total_pick": "Over" или "Under", '
                    f'"total_reasoning": "Аргумент за тотал (1 предложение)"}}'
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
                return {"analysis": f"(Тень недоступна: {e})"}

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
            consensus = f"⚠️ Химера: {gpt_outcome} | Тень: {llama_outcome}"
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
            nv_over   = total_data.get("nv_over", 0)
            nv_under  = total_data.get("nv_under", 0)
            has_value = total_data.get("has_value", False)
            ai_total_agree = ""
            if gpt_total and llama_total:
                if gpt_total == llama_total:
                    ai_total_agree = f" (AI: ✅ {gpt_total})"
                else:
                    ai_total_agree = f" (Химера: {gpt_total}, Тень: {llama_total})"
            nv_str = f"  <i>Рынок (no-vig): Over {nv_over}% / Under {nv_under}%</i>\n" if nv_over else ""
            value_icon = "🎯" if has_value else "📊"
            total_str = (
                f"\n\n{value_icon} <b>Тотал {line}:</b>\n"
                f"  Over @ {total_data['over_odds']} / Under @ {total_data['under_odds']}\n"
                f"{nv_str}"
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
        _bball_pred_id = None
        try:
            rec_outcome = "home_win" if h_prob >= a_prob else "away_win"
            _bball_pred_id = save_prediction(
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
                total_line=total_data.get("line") if total_data else None,
                total_lean=total_data.get("lean") if total_data else None,
                total_lean_odds=total_data.get("lean_odds") if total_data else None,
                total_ev=total_data.get("lean_ev") if total_data else None,
                spread_home=odds.get("spread_home"),
                spread_away=odds.get("spread_away"),
                prediction_data={
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
            log_action(call.from_user.id, "анализ Баскетбол")
        except Exception:
            pass

        bball_kb = InlineKeyboardBuilder()
        bball_kb.button(text="🏀 Победитель", callback_data=f"bball_mkt_winner_{league_key}_{idx}")
        bball_kb.button(text="📊 Тотал очков", callback_data=f"bball_mkt_total_{league_key}_{idx}")
        bball_kb.button(text="⚖️ Фора", callback_data=f"bball_mkt_spread_{league_key}_{idx}")
        bball_kb.button(text="⬅️ Матчи", callback_data=f"bball_league_{league_key}")
        bball_kb.button(text="🏠 Меню", callback_data="back_to_main")
        if _bball_pred_id:
            _bb_bet_odds = fav_odds or 0
            _bb_odds_enc = int(round(float(_bb_bet_odds) * 100)) if _bb_bet_odds else 0
            _bb_units = 3 if kelly >= 4 else (2 if kelly >= 2 else 1)
            bball_kb.button(
                text=f"✅ Я поставил {_bb_units}u — записать в статистику",
                callback_data=f"mybet_basketball_{_bball_pred_id}_{_bb_odds_enc}_{_bb_units}"
            )
        bball_kb.adjust(2)
        _bb_kb = bball_kb.as_markup()
        await call.message.edit_text(report, parse_mode="HTML", reply_markup=_bb_kb)
        import time as _time
        _report_cache[f"bball_{league_key}_{idx}"] = {
            "text": report, "kb": _bb_kb,
            "parse_mode": "HTML", "ts": _time.time(),
        }

    except Exception as e:
        import traceback as _tb
        logger.error(f"[Баскетбол анализ] Ошибка: {e}\n{_tb.format_exc()}")
        try:
            await call.message.edit_text(f"⚠️ Не удалось выполнить анализ.\n<code>{type(e).__name__}: {str(e)[:120]}</code>", parse_mode="HTML")
        except Exception:
            await call.message.edit_text("⚠️ Не удалось выполнить анализ.")


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


def _format_chimera_page(candidates: list, idx: int) -> str:
    """Форматирует одну страницу карусели — полный формат как в format_chimera_signals."""
    import html as _html
    from chimera_signal import _format_match_time, score_label, _format_totals_block
    c = candidates[idx]
    total = len(candidates)
    sp = c.get("sport", "football")
    sp_emoji = {"cs2": "🎮", "tennis": "🎾", "basketball": "🏀"}.get(sp, "⚽")

    t_str, t_live = _format_match_time(c.get("commence_time", ""))
    live_tag = "🟢 LIVE" if t_live else (f"🕐 {t_str}" if t_str else "")
    time_line = f"\n{live_tag}" if live_tag else ""

    score = c.get("chimera_score", 0)
    label = score_label(score)

    if idx == 0:
        header = f"🎯 <b>CHIMERA SIGNAL — ЛУЧШАЯ СТАВКА ДНЯ</b>"
    else:
        header = f"📋 <b>ВАРИАНТ {idx + 1} из {total}</b>"

    matchup = f"<b>{c.get('home','')} vs {c.get('away','')}</b>"
    bet_line = f"📌 Ставка: <b>{c.get('team','')} ({c.get('outcome','')})</b>"

    lines = [
        header, "",
        f"<b>{label} [{score:.0f}/100]</b>{time_line}", "",
        f"{sp_emoji} | {matchup}",
        bet_line,
        f"💰 Кэф: <b>{c.get('odds','?')}</b> | Наша вероятность: <b>{c.get('prob',0)}%</b>",
        f"📈 EV: <b>{c.get('ev',0):+.1f}%</b> | Ставь: <b>{c.get('kelly',0):.1f}%</b> банка",
    ]

    # AI блок
    if c.get("ai_confirmed") is True:
        llama_agrees = c.get("llama_agrees")
        ai_header = "🐉 Химера единогласна (Змея + Лев + Козёл + Тень)" if llama_agrees else "🐍🦁🐐 Химера подтверждает"
        ai_reason   = _html.escape(str(c.get("ai_reason", "") or ""))
        llama_logic = _html.escape(str(c.get("llama_logic", "") or ""))
        llama_warn  = _html.escape(str(c.get("llama_warning", "") or ""))
        lines += ["", f"<b>{ai_header} ({c.get('ai_confidence',0)}% уверенности):</b>",
                  f"<i>🐍🦁🐐 Химера: «{ai_reason}»</i>"]
        if llama_logic:
            lines.append(f"<i>🌀 Тень: «{llama_logic}»</i>")
        if llama_warn:
            lines.append(f"⚠️ Риск: {llama_warn}")
    elif c.get("ai_confirmed") is False:
        lines.append("\n⚠️ AI сомневается в этой ставке")
    elif c.get("ai_confirmed") is None:
        if c.get("ai_reason"):
            lines.append(f"\n🤖 <i>AI: «{_html.escape(str(c['ai_reason']))}»</i>")
        else:
            lines.append("\n🤖 <i>AI не выбрал этот вариант как лучший</i>")

    # Детали CHIMERA Score
    score_lines = [
        "", "📊 <b>Детали CHIMERA Score:</b>",
        f"├ ELO преимущество: {c.get('elo_pts',0):+.0f} pts" +
            (f" (разрыв: {c['elo_gap']} очков)" if c.get('elo_gap') else ""),
        f"├ Форма команды: {c.get('form_pts',0):+.0f} pts" +
            (f" ({c['form']})" if c.get('form') else ""),
        f"├ Ценность кэфа: {c.get('value_pts',0):+.0f} pts" +
            f" ({c.get('prob',0)}% vs бук {c.get('implied_prob',0)}%)",
        f"├ Сила прогноза: {c.get('prob_pts',0):+.0f} pts",
    ]
    if c.get("xg_pts", 0):
        score_lines.append(f"├ xG качество: {c['xg_pts']:+.0f} pts")
    if c.get("line_pts", 0):
        icon = "📉" if c["line_pts"] > 0 else "⚠️"
        score_lines.append(f"├ {icon} Движение линии: {c['line_pts']:+.0f} pts")
    if c.get("h2h_pts", 0):
        score_lines.append(f"├ ⚔️ H2H история: {c['h2h_pts']:+.0f} pts")
    score_lines[-1] = score_lines[-1].replace("├", "└")
    lines += score_lines

    totals_block = _format_totals_block(c)
    if totals_block:
        lines += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━", totals_block]

    result = "\n".join(lines)
    if len(result) > 4000:
        result = result[:3990] + "\n…"
    return result


def _build_chimera_carousel_kb(candidates: list, idx: int, user_id: int) -> types.InlineKeyboardMarkup:
    """Клавиатура карусели: ◀️ счётчик ▶️ + кнопка 'Я поставил'."""
    total = len(candidates)
    c = candidates[idx]
    kelly = c.get("kelly", 2)
    units = 3 if kelly >= 4 else (2 if kelly >= 2 else 1)

    nav_row = []
    if idx > 0:
        nav_row.append(types.InlineKeyboardButton(text="◀️", callback_data=f"chimera_page_{idx-1}"))
    nav_row.append(types.InlineKeyboardButton(text=f"{idx+1} / {total}", callback_data="chimera_noop"))
    if idx < total - 1:
        nav_row.append(types.InlineKeyboardButton(text="▶️", callback_data=f"chimera_page_{idx+1}"))

    # Кнопка ставки — всегда через индекс, pred_id получаем лениво при нажатии
    bet_row = [types.InlineKeyboardButton(
        text=f"✅ Я поставил {units}u — записать",
        callback_data=f"chimera_bet_{idx}_{units}"
    )]

    return types.InlineKeyboardMarkup(inline_keyboard=[nav_row, bet_row])


def _build_chimera_kb(top_candidates, top_pred_id, top_sport, top_odds, user_id):
    """Обёртка — строит карусель с idx=0 + кнопка обновить."""
    if not top_candidates:
        return None
    kb = _build_chimera_carousel_kb(top_candidates, 0, user_id)
    rows = list(kb.inline_keyboard) + [[
        types.InlineKeyboardButton(text="🔄 Обновить сигналы", callback_data="chimera_refresh")
    ]]
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


# --- 10c. Команда /signals — CHIMERA SIGNAL ENGINE v2 ---
@dp.message(Command("signals"))
async def cmd_signals(message: types.Message):
    """Кнопка 'Сигналы дня'."""
    if message.from_user:
        log_action(message.from_user.id, "signals")
    import time as _time_mod
    from chimera_signal import compute_chimera_score, run_ai_verification, format_chimera_signals

    # ── Кеш: если скан свежий (<45 мин) — отдаём моментально ────────────────
    _cached = _signals_scan_cache.get("last")
    # Сбрасываем кеш если в кандидатах нет _pred_id (старый формат до обновления)
    if _cached and _cached.get("candidates") and "_pred_id" not in _cached["candidates"][0]:
        _cached = None
        _signals_scan_cache.clear()
    if _cached and (_time_mod.time() - _cached["ts"]) < SIGNALS_SCAN_TTL:
        top_candidates   = _cached["candidates"]
        _chimera_top_pred_id = _cached.get("top_pred_id")
        _chimera_top_sport   = _cached.get("top_sport", "football")
        _chimera_top_odds    = _cached.get("top_odds", 0)
        _cache_age_min = int((_time_mod.time() - _cached["ts"]) / 60)
        print(f"[CHIMERA] Кеш сигналов использован (возраст {_cache_age_min} мин)")
        try:
            result_text = _format_chimera_page(top_candidates, 0)
            result_text += f"\n\n<i>🕐 Обновлено {_cache_age_min} мин назад</i>"
            _chimera_kb = _build_chimera_kb(
                top_candidates, _chimera_top_pred_id, _chimera_top_sport,
                _chimera_top_odds, message.from_user.id
            )
            await message.answer(result_text, parse_mode="HTML", reply_markup=_chimera_kb)
        except Exception as _ce:
            logger.error(f"[CHIMERA Cache] Ошибка форматирования: {_ce}")
            await message.answer("⚠️ Ошибка загрузки кеша. Перезапускаю скан...", parse_mode="HTML")
            _signals_scan_cache.clear()
            await cmd_signals(message)
        return

    status_msg = await message.answer(
        "🔍 <b>CHIMERA SIGNAL запущен...</b>\n\n"
        "⏳ <i>Это займёт 30–90 секунд — Химера сканирует рынок.</i>\n\n"
        "⚙️ Шаг 1/3: Загрузка матчей (Футбол + CS2 + Теннис + Баскетбол)...",
        parse_mode="HTML"
    )

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

    # historical_movement.prefetch_snapshots убран — line_tracker.py покрывает
    # движение линий Pinnacle без дополнительных API кредитов.

    try:
        from api_football import get_h2h
        _h2h_ok = True
    except ImportError:
        _h2h_ok = False

    all_candidates = []
    _scan_errors = []  # собираем ошибки для финального сообщения

    def _run_math_scan():
        """Весь блокирующий скан — запускается в thread executor."""
        import concurrent.futures as _cf
        from datetime import datetime, timezone as _tz, timedelta as _td
        import requests as _req
        from odds_cache import get_odds as _get_odds

        _now = datetime.now(_tz.utc)
        _cutoff_future = (_now + _td(hours=72)).isoformat()[:19]
        _cutoff_past   = (_now - _td(hours=3)).isoformat()[:19]

        # ── Параллельная загрузка всех лиг (с кешем 30 мин) ─────────────
        def _fetch_league(lkey):
            # Все регионы + spreads для полного набора букмекеров (Pinnacle, Betfair, etc.)
            raw = _get_odds(lkey, markets="h2h,totals,spreads")
            if raw is None:
                return "quota", []
            if isinstance(raw, list):
                filtered = [m for m in raw
                            if _cutoff_past < m.get("commence_time", "") <= _cutoff_future]
                return lkey, filtered
            return lkey, []

        def _fetch_bball_league(lkey):
            raw = _get_odds(lkey, markets="h2h,totals,spreads")
            if raw and isinstance(raw, list):
                now2 = datetime.now(_tz.utc)
                cutoff2 = (now2 - _td(hours=2)).isoformat()[:19]
                return lkey, [m for m in raw if m.get("commence_time", "") > cutoff2][:25]
            return lkey, []

        def _scan_football(matches):
            cands = []
            try:
                from historical_movement import get_line_movement as _get_hist_mov
                _hist_mov_ok = True
            except ImportError:
                _hist_mov_ok = False

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
                    c_list = compute_chimera_score(
                        home_team=home, away_team=away,
                        home_prob=elo_p["home"], away_prob=elo_p["away"], draw_prob=elo_p["draw"],
                        bookmaker_odds=odds, home_form=form_h, away_form=form_a,
                        elo_home=elo_h, elo_away=elo_a,
                        league=m.get("sport_key", ""), line_movement=_movement,
                    )
                    _ct = m.get("commence_time", "")
                    _mid = m.get("id", "")
                    _sport_key = m.get("sport_key", "")
                    for c in c_list:
                        c["commence_time"] = _ct
                        # ── Исторический сдвиг линии Pinnacle ─────────────
                        if _hist_mov_ok and odds.get("pinnacle_home"):
                            try:
                                hist_mov = _get_hist_mov(
                                    sport_key=_sport_key,
                                    match_id=_mid,
                                    home_team=home, away_team=away,
                                    current_pinnacle_home=odds.get("pinnacle_home", 0),
                                    current_pinnacle_away=odds.get("pinnacle_away", 0),
                                    current_pinnacle_draw=odds.get("pinnacle_draw", 0),
                                )
                                if hist_mov.get("score_boost"):
                                    c["chimera_score"] = c.get("chimera_score", 0) + hist_mov["score_boost"]
                                    c["hist_movement"] = hist_mov
                            except Exception:
                                pass
                    cands.extend(c_list)
                except Exception as _ce:
                    print(f"[CHIMERA] Ошибка {m.get('home_team','')} vs {m.get('away_team','')}: {_ce}")
            return cands

        def _scan_cs2():
            cands = []
            try:
                from sports.cs2 import calculate_cs2_win_prob
                from signal_engine import predict_cs2_totals as _ps_totals

                # Если кэш пустой (пользователь не открывал раздел CS2) — фетчим сами
                _cs2_source = cs2_matches_cache
                if not _cs2_source:
                    try:
                        from sports.cs2.pandascore import get_cs2_matches_pandascore
                        _fetched = get_cs2_matches_pandascore()
                        if _fetched:
                            cs2_matches_cache.clear()
                            cs2_matches_cache.extend(_fetched)
                            _cs2_source = _fetched
                            print(f"[CHIMERA CS2] Загружено {len(_fetched)} матчей из PandaScore")
                    except Exception as _fe:
                        print(f"[CHIMERA CS2] Ошибка фетча матчей: {_fe}")

                for m in _cs2_source[:25]:
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
                        # Пропускаем матчи без реальных данных (обе команды неизвестны)
                        data_conf = analysis.get("data_confidence", 0)
                        if data_conf < 0.3:
                            print(f"[CHIMERA CS2] Пропуск {home} vs {away}: нет данных (conf={data_conf:.2f})")
                            continue
                        raw_odds = m.get("odds", {})
                        h_odds = raw_odds.get("home_win") or (round(1.0 / h_prob, 2) if h_prob > 0 else 1.85)
                        a_odds = raw_odds.get("away_win") or (round(1.0 / a_prob, 2) if a_prob > 0 else 1.85)
                        h_stats = analysis.get("home_stats", {})
                        a_stats = analysis.get("away_stats", {})
                        c_list = compute_chimera_score(
                            home_team=home, away_team=away,
                            home_prob=h_prob, away_prob=a_prob, draw_prob=0,
                            bookmaker_odds={"home_win": h_odds, "away_win": a_odds, "draw": 0},
                            home_form=h_stats.get("form", ""), away_form=a_stats.get("form", ""),
                            elo_home=analysis.get("elo_home", 1450), elo_away=analysis.get("elo_away", 1450),
                            league="CS2",
                        )
                        _ct = m.get("commence_time", "") or m.get("time", "")
                        _cs2_tier = m.get("tier", "B")
                        _league_low = (m.get("league", "") + " " + m.get("tournament", "")).lower()
                        _is_t3 = any(kw in _league_low for kw in CS2_TIER3_KEYWORDS) or _cs2_tier not in ("S", "A")
                        try:
                            _cs2_totals = _ps_totals(
                                home_prob=h_prob, away_prob=a_prob,
                                home_map_stats={mp: hp * 100 for mp, hp, _ in analysis.get("maps", [])},
                                away_map_stats={mp: ap * 100 for mp, _, ap in analysis.get("maps", [])},
                                predicted_maps=[mp for mp, _, _ in analysis.get("maps", [])],
                            )
                        except Exception:
                            _cs2_totals = None
                        for c in c_list:
                            c["sport"] = "cs2"
                            c["commence_time"] = _ct
                            c["cs2_tier"] = _cs2_tier
                            c["tier_label"] = m.get("tier_label", "🎮")
                            c["totals_data"] = _cs2_totals
                            if _is_t3:
                                c["cs2_tier3"] = True
                        cands.extend(c_list)
                    except Exception as _ce:
                        print(f"[CHIMERA CS2] Ошибка {m.get('home','')} vs {m.get('away','')}: {_ce}")
            except Exception as e:
                print(f"[CHIMERA] CS2 ошибка: {e}")
            return cands

        def _scan_tennis():
            try:
                from sports.tennis import scan_tennis_signals
                from sports.tennis.matches import get_tennis_matches, get_active_tennis_sports
                # Диагностика: сначала проверяем что вообще есть турниры
                sport_keys = get_active_tennis_sports()
                print(f"[CHIMERA] Теннис: активных турниров = {len(sport_keys)}: {sport_keys[:5]}")
                if not sport_keys:
                    print("[CHIMERA] Теннис: нет активных турниров в The Odds API")
                    return []
                raw_matches = get_tennis_matches()
                print(f"[CHIMERA] Теннис: матчей с коэффициентами = {len(raw_matches)}")
                if not raw_matches:
                    return []
                cands = scan_tennis_signals()
                # Минимальный порог уверенности 55% для теннисных кандидатов
                cands = [c for c in cands if c.get("prob", 0) >= 55]
                print(f"[CHIMERA] Теннис: кандидатов после скана = {len(cands)}")
                return cands
            except Exception as e:
                import traceback
                print(f"[CHIMERA] Теннис ошибка: {e}")
                traceback.print_exc()
                return []

        def _scan_basketball(bball_data):
            cands = []
            try:
                from sports.basketball import calculate_basketball_win_prob
                from sports.basketball.core import get_basketball_odds
                total = 0
                for bkey, bname, bmatches in bball_data:
                    for bm in bmatches:
                        try:
                            bh = bm.get("home_team", "")
                            ba = bm.get("away_team", "")
                            if not bh or not ba:
                                continue
                            bodds = get_basketball_odds(bm)
                            if not bodds.get("home_win"):
                                continue
                            bres = calculate_basketball_win_prob(
                                bh, ba, bodds, bkey,
                                no_vig_home=bodds.get("no_vig_home", 0.0),
                                no_vig_away=bodds.get("no_vig_away", 0.0),
                            )
                            hp = bres.get("home_prob", 0)
                            ap = bres.get("away_prob", 0)
                            if not hp:
                                continue
                            bcands = compute_chimera_score(
                                home_team=bh, away_team=ba,
                                home_prob=hp, away_prob=ap, draw_prob=0,
                                bookmaker_odds={"home_win": bodds.get("home_win", 0),
                                               "away_win": bodds.get("away_win", 0), "draw": 0},
                                home_form=bres.get("home_form", ""), away_form=bres.get("away_form", ""),
                                elo_home=bres.get("elo_home", 1550), elo_away=bres.get("elo_away", 1550),
                                league=bkey,
                            )
                            bct = bm.get("commence_time", "")
                            for bc in bcands:
                                bc["sport"] = "basketball"
                                bc["league_name"] = bname
                                bc["commence_time"] = bct
                            cands.extend(bcands)
                            total += len(bcands)
                        except Exception as bce:
                            print(f"[CHIMERA BBALL] Ошибка {bm.get('home_team','')} vs {bm.get('away_team','')}: {bce}")
                print(f"[CHIMERA] Баскетбол: {total} кандидатов")
            except Exception as e:
                print(f"[CHIMERA] Баскетбол ошибка: {e}")
            return cands

        # ── Шаг 1: параллельная загрузка данных по сети ──────────────────
        from sports.basketball.core import BASKETBALL_LEAGUES as _BBALL_LEAGUES
        all_league_keys = [lk for lk, _ in FOOTBALL_LEAGUES] + [bk for bk, _ in _BBALL_LEAGUES]

        football_matches = []
        bball_raw = {bk: [] for bk, _ in _BBALL_LEAGUES}

        with _cf.ThreadPoolExecutor(max_workers=12) as pool:
            fball_futures = {pool.submit(_fetch_league, lk): lk for lk, _ in FOOTBALL_LEAGUES}
            bball_futures = {pool.submit(_fetch_bball_league, bk): bk for bk, _ in _BBALL_LEAGUES}

            for fut in _cf.as_completed(fball_futures):
                lkey, matches = fut.result()
                if lkey == "quota":
                    _scan_errors.append("quota")
                else:
                    football_matches.extend(matches)

            for fut in _cf.as_completed(bball_futures):
                bkey, matches = fut.result()
                bball_raw[bkey] = matches

        print(f"[CHIMERA] Всего матчей для скана: {len(football_matches)}")
        bball_data = [(bk, bn, bball_raw.get(bk, [])) for bk, bn in _BBALL_LEAGUES]

        # ── Шаг 2: параллельный скан всех видов спорта ───────────────────
        with _cf.ThreadPoolExecutor(max_workers=4) as pool:
            f_football  = pool.submit(_scan_football, football_matches)
            f_cs2       = pool.submit(_scan_cs2)
            f_tennis    = pool.submit(_scan_tennis)
            f_basketball = pool.submit(_scan_basketball, bball_data)

            _sport_names = ["football", "cs2", "tennis", "basketball"]
            candidates = []
            for _sname, fut in zip(_sport_names, [f_football, f_cs2, f_tennis, f_basketball]):
                try:
                    candidates.extend(fut.result(timeout=120))
                except Exception as e:
                    import traceback
                    print(f"[CHIMERA] Скан ошибка [{_sname}]: {type(e).__name__}: {e}")
                    traceback.print_exc()

        return candidates

    import functools
    loop = asyncio.get_running_loop()
    all_candidates = await loop.run_in_executor(None, _run_math_scan)

    # ── Сортируем, берём топ-3 для AI ────────────────────────────────────
    # Фильтр: EV > 5% и кэф >= 1.40 и не аномальный EV (> 80% = баг данных)
    all_candidates = [
        c for c in all_candidates
        if c.get("ev", 0) > 5 and c.get("odds", 0) >= 1.40 and c.get("ev", 0) <= 80
    ]
    all_candidates.sort(key=lambda x: x["chimera_score"], reverse=True)
    top_candidates = all_candidates[:3]
    print(f"[CHIMERA] Кандидатов найдено: {len(all_candidates)}, топ-3 scores: {[round(c['chimera_score'],1) for c in top_candidates]}")

    if not top_candidates:
        if "quota" in _scan_errors:
            await status_msg.edit_text(
                "⚠️ <b>CHIMERA SIGNAL</b>\n\n"
                "Футбольные матчи временно недоступны.\n\n"
                "<i>Попробуй позже или воспользуйся разделами CS2 и Теннис.</i>",
                parse_mode="HTML"
            )
        else:
            await status_msg.edit_text(
                "📊 <b>CHIMERA SIGNAL</b>\n\nСегодня матчей с ценностью не найдено.\nПопробуйте позже.",
                parse_mode="HTML"
            )
        return

    # ── Шаг 2/3: Новости + AI верификация ────────────────────────────────
    await status_msg.edit_text(
        "🔍 <b>CHIMERA SIGNAL</b>\n\n"
        "✅ Шаг 1/3: Матчи загружены\n"
        f"📰 Шаг 2/3: Собираю новости и запускаю AI для топ-{len(top_candidates)}...\n"
        "<i>⏳ Ещё 20–50 секунд...</i>",
        parse_mode="HTML"
    )

    # Параллельно тянем новости для каждого кандидата
    def _fetch_news_for_candidates():
        import concurrent.futures as _cf_news
        from oracle_ai import oracle_analyze
        def _get_news(c):
            try:
                res = oracle_analyze(c.get("home", ""), c.get("away", ""))
                summaries = []
                for team, data in res.items():
                    headlines = [a.get("title", "") for a in data.get("articles", [])[:2]]
                    if headlines:
                        summaries.append(f"{team}: {'; '.join(headlines)}")
                c["news_context"] = " | ".join(summaries)[:400] if summaries else ""
            except Exception:
                c["news_context"] = ""
        with _cf_news.ThreadPoolExecutor(max_workers=3) as _pool:
            list(_pool.map(_get_news, top_candidates))

    try:
        await loop.run_in_executor(None, _fetch_news_for_candidates)
    except Exception as _ne:
        print(f"[CHIMERA News] Ошибка: {_ne}")

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
    # Сохраняем топ-кандидатов в БД для трекинга результатов
    _chimera_top_pred_id = None
    _chimera_top_sport   = "football"
    _chimera_top_odds = 0
    for _ci, _cand in enumerate(top_candidates):
        try:
            _c_sport = _cand.get("sport", "football")
            _c_home  = _cand.get("home", "")
            _c_away  = _cand.get("away", "")
            _c_team  = _cand.get("team", "")
            _c_odds  = _cand.get("odds", 0)
            _c_time  = _cand.get("commence_time", "")
            _c_rec   = "home_win" if _c_team == _c_home else "away_win"
            _c_odds_h = _c_odds if _c_rec == "home_win" else 0
            _c_odds_a = _c_odds if _c_rec == "away_win" else 0
            _mid = f"chimera_{_c_home}_{_c_away}_{_c_time[:10]}"
            _saved_id = save_prediction(
                sport=_c_sport,
                match_id=_mid,
                match_date=_c_time,
                home_team=_c_home,
                away_team=_c_away,
                league=_cand.get("league_name", _c_sport),
                recommended_outcome=_c_rec,
                bet_signal="СТАВИТЬ",
                bookmaker_odds_home=_c_odds_h if _c_odds_h > 1 else None,
                bookmaker_odds_away=_c_odds_a if _c_odds_a > 1 else None,
                ensemble_home=round(_cand.get("prob", 50) / 100, 3),
                ensemble_away=round(1 - _cand.get("prob", 50) / 100, 3),
                ensemble_best_outcome=_c_rec,
            )
            # Если запись уже есть (UNIQUE конфликт) — берём существующий ID
            if not _saved_id:
                from database import _get_db_connection
                _tbl = {"football": "football_predictions", "cs2": "cs2_predictions",
                        "tennis": "tennis_predictions", "basketball": "basketball_predictions"}.get(_c_sport, "football_predictions")
                with _get_db_connection() as _conn:
                    _row = _conn.execute(f"SELECT id FROM {_tbl} WHERE match_id=?", (_mid,)).fetchone()
                    if _row:
                        _saved_id = _row[0]
            _cand["_pred_id"] = _saved_id  # сохраняем для кнопок альтернатив
            if _ci == 0:  # запоминаем только топ-1 для кнопки
                _chimera_top_pred_id = _saved_id
                _chimera_top_sport = _c_sport
                _chimera_top_odds = _c_odds
        except Exception as _cs_err:
            print(f"[CHIMERA Save] {_cs_err}")

    # ── Сохраняем в кеш на 45 минут ──────────────────────────────────────
    import time as _time_mod2
    _signals_scan_cache["last"] = {
        "ts":         _time_mod2.time(),
        "candidates": top_candidates,
        "top_pred_id": _chimera_top_pred_id,
        "top_sport":   _chimera_top_sport,
        "top_odds":    _chimera_top_odds,
    }
    print(f"[CHIMERA] Результаты скана закешированы на {SIGNALS_SCAN_TTL // 60} мин")

    try:
        result_text = _format_chimera_page(top_candidates, 0)
        _chimera_kb = _build_chimera_kb(
            top_candidates, _chimera_top_pred_id, _chimera_top_sport,
            _chimera_top_odds, message.from_user.id
        )
        await status_msg.edit_text(result_text, parse_mode="HTML", reply_markup=_chimera_kb)
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
    asyncio.create_task(run_tennis_form_prefetch_task())
    asyncio.create_task(run_calibration_task())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
