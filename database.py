import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

DB_FILE = "chimera_predictions.db"

_VALID_SPORTS = {"football", "cs2", "tennis", "basketball"}

def _validate_sport(sport: str) -> str:
    if sport not in _VALID_SPORTS:
        raise ValueError(f"Invalid sport: {sport!r}")
    return sport

def _get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def _migrate_old_predictions():
    """Мигрирует данные из старой таблицы 'predictions' в новые 'football_predictions' и 'cs2_predictions'."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()

        # Проверяем, существует ли старая таблица predictions
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
        if not cursor.fetchone():
            print("[database] Old 'predictions' table not found, migration not needed.")
            return

        print("[database] Migrating data from old 'predictions' table...")

        # Извлекаем все данные из старой таблицы
        old_predictions = cursor.execute("SELECT * FROM predictions;").fetchall()
        old_columns = [description[0] for description in cursor.description]

        for row in old_predictions:
            old_data = dict(zip(old_columns, row))
            league = old_data.get('league', 'soccer_epl')
            sport = 'cs2' if 'cs2' in league.lower() or 'esports' in league.lower() else 'football'

            prediction_data = json.loads(old_data.get('prediction_data', '{}'))

            # Определяем result_checked_at для миграции
            migrated_result_checked_at = old_data.get('result_checked_at', old_data.get('created_at'))

            if sport == 'football':
                # Подготовка данных для football_predictions
                football_data = {
                    'match_id': old_data['match_id'],
                    'match_date': old_data['match_date'],
                    'home_team': old_data['home_team'],
                    'away_team': old_data['away_team'],
                    'league': league,
                    'gpt_verdict': old_data.get('gpt_verdict'),
                    'llama_verdict': old_data.get('llama_verdict'),
                    'mixtral_verdict': old_data.get('mixtral_verdict'),
                    'gpt_confidence': old_data.get('gpt_confidence'),
                    'llama_confidence': old_data.get('llama_confidence'),
                    'mixtral_confidence': old_data.get('mixtral_confidence'),
                    'bet_signal': old_data.get('bet_signal'),
                    'recommended_outcome': old_data.get('recommended_outcome'),
                    'total_goals_prediction': old_data.get('total_goals_prediction'),
                    'btts_prediction': old_data.get('btts_prediction'),
                    'poisson_home_win': old_data.get('poisson_home_win'),
                    'poisson_draw': old_data.get('poisson_draw'),
                    'poisson_away_win': old_data.get('poisson_away_win'),
                    'poisson_over25': old_data.get('poisson_over25'),
                    'poisson_btts': old_data.get('poisson_btts'),
                    'poisson_data_source': old_data.get('poisson_data_source'),
                    'elo_home': old_data.get('elo_home'),
                    'elo_away': old_data.get('elo_away'),
                    'elo_home_win': old_data.get('elo_home_win'),
                    'elo_draw': old_data.get('elo_draw'),
                    'elo_away_win': old_data.get('elo_away_win'),
                    'ensemble_home': old_data.get('ensemble_home'),
                    'ensemble_draw': old_data.get('ensemble_draw'),
                    'ensemble_away': old_data.get('ensemble_away'),
                    'ensemble_best_outcome': old_data.get('ensemble_best_outcome'),
                    'value_bet_outcome': old_data.get('value_bet_outcome'),
                    'value_bet_odds': old_data.get('value_bet_odds'),
                    'value_bet_ev': old_data.get('value_bet_ev'),
                    'value_bet_kelly': old_data.get('value_bet_kelly'),
                    'value_bet_correct': old_data.get('value_bet_correct'),
                    'bookmaker_odds_home': old_data.get('bookmaker_odds_home'),
                    'bookmaker_odds_draw': old_data.get('bookmaker_odds_draw'),
                    'bookmaker_odds_away': old_data.get('bookmaker_odds_away'),
                    'bookmaker_odds_over25': old_data.get('bookmaker_odds_over25'),
                    'bookmaker_odds_under25': old_data.get('bookmaker_odds_under25'),
                    'real_home_score': old_data.get('real_home_score'),
                    'real_away_score': old_data.get('real_away_score'),
                    'real_outcome': old_data.get('real_outcome'),
                    'is_correct': old_data.get('is_correct'),
                    'is_goals_correct': old_data.get('is_goals_correct'),
                    'is_btts_correct': old_data.get('is_btts_correct'),
                    'is_ensemble_correct': old_data.get('is_ensemble_correct'),
                    'roi_outcome': old_data.get('roi_outcome'),
                    'roi_value_bet': old_data.get('roi_value_bet'),
                    'model_weights_at_prediction': prediction_data.get('model_weights_at_prediction', '{}'),
                    'prediction_data': json.dumps(prediction_data, ensure_ascii=False),
                    'created_at': old_data.get('created_at'),
                    'result_checked_at': migrated_result_checked_at, # Добавлено
                }
                columns = ', '.join(football_data.keys())
                placeholders = ', '.join(['?' for _ in football_data.keys()])
                cursor.execute(f"INSERT OR REPLACE INTO football_predictions ({columns}) VALUES ({placeholders})", list(football_data.values()))
            elif sport == 'cs2':
                # Подготовка данных для cs2_predictions
                cs2_data = {
                    'match_id': old_data['match_id'],
                    'match_date': old_data['match_date'],
                    'home_team': old_data['home_team'],
                    'away_team': old_data['away_team'],
                    'league': league,
                    'gpt_verdict': old_data.get('gpt_verdict'),
                    'llama_verdict': old_data.get('llama_verdict'),
                    'mixtral_verdict': old_data.get('mixtral_verdict'),
                    'gpt_confidence': old_data.get('gpt_confidence'),
                    'llama_confidence': old_data.get('llama_confidence'),
                    'mixtral_confidence': old_data.get('mixtral_confidence'),
                    'bet_signal': old_data.get('bet_signal'),
                    'recommended_outcome': old_data.get('recommended_outcome'),
                    'elo_home': old_data.get('elo_home'),
                    'elo_away': old_data.get('elo_away'),
                    'elo_home_win': old_data.get('elo_home_win'),
                    'elo_away_win': old_data.get('elo_away_win'),
                    'ensemble_home': old_data.get('ensemble_home'),
                    'ensemble_away': old_data.get('ensemble_away'),
                    'ensemble_best_outcome': old_data.get('ensemble_best_outcome'),
                    'value_bet_outcome': old_data.get('value_bet_outcome'),
                    'value_bet_odds': old_data.get('value_bet_odds'),
                    'value_bet_ev': old_data.get('value_bet_ev'),
                    'value_bet_kelly': old_data.get('value_bet_kelly'),
                    'value_bet_correct': old_data.get('value_bet_correct'),
                    'bookmaker_odds_home': old_data.get('bookmaker_odds_home'),
                    'bookmaker_odds_away': old_data.get('bookmaker_odds_away'),
                    'predicted_maps': prediction_data.get('predicted_maps', '[]'),
                    'map_advantage_score': prediction_data.get('map_advantage_score'),
                    'key_player_advantage': prediction_data.get('key_player_advantage'),
                    'ai_signal_reason': prediction_data.get('ai_signal_reason'),
                    'home_map_winrates': prediction_data.get('home_map_winrates', '{}'),
                    'away_map_winrates': prediction_data.get('away_map_winrates', '{}'),
                    'home_player_ratings': prediction_data.get('home_player_ratings', '[]'),
                    'away_player_ratings': prediction_data.get('away_player_ratings', '[]'),
                    'real_home_score': old_data.get('real_home_score'),
                    'real_away_score': old_data.get('real_away_score'),
                    'real_outcome': old_data.get('real_outcome'),
                    'is_correct': old_data.get('is_correct'),
                    'is_ensemble_correct': old_data.get('is_ensemble_correct'),
                    'roi_outcome': old_data.get('roi_outcome'),
                    'roi_value_bet': old_data.get('roi_value_bet'),
                    'model_weights_at_prediction': prediction_data.get('model_weights_at_prediction', '{}'),
                    'prediction_data': json.dumps(prediction_data, ensure_ascii=False),
                    'created_at': old_data.get('created_at'),
                    'result_checked_at': migrated_result_checked_at, # Добавлено
                }
                columns = ', '.join(cs2_data.keys())
                placeholders = ', '.join(['?' for _ in cs2_data.keys()])
                cursor.execute(f"INSERT OR REPLACE INTO cs2_predictions ({columns}) VALUES ({placeholders})", list(cs2_data.values()))

        # Удаляем старую таблицу после успешной миграции
        cursor.execute("DROP TABLE predictions;")
        conn.commit()
        print("[database] Migration complete. Old 'predictions' table dropped.")

def init_db():
    """Инициализирует базу данных и создаёт таблицы."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()

        # Создание таблицы football_predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS football_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            gpt_verdict TEXT,
            llama_verdict TEXT,
            mixtral_verdict TEXT,
            gpt_confidence INTEGER,
            llama_confidence INTEGER,
            mixtral_confidence INTEGER,
            bet_signal TEXT,
            recommended_outcome TEXT,
            total_goals_prediction TEXT,
            btts_prediction TEXT,
            poisson_home_win REAL,
            poisson_draw REAL,
            poisson_away_win REAL,
            poisson_over25 REAL,
            poisson_btts REAL,
            poisson_data_source TEXT,
            elo_home INTEGER,
            elo_away INTEGER,
            elo_home_win REAL,
            elo_draw REAL,
            elo_away_win REAL,
            ensemble_home REAL,
            ensemble_draw REAL,
            ensemble_away REAL,
            ensemble_best_outcome TEXT,
            value_bet_outcome TEXT,
            value_bet_odds REAL,
            value_bet_ev REAL,
            value_bet_kelly REAL,
            value_bet_correct INTEGER,
            bookmaker_odds_home REAL,
            bookmaker_odds_draw REAL,
            bookmaker_odds_away REAL,
            bookmaker_odds_over25 REAL,
            bookmaker_odds_under25 REAL,
            real_home_score INTEGER,
            real_away_score INTEGER,
            real_outcome TEXT,
            is_correct INTEGER,
            is_goals_correct INTEGER,
            is_btts_correct INTEGER,
            is_ensemble_correct INTEGER,
            roi_outcome REAL,
            roi_value_bet REAL,
            model_weights_at_prediction TEXT,
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_checked_at TIMESTAMP -- Добавлено
        );
        """)

        # Создание таблицы cs2_predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cs2_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            gpt_verdict TEXT,
            llama_verdict TEXT,
            mixtral_verdict TEXT,
            gpt_confidence INTEGER,
            llama_confidence INTEGER,
            mixtral_confidence INTEGER,
            bet_signal TEXT,
            recommended_outcome TEXT,
            elo_home INTEGER,
            elo_away INTEGER,
            elo_home_win REAL,
            elo_away_win REAL,
            ensemble_home REAL,
            ensemble_away REAL,
            ensemble_best_outcome TEXT,
            value_bet_outcome TEXT,
            value_bet_odds REAL,
            value_bet_ev REAL,
            value_bet_kelly REAL,
            value_bet_correct INTEGER,
            bookmaker_odds_home REAL,
            bookmaker_odds_away REAL,
            predicted_maps TEXT, -- JSON строка
            map_advantage_score REAL,
            key_player_advantage REAL,
            ai_signal_reason TEXT,
            home_map_winrates TEXT, -- JSON строка
            away_map_winrates TEXT, -- JSON строка
            home_player_ratings TEXT, -- JSON строка
            away_player_ratings TEXT, -- JSON строка
            real_home_score INTEGER,
            real_away_score INTEGER,
            real_outcome TEXT,
            is_correct INTEGER,
            is_ensemble_correct INTEGER,
            roi_outcome REAL,
            roi_value_bet REAL,
            model_weights_at_prediction TEXT,
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_checked_at TIMESTAMP -- Добавлено
        );
        """)
        # Добавляем новые колонки CS2 если ещё нет (миграция без пересоздания)
        existing = [r[1] for r in cursor.execute("PRAGMA table_info(cs2_predictions)").fetchall()]
        new_cols = {
            "total_prediction":  "TEXT",
            "total_odds":        "REAL",
            "top_bet_type":      "TEXT",
            "top_bet_odds":      "REAL",
            "top_bet_ev":        "REAL",
            "is_total_correct":  "INTEGER",
            "actual_total":      "TEXT",
        }
        for col, col_type in new_cols.items():
            if col not in existing:
                cursor.execute(f"ALTER TABLE cs2_predictions ADD COLUMN {col} {col_type}")

        # Создание таблицы tennis_predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tennis_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            recommended_outcome TEXT,
            ensemble_home REAL,
            ensemble_away REAL,
            ensemble_best_outcome TEXT,
            bookmaker_odds_home REAL,
            bookmaker_odds_away REAL,
            real_outcome TEXT,
            is_correct INTEGER,
            roi_outcome REAL,
            value_bet_correct INTEGER,
            roi_value_bet REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_checked_at TIMESTAMP
        );
        """)

        # Создание таблицы basketball_predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS basketball_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            gpt_verdict TEXT,
            llama_verdict TEXT,
            bet_signal TEXT,
            recommended_outcome TEXT,
            elo_home INTEGER,
            elo_away INTEGER,
            elo_home_win REAL,
            elo_away_win REAL,
            ensemble_home REAL,
            ensemble_away REAL,
            ensemble_best_outcome TEXT,
            bookmaker_odds_home REAL,
            bookmaker_odds_away REAL,
            total_line REAL,
            total_lean TEXT,
            total_lean_odds REAL,
            total_ev REAL,
            spread_home REAL,
            spread_away REAL,
            real_outcome TEXT,
            is_correct INTEGER,
            roi_outcome REAL,
            value_bet_correct INTEGER,
            roi_value_bet REAL,
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_checked_at TIMESTAMP
        );
        """)
        # Миграция: добавляем value_bet колонки если их нет
        _bb_cols = [r[1] for r in cursor.execute("PRAGMA table_info(basketball_predictions)").fetchall()]
        if "value_bet_correct" not in _bb_cols:
            cursor.execute("ALTER TABLE basketball_predictions ADD COLUMN value_bet_correct INTEGER")
        if "roi_value_bet" not in _bb_cols:
            cursor.execute("ALTER TABLE basketball_predictions ADD COLUMN roi_value_bet REAL")

        # Таблица пользователей (личный кабинет)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY,
            username    TEXT,
            first_name  TEXT,
            first_seen  TEXT,
            last_active TEXT,
            analyses_football    INTEGER DEFAULT 0,
            analyses_cs2         INTEGER DEFAULT 0,
            analyses_tennis      INTEGER DEFAULT 0,
            analyses_basketball  INTEGER DEFAULT 0,
            analyses_total       INTEGER DEFAULT 0
        );
        """)
        # Миграция: добавляем недостающие колонки
        _u_cols = [r[1] for r in cursor.execute("PRAGMA table_info(users)").fetchall()]
        if "analyses_basketball" not in _u_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN analyses_basketball INTEGER DEFAULT 0")
        if "language" not in _u_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN language TEXT DEFAULT 'ru'")
        if "bankroll" not in _u_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN bankroll REAL DEFAULT NULL")

        conn.commit()
        print("[database] Tables initialized.")

    _migrate_old_predictions()

def save_prediction(
    sport: str,
    match_id: str,
    match_date: str,
    home_team: str,
    away_team: str,
    league: str,
    gpt_verdict: Optional[str] = None,
    llama_verdict: Optional[str] = None,
    mixtral_verdict: Optional[str] = None,
    gpt_confidence: Optional[int] = None,
    llama_confidence: Optional[int] = None,
    mixtral_confidence: Optional[int] = None,
    bet_signal: Optional[str] = None,
    recommended_outcome: Optional[str] = None,
    total_goals_prediction: Optional[str] = None,
    btts_prediction: Optional[str] = None,
    poisson_home_win: Optional[float] = None,
    poisson_draw: Optional[float] = None,
    poisson_away_win: Optional[float] = None,
    poisson_over25: Optional[float] = None,
    poisson_btts: Optional[float] = None,
    poisson_data_source: Optional[str] = None,
    elo_home: Optional[int] = None,
    elo_away: Optional[int] = None,
    elo_home_win: Optional[float] = None,
    elo_draw: Optional[float] = None,
    elo_away_win: Optional[float] = None,
    ensemble_home: Optional[float] = None,
    ensemble_draw: Optional[float] = None,
    ensemble_away: Optional[float] = None,
    ensemble_best_outcome: Optional[str] = None,
    value_bet_outcome: Optional[str] = None,
    value_bet_odds: Optional[float] = None,
    value_bet_ev: Optional[float] = None,
    value_bet_kelly: Optional[float] = None,
    value_bet_correct: Optional[int] = None,
    bookmaker_odds_home: Optional[float] = None,
    bookmaker_odds_draw: Optional[float] = None,
    bookmaker_odds_away: Optional[float] = None,
    bookmaker_odds_over25: Optional[float] = None,
    bookmaker_odds_under25: Optional[float] = None,
    # CS2 Specific
    predicted_maps: Optional[List[str]] = None,
    map_advantage_score: Optional[float] = None,
    key_player_advantage: Optional[float] = None,
    ai_signal_reason: Optional[str] = None,
    home_map_winrates: Optional[Dict[str, float]] = None,
    away_map_winrates: Optional[Dict[str, float]] = None,
    home_player_ratings: Optional[List[float]] = None,
    away_player_ratings: Optional[List[float]] = None,
    model_weights_at_prediction: Optional[Dict] = None,
    prediction_data: Optional[Dict] = None,
):
    """Сохраняет прогноз в соответствующую таблицу базы данных."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        current_time = datetime.now(timezone.utc).isoformat()

        if sport == 'football':
            table_name = 'football_predictions'
            data = {
                'match_id': match_id,
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'gpt_verdict': gpt_verdict,
                'llama_verdict': llama_verdict,
                'mixtral_verdict': mixtral_verdict,
                'gpt_confidence': gpt_confidence,
                'llama_confidence': llama_confidence,
                'mixtral_confidence': mixtral_confidence,
                'bet_signal': bet_signal,
                'recommended_outcome': recommended_outcome,
                'total_goals_prediction': total_goals_prediction,
                'btts_prediction': btts_prediction,
                'poisson_home_win': poisson_home_win,
                'poisson_draw': poisson_draw,
                'poisson_away_win': poisson_away_win,
                'poisson_over25': poisson_over25,
                'poisson_btts': poisson_btts,
                'poisson_data_source': poisson_data_source,
                'elo_home': elo_home,
                'elo_away': elo_away,
                'elo_home_win': elo_home_win,
                'elo_draw': elo_draw,
                'elo_away_win': elo_away_win,
                'ensemble_home': ensemble_home,
                'ensemble_draw': ensemble_draw,
                'ensemble_away': ensemble_away,
                'ensemble_best_outcome': ensemble_best_outcome,
                'value_bet_outcome': value_bet_outcome,
                'value_bet_odds': value_bet_odds,
                'value_bet_ev': value_bet_ev,
                'value_bet_kelly': value_bet_kelly,
                'value_bet_correct': value_bet_correct,
                'bookmaker_odds_home': bookmaker_odds_home,
                'bookmaker_odds_draw': bookmaker_odds_draw,
                'bookmaker_odds_away': bookmaker_odds_away,
                'bookmaker_odds_over25': bookmaker_odds_over25,
                'bookmaker_odds_under25': bookmaker_odds_under25,
                'model_weights_at_prediction': json.dumps(model_weights_at_prediction) if model_weights_at_prediction else '{}',
                'prediction_data': json.dumps(prediction_data) if prediction_data else '{}',
                'created_at': current_time,
                'result_checked_at': None, # Изначально null
            }
        elif sport == 'cs2':
            table_name = 'cs2_predictions'
            data = {
                'match_id': match_id,
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'gpt_verdict': gpt_verdict,
                'llama_verdict': llama_verdict,
                'mixtral_verdict': mixtral_verdict,
                'gpt_confidence': gpt_confidence,
                'llama_confidence': llama_confidence,
                'mixtral_confidence': mixtral_confidence,
                'bet_signal': bet_signal,
                'recommended_outcome': recommended_outcome,
                'elo_home': elo_home,
                'elo_away': elo_away,
                'elo_home_win': elo_home_win,
                'elo_away_win': elo_away_win,
                'ensemble_home': ensemble_home,
                'ensemble_away': ensemble_away,
                'ensemble_best_outcome': ensemble_best_outcome,
                'value_bet_outcome': value_bet_outcome,
                'value_bet_odds': value_bet_odds,
                'value_bet_ev': value_bet_ev,
                'value_bet_kelly': value_bet_kelly,
                'value_bet_correct': value_bet_correct,
                'bookmaker_odds_home': bookmaker_odds_home,
                'bookmaker_odds_away': bookmaker_odds_away,
                'predicted_maps': json.dumps(predicted_maps) if predicted_maps else '[]',
                'map_advantage_score': map_advantage_score,
                'key_player_advantage': key_player_advantage,
                'ai_signal_reason': ai_signal_reason,
                'home_map_winrates': json.dumps(home_map_winrates) if home_map_winrates else '{}',
                'away_map_winrates': json.dumps(away_map_winrates) if away_map_winrates else '{}',
                'home_player_ratings': json.dumps(home_player_ratings) if home_player_ratings else '[]',
                'away_player_ratings': json.dumps(away_player_ratings) if away_player_ratings else '[]',
                'model_weights_at_prediction': json.dumps(model_weights_at_prediction) if model_weights_at_prediction else '{}',
                'prediction_data': json.dumps(prediction_data) if prediction_data else '{}',
                'created_at': current_time,
                'result_checked_at': None, # Изначально null
            }
        elif sport == 'tennis':
            table_name = 'tennis_predictions'
            data = {
                'match_id': match_id,
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'recommended_outcome': recommended_outcome,
                'ensemble_home': ensemble_home,
                'ensemble_away': ensemble_away,
                'ensemble_best_outcome': ensemble_best_outcome,
                'bookmaker_odds_home': bookmaker_odds_home,
                'bookmaker_odds_away': bookmaker_odds_away,
                'created_at': current_time,
                'result_checked_at': None,
            }
        elif sport == 'basketball':
            table_name = 'basketball_predictions'
            data = {
                'match_id': match_id,
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'gpt_verdict': gpt_verdict,
                'llama_verdict': llama_verdict,
                'bet_signal': bet_signal,
                'recommended_outcome': recommended_outcome,
                'elo_home': elo_home,
                'elo_away': elo_away,
                'elo_home_win': elo_home_win,
                'elo_away_win': elo_away_win,
                'ensemble_home': ensemble_home,
                'ensemble_away': ensemble_away,
                'ensemble_best_outcome': ensemble_best_outcome,
                'bookmaker_odds_home': bookmaker_odds_home,
                'bookmaker_odds_away': bookmaker_odds_away,
                'prediction_data': json.dumps(prediction_data) if prediction_data else '{}',
                'created_at': current_time,
                'result_checked_at': None,
            }
        else:
            raise ValueError(f"Неизвестный вид спорта: {sport}")

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        cursor.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", list(data.values()))
        conn.commit()
        return cursor.lastrowid

def update_result(
    sport: str,
    match_id: str,
    real_home_score: Optional[int],
    real_away_score: Optional[int],
    real_outcome: str,
    is_correct: Optional[int] = None,
    is_ensemble_correct: Optional[int] = None,
    is_goals_correct: Optional[int] = None,
    is_btts_correct: Optional[int] = None,
    roi_outcome: Optional[float] = None,
    roi_value_bet: Optional[float] = None,
    value_bet_correct: Optional[int] = None,
):
    """Обновляет результаты прогноза в соответствующей таблице."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        table_name = f"{_validate_sport(sport)}_predictions"
        current_time = datetime.now(timezone.utc).isoformat()

        # Получаем реальные колонки таблицы
        cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]

        # Строим UPDATE динамически — только существующие колонки
        # Используем append чтобы сохранить порядок set_parts == values
        set_parts = ["real_outcome = ?", "result_checked_at = ?"]
        values    = [real_outcome, current_time]

        if "real_home_score" in cols:
            set_parts.append("real_home_score = ?")
            values.append(real_home_score)
        if "real_away_score" in cols:
            set_parts.append("real_away_score = ?")
            values.append(real_away_score)
        # is_correct может быть None (нет рекомендации — не считать)
        if "is_correct" in cols:
            set_parts.append("is_correct = ?")
            values.append(is_correct)
        if "is_ensemble_correct" in cols and is_ensemble_correct is not None:
            set_parts.append("is_ensemble_correct = ?")
            values.append(is_ensemble_correct)
        if "is_goals_correct" in cols and is_goals_correct is not None:
            set_parts.append("is_goals_correct = ?")
            values.append(is_goals_correct)
        if "is_btts_correct" in cols and is_btts_correct is not None:
            set_parts.append("is_btts_correct = ?")
            values.append(is_btts_correct)
        if "roi_outcome" in cols:
            set_parts.append("roi_outcome = ?")
            values.append(roi_outcome)
        if "roi_value_bet" in cols and roi_value_bet is not None:
            set_parts.append("roi_value_bet = ?")
            values.append(roi_value_bet)
        if "value_bet_correct" in cols and value_bet_correct is not None:
            set_parts.append("value_bet_correct = ?")
            values.append(value_bet_correct)

        values.append(match_id)
        cursor.execute(
            f"UPDATE {table_name} SET {', '.join(set_parts)} WHERE match_id = ?",
            values
        )
        conn.commit()

def get_pending_predictions(sport: str) -> List[Dict]:
    """Возвращает список прогнозов, для которых ещё нет результатов."""
    with _get_db_connection() as conn:
        conn.row_factory = sqlite3.Row # Для доступа к колонкам по имени
        cursor = conn.cursor()
        table_name = f"{_validate_sport(sport)}_predictions"
        cursor.execute(f"SELECT * FROM {table_name} WHERE real_outcome IS NULL")
        return [dict(row) for row in cursor.fetchall()]

def get_statistics(sport: Optional[str] = None) -> Dict:
    """Возвращает статистику по прогнозам. Если sport не указан, возвращает общую статистику."""
    stats = {}
    sports_to_check = [sport] if sport else ['football', 'cs2', 'tennis', 'basketball']

    with _get_db_connection() as conn:
        cursor = conn.cursor()

        for s in sports_to_check:
            table_name = f"{s}_predictions"
            try:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if not cursor.fetchone():
                    continue
            except Exception:
                continue

            try:
                # Общая статистика
                total = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                correct = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE is_correct = 1").fetchone()[0]
                total_checked = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE is_correct IS NOT NULL").fetchone()[0]

                # Статистика по Value Bets
                cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]
                has_vb = "value_bet_correct" in cols
                has_roi_vb = "roi_value_bet" in cols

                vb_checked = cursor.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE value_bet_correct IS NOT NULL"
                ).fetchone()[0] if has_vb else 0
                vb_correct_count = cursor.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE value_bet_correct = 1"
                ).fetchone()[0] if has_vb else 0
                roi_vb_row = cursor.execute(
                    f"SELECT SUM(roi_value_bet) FROM {table_name} WHERE roi_value_bet IS NOT NULL"
                ).fetchone()[0] if has_roi_vb else None
                roi_main_row = cursor.execute(
                    f"SELECT SUM(roi_outcome) FROM {table_name} WHERE roi_outcome IS NOT NULL"
                ).fetchone()[0] if "roi_outcome" in cols else None
                vb_accuracy = (vb_correct_count / vb_checked * 100) if vb_checked > 0 else 0

                # Последние результаты — выбираем только существующие колонки
                base_cols = "home_team, away_team, recommended_outcome, is_correct, created_at"
                extra_cols = ""
                if "real_home_score" in cols:
                    extra_cols += ", real_home_score, real_away_score"
                else:
                    extra_cols += ", NULL as real_home_score, NULL as real_away_score"
                if "is_ensemble_correct" in cols:
                    extra_cols += ", is_ensemble_correct"
                else:
                    extra_cols += ", NULL as is_ensemble_correct"
                if "value_bet_outcome" in cols:
                    extra_cols += ", value_bet_outcome, value_bet_correct, value_bet_odds"
                else:
                    extra_cols += ", NULL as value_bet_outcome, NULL as value_bet_correct, NULL as value_bet_odds"

                recent = cursor.execute(f"""
                SELECT {base_cols}{extra_cols}
                FROM {table_name}
                WHERE is_correct IS NOT NULL
                ORDER BY result_checked_at DESC
                LIMIT 10
                """).fetchall()

                # По месяцам
                ens_expr = "SUM(CASE WHEN is_ensemble_correct = 1 THEN 1 ELSE 0 END)" if "is_ensemble_correct" in cols else "0"
                vb_expr = "SUM(CASE WHEN value_bet_correct = 1 THEN 1 ELSE 0 END)" if has_vb else "0"
                roi_vb_expr = "SUM(COALESCE(roi_value_bet, 0))" if has_roi_vb else "0"
                monthly = cursor.execute(f"""
                SELECT strftime('%Y-%m', created_at) as month,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                       {ens_expr} as ens_correct,
                       {vb_expr} as vb_correct,
                       {roi_vb_expr} as roi_vb
                FROM {table_name}
                WHERE is_correct IS NOT NULL
                GROUP BY month
                ORDER BY month DESC
                LIMIT 6
                """).fetchall()

                stats[s] = {
                    "total": total,
                    "total_checked": total_checked,
                    "correct": correct,
                    "accuracy": (correct / total_checked * 100) if total_checked > 0 else 0,
                    "roi_main": round(roi_main_row, 2) if roi_main_row is not None else 0,
                    "vb_checked": vb_checked,
                    "vb_correct": vb_correct_count,
                    "vb_accuracy": round(vb_accuracy, 2),
                    "roi_value_bet": round(roi_vb_row, 2) if roi_vb_row is not None else 0,
                    "recent": [dict(row) for row in recent],
                    "monthly": [dict(row) for row in monthly],
                }
            except Exception as _stat_err:
                import logging as _log
                _log.getLogger(__name__).warning(f"[DB] get_statistics {s} error: {_stat_err}")

    return stats


# ─── Личный кабинет пользователя ─────────────────────────────────────────────

def upsert_user(user_id: int, username: str = "", first_name: str = ""):
    """Создаёт или обновляет запись пользователя."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_db_connection() as conn:
        conn.execute("""
            INSERT INTO users (user_id, username, first_name, first_seen, last_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username    = excluded.username,
                first_name  = excluded.first_name,
                last_active = excluded.last_active
        """, (user_id, username or "", first_name or "", now, now))
        conn.commit()


def track_analysis(user_id: int, sport: str):
    """Увеличивает счётчик анализов пользователя по виду спорта."""
    col_map = {"football": "analyses_football", "cs2": "analyses_cs2", "tennis": "analyses_tennis", "basketball": "analyses_basketball"}
    col = col_map.get(sport, "analyses_football")
    now = datetime.now(timezone.utc).isoformat()
    with _get_db_connection() as conn:
        conn.execute(f"""
            UPDATE users SET {col} = {col} + 1,
                             analyses_total = analyses_total + 1,
                             last_active = ?
            WHERE user_id = ?
        """, (now, user_id))
        conn.commit()


def get_user_profile(user_id: int) -> Optional[Dict]:
    """Возвращает профиль пользователя или None если не найден."""
    with _get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def get_user_language(user_id: int) -> str:
    """Возвращает язык пользователя (ru/en), по умолчанию ru."""
    with _get_db_connection() as conn:
        row = conn.execute("SELECT language FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return (row["language"] or "ru") if row else "ru"


def set_user_language(user_id: int, lang: str):
    """Сохраняет выбранный язык пользователя."""
    with _get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users (user_id, language) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET language = excluded.language",
            (user_id, lang)
        )
        conn.commit()


def set_user_bankroll(user_id: int, amount: float):
    """Сохраняет размер банка пользователя."""
    with _get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users (user_id, bankroll) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET bankroll = excluded.bankroll",
            (user_id, amount)
        )
        conn.commit()


def get_user_bankroll(user_id: int) -> Optional[float]:
    """Возвращает банк пользователя или None."""
    with _get_db_connection() as conn:
        row = conn.execute(
            "SELECT bankroll FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else None


def get_pl_stats(days: int = 30) -> dict:
    """
    Считает P&L статистику бота за последние N дней.
    Только сигналы где bet_signal='СТАВИТЬ' и есть real_outcome.
    Возвращает: total, wins, losses, roi, profit_units, best_streak, current_streak
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    tables = ["football_predictions", "cs2_predictions", "tennis_predictions", "basketball_predictions"]

    all_bets = []
    with _get_db_connection() as conn:
        for table in tables:
            try:
                rows = conn.execute(f"""
                    SELECT recommended_outcome, real_outcome,
                           bookmaker_odds_home, bookmaker_odds_away,
                           created_at
                    FROM {table}
                    WHERE bet_signal = 'СТАВИТЬ'
                      AND real_outcome IS NOT NULL
                      AND real_outcome != ''
                      AND real_outcome != 'expired'
                      AND created_at >= ?
                    ORDER BY created_at ASC
                """, (cutoff,)).fetchall()
                all_bets.extend(rows)
            except Exception:
                continue

    if not all_bets:
        return {"total": 0, "wins": 0, "losses": 0, "roi": 0.0,
                "profit_units": 0.0, "best_streak": 0, "current_streak": 0}

    wins = 0
    losses = 0
    profit = 0.0
    best_streak = 0
    current_streak = 0
    streak_sign = None  # 'w' or 'l'

    for rec_outcome, real_outcome, odds_home, odds_away, _ in all_bets:
        is_win = (rec_outcome == real_outcome)
        # Определяем коэффициент рекомендованного исхода
        if rec_outcome == "home_win":
            odds = float(odds_home or 0)
        elif rec_outcome == "away_win":
            odds = float(odds_away or 0)
        else:
            odds = 0.0
        if odds < 1.02:
            odds = 1.85  # fallback

        if is_win:
            wins += 1
            profit += (odds - 1)
            if streak_sign == 'w':
                current_streak += 1
            else:
                current_streak = 1
                streak_sign = 'w'
        else:
            losses += 1
            profit -= 1.0
            if streak_sign == 'l':
                current_streak += 1
            else:
                current_streak = 1
                streak_sign = 'l'

        if streak_sign == 'w' and current_streak > best_streak:
            best_streak = current_streak

    total = wins + losses
    roi = round(profit / total * 100, 1) if total > 0 else 0.0

    return {
        "total":          total,
        "wins":           wins,
        "losses":         losses,
        "roi":            roi,
        "profit_units":   round(profit, 2),
        "best_streak":    best_streak,
        "current_streak": current_streak if streak_sign == 'w' else 0,
        "days":           days,
    }


if __name__ == "__main__":
    init_db()
    # Пример использования (можно раскомментировать для ручного тестирования)
    # save_prediction(
    #     sport='football',
    #     match_id='match_123',
    #     match_date='2024-03-15',
    #     home_team='Team A',
    #     away_team='Team B',
    #     league='EPL',
    #     recommended_outcome='home_win',
    #     value_bet_outcome='home_win',
    #     value_bet_odds=1.8,
    #     value_bet_ev=0.1,
    #     value_bet_kelly=0.05,
    #     bookmaker_odds_home=1.8,
    #     bookmaker_odds_draw=3.5,
    #     bookmaker_odds_away=4.0,
    # )
    # update_result(
    #     sport='football',
    #     match_id='match_123',
    #     real_home_score=2,
    #     real_away_score=1,
    #     real_outcome='home_win',
    #     is_correct=1,
    #     roi_outcome=0.8,
    #     roi_value_bet=0.8,
    #     value_bet_correct=1,
    # )
    # print(get_statistics('football'))

    # save_prediction(
    #     sport='cs2',
    #     match_id='cs2_match_456',
    #     match_date='2024-03-15',
    #     home_team='Navi',
    #     away_team='G2',
    #     league='ESL Pro League',
    #     recommended_outcome='home_win',
    #     value_bet_outcome='home_win',
    #     value_bet_odds=1.7,
    #     value_bet_ev=0.08,
    #     value_bet_kelly=0.04,
    #     bookmaker_odds_home=1.7,
    #     bookmaker_odds_away=2.1,
    #     predicted_maps=['Inferno', 'Nuke'],
    #     map_advantage_score=0.12,
    #     key_player_advantage=0.1,
    # )
    # update_result(
    #     sport='cs2',
    #     match_id='cs2_match_456',
    #     real_home_score=2,
    #     real_away_score=0,
    #     real_outcome='home_win',
    #     is_correct=1,
    #     roi_outcome=0.7,
    #     roi_value_bet=0.7,
    #     value_bet_correct=1,
    # )
    # print(get_statistics('cs2'))

    # print(get_statistics())
