import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict

DB_FILE = "chimera_predictions.db"

def _get_db_connection():
    return sqlite3.connect(DB_FILE)

def _migrate_old_predictions():
    """Мигрирует данные из старой таблицы 'predictions' в новые 'football_predictions' и 'cs2_predictions'."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()

        # Проверяем, существует ли старая таблица predictions
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
        if not cursor.fetchone():
            print("[База данных] Старая таблица 'predictions' не найдена, миграция не требуется.")
            return

        print("[База данных] Начинается миграция данных из старой таблицы 'predictions'...")

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
        print("[База данных] Миграция завершена. Старая таблица 'predictions' удалена.")

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
        conn.commit()
        print("[База данных] Новые таблицы 'football_predictions' и 'cs2_predictions' инициализированы.")

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
    real_home_score: int,
    real_away_score: int,
    real_outcome: str,
    is_correct: int,
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
        table_name = f"{sport}_predictions"
        current_time = datetime.now(timezone.utc).isoformat()

        update_query = f"""
            UPDATE {table_name} SET
                real_home_score = ?,
                real_away_score = ?,
                real_outcome = ?,
                is_correct = ?,
                is_ensemble_correct = ?,
                is_goals_correct = ?,
                is_btts_correct = ?,
                roi_outcome = ?,
                roi_value_bet = ?,
                value_bet_correct = ?,
                result_checked_at = ? -- Обновляем время проверки результата
            WHERE match_id = ?
        """
        cursor.execute(update_query, (
            real_home_score, real_away_score, real_outcome, is_correct,
            is_ensemble_correct, is_goals_correct, is_btts_correct,
            roi_outcome, roi_value_bet, value_bet_correct, current_time, match_id
        ))
        conn.commit()

def get_pending_predictions(sport: str) -> List[Dict]:
    """Возвращает список прогнозов, для которых ещё нет результатов."""
    with _get_db_connection() as conn:
        conn.row_factory = sqlite3.Row # Для доступа к колонкам по имени
        cursor = conn.cursor()
        table_name = f"{sport}_predictions"
        cursor.execute(f"SELECT * FROM {table_name} WHERE real_outcome IS NULL")
        return [dict(row) for row in cursor.fetchall()]

def get_statistics(sport: Optional[str] = None) -> Dict:
    """Возвращает статистику по прогнозам. Если sport не указан, возвращает общую статистику."""
    stats = {}
    sports_to_check = [sport] if sport else ['football', 'cs2']

    with _get_db_connection() as conn:
        cursor = conn.cursor()

        for s in sports_to_check:
            table_name = f"{s}_predictions"

            # Общая статистика
            total = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            correct = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE is_correct = 1").fetchone()[0]
            total_checked = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE is_correct IS NOT NULL").fetchone()[0]

            # Статистика по Value Bets
            vb_checked = cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE value_bet_correct IS NOT NULL"
            ).fetchone()[0]
            vb_correct_count = cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE value_bet_correct = 1"
            ).fetchone()[0]
            roi_vb_row = cursor.execute(
                f"SELECT SUM(roi_value_bet) FROM {table_name} WHERE roi_value_bet IS NOT NULL"
            ).fetchone()[0]
            roi_main_row = cursor.execute(
                f"SELECT SUM(roi_outcome) FROM {table_name} WHERE roi_outcome IS NOT NULL"
            ).fetchone()[0]
            vb_accuracy = (vb_correct_count / vb_checked * 100) if vb_checked > 0 else 0

            # Последние результаты
            recent = cursor.execute(f"""
            SELECT home_team, away_team, real_home_score, real_away_score,
                   recommended_outcome, is_correct, created_at,
                   is_ensemble_correct, value_bet_outcome, value_bet_correct, value_bet_odds
            FROM {table_name}
            WHERE is_correct IS NOT NULL
            ORDER BY result_checked_at DESC -- Используем новую колонку
            LIMIT 10
            """
            ).fetchall()

            # По месяцам
            monthly = cursor.execute(f"""
            SELECT strftime('%Y-%m', created_at) as month,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN is_ensemble_correct = 1 THEN 1 ELSE 0 END) as ens_correct,
                   SUM(CASE WHEN value_bet_correct = 1 THEN 1 ELSE 0 END) as vb_correct,
                   SUM(COALESCE(roi_value_bet, 0)) as roi_vb
            FROM {table_name}
            WHERE is_correct IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            LIMIT 6
            """
            ).fetchall()

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

    return stats


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
