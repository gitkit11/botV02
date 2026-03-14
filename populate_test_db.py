
import sqlite3
import random
import os

def populate_cs2_test_data(db_path: str = "chimera_predictions.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Удаляем таблицу, если она существует, чтобы гарантировать актуальную схему
    cursor.execute("DROP TABLE IF EXISTS cs2_predictions;")

    # Убедимся, что таблица cs2_predictions существует с нужными колонками
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cs2_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        sport TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        outcome TEXT NOT NULL, -- П1 или П2
        odds REAL NOT NULL,
        prob REAL NOT NULL,
        ev REAL NOT NULL,
        kelly REAL NOT NULL,
        result TEXT, -- \'П1\', \'П2\', или \'CANCEL\'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result_checked_at TIMESTAMP -- Добавлено
    );
    """)

    test_data = []
    # Сгенерируем 100 тестовых прогнозов
    for i in range(100):
        ev = random.uniform(0.01, 0.30) # EV от 1% до 30%
        odds = random.uniform(1.5, 3.0)
        prob = (ev + 1) / odds
        kelly = random.uniform(0.05, 0.15)
        outcome = "П1"

        # Создадим сценарий: прогнозы с EV > 10% более точные
        is_correct = False
        if ev > 0.10:
            # 70% шанс на правильный исход, если EV > 10%
            if random.random() < 0.7:
                is_correct = True
        else:
            # 45% шанс на правильный исход, если EV <= 10%
            if random.random() < 0.45:
                is_correct = True

        result = "П1" if is_correct else "П2"
        result_checked_at = "2024-03-14 12:00:00" # Для тестовых данных

        test_data.append((
            f"test_match_{i}", "cs2", "Team A", "Team B", outcome, odds, prob, ev, kelly, result, result_checked_at
        ))

    cursor.executemany(
        "INSERT INTO cs2_predictions (match_id, sport, home_team, away_team, outcome, odds, prob, ev, kelly, result, result_checked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        test_data
    )

    conn.commit()
    conn.close()
    print(f"База данных заполнена {len(test_data)} тестовыми записями для CS2.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "chimera_predictions.db")
    populate_cs2_test_data(db_path=db_path)
