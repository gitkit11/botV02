import sqlite3
import random
import os

def populate_test_data(db_path: str = "chimera_predictions.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for sport in ['football', 'cs2']:
        table_name = f"{sport}_predictions"
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [c[1] for c in cursor.fetchall()]
        if not cols: continue
            
        cursor.execute(f"DELETE FROM {table_name}")
        
        test_data = []
        for i in range(50):
            ev = random.uniform(0.01, 0.25)
            is_correct = (random.random() < 0.65) if ev >= 0.10 else (random.random() < 0.30)
            
            # Определяем колонку результата
            res_col = "real_outcome" if "real_outcome" in cols else "result"
            
            data = {
                "match_id": f"test_{sport}_{i}",
                "match_date": "2026-03-14",
                "home_team": "Team A",
                "away_team": "Team B",
                "league": "Test League",
                res_col: "П1" if is_correct else "П2",
                "value_bet_outcome": "П1",
                "value_bet_odds": 2.0,
                "value_bet_ev": ev,
                "value_bet_kelly": 0.1,
                "sport": sport,
                "outcome": "П1",
                "odds": 2.0,
                "prob": 0.5,
                "ev": ev,
                "kelly": 0.1
            }
            
            row = []
            valid_cols = []
            for k, v in data.items():
                if k in cols:
                    valid_cols.append(k)
                    row.append(v)
            
            test_data.append(tuple(row))
            
        col_str = ", ".join(valid_cols)
        placeholders = ", ".join(["?"] * len(valid_cols))
        cursor.executemany(f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})", test_data)

    conn.commit()
    conn.close()
    print("База данных заполнена.")

if __name__ == "__main__":
    populate_test_data()
