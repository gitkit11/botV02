# -*- coding: utf-8 -*-
"""
prophet_loader.py — загрузка нейросети Пророк и get_prophet_prediction.
Вынесено из main.py чтобы не загрязнять пространство имён handlers.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

prophet_model = None
scaler = None
team_encoder: dict = {}
data = None
feature_cols: list = []


def init():
    """Загружает Prophet модель и датасет. Вызывается один раз при старте."""
    global prophet_model, scaler, team_encoder, data, feature_cols

    # TensorFlow — импортируем здесь чтобы не замедлять остальной старт
    try:
        import tensorflow as tf
        prophet_model = tf.keras.models.load_model("prophet_model.keras")
        print("[Prophet] Модель загружена.")
    except Exception as e:
        print(f"[Prophet] WARN: не удалось загрузить модель: {e}")
        prophet_model = None

    try:
        import json
        _base = os.path.dirname(os.path.abspath(__file__))
        _csv_paths = [
            os.path.join(_base, "ml", "data", "all_matches_featured.csv"),
            os.path.join(_base, "all_matches_featured.csv"),
        ]
        _csv_path = next((p for p in _csv_paths if os.path.exists(p)), None)
        if _csv_path is None:
            raise FileNotFoundError("all_matches_featured.csv не найден")
        data = pd.read_csv(_csv_path, index_col=0)
        feature_cols = [c for c in data.columns if c not in ('FTR', 'label', 'HomeTeam', 'AwayTeam')]
        scaler = MinMaxScaler()
        scaler.fit(data[feature_cols])
        _enc_path = os.path.join(_base, "team_encoder.json")
        with open(_enc_path, 'r', encoding='utf-8') as f:
            team_encoder = json.load(f)
        print(f"[Prophet] Датасет готов ({_csv_path}). Команд: {len(team_encoder)}")
    except Exception as e:
        print(f"[Prophet] Датасет не найден (некритично): {e}")
        data = None
        scaler = None
        team_encoder = {}


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


def normalize_team(name: str) -> str:
    """Нормализует название команды для поиска в датасете АПЛ."""
    return TEAM_NAME_MAP.get(name, name)


def get_prophet_prediction(home_team: str, away_team: str):
    """Возвращает предсказание нейросети Пророк [draw%, home%, away%] или None."""
    if not prophet_model or data is None or scaler is None:
        return [0.33, 0.33, 0.34]
    try:
        home_norm = normalize_team(home_team)
        away_norm = normalize_team(away_team)
        home_id = team_encoder.get(home_norm)
        away_id = team_encoder.get(away_norm)
        if home_id is None or away_id is None:
            print(f"[Пророк] Команды вне АПЛ: '{home_team}'/'{away_team}' — пропускаем")
            return None
        home_data = data[data['HomeTeam_encoded'] == home_id].tail(5)
        away_data = data[data['AwayTeam_encoded'] == away_id].tail(5)
        if len(home_data) < 3 or len(away_data) < 3:
            sample = data.tail(10)
        else:
            sample = pd.concat([home_data, away_data]).tail(10)
        if len(sample) < 10:
            sample = pd.concat([sample, data.tail(10 - len(sample))])
        sample = sample[feature_cols].tail(10)
        scaled = scaler.transform(sample)
        prediction = prophet_model.predict(np.array([scaled]), verbose=0)[0]
        print(f"[Пророк] {home_team} vs {away_team}: П1={prediction[1]:.2f} Х={prediction[0]:.2f} П2={prediction[2]:.2f}")
        return [float(prediction[0]), float(prediction[1]), float(prediction[2])]
    except Exception as e:
        print(f"[Пророк] Ошибка: {e}")
        return [0.33, 0.33, 0.34]
