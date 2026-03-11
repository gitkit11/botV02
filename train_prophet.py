import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Загрузка и объединение данных ---
def load_and_merge_data(data_path):
    all_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv') and file[:2] in ["E0", "SP", "D1", "I1", "F1"]:
                all_files.append(os.path.join(root, file))
    df_list = [pd.read_csv(file, encoding='latin1', on_bad_lines='skip') for file in all_files]
    df_list = [df for df in df_list if not df.empty]
    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

# --- 2. Очистка и базовая подготовка ---
def initial_preprocess(df):
    features = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    if 'HG' in df.columns and 'FTHG' not in df.columns:
        df.rename(columns={'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'}, inplace=True)
    if not all(col in df.columns for col in features):
        return pd.DataFrame()
    df = df[features].copy()
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df['Result'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 2 if x == 'A' else 0)
    return df

# --- 3. Feature Engineering ---
def create_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_map = {team: i for i, team in enumerate(teams)}
    df['HomeTeamID'] = df['HomeTeam'].map(team_map)
    df['AwayTeamID'] = df['AwayTeam'].map(team_map)
    team_stats = {}
    home_features_list, away_features_list = [], []
    for index, row in df.iterrows():
        home_id, away_id = row['HomeTeamID'], row['AwayTeamID']
        if home_id not in team_stats: team_stats[home_id] = {'goals_scored': [], 'goals_conceded': []}
        if away_id not in team_stats: team_stats[away_id] = {'goals_scored': [], 'goals_conceded': []}
        home_hist_gs, home_hist_gc = team_stats[home_id]['goals_scored'], team_stats[home_id]['goals_conceded']
        home_features = {
            'avg_gs_5': np.mean(home_hist_gs[-5:]) if len(home_hist_gs) >= 5 else 0,
            'avg_gc_5': np.mean(home_hist_gc[-5:]) if len(home_hist_gc) >= 5 else 0,
            'avg_gs_10': np.mean(home_hist_gs[-10:]) if len(home_hist_gs) >= 10 else 0,
            'avg_gc_10': np.mean(home_hist_gc[-10:]) if len(home_hist_gc) >= 10 else 0
        }
        home_features_list.append(home_features)
        away_hist_gs, away_hist_gc = team_stats[away_id]['goals_scored'], team_stats[away_id]['goals_conceded']
        away_features = {
            'avg_gs_5': np.mean(away_hist_gs[-5:]) if len(away_hist_gs) >= 5 else 0,
            'avg_gc_5': np.mean(away_hist_gc[-5:]) if len(away_hist_gc) >= 5 else 0,
            'avg_gs_10': np.mean(away_hist_gs[-10:]) if len(away_hist_gs) >= 10 else 0,
            'avg_gc_10': np.mean(away_hist_gc[-10:]) if len(away_hist_gc) >= 10 else 0
        }
        away_features_list.append(away_features)
        team_stats[home_id]['goals_scored'].append(row['FTHG'])
        team_stats[home_id]['goals_conceded'].append(row['FTAG'])
        team_stats[away_id]['goals_scored'].append(row['FTAG'])
        team_stats[away_id]['goals_conceded'].append(row['FTHG'])
    home_df = pd.DataFrame(home_features_list, index=df.index).add_prefix('H_')
    away_df = pd.DataFrame(away_features_list, index=df.index).add_prefix('A_')
    df = pd.concat([df, home_df, away_df], axis=1)
    return df, team_map

# --- 4. Создание последовательностей для LSTM ---
def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# --- Основной блок ---
if __name__ == '__main__':
    # Загрузка и предобработка данных (если файл еще не создан)
    if not os.path.exists('featured_football_data.csv'):
        DATA_PATH = '/home/ubuntu/.cache/kagglehub/datasets/mexwell/historical-football-resultsbetting-odds-data/versions/2'
        print("Шаг 1-3: Загрузка, обработка и создание признаков...")
        raw_data = load_and_merge_data(DATA_PATH)
        processed_data = initial_preprocess(raw_data)
        featured_data, _ = create_features(processed_data)
        featured_data.to_csv('featured_football_data.csv', index=False)
        print("Данные с признаками сохранены.")
    else:
        print("Найден файл 'featured_football_data.csv', используем его.")
        featured_data = pd.read_csv('featured_football_data.csv')

    # --- 4. Подготовка данных для модели ---
    print("\nШаг 4: Подготовка данных для нейросети...")
    # Выбираем только числовые фичи для модели
    features_to_scale = [col for col in featured_data.columns if col.startswith(('H_', 'A_'))]
    target = 'Result'
    
    # Масштабирование фичей
    scaler = StandardScaler()
    featured_data[features_to_scale] = scaler.fit_transform(featured_data[features_to_scale])

    # Создание последовательностей
    SEQUENCE_LENGTH = 10
    X, y = create_sequences(featured_data[features_to_scale + [target]].values, SEQUENCE_LENGTH)
    
    # Разделяем X и y
    X_data = X[:, :, :-1]
    y_data = X[:, -1, -1] # Результат последнего матча в последовательности

    # Разделение на обучающую, валидационную и тестовую выборки
    X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # --- 5. Построение и обучение LSTM-модели ---
    print("\nШаг 5: Построение и обучение LSTM-модели...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax') # 3 класса: ничья, победа хозяев, победа гостей
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # Ранняя остановка для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    # --- 6. Оценка и сохранение модели ---
    print("\nШаг 6: Оценка и сохранение модели...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nТочность на тестовых данных: {test_acc:.4f}')

    # Сохраняем модель
    model.save('prophet_model.keras')
    print("Модель ИИ #1 'Пророк' успешно обучена и сохранена в 'prophet_model.keras'")
