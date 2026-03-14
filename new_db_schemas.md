## Предлагаемые SQL-схемы для новых таблиц базы данных

Для реализации самообучения и более детального анализа по видам спорта, я предлагаю создать две отдельные таблицы: `football_predictions` и `cs2_predictions`. Это позволит хранить специфические данные для каждого вида спорта и упростит процесс обучения ИИ.

### 1. Таблица `football_predictions`

Эта таблица будет содержать все прогнозы и результаты для футбольных матчей. Она будет во многом повторять текущую структуру `predictions`, но будет явно отделена для футбола.

```sql
CREATE TABLE IF NOT EXISTS football_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE, -- Уникальный идентификатор матча
    match_date TEXT,      -- Дата и время матча
    home_team TEXT,       -- Домашняя команда
    away_team TEXT,       -- Гостевая команда
    league TEXT,          -- Лига (например, soccer_epl)

    -- Прогнозы агентов
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

    -- Математические модели
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

    -- Ансамбль
    ensemble_home REAL,
    ensemble_draw REAL,
    ensemble_away REAL,
    ensemble_best_outcome TEXT,

    -- Value ставки (первая/лучшая)
    value_bet_outcome TEXT,
    value_bet_odds REAL,
    value_bet_ev REAL,
    value_bet_kelly REAL,
    value_bet_correct INTEGER,

    -- Букмекерские коэффициенты
    bookmaker_odds_home REAL,
    bookmaker_odds_draw REAL,
    bookmaker_odds_away REAL,
    bookmaker_odds_over25 REAL,
    bookmaker_odds_under25 REAL,

    -- Результат матча
    real_home_score INTEGER,
    real_away_score INTEGER,
    real_outcome TEXT,
    is_correct INTEGER,
    is_goals_correct INTEGER,
    is_btts_correct INTEGER,
    is_ensemble_correct INTEGER,
    value_bet_correct INTEGER,
    result_checked_at TIMESTAMP,

    -- ROI расчёт
    roi_outcome REAL,
    roi_value_bet REAL,

    -- Мета-данные для обучения ИИ
    model_weights_at_prediction TEXT, -- JSON-строка с весами моделей на момент прогноза
    prediction_data TEXT,             -- Полные данные прогноза в JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Таблица `cs2_predictions`

Эта таблица будет содержать прогнозы и результаты для матчей CS2, включая специфические поля для анализа киберспорта.

```sql
CREATE TABLE IF NOT EXISTS cs2_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE, -- Уникальный идентификатор матча
    match_date TEXT,      -- Дата и время матча
    home_team TEXT,       -- Домашняя команда
    away_team TEXT,       -- Гостевая команда
    league TEXT,          -- Лига/Турнир (например, ESL Pro League)

    -- Прогнозы агентов
    gpt_verdict TEXT,
    llama_verdict TEXT,
    mixtral_verdict TEXT,
    gpt_confidence INTEGER,
    llama_confidence INTEGER,
    mixtral_confidence INTEGER,
    bet_signal TEXT,
    recommended_outcome TEXT,

    -- Математические модели (для CS2 могут быть свои)
    elo_home INTEGER,
    elo_away INTEGER,
    elo_home_win REAL,
    elo_away_win REAL,

    -- Ансамбль
    ensemble_home REAL,
    ensemble_away REAL,
    ensemble_best_outcome TEXT,

    -- Value ставки
    value_bet_outcome TEXT,
    value_bet_odds REAL,
    value_bet_ev REAL,
    value_bet_kelly REAL,
    value_bet_correct INTEGER,

    -- Букмекерские коэффициенты
    bookmaker_odds_home REAL,
    bookmaker_odds_away REAL,

    -- Специфические поля для CS2
    predicted_maps TEXT,              -- JSON-строка с предсказанными картами вето
    map_advantage_score REAL,         -- Балл за преимущество на картах
    key_player_advantage REAL,        -- Балл за преимущество ключевых игроков
    ai_signal_reason TEXT,            -- Краткое обоснование от AI-агента для сигнала
    home_map_winrates TEXT,           -- JSON-строка с винрейтами домашней команды по картам
    away_map_winrates TEXT,           -- JSON-строка с винрейтами гостевой команды по картам
    home_player_ratings TEXT,         -- JSON-строка с рейтингами игроков домашней команды
    away_player_ratings TEXT,         -- JSON-строка с рейтингами игроков гостевой команды

    -- Результат матча
    real_home_score INTEGER,
    real_away_score INTEGER,
    real_outcome TEXT,
    is_correct INTEGER,
    is_ensemble_correct INTEGER,
    result_checked_at TIMESTAMP,

    -- ROI расчёт
    roi_outcome REAL,
    roi_value_bet REAL,

    -- Мета-данные для обучения ИИ
    model_weights_at_prediction TEXT, -- JSON-строка с весами моделей на момент прогноза
    prediction_data TEXT,             -- Полные данные прогноза в JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. Миграция данных

После создания новых таблиц, мы перенесем существующие данные из таблицы `predictions` в соответствующие новые таблицы. Это будет сделано скриптом, который я подготовлю после твоего одобрения этих схем.

### 4. Удаление старой таблицы

После успешной миграции, старая таблица `predictions` будет удалена.

Пожалуйста, просмотри предложенные схемы. Все ли поля соответствуют твоим ожиданиям? Есть ли что-то, что нужно добавить или изменить? Я готов к твоим комментариям.
