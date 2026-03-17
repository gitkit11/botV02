# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Обзор проекта

**Chimera AI** — Telegram-бот для прогнозов на спортивные ставки. Архитектура «Три Головы»: три независимых ИИ-агента совместно анализируют матчи и выдают сигналы. Поддерживает Футбол, CS2 и Теннис. Написан на Python, с Node.js-хелпером для статистики CS2 (HLTV).

## Команды

```bash
# Запуск бота
python main.py

# Проверка синтаксиса перед коммитом
python -m py_compile main.py

# Ручная рекалибровка ELO (автоматически запускается в понедельник в 3:00)
python elo_calibrate.py

# Диагностические скрипты
python test_groq_llama.py
python test_cs2_api.py
python test_meta_learner.py

# Установка Python-зависимостей
pip install -r requirements.txt

# Установка Node.js-зависимостей (для статистики HLTV)
npm install
```

## Необходимые переменные окружения

Скопируй `.env.example` → `.env` и заполни все 7 ключей:

| Переменная | Сервис |
|---|---|
| `TELEGRAM_TOKEN` | Telegram Bot API |
| `THE_ODDS_API_KEY` | The Odds API (данные матчей + коэффициенты букмекеров) |
| `OPENAI_API_KEY` | OpenAI GPT-4.1-mini (Статистик, Скаут, Арбитр) |
| `GROQ_API_KEY` | Groq Llama 3.3 70B (4-й независимый агент) |
| `API_FOOTBALL_KEY` | API-Football (статистика команд) |
| `RAPID_API_KEY` | RapidAPI hub |
| `PANDASCORE_API_KEY` | PandaScore (данные матчей CS2) |

## Архитектура

### «Три Головы» + Llama

Четыре независимых агента в `agents.py`. Ни один не является запасным для другого — разногласие является намеренным сигналом риска:

1. **Статистик** (GPT-4.1-mini) — Математический анализ: вероятности Пуассона/ELO, xG, форма команды.
2. **Скаут** (GPT-4.1-mini) — Качественный анализ: новостной сентимент, травмы, мотивация.
3. **Арбитр** (GPT-4.1-mini) — Синтезирует Статистика + Скаута + коэффициенты букмекеров → критерий Келли, финальный вердикт `СТАВИТЬ/НЕ СТАВИТЬ`.
4. **Llama 3.3 70B** (Groq) — Полностью независимое второе мнение. Если Groq возвращает 403 (региональное ограничение), система работает в «ограниченном режиме» — НЕ добавлять запасной вариант на GPT.

### Поток данных (Футбол)

```
The Odds API → список матчей
  → math_model.py (Пуассон + ELO + Dixon-Coles) → вероятности
  → prophet_model.keras (нейросеть, 66k матчей АПЛ) → P1/X/P2
  → understat_stats.py (xG) + oracle_ai.py (новости) + injuries.py
  → agents.py: Статистик + Скаут → Арбитр + Llama (параллельно)
  → signal_engine.py: минимум 4/6 сигналов для прохождения
  → database.py → сообщение в Telegram
```

### Веса ансамбля

- Пуассон: 40%, ELO: 30%, ИИ-агенты: 15%, Букмекеры: 10%, Prophet-нейросеть: 5%

### Движок сигналов (`signal_engine.py`)

- **Футбол**: минимум **4 из 6** сигналов для рекомендации ставки
- **CS2**: минимум **6 из 10** сигналов (строже)
- Порог ценной ставки: EV > 10%, коэффициент ≥ 1.55, уверенность ≥ 52%

## Структура модулей

```
main.py                     # Точка входа Telegram-бота, все обработчики
config.py                   # Загрузчик .env через python-dotenv
agents.py                   # Четыре ИИ-агента (футбол): GPT×3 + Llama
chimera_multi_agent.py      # Bayesian Multi-Agent Engine (используется agents.py и sports/)
math_model.py               # Пуассон + ELO + Dixon-Coles (футбол)
maestro_ai.py               # Ценные ставки + построитель ансамбля
oracle_ai.py                # GNews + сентимент-анализ DistilBERT
injuries.py                 # Данные о травмах/дисквалификациях команд
understat_stats.py          # xG-данные с Understat
api_football.py             # Интеграция с API-Football
signal_engine.py            # Система подсчёта сигналов (футбол + CS2 + тоталы)
chimera_signal.py           # CHIMERA Score engine (Сигнал дня, рейтинг кандидатов)
express_builder.py          # Построитель экспрессов (победители + тоталы)
meta_learner.py             # Анализ ROI после матчей, авто-настройка порогов
line_movement.py            # Отслеживание движения линий букмекеров
database.py                 # SQLite (авто-миграция таблиц по видам спорта)
elo_calibrate.py            # Рекалибровка ELO (запускается вручную или авто пн 3:00)
i18n.py                     # Переводы RU/EN

sports/cs2/                 # Модуль CS2 (изолированный)
  __init__.py               # Публичный API модуля
  core.py                   # Вероятности победы (MIS+ELO+форма+рейтинг), формат отчёта
  pandascore.py             # PandaScore API + алиасы команд + fuzzy matching
  hltv_stats.py             # Статика HLTV: карты, игроки, алиасы команд
  hltv_scraper.py           # Обновление hltv_cache.json
  hltv_sync.py              # Синхронизация HLTV через Node.js
  hltv_odds.py              # Коэффициенты HLTV
  veto_logic.py             # Симуляция вето карт (BO3)
  agents.py                 # CS2: GPT + Llama анализ
  pinnacle_cs2.py           # Pinnacle как fallback для коэффициентов CS2
  results_tracker.py        # Авто-запись результатов + ELO-обучение CS2

sports/tennis/              # Модуль теннис
  __init__.py               # Публичный API + scan_tennis_signals()
  model.py                  # ELO из рейтингов, поверхность, тоталы геймов
  agents.py                 # GPT + Llama анализ, format_tennis_full_report()
  api_tennis.py             # api-tennis.com: форма, H2H, рейтинги
  matches.py                # The Odds API: теннисные матчи
  rankings.py               # ATP/WTA рейтинги, surface detection
  pinnacle.py               # Pinnacle как fallback
  results_tracker.py        # Авто-запись результатов

sports/basketball/          # Модуль баскетбол
  __init__.py               # Публичный API
  core.py                   # ELO, вероятности, тотал очков
  results_tracker.py        # Авто-запись результатов

sports/football/__init__.py # Реэкспорт из корневых футбольных модулей

ml/                         # ML: обучение XGBoost
  train_model.py            # Обучение + retrain_incremental() по реальным матчам
  predictor.py              # Загрузка и использование обученных моделей
  build_features.py         # Построение feature matrix
  dixon_coles.py            # Dixon-Coles поправка
  download_data.py          # Загрузка исторических данных
  data/                     # CSV данные (100+ сезонов, all_matches_raw.csv, live_matches.csv)
  models/                   # Обученные XGBoost модели (.pkl)

scripts/                    # Утилиты и диагностика (не запускаются ботом)
  run_meta_learner.py       # Ручной запуск MetaLearner
  update_hltv_stats.py      # Обновление статики HLTV
  update_hltv_daily.mjs     # Node.js скрипт синхронизации HLTV
  elo_basketball_calibrate.py
  train_epl.py / train_prophet.py
  check_keys.py / find_cs2_sport.py / list_all_sports.py
  populate_test_db.py / test_*.py
```

## Автоматические задачи (внутри `main.py`)

- **Каждый час**: Проверка результатов матчей, обновление ROI в БД
- **Понедельник 3:00**: Рекалибровка ELO
- **Ежедневно**: Синхронизация статистики HLTV (CS2)

## Файлы состояния (генерируются автоматически, в .gitignore)

- `chimera_predictions.db` — SQLite база данных прогнозов
- `elo_ratings.json` — Текущие ELO-рейтинги для 96 команд из 5 лиг
- `team_form.json` — Результаты последних 5 матчей на команду
- `prophet_model.keras` — Предобученная нейросеть (должна присутствовать для запуска)
- `team_encoder.json` — Энкодер признаков для модели Prophet

## Ключевые правила

- **Изоляция модулей**: Код CS2 живёт только в `sports/cs2/`; не создавать зависимости между видами спорта. Публичный API экспортируется через `sports/cs2/__init__.py`.
- **Нет запасного варианта для Llama**: Если Groq недоступен — возвращать словарь с ошибкой, никогда не заменять Llama другим вызовом GPT.
- **Миграция БД**: `database.py` при запуске автоматически мигрирует старую таблицу `predictions` в `football_predictions`/`cs2_predictions` — не ломать эту логику.
- **Изменения в main.py**: Это монолитная точка входа; изменения здесь затрагивают все виды спорта и все обработчики Telegram. Будь осторожен.
