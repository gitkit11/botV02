# Chimera AI — Руководство для разработчиков

## Архитектура проекта

Проект разделён на независимые модули по видам спорта. Каждый модуль живёт в своей папке и не влияет на другие.

```
chimera_ai/
├── main.py                  # Главный файл бота (точка входа)
├── config.py                # Конфигурация (читает .env)
├── agents.py                # AI агенты для футбола (GPT, Llama)
├── math_model.py            # Пуассон, ELO, Dixon-Coles
├── database.py              # SQLite — прогнозы и ROI
├── injuries.py              # Травмы (GNews + GPT)
├── oracle_ai.py             # Новостной анализ
├── understat_stats.py       # xG статистика
├── elo_calibrate.py         # Пересчёт ELO по сезону
│
├── sports/
│   ├── football/            # ✅ Футбол (реализован)
│   │   └── __init__.py      # Публичный интерфейс
│   │
│   ├── cs2/                 # ✅ CS2 (реализован)
│   │   ├── __init__.py      # Публичный интерфейс
│   │   ├── pandascore.py    # PandaScore API — матчи
│   │   ├── core.py          # Расчёт вероятностей (Veto + MIS)
│   │   ├── agents.py        # AI агенты для CS2
│   │   ├── veto_logic.py    # Симуляция мап-вето BO3
│   │   ├── math_model.py    # Математическая модель CS2
│   │   └── parser.py        # Парсер данных
│   │
│   └── tennis/              # ⏳ Теннис (в разработке)
│       └── __init__.py      # Публичный интерфейс (заглушка)
```

---

## Как добавить новую функцию

### Для CS2:
1. Работай только в `sports/cs2/`
2. Не трогай `main.py` — только если нужно добавить новую кнопку
3. Экспортируй новые функции через `sports/cs2/__init__.py`

### Для Теннис:
1. Работай только в `sports/tennis/`
2. Создай `matches.py`, `analysis.py`, `report.py`
3. Зарегистрируй функции в `sports/tennis/__init__.py`
4. Добавь обработчик в `main.py` (раздел "🎾 Теннис")

### Для Футбол:
1. Основные файлы в корне: `agents.py`, `math_model.py`, `understat_stats.py`
2. Не меняй структуру `format_main_report()` без согласования

---

## API ключи (.env)

```
TELEGRAM_TOKEN=           # Telegram Bot Token
THE_ODDS_API_KEY=         # The Odds API (футбол)
OPENAI_API_KEY=           # OpenAI GPT-4o
GROQ_API_KEY=             # Groq (Llama 3.3 70B)
RAPID_API_KEY=            # RapidAPI
PANDASCORE_API_KEY=       # PandaScore (CS2, Теннис)
```

---

## Правила работы с Git

- **Никогда** не коммить `.env` — он в `.gitignore`
- Перед пушем делай `python -m py_compile main.py` для проверки синтаксиса
- Если меняешь `main.py` — предупреди других разработчиков
- Каждый модуль (`cs2/`, `tennis/`) можно менять независимо

---

## Запуск

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить бота
python main.py

# Обновить ELO рейтинги вручную
python elo_calibrate.py
```
