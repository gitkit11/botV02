"""
config.py — Конфигурация Chimera AI
Все ключи загружаются из .env файла (НЕ хардкодить в коде!)
Создай файл .env на основе .env.example и заполни своими ключами.
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — используем переменные окружения напрямую

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', '')
RAPID_API_KEY = os.getenv('RAPID_API_KEY', '')
PANDASCORE_API_KEY = os.getenv('PANDASCORE_API_KEY', '')

# Проверка при запуске
if not TELEGRAM_TOKEN:
    print("[WARN] TELEGRAM_TOKEN не задан в .env!")
if not THE_ODDS_API_KEY:
    print("[WARN] THE_ODDS_API_KEY не задан в .env!")
