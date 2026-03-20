# state.py — глобальное состояние бота
# Все глобальные переменные, НЕ зависящие от asyncio

# --- Кэш матчей ---
matches_cache: list = []
_last_matches_refresh: float = 0.0
_current_league: str = "soccer_epl"
_league_matches_cache: dict = {}  # league_key → {"matches": [...], "ts": float}
_LEAGUE_CACHE_TTL: int = 1200    # 20 минут

cs2_matches_cache: list = []     # Кэш матчей CS2
tennis_matches_cache: list = []  # Кэш матчей тенниса
analysis_cache: dict = {}        # Хранит результаты анализа по match_id

# Кэш готовых HTML-отчётов: {key: {"text": str, "kb": markup, "parse_mode": str, "ts": float}}
# Ключи: "football_{idx}", "cs2_{idx}", "tennis_{sport_key}_{idx}", "bball_{league}_{idx}"
_report_cache: dict = {}
_REPORT_CACHE_TTL: int = 2700  # 45 минут

# ── Кеш результатов скана "Сигналы дня" ──────────────────────────────────────
# top_candidates после AI-верификации. TTL 45 мин = свежие сигналы + экономия.
# Структура: {"ts": float, "candidates": list, "result_text": str,
#             "top_pred_id": int|None, "top_sport": str, "top_odds": float}
_signals_scan_cache: dict = {}
SIGNALS_SCAN_TTL: int = 45 * 60  # 45 минут

# Химера-чат: пользователи ожидающие ввода вопроса + дневной лимит
_chimera_waiting: set = set()       # user_id ожидают ввода вопроса
_chimera_daily: dict = {}           # {user_id: (date, count)}
_chimera_history: dict = {}         # {user_id: [{"role": ..., "content": ...}, ...]}

# Пользователи ожидающие ввода банка
_awaiting_bankroll: set = set()

# Константы
CHIMERA_DAILY_LIMIT: int = 7
ADMIN_IDS: set = {6852160892, 608064556}
