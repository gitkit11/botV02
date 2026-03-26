# -*- coding: utf-8 -*-
"""
ИИ #2 "Оракул" - Модуль анализа новостей и настроений
Собирает новости по командам и анализирует их тональность.

Источники (по приоритету):
  1. RSS: BBC, Sky Sports, Guardian, ESPN — быстро, без лимитов
  2. DuckDuckGo — бесплатный, без API ключа, живой поиск
  3. GNews — как запасной вариант

Парсинг полного текста статей через BeautifulSoup — агенты
видят реальный контент, а не только заголовки.
"""

import json
import re
import time
import logging
import requests
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

try:
    import feedparser
    _FEEDPARSER_OK = True
except ImportError:
    _FEEDPARSER_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

try:
    from duckduckgo_search import DDGS
    _DDG_OK = True
except ImportError:
    _DDG_OK = False

try:
    from gnews import GNews
    _GNEWS_OK = True
except ImportError:
    _GNEWS_OK = False

from transformers import pipeline

# Загружаем модель для анализа тональности (один раз при старте)
print("[oracle_ai] Loading sentiment model...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    print("[oracle_ai] Sentiment model loaded.")
except Exception as e:
    print(f"[oracle_ai] Model load error: {e}")
    sentiment_analyzer = None

# RSS источники
_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.skysports.com/rss/12040",
    "https://www.theguardian.com/football/rss",
    "https://www.espn.com/espn/rss/soccer/news",
]

_rss_cache: dict = {}
_RSS_CACHE_TTL = 3 * 3600  # 3 часа

# Кеш полного текста статей (по URL, TTL 6 часов)
_content_cache: dict = {}
_CONTENT_CACHE_TTL = 6 * 3600

_REQUESTS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ── Парсинг полного текста статьи ────────────────────────────────────────────

def _fetch_article_text(url: str, max_chars: int = 2000) -> str:
    """Загружает и парсит полный текст статьи по URL. Возвращает '' при ошибке."""
    if not _BS4_OK or not url:
        return ""
    cached = _content_cache.get(url)
    if cached and (time.time() - cached["ts"]) < _CONTENT_CACHE_TTL:
        return cached["text"]
    try:
        r = requests.get(url, headers=_REQUESTS_HEADERS, timeout=5)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Убираем мусор
        for tag in soup(["script", "style", "nav", "footer", "aside", "ads"]):
            tag.decompose()
        # Ищем основной контент
        content = (
            soup.find("article") or
            soup.find(attrs={"role": "main"}) or
            soup.find("main") or
            soup.find(class_=re.compile(r"article|content|story|body", re.I))
        )
        text = (content or soup.body or soup).get_text(separator=" ", strip=True)
        # Нормализуем пробелы и обрезаем
        text = re.sub(r'\s+', ' ', text).strip()[:max_chars]
        _content_cache[url] = {"ts": time.time(), "text": text}
        return text
    except Exception:
        return ""


# ── Источники новостей ────────────────────────────────────────────────────────

def _get_rss_articles(team_name: str) -> list:
    """RSS: возвращает статьи с заголовком и URL."""
    if not _FEEDPARSER_OK:
        return []
    results = []
    team_lower = team_name.lower()
    short_names = [team_lower]
    if " " in team_lower:
        short_names.append(team_lower.split()[0])
        short_names.append(team_lower.split()[-1])

    for feed_url in _RSS_FEEDS:
        try:
            cached = _rss_cache.get(feed_url)
            if not cached or (time.time() - cached["ts"]) > _RSS_CACHE_TTL:
                feed = feedparser.parse(feed_url)
                _rss_cache[feed_url] = {"ts": time.time(), "entries": feed.entries[:40]}
            for entry in _rss_cache[feed_url]["entries"]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()
                if any(sn in text for sn in short_names):
                    results.append({
                        "title": title,
                        "url": entry.get("link", ""),
                        "publisher": {"title": feed_url.split("/")[2]},
                    })
        except Exception:
            continue
    return results[:6]


def _get_ddg_articles(team_name: str, max_results: int = 5) -> list:
    """DuckDuckGo: бесплатный поиск без API ключа."""
    if not _DDG_OK:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{team_name} football news",
                max_results=max_results,
                timelimit="w",  # только за последнюю неделю
            ))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "publisher": {"title": r.get("source", "DDG")},
            }
            for r in results if r.get("title")
        ]
    except Exception as e:
        logger.debug(f"[DDG] Ошибка для {team_name}: {e}")
        return []


def _get_gnews_articles(team_name: str, max_results: int = 5) -> list:
    """GNews: резервный источник."""
    if not _GNEWS_OK:
        return []
    try:
        gn = GNews(language='en', country='US', max_results=max_results)
        articles = gn.get_news(f"{team_name} football") or []
        return [
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "publisher": a.get("publisher", {}),
            }
            for a in articles
        ]
    except Exception as e:
        logger.debug(f"[GNews] Ошибка для {team_name}: {e}")
        return []


def get_team_news(team_name: str, max_results: int = 10) -> list:
    """
    Собирает новости из всех источников.
    Возвращает список статей с полным текстом для агентов.
    """
    articles = []

    # 1. RSS (быстро, надёжно)
    articles.extend(_get_rss_articles(team_name))

    # 2. DuckDuckGo (бесплатно, живой поиск)
    if len(articles) < 4 and _DDG_OK:
        articles.extend(_get_ddg_articles(team_name, max_results=5))

    # 3. GNews (резерв)
    if len(articles) < 4 and _GNEWS_OK:
        articles.extend(_get_gnews_articles(team_name, max_results=5))

    # Дедупликация по заголовку
    seen = set()
    unique = []
    for a in articles:
        title = a.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            unique.append(a)

    # Парсим полный текст для топ-3 статей (не все — экономим время)
    for i, article in enumerate(unique[:3]):
        url = article.get("url", "")
        if url and _BS4_OK:
            full_text = _fetch_article_text(url, max_chars=1500)
            if full_text:
                article["full_text"] = full_text

    return unique[:max_results]


def get_team_injuries_rss(team_name: str) -> list:
    """Ищет новости о травмах через все источники."""
    injury_keywords = ["injury", "injured", "doubt", "miss", "suspension",
                       "suspended", "out", "ruled out", "fitness", "травм"]
    results = []
    team_lower = team_name.lower()

    # RSS
    for feed_url in _RSS_FEEDS:
        try:
            cached = _rss_cache.get(feed_url)
            if not cached or (time.time() - cached["ts"]) > _RSS_CACHE_TTL:
                feed = feedparser.parse(feed_url)
                _rss_cache[feed_url] = {"ts": time.time(), "entries": feed.entries[:40]}
            for entry in _rss_cache[feed_url]["entries"]:
                title = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                if team_lower.split()[0] in title and any(kw in title for kw in injury_keywords):
                    results.append(entry.get("title", ""))
        except Exception:
            continue

    # DuckDuckGo для травм
    if len(results) < 2 and _DDG_OK:
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(
                    f"{team_name} injury suspension miss",
                    max_results=4, timelimit="w"
                ))
            for r in ddg_results:
                title = r.get("title", "")
                if any(kw in title.lower() for kw in injury_keywords):
                    results.append(title)
        except Exception:
            pass

    # GNews для травм
    if len(results) < 2 and _GNEWS_OK:
        try:
            gn = GNews(language='en', country='US', max_results=5)
            inj = gn.get_news(f"{team_name} injury suspension miss") or []
            results.extend([a.get("title", "") for a in inj])
        except Exception:
            pass

    return results[:5]


# ── Анализ тональности ────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> float:
    """Анализирует тональность текста. От -1.0 (негатив) до +1.0 (позитив)."""
    if not sentiment_analyzer or not text:
        return 0.0
    try:
        result = sentiment_analyzer(text[:512])[0]
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = -score
        return round(score, 4)
    except Exception:
        return 0.0


# ── Главная функция ───────────────────────────────────────────────────────────

def oracle_analyze(home_team: str, away_team: str) -> dict:
    """
    Анализирует новостной фон для обеих команд.
    Возвращает словарь с sentiment, новостями и флагом травм.
    """
    logger.info(f"[Оракул] Анализирую: {home_team} vs {away_team}")
    results = {}

    for team in [home_team, away_team]:
        news_articles = get_team_news(team, max_results=10)

        if not news_articles:
            results[team] = {
                'sentiment_score': 0.0, 'sentiment': 0.0,
                'news_count': 0, 'verdict': 'нейтральный',
                'articles': [], 'injury_news': [], 'has_injuries': False,
            }
            continue

        sentiments = []
        articles_summary = []

        for article in news_articles:
            title = article.get('title', '')
            if not title:
                continue
            # Анализируем полный текст если есть, иначе заголовок
            text_for_sentiment = article.get('full_text') or title
            score = analyze_sentiment(text_for_sentiment)
            sentiments.append(score)
            articles_summary.append({
                'title': title,
                'sentiment': score,
                'publisher': article.get('publisher', {}).get('title', 'N/A'),
                'has_full_text': bool(article.get('full_text')),
            })

        avg_sentiment = round(sum(sentiments) / len(sentiments), 4) if sentiments else 0.0

        verdict = ('позитивный' if avg_sentiment > 0.2
                   else 'негативный' if avg_sentiment < -0.2
                   else 'нейтральный')

        injury_news = get_team_injuries_rss(team)

        results[team] = {
            'sentiment_score': avg_sentiment,
            'sentiment': avg_sentiment,  # алиас для обратной совместимости
            'news_count': len(articles_summary),
            'verdict': verdict,
            'articles': articles_summary[:5],
            'injury_news': injury_news,
            'has_injuries': len(injury_news) > 0,
        }

        rich = sum(1 for a in articles_summary if a.get('has_full_text'))
        logger.info(f"[Оракул] {team}: {len(articles_summary)} новостей "
                    f"({rich} с полным текстом), настроение: {avg_sentiment:.3f} ({verdict})")

    return results


def format_oracle_report(home_team: str, away_team: str, analysis_results: dict) -> str:
    """Форматирует результаты анализа в читаемый отчёт."""
    home_data = analysis_results.get(home_team, {})
    away_data = analysis_results.get(away_team, {})

    home_score = home_data.get('sentiment_score', 0.0)
    away_score = away_data.get('sentiment_score', 0.0)

    report = (
        f"\n╔══════════════════════════════════════════╗\n"
        f"║         ИИ #2 \"ОРАКУЛ\" - ОТЧЕТ          ║\n"
        f"╚══════════════════════════════════════════╝\n\n"
        f"Матч: {home_team} vs {away_team}\n\n"
        f"📰 НОВОСТНОЙ ФОН:\n\n"
        f"🏠 {home_team}:\n"
        f"   Новостей: {home_data.get('news_count', 0)} | Настроение: {home_score:+.3f} ({home_data.get('verdict','нейтральный').upper()})\n\n"
        f"✈️  {away_team}:\n"
        f"   Новостей: {away_data.get('news_count', 0)} | Настроение: {away_score:+.3f} ({away_data.get('verdict','нейтральный').upper()})\n\n"
        f"📊 ИТОГ ОРАКУЛА:\n"
    )

    diff = home_score - away_score
    if diff > 0.1:
        report += f"   Фон благоприятнее для {home_team} (+{diff:.3f})\n"
    elif diff < -0.1:
        report += f"   Фон благоприятнее для {away_team} ({diff:.3f})\n"
    else:
        report += f"   Новостной фон примерно одинаков.\n"

    for team, data in [(home_team, home_data), (away_team, away_data)]:
        inj = data.get("injury_news", [])
        if inj:
            report += f"\n🚑 Травмы/сомнения {team}:\n"
            for h in inj[:3]:
                report += f"   • {h}\n"

    return report


if __name__ == "__main__":
    analysis = oracle_analyze("Manchester City", "Arsenal")
    print(format_oracle_report("Manchester City", "Arsenal", analysis))
    with open('oracle_result.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
