# -*- coding: utf-8 -*-
"""
ИИ #2 "Оракул" - Модуль анализа новостей и настроений
Собирает новости по командам и анализирует их тональность
с помощью предобученной модели трансформера.
"""

import json
import re
from gnews import GNews
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

try:
    import feedparser
    _FEEDPARSER_OK = True
except ImportError:
    _FEEDPARSER_OK = False

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


# RSS источники: надёжные футбольные новости
_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",          # BBC Sport Football
    "https://www.skysports.com/rss/12040",                       # Sky Sports Football
    "https://www.theguardian.com/football/rss",                  # The Guardian Football
    "https://www.espn.com/espn/rss/soccer/news",                 # ESPN Soccer
]

_rss_cache: dict = {}   # {feed_url: [articles]}


def _get_rss_articles(team_name: str) -> list:
    """Ищет упоминания команды в RSS лентах. Возвращает список заголовков."""
    if not _FEEDPARSER_OK:
        return []
    results = []
    team_lower = team_name.lower()
    # Сокращённые имена для поиска в RSS
    short_names = [team_lower]
    if " " in team_lower:
        short_names.append(team_lower.split()[0])  # первое слово
        short_names.append(team_lower.split()[-1]) # последнее слово

    for feed_url in _RSS_FEEDS:
        try:
            if feed_url not in _rss_cache:
                feed = feedparser.parse(feed_url)
                _rss_cache[feed_url] = feed.entries[:40]
            for entry in _rss_cache[feed_url]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()
                if any(sn in text for sn in short_names):
                    results.append({
                        "title": title,
                        "publisher": {"title": feed_url.split("/")[2]},
                    })
        except Exception:
            continue
    return results[:8]


def get_team_news(team_name, max_results=10):
    """
    Получает новости для команды из нескольких источников:
    1. RSS: BBC Sport, Sky Sports, Guardian, ESPN (быстро, надёжно)
    2. Google News как резервный источник
    """
    articles = []

    # Источник 1: RSS (быстро, без лимитов)
    rss_articles = _get_rss_articles(team_name)
    articles.extend(rss_articles)

    # Источник 2: Google News (если RSS дал мало)
    if len(articles) < 5:
        try:
            google_news = GNews(language='en', country='US', max_results=max_results)
            gnews = google_news.get_news(f"{team_name} football")
            articles.extend(gnews or [])
        except Exception as e:
            print(f"  [Оракул] Google News ошибка для '{team_name}': {e}")

    return articles[:max_results]


def get_team_injuries_rss(team_name: str) -> list:
    """
    Ищет новости о травмах через RSS + Google News.
    Возвращает список заголовков с ключевыми словами травм.
    """
    injury_keywords = ["injury", "injured", "doubt", "miss", "suspension",
                       "suspended", "out", "ruled out", "fitness", "травм"]
    results = []
    team_lower = team_name.lower()

    # RSS scan
    for feed_url in _RSS_FEEDS:
        try:
            if feed_url not in _rss_cache:
                feed = feedparser.parse(feed_url)
                _rss_cache[feed_url] = feed.entries[:40]
            for entry in _rss_cache[feed_url]:
                title = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                if team_lower.split()[0] in title and any(kw in title for kw in injury_keywords):
                    results.append(entry.get("title", ""))
        except Exception:
            continue

    # Google News для травм
    if len(results) < 3 and _FEEDPARSER_OK is not None:
        try:
            gn = GNews(language='en', country='US', max_results=5)
            inj_news = gn.get_news(f"{team_name} injury suspension miss")
            results.extend([a.get("title", "") for a in (inj_news or [])])
        except Exception:
            pass

    return results[:5]


def analyze_sentiment(text):
    """
    Анализирует тональность текста.
    Возвращает числовое значение от -1.0 (негатив) до +1.0 (позитив).
    """
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


def oracle_analyze(home_team, away_team):
    """
    Главная функция "Оракула".
    Анализирует новостной фон для обеих команд и возвращает
    итоговый "индекс настроения" для каждой.
    """
    print(f"\n[Оракул] Анализирую новостной фон: {home_team} vs {away_team}")

    results = {}

    for team in [home_team, away_team]:
        print(f"  [Оракул] Ищу новости для: {team}...")
        news_articles = get_team_news(team, max_results=10)

        if not news_articles:
            print(f"  [Оракул] Новостей для '{team}' не найдено. Нейтральный фон.")
            results[team] = {
                'sentiment_score': 0.0,
                'news_count': 0,
                'verdict': 'нейтральный',
                'articles': []
            }
            continue

        sentiments = []
        articles_summary = []

        for article in news_articles:
            title = article.get('title', '')
            if not title:
                continue
            score = analyze_sentiment(title)
            sentiments.append(score)
            articles_summary.append({
                'title': title,
                'sentiment': score,
                'publisher': article.get('publisher', {}).get('title', 'N/A')
            })

        if sentiments:
            avg_sentiment = round(sum(sentiments) / len(sentiments), 4)
        else:
            avg_sentiment = 0.0

        if avg_sentiment > 0.2:
            verdict = 'позитивный'
        elif avg_sentiment < -0.2:
            verdict = 'негативный'
        else:
            verdict = 'нейтральный'

        # Травмы из RSS
        injury_news = get_team_injuries_rss(team)
        injury_flag = len(injury_news) > 0

        results[team] = {
            'sentiment_score': avg_sentiment,
            'news_count': len(articles_summary),
            'verdict': verdict,
            'articles': articles_summary[:5],
            'injury_news': injury_news,
            'has_injuries': injury_flag,
        }

        print(f"  [Оракул] {team}: {len(articles_summary)} новостей, "
              f"средний индекс настроения: {avg_sentiment:.4f} ({verdict})")

    return results


def format_oracle_report(home_team, away_team, analysis_results):
    """
    Форматирует результаты анализа в читаемый отчет.
    """
    home_data = analysis_results.get(home_team, {})
    away_data = analysis_results.get(away_team, {})

    home_score = home_data.get('sentiment_score', 0.0)
    away_score = away_data.get('sentiment_score', 0.0)

    report = f"""
╔══════════════════════════════════════════╗
║         ИИ #2 "ОРАКУЛ" - ОТЧЕТ          ║
╚══════════════════════════════════════════╝

Матч: {home_team} vs {away_team}

📰 НОВОСТНОЙ ФОН:

🏠 {home_team}:
   Новостей проанализировано: {home_data.get('news_count', 0)}
   Индекс настроения: {home_score:+.4f}
   Вердикт: {home_data.get('verdict', 'нейтральный').upper()}

✈️  {away_team}:
   Новостей проанализировано: {away_data.get('news_count', 0)}
   Индекс настроения: {away_score:+.4f}
   Вердикт: {away_data.get('verdict', 'нейтральный').upper()}

📊 ИТОГ ОРАКУЛА:
"""

    diff = home_score - away_score
    if diff > 0.1:
        report += f"   Новостной фон благоприятнее для {home_team} (+{diff:.4f})\n"
    elif diff < -0.1:
        report += f"   Новостной фон благоприятнее для {away_team} ({diff:.4f})\n"
    else:
        report += f"   Новостной фон примерно одинаков для обеих команд.\n"

    # Травмы
    for team, data in [(home_team, home_data), (away_team, away_data)]:
        inj = data.get("injury_news", [])
        if inj:
            report += f"\n🚑 Травмы/сомнения {team}:\n"
            for headline in inj[:3]:
                report += f"   • {headline}\n"

    return report


if __name__ == "__main__":
    # Тестовый запуск
    home = "Manchester City"
    away = "Arsenal"

    analysis = oracle_analyze(home, away)
    report = format_oracle_report(home, away, analysis)
    print(report)

    # Сохраняем результат в JSON для интеграции с другими модулями
    with open('oracle_result.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print("\n[Оракул] Результаты сохранены в oracle_result.json")
