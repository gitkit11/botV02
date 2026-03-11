# -*- coding: utf-8 -*-
"""
ИИ #2 "Оракул" - Модуль анализа новостей и настроений
Собирает новости по командам и анализирует их тональность
с помощью предобученной модели трансформера.
"""

import json
from gnews import GNews
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Загружаем модель для анализа тональности (один раз при старте)
print("Загрузка модели анализа тональности (может занять несколько секунд)...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    print("Модель тональности загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    sentiment_analyzer = None


def get_team_news(team_name, max_results=10):
    """
    Получает новости для конкретной команды через Google News.
    """
    try:
        google_news = GNews(language='en', country='US', max_results=max_results)
        news = google_news.get_news(f"{team_name} football")
        return news if news else []
    except Exception as e:
        print(f"  [Оракул] Ошибка получения новостей для '{team_name}': {e}")
        return []


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

        results[team] = {
            'sentiment_score': avg_sentiment,
            'news_count': len(articles_summary),
            'verdict': verdict,
            'articles': articles_summary[:5]  # Сохраняем только топ-5 для отчета
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
