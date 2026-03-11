# -*- coding: utf-8 -*-
"""
ИИ #3 "Маэстро" - Ансамблевая система и поиск выгодных ставок (Value Bets)
Объединяет прогнозы от ИИ #1 "Пророк" и ИИ #2 "Оракул",
сравнивает с коэффициентами букмекеров и находит ставки с положительным EV.
"""

import numpy as np
import json
import os
import requests
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# НАСТРОЙКИ
# ============================================================
THE_ODDS_API_KEY = "0eadec3556f8784ba3f52adaab13c093"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"

# Веса для каждого ИИ (можно настраивать)
PROPHET_WEIGHT = 0.65   # ИИ #1 "Пророк" - исторические данные
ORACLE_WEIGHT = 0.35    # ИИ #2 "Оракул" - новостной фон


# ============================================================
# ПОЛУЧЕНИЕ КОЭФФИЦИЕНТОВ БУКМЕКЕРОВ
# ============================================================
def get_bookmaker_odds(home_team, away_team):
    """
    Получает актуальные коэффициенты букмекеров через The Odds API.
    Возвращает словарь с вероятностями для каждого исхода.
    """
    try:
        params = {
            'apiKey': THE_ODDS_API_KEY,
            'regions': 'eu',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }
        response = requests.get(THE_ODDS_API_URL, params=params, timeout=10)
        data = response.json()

        for game in data:
            if (home_team.lower() in game.get('home_team', '').lower() or
                    away_team.lower() in game.get('away_team', '').lower()):
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = {o['name']: o['price'] for o in market['outcomes']}
                            return outcomes

        return None
    except Exception as e:
        print(f"  [Маэстро] Ошибка получения коэффициентов: {e}")
        return None


def odds_to_probability(odds):
    """Конвертирует десятичный коэффициент в вероятность."""
    if odds and odds > 1:
        return round(1 / odds, 4)
    return 0.0


# ============================================================
# АНСАМБЛЕВОЕ ОБЪЕДИНЕНИЕ ПРОГНОЗОВ
# ============================================================
def ensemble_predictions(prophet_probs, oracle_sentiment_home, oracle_sentiment_away):
    """
    Объединяет вероятности от Пророка с данными Оракула.
    
    prophet_probs: [prob_draw, prob_home_win, prob_away_win]
    oracle_sentiment_home: float от -1 до +1
    oracle_sentiment_away: float от -1 до +1
    
    Возвращает скорректированные вероятности [draw, home_win, away_win]
    """
    # Базовые вероятности от Пророка
    prob_draw = prophet_probs[0]
    prob_home = prophet_probs[1]
    prob_away = prophet_probs[2]

    # Корректировка на основе новостного фона от Оракула
    # Позитивный фон для хозяев немного увеличивает их шансы
    sentiment_diff = oracle_sentiment_home - oracle_sentiment_away
    oracle_home_boost = sentiment_diff * 0.05  # Максимальный буст ±5%

    # Применяем корректировку
    adjusted_home = prob_home + oracle_home_boost
    adjusted_away = prob_away - oracle_home_boost
    adjusted_draw = prob_draw  # Ничья не меняется

    # Нормализуем, чтобы сумма была 1.0
    total = adjusted_home + adjusted_away + adjusted_draw
    if total > 0:
        adjusted_home /= total
        adjusted_away /= total
        adjusted_draw /= total

    # Финальный ансамбль с весами
    final_home = PROPHET_WEIGHT * adjusted_home + ORACLE_WEIGHT * (0.5 + sentiment_diff * 0.3)
    final_away = PROPHET_WEIGHT * adjusted_away + ORACLE_WEIGHT * (0.5 - sentiment_diff * 0.3)
    final_draw = PROPHET_WEIGHT * adjusted_draw + ORACLE_WEIGHT * 0.2

    # Финальная нормализация
    total_final = final_home + final_away + final_draw
    if total_final > 0:
        final_home /= total_final
        final_away /= total_final
        final_draw /= total_final

    return {
        'home_win': round(final_home, 4),
        'draw': round(final_draw, 4),
        'away_win': round(final_away, 4)
    }


# ============================================================
# ПОИСК VALUE BETS
# ============================================================
def find_value_bets(ai_probs, bookmaker_odds, home_team, away_team):
    """
    Ищет ставки с положительным ожидаемым значением (Value Bets).
    
    Value Bet: когда наша оценка вероятности выше, чем подразумевает коэффициент.
    EV = (наша_вероятность * коэффициент) - 1
    Если EV > 0, ставка выгодна.
    """
    value_bets = []

    if not bookmaker_odds:
        print("  [Маэстро] Коэффициенты букмекеров недоступны, поиск value bets пропущен.")
        return value_bets

    outcomes_map = {
        'home_win': home_team,
        'draw': 'Draw',
        'away_win': away_team
    }

    for outcome_key, team_name in outcomes_map.items():
        our_prob = ai_probs.get(outcome_key, 0)
        bookie_odds = bookmaker_odds.get(team_name, None)

        if bookie_odds and our_prob > 0:
            ev = (our_prob * bookie_odds) - 1
            implied_prob = odds_to_probability(bookie_odds)

            if ev > 0.05:  # Порог: EV > 5%
                value_bets.append({
                    'outcome': outcome_key,
                    'team': team_name,
                    'our_probability': our_prob,
                    'implied_probability': implied_prob,
                    'bookmaker_odds': bookie_odds,
                    'expected_value': round(ev, 4),
                    'edge': round(our_prob - implied_prob, 4)
                })

    # Сортируем по EV (самые выгодные первыми)
    value_bets.sort(key=lambda x: x['expected_value'], reverse=True)
    return value_bets


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ МАЭСТРО
# ============================================================
def maestro_analyze(home_team, away_team, prophet_probs, oracle_results):
    """
    Главная функция ИИ #3 "Маэстро".
    Принимает результаты от Пророка и Оракула, объединяет их
    и ищет выгодные ставки.
    """
    print(f"\n[Маэстро] Начинаю финальный анализ: {home_team} vs {away_team}")

    # Извлекаем данные Оракула
    home_sentiment = oracle_results.get(home_team, {}).get('sentiment_score', 0.0)
    away_sentiment = oracle_results.get(away_team, {}).get('sentiment_score', 0.0)

    print(f"  [Маэстро] Данные Пророка: Ничья={prophet_probs[0]:.4f}, "
          f"Хозяева={prophet_probs[1]:.4f}, Гости={prophet_probs[2]:.4f}")
    print(f"  [Маэстро] Данные Оракула: {home_team}={home_sentiment:+.4f}, "
          f"{away_team}={away_sentiment:+.4f}")

    # Ансамблевое объединение
    final_probs = ensemble_predictions(prophet_probs, home_sentiment, away_sentiment)
    print(f"  [Маэстро] Финальные вероятности: {final_probs}")

    # Получаем коэффициенты букмекеров
    print(f"  [Маэстро] Запрашиваю коэффициенты букмекеров...")
    bookmaker_odds = get_bookmaker_odds(home_team, away_team)

    # Ищем value bets
    value_bets = find_value_bets(final_probs, bookmaker_odds, home_team, away_team)

    # Определяем основной прогноз
    best_outcome = max(final_probs, key=final_probs.get)
    outcome_names = {
        'home_win': f'Победа {home_team}',
        'draw': 'Ничья',
        'away_win': f'Победа {away_team}'
    }
    main_prediction = outcome_names[best_outcome]
    confidence = final_probs[best_outcome]

    result = {
        'home_team': home_team,
        'away_team': away_team,
        'final_probabilities': final_probs,
        'main_prediction': main_prediction,
        'confidence': confidence,
        'bookmaker_odds': bookmaker_odds,
        'value_bets': value_bets,
        'prophet_weight': PROPHET_WEIGHT,
        'oracle_weight': ORACLE_WEIGHT
    }

    return result


def format_maestro_report(analysis):
    """
    Форматирует финальный отчет Маэстро.
    """
    home = analysis['home_team']
    away = analysis['away_team']
    probs = analysis['final_probabilities']
    value_bets = analysis['value_bets']

    confidence_pct = analysis['confidence'] * 100

    report = f"""
╔══════════════════════════════════════════╗
║        ИИ #3 "МАЭСТРО" - ФИНАЛЬНЫЙ      ║
║              АНАЛИЗ МАТЧА                ║
╚══════════════════════════════════════════╝

⚽ {home} vs {away}

📊 ФИНАЛЬНЫЕ ВЕРОЯТНОСТИ (ансамбль):
   🏠 Победа {home}: {probs['home_win']*100:.1f}%
   🤝 Ничья: {probs['draw']*100:.1f}%
   ✈️  Победа {away}: {probs['away_win']*100:.1f}%

🎯 ОСНОВНОЙ ПРОГНОЗ:
   {analysis['main_prediction']} (уверенность: {confidence_pct:.1f}%)

⚖️  ВЕСА СИСТЕМ:
   ИИ #1 "Пророк" (история): {analysis['prophet_weight']*100:.0f}%
   ИИ #2 "Оракул" (новости): {analysis['oracle_weight']*100:.0f}%
"""

    if value_bets:
        report += "\n💰 VALUE BETS (ставки с положительным EV):\n"
        for vb in value_bets:
            report += f"""
   ✅ {vb['team']} ({vb['outcome']})
      Наша вероятность: {vb['our_probability']*100:.1f}%
      Вероятность букмекера: {vb['implied_probability']*100:.1f}%
      Коэффициент: {vb['bookmaker_odds']:.2f}
      Ожидаемая ценность (EV): +{vb['expected_value']*100:.1f}%
      Наше преимущество: +{vb['edge']*100:.1f}%
"""
    else:
        report += "\n💰 VALUE BETS: Выгодных ставок не обнаружено.\n"

    report += "\n" + "═" * 44 + "\n"
    return report


# ============================================================
# ТЕСТОВЫЙ ЗАПУСК
# ============================================================
if __name__ == "__main__":
    # Симулируем данные от Пророка (в реальности они придут из prophet_model.keras)
    # [prob_draw, prob_home_win, prob_away_win]
    test_prophet_probs = [0.177, 0.720, 0.103]

    # Симулируем данные от Оракула (в реальности они придут из oracle_ai.py)
    test_oracle_results = {
        "Manchester City": {"sentiment_score": 0.5811},
        "Arsenal": {"sentiment_score": -0.0292}
    }

    analysis = maestro_analyze(
        "Manchester City",
        "Arsenal",
        test_prophet_probs,
        test_oracle_results
    )

    report = format_maestro_report(analysis)
    print(report)

    # Сохраняем результат
    with open('maestro_result.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
    print("[Маэстро] Результаты сохранены в maestro_result.json")
