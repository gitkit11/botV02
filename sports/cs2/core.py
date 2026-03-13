# -*- coding: utf-8 -*-
from .veto_logic import simulate_bo3_veto, get_map_impact_score
import json

def calculate_cs2_win_prob(home_team, away_team):
    """
    Уникальный расчет вероятности на основе Veto-логики и MIS.
    """
    # 1. Симулируем мап-вето
    maps, veto_log = simulate_bo3_veto(home_team, away_team)
    
    # 2. Считаем MIS для каждой карты
    map_scores = []
    for m in maps:
        h_mis = get_map_impact_score(home_team, m)
        a_mis = get_map_impact_score(away_team, m)
        
        # Сила на карте (0-100)
        total = h_mis + a_mis
        if total == 0:
            h_prob, a_prob = 0.5, 0.5
        else:
            h_prob = h_mis / total
            a_prob = a_mis / total
        
        map_scores.append((m, h_prob, a_prob))
        
    # 3. Финальная вероятность победы в матче (взвешенная по картам)
    # Пики команд имеют больший вес, десайдер — средний
    weights = [0.35, 0.35, 0.30]
    final_h_prob = sum(map_scores[i][1] * weights[i] for i in range(3))
    final_a_prob = 1 - final_h_prob
    
    return {
        "home_prob": round(final_h_prob, 2),
        "away_prob": round(final_a_prob, 2),
        "maps": map_scores,
        "veto_log": veto_log
    }

def get_golden_signal(analysis_data, bookmaker_odds):
    """
    Алгоритм "Золотого сигнала" с высокой проходимостью.
    Условия:
    1. Наша вероятность > 60%
    2. Коэффициент букмекера > 1.60
    3. EV (Expected Value) > 15%
    """
    h_prob = analysis_data["home_prob"]
    a_prob = analysis_data["away_prob"]
    
    h_odds = bookmaker_odds.get("home_win", 0)
    a_odds = bookmaker_odds.get("away_win", 0)
    
    signals = []
    
    # Проверка для хозяев
    if h_odds > 1.0:
        h_ev = (h_prob * h_odds) - 1
        if h_prob >= 0.60 and h_odds >= 1.60 and h_ev >= 0.15:
            signals.append({
                "type": "GOLDEN",
                "team": analysis_data["home_team"],
                "outcome": "Победа (П1)",
                "odds": h_odds,
                "ev": round(h_ev * 100, 1),
                "confidence": int(h_prob * 100)
            })
            
    # Проверка для гостей
    if a_odds > 1.0:
        a_ev = (a_prob * a_odds) - 1
        if a_prob >= 0.60 and a_odds >= 1.60 and a_ev >= 0.15:
            signals.append({
                "type": "GOLDEN",
                "team": analysis_data["away_team"],
                "outcome": "Победа (П2)",
                "odds": a_odds,
                "ev": round(a_ev * 100, 1),
                "confidence": int(a_prob * 100)
            })
            
    return signals

def format_cs2_full_report(home_team, away_team, analysis, gpt_analysis, llama_analysis, golden_signals):
    """Форматирует профессиональный отчет для CS2 v4.4."""
    report = f"🎮 *CHIMERA AI CS2 v4.4 — УНИКАЛЬНЫЙ АНАЛИЗ*\n"
    report += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    report += f"⚔️ *{home_team} vs {away_team}*\n\n"
    
    report += f"🗺 *СИМУЛЯЦИЯ VETO (BO3):*\n"
    for log in analysis["veto_log"]:
        report += f" {log}\n"
    report += "\n"
    
    report += f"📊 *ВЕРОЯТНОСТЬ ПО КАРТАМ (MIS):*\n"
    for m, hp, ap in analysis["maps"]:
        report += f" • {m}: {int(hp*100)}% — {int(ap*100)}%\n"
    report += "\n"
    
    report += f"📈 *ИТОГОВЫЙ РАСЧЕТ (Chimera Core):*\n"
    report += f" 🔹 {home_team}: {int(analysis['home_prob']*100)}%\n"
    report += f" 🔸 {away_team}: {int(analysis['away_prob']*100)}%\n\n"
    
    report += f"🧠 *GPT-4o (Стратег):*\n_{gpt_analysis}_\n\n"
    report += f"🤖 *Llama 3.3 (Тактик):*\n_{llama_analysis}_\n\n"
    
    if golden_signals:
        report += f"🌟 *ЗОЛОТОЙ СИГНАЛ (High Confidence):*\n"
        for sig in golden_signals:
            report += f"🔥 {sig['outcome']} {sig['team']} @ {sig['odds']}\n"
            report += f"   (Уверенность: {sig['confidence']}% | EV: +{sig['ev']}%)\n"
    else:
        report += f"⏸ *СИГНАЛ: ПРОПУСТИТЬ (Низкая уверенность)*\n"
        
    return report
