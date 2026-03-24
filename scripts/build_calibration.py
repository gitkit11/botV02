# -*- coding: utf-8 -*-
"""
scripts/build_calibration.py — Одноразовая калибровка модели по историческим данным
=====================================================================================
Скачивает 3 месяца коэффициентов Pinnacle + реальные результаты.
Строит таблицу: "когда Pinnacle даёт 60% — реально побеждают 57%?"
Сохраняет в calibration_table.json — используется при каждом прогнозе.

Запуск: python scripts/build_calibration.py
Стоимость: ~8000–12000 кредитов (одноразово из 100k).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict

try:
    from config import THE_ODDS_API_KEY
except ImportError:
    THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASE_URL = "https://api.the-odds-api.com/v4"

LEAGUES = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
]

OUTPUT_FILE = "calibration_table.json"

# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _log_quota(r):
    print(f"  Квота: использовано={r.headers.get('x-requests-used','?')}, "
          f"осталось={r.headers.get('x-requests-remaining','?')}, "
          f"этот запрос={r.headers.get('x-requests-last','?')} кредитов")


def get_historical_odds_snapshot(sport_key: str, snap_date: str) -> list:
    """Получает снапшот коэффициентов Pinnacle для лиги в заданную дату."""
    try:
        r = requests.get(
            f"{BASE_URL}/historical/sports/{sport_key}/odds/",
            params={
                "apiKey":     THE_ODDS_API_KEY,
                "date":       snap_date,
                "regions":    "eu,uk",
                "markets":    "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        _log_quota(r)
        if r.ok:
            data = r.json()
            return data.get("data", [])
    except Exception as e:
        print(f"  Ошибка: {e}")
    return []


def get_scores(sport_key: str, days_from: int = 3) -> list:
    """Получает результаты матчей."""
    try:
        r = requests.get(
            f"{BASE_URL}/sports/{sport_key}/scores/",
            params={"apiKey": THE_ODDS_API_KEY, "daysFrom": days_from, "dateFormat": "iso"},
            timeout=10,
        )
        _log_quota(r)
        if r.ok:
            return [m for m in r.json() if m.get("completed")]
    except Exception as e:
        print(f"  Ошибка scores: {e}")
    return []


def extract_pinnacle_probs(event: dict) -> dict:
    """Извлекает no-vig вероятности из Pinnacle для одного матча."""
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    for bm in event.get("bookmakers", []):
        if "pinnacle" not in bm.get("key", "").lower():
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            oc = {o["name"]: float(o.get("price", 0) or 0) for o in market.get("outcomes", [])}
            ph = oc.get(home, 0)
            pd = oc.get("Draw", 0)
            pa = oc.get(away, 0)
            if ph >= 1.02 and pa >= 1.02:
                ih = 1 / ph
                id_ = 1 / pd if pd >= 1.02 else 0
                ia = 1 / pa
                total = ih + id_ + ia
                return {
                    "home_prob": round(ih / total, 4),
                    "draw_prob": round(id_ / total, 4) if id_ else 0,
                    "away_prob": round(ia / total, 4),
                    "home_team": home,
                    "away_team": away,
                    "event_id":  event.get("id", ""),
                    "commence":  event.get("commence_time", ""),
                }
    return {}


# ─── Основная логика ──────────────────────────────────────────────────────────

def collect_data():
    """
    Собирает данные за последние 90 дней.
    Для каждой недели: снапшот коэффициентов за 2 дня до матча + реальный результат.
    """
    all_data = []  # [{home_prob, draw_prob, away_prob, outcome}]

    now = datetime.now(timezone.utc)

    for league in LEAGUES:
        print(f"\n=== {league} ===")

        # Получаем результаты за последние 3 дня (максимум API)
        # Для старых матчей нужен другой подход — через исторические снапшоты
        scores_map = {}  # match_id → outcome

        # Стратегия: берём снапшоты с шагом 1 неделя за 12 недель
        # Каждый снапшот стоит 10 кредитов и даёт ~10 матчей
        # 12 недель × 1 снапшот = 120 кредитов на лигу × 5 лиг = 600 кредитов

        for week_ago in range(1, 13):  # 1-12 недель назад
            # Снапшот за 2 дня до предполагаемых матчей этой недели
            snap_dt = now - timedelta(weeks=week_ago, days=2)
            snap_ts = snap_dt.strftime("%Y-%m-%dT12:00:00Z")

            print(f"  Неделя -{week_ago}: снапшот {snap_ts[:10]}")
            events = get_historical_odds_snapshot(league, snap_ts)

            if not events:
                print(f"  → Нет данных")
                time.sleep(0.5)
                continue

            print(f"  → {len(events)} матчей")

            for event in events:
                probs = extract_pinnacle_probs(event)
                if not probs:
                    continue

                # Ищем результат этого матча в исторических scores
                # Используем event_id если есть, иначе по командам
                commence = probs.get("commence", "")
                if not commence:
                    continue

                # Пробуем найти результат через scores API
                # (только для последних 3 дней — для старых нет прямого способа)
                # Сохраняем пока только вероятности, результаты найдём через исторические scores
                event_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                days_since = (now - event_dt).days

                # Для матчей которые уже прошли — записываем данные
                if days_since > 0:
                    all_data.append({
                        "event_id":   probs["event_id"],
                        "home_team":  probs["home_team"],
                        "away_team":  probs["away_team"],
                        "home_prob":  probs["home_prob"],
                        "draw_prob":  probs["draw_prob"],
                        "away_prob":  probs["away_prob"],
                        "league":     league,
                        "commence":   commence,
                        "outcome":    None,  # заполним позже
                    })

            time.sleep(0.3)  # не слишком быстро

    return all_data


def enrich_with_scores(data: list) -> list:
    """
    Добавляет реальные результаты к данным.
    Использует /scores/ для последних 3 дней и исторические снапшоты для старых.
    """
    print("\n=== Загрузка результатов ===")

    # Сначала загружаем доступные результаты через /scores/
    scores_by_id   = {}  # event_id → outcome
    scores_by_name = {}  # (home, away) → outcome

    for league in LEAGUES:
        scores = get_scores(league, days_from=3)
        for s in scores:
            home = s.get("home_team", "")
            away = s.get("away_team", "")
            raw_scores = s.get("scores") or []
            score_map = {sc["name"]: int(sc["score"]) for sc in raw_scores
                         if sc.get("name") and sc.get("score")}
            h_score = score_map.get(home, -1)
            a_score = score_map.get(away, -1)
            if h_score < 0 or a_score < 0:
                continue
            if h_score > a_score:
                outcome = "home_win"
            elif a_score > h_score:
                outcome = "away_win"
            else:
                outcome = "draw"
            scores_by_id[s.get("id", "")] = outcome
            scores_by_name[(home.lower(), away.lower())] = outcome
        time.sleep(0.2)

    # Матчуем
    matched = 0
    for item in data:
        eid = item.get("event_id", "")
        if eid in scores_by_id:
            item["outcome"] = scores_by_id[eid]
            matched += 1
            continue
        key = (item["home_team"].lower(), item["away_team"].lower())
        if key in scores_by_name:
            item["outcome"] = scores_by_name[key]
            matched += 1

    print(f"Сматчировано результатов: {matched}/{len(data)}")
    return [d for d in data if d["outcome"]]  # только с известными результатами


def build_calibration_table(data: list) -> dict:
    """
    Строит калибровочную таблицу из собранных данных.
    Группирует вероятности по бинам (50-55%, 55-60% и т.д.)
    и считает реальную частоту побед в каждом бине.
    """
    print(f"\n=== Построение таблицы по {len(data)} матчам ===")

    bins = defaultdict(lambda: {"predicted": [], "actual_win": 0, "total": 0})

    for item in data:
        outcome = item["outcome"]

        for side in ["home", "away", "draw"]:
            prob = item.get(f"{side}_prob", 0)
            if not prob:
                continue

            # Бин: 5% шаги от 10% до 90%
            bin_key = round(prob * 20) / 20  # округляем до 0.05

            won = (
                (side == "home" and outcome == "home_win") or
                (side == "away" and outcome == "away_win") or
                (side == "draw" and outcome == "draw")
            )

            bins[bin_key]["predicted"].append(prob)
            bins[bin_key]["total"] += 1
            if won:
                bins[bin_key]["actual_win"] += 1

    # Строим таблицу
    table = {}
    for bin_key, stats in sorted(bins.items()):
        if stats["total"] < 5:  # нужно минимум 5 матчей для надёжности
            continue
        actual_rate = stats["actual_win"] / stats["total"]
        correction = actual_rate / bin_key if bin_key > 0 else 1.0
        table[str(round(bin_key, 2))] = {
            "predicted":   round(bin_key, 2),
            "actual_rate": round(actual_rate, 4),
            "correction":  round(correction, 4),
            "sample_size": stats["total"],
        }
        print(f"  Pinnacle {bin_key*100:.0f}% → реально {actual_rate*100:.1f}% "
              f"(поправка ×{correction:.3f}, n={stats['total']})")

    return table


def main():
    print("=" * 60)
    print("CHIMERA Calibration Builder")
    print("Собираем исторические данные Pinnacle...")
    print("=" * 60)

    # Шаг 1: Собираем коэффициенты
    data = collect_data()
    print(f"\nСобрано {len(data)} матчей с коэффициентами")

    if not data:
        print("Нет данных. Проверь API ключ.")
        return

    # Шаг 2: Добавляем результаты
    data = enrich_with_scores(data)
    print(f"После матчинга с результатами: {len(data)} матчей")

    if len(data) < 20:
        print("Мало данных для калибровки (нужно минимум 20 матчей с результатами).")
        print("Это нормально для первого запуска — результаты приходят постепенно.")
        print("Запусти скрипт снова через несколько дней.")
    else:
        # Шаг 3: Строим таблицу
        table = build_calibration_table(data)

        # Сохраняем
        output = {
            "built_at":   datetime.now(timezone.utc).isoformat(),
            "sample_size": len(data),
            "leagues":    LEAGUES,
            "table":      table,
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_FILE)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nТаблица сохранена → {out_path}")
        print(f"Бинов: {len(table)}, матчей: {len(data)}")

    # Сохраняем сырые данные для анализа
    raw_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сырые данные → {raw_path}")


if __name__ == "__main__":
    main()
