import sqlite3
from typing import Dict, List, Optional
import json
import os
import re
import logging

logger = logging.getLogger(__name__)

_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_thresholds.json")


def _load_json_cfg() -> dict:
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_cfg(data: dict) -> None:
    """Атомарная запись: пишем во временный файл, потом rename."""
    tmp = _JSON_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _JSON_PATH)  # атомарная замена на всех ОС


class MetaLearner:
    def __init__(self, db_path: str = 'chimera_predictions.db', signal_engine_path: str = 'signal_engine.py'):
        self.db_path = db_path
        self.signal_engine_path = signal_engine_path
        self.current_cfgs = self._load_current_cfgs()

    def _load_current_cfgs(self) -> Dict:
        """Загружает текущие пороги из config_thresholds.json."""
        try:
            data = _load_json_cfg()
            return {
                "FOOTBALL_CFG":    dict(data.get("FOOTBALL_CFG", {})),
                "CS2_CFG":         dict(data.get("CS2_CFG", {})),
                "BASKETBALL_CFG":  dict(data.get("BASKETBALL_CFG", {})),
                "HOCKEY_CFG":      dict(data.get("HOCKEY_CFG", {})),
            }
        except Exception as e:
            logger.error(f"Ошибка при загрузке CFG из config_thresholds: {e}")
            return {"FOOTBALL_CFG": {}, "CS2_CFG": {}, "BASKETBALL_CFG": {}, "HOCKEY_CFG": {}}

    def analyze_performance(self, sport: str) -> Dict:
        if not os.path.exists(self.db_path):
            return {"error": "Database not found"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        table_name = f"{sport}_predictions"
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            cols = [c[1] for c in cursor.fetchall()]
            if not cols: return {"error": f"Table {table_name} not found"}

            # Определяем колонку результата (result или real_outcome)
            res_col = "real_outcome" if "real_outcome" in cols else "result"
            
            cursor.execute(f"SELECT * FROM {table_name} WHERE {res_col} IS NOT NULL AND {res_col} != ''")
            rows = cursor.fetchall()
            
            if not rows: return {"total": 0, "msg": "No completed matches found"}

            total_bets = 0
            wins = 0
            total_profit = 0
            factor_analysis = {"ev_groups": {}}

            for row in rows:
                pred = dict(zip(cols, row))
                # Используем recommended_outcome как прогноз
                outcome = pred.get('recommended_outcome')
                if not outcome: continue
                # Пропускаем если нет результата
                real = pred.get(res_col)
                if not real: continue

                is_win = (outcome == real)

                # Котировка — берём ту что соответствует прогнозу
                if outcome == 'home_win':
                    odds = float(pred.get('bookmaker_odds_home') or 0)
                elif outcome == 'away_win':
                    odds = float(pred.get('bookmaker_odds_away') or 0)
                else:
                    odds = float(pred.get('bookmaker_odds_draw') or 0)
                # Если котировка невалидна — используем среднюю 1.85
                if odds < 1.02:
                    odds = 1.85

                total_bets += 1
                if is_win:
                    wins += 1
                    total_profit += (odds - 1)   # +прибыль на 1 единицу ставки
                else:
                    total_profit -= 1.0           # -1 единица ставки

                # EV = (est_prob * odds - 1) * 100
                # est_prob: берём из ансамбля, fallback на implied (1/odds)
                if outcome == 'home_win':
                    est_prob = float(pred.get('ensemble_home') or 0)
                elif outcome == 'away_win':
                    est_prob = float(pred.get('ensemble_away') or 0)
                else:
                    est_prob = float(pred.get('ensemble_draw') or 0)
                if est_prob <= 0 or est_prob >= 1:
                    est_prob = 1.0 / odds  # fallback на implied probability
                ev = round((est_prob * odds - 1) * 100, 1) if odds > 1 else 0.0
                ev_bucket = int(ev // 5) * 5
                if ev_bucket not in factor_analysis["ev_groups"]:
                    factor_analysis["ev_groups"][ev_bucket] = {"count": 0, "wins": 0, "profit": 0}
                factor_analysis["ev_groups"][ev_bucket]["count"] += 1
                if is_win: factor_analysis["ev_groups"][ev_bucket]["wins"] += 1
                factor_analysis["ev_groups"][ev_bucket]["profit"] += (odds - 1) if is_win else -1.0

            # ROI = прибыль / (total_bets * 1 ед. ставки) * 100%
            roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
            accuracy = (wins / total_bets * 100) if total_bets > 0 else 0
            checked = total_bets  # только матчи с is_correct IS NOT NULL

            return {
                "sport":    sport,
                "total":    total_bets,
                "checked":  checked,
                "accuracy": round(accuracy, 2),
                "roi":      round(roi, 2),
                "factors":  factor_analysis,
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    def suggest_updates(self, sport: str, perf: Dict) -> Dict:
        if "error" in perf or perf.get("total", 0) < 10: return {}
        updates = {}
        ev_groups = perf.get("factors", {}).get("ev_groups", {})
        if perf["roi"] < 0:
            sorted_ev = sorted(ev_groups.keys())
            for ev_threshold in sorted_ev:
                sub_profit = sum(ev_groups[e]["profit"] for e in sorted_ev if e >= ev_threshold)
                sub_count = sum(ev_groups[e]["count"] for e in sorted_ev if e >= ev_threshold)
                if sub_count > 5 and (sub_profit / sub_count) > 0:
                    new_min_ev = ev_threshold / 100.0
                    current_min_ev = self.current_cfgs.get(f"{sport.upper()}_CFG", {}).get("min_ev", 0)
                    if new_min_ev > current_min_ev:
                        updates["min_ev"] = new_min_ev
                        break
        return updates

    def apply_updates(self, sport: str, updates: Dict):
        """Обновляет пороги в config_thresholds.json атомарно."""
        if not updates:
            return
        cfg_name = f"{sport.upper()}_CFG"
        try:
            data = _load_json_cfg()
            if cfg_name not in data:
                logger.warning(f"[MetaLearner] {cfg_name} не найден в JSON")
                return
            for key, val in updates.items():
                old = data[cfg_name].get(key)
                data[cfg_name][key] = val
                logger.info(f"[MetaLearner] {cfg_name}['{key}']: {old} → {val}")
            _save_json_cfg(data)
            # Перезагружаем модуль чтобы изменения применились сразу
            import importlib, sys
            if 'config_thresholds' in sys.modules:
                importlib.reload(sys.modules['config_thresholds'])
        except Exception as e:
            logger.error(f"[MetaLearner] Ошибка при обновлении config_thresholds.json: {e}")

    def analyze_hockey_weights(self) -> Dict:
        """
        Анализирует какой компонент (ELO или Odds) был точнее для хоккея.
        Возвращает рекомендованные веса для HOCKEY_CFG.
        """
        if not os.path.exists(self.db_path):
            return {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ensemble_home, ensemble_away, elo_home, elo_away,
                       bookmaker_odds_home, bookmaker_odds_away,
                       recommended_outcome, real_outcome
                FROM hockey_predictions
                WHERE real_outcome IS NOT NULL AND real_outcome != 'expired'
            """)
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 10:
                return {}

            elo_correct = 0
            odds_correct = 0
            total = 0

            for row in rows:
                ens_h, ens_a, elo_h, elo_a, odds_h, odds_a, pred, real = row
                if not real or not pred:
                    continue
                total += 1

                if elo_h and elo_a:
                    elo_pred = "home_win" if elo_h > elo_a else "away_win"
                    if elo_pred == real:
                        elo_correct += 1

                if odds_h and odds_a and odds_h > 1.02 and odds_a > 1.02:
                    odds_pred = "home_win" if (1 / odds_h) > (1 / odds_a) else "away_win"
                    if odds_pred == real:
                        odds_correct += 1

            if total == 0:
                return {}

            elo_acc  = elo_correct  / total
            odds_acc = odds_correct / total
            total_acc = elo_acc + odds_acc

            if total_acc > 0:
                new_elo  = round(0.70 * elo_acc  / total_acc, 2)
                new_odds = round(0.70 * odds_acc / total_acc, 2)
                new_elo  = max(0.15, min(0.45, new_elo))
                new_odds = round(0.70 - new_elo, 2)
                logger.info(
                    f"[Hockey Meta] ELO точность: {elo_acc*100:.1f}%, "
                    f"Odds точность: {odds_acc*100:.1f}% | "
                    f"Новые веса: ELO={new_elo}, Odds={new_odds}"
                )
                return {"weight_elo": new_elo, "weight_odds": new_odds}
        except Exception as e:
            logger.error(f"[Hockey Meta] Ошибка: {e}")
        return {}

    def analyze_basketball_weights(self) -> Dict:
        """
        Анализирует какой компонент (ELO или Odds) был точнее для баскетбола.
        Возвращает рекомендованные веса для BASKETBALL_CFG.
        """
        if not os.path.exists(self.db_path):
            return {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ensemble_home, ensemble_away, elo_home_win, elo_away_win,
                       bookmaker_odds_home, bookmaker_odds_away,
                       recommended_outcome, real_outcome
                FROM basketball_predictions
                WHERE real_outcome IS NOT NULL AND real_outcome != 'expired'
            """)
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 10:
                return {}

            elo_correct = 0
            odds_correct = 0
            total = 0

            for row in rows:
                ens_h, ens_a, elo_h, elo_a, odds_h, odds_a, pred, real = row
                if not real or not pred:
                    continue
                total += 1

                # ELO прогноз
                if elo_h and elo_a:
                    elo_pred = "home_win" if elo_h > elo_a else "away_win"
                    if elo_pred == real:
                        elo_correct += 1

                # Odds прогноз (обратная вероятность)
                if odds_h and odds_a and odds_h > 1.02 and odds_a > 1.02:
                    odds_pred = "home_win" if (1/odds_h) > (1/odds_a) else "away_win"
                    if odds_pred == real:
                        odds_correct += 1

            if total == 0:
                return {}

            elo_acc  = elo_correct  / total
            odds_acc = odds_correct / total
            total_acc = elo_acc + odds_acc

            # Перераспределяем веса пропорционально точности (сохраняя сумму 0.70)
            if total_acc > 0:
                new_elo  = round(0.70 * elo_acc  / total_acc, 2)
                new_odds = round(0.70 * odds_acc / total_acc, 2)
                # Нормализуем чтобы сумма была ровно 0.70
                new_elo  = max(0.20, min(0.50, new_elo))
                new_odds = round(0.70 - new_elo, 2)
                logger.info(
                    f"[Basketball Meta] ELO точность: {elo_acc*100:.1f}%, "
                    f"Odds точность: {odds_acc*100:.1f}% | "
                    f"Новые веса: ELO={new_elo}, Odds={new_odds}"
                )
                return {"weight_elo": new_elo, "weight_odds": new_odds}
        except Exception as e:
            logger.error(f"[Basketball Meta] Ошибка: {e}")
        return {}


if __name__ == "__main__":
    ml = MetaLearner()
    for s in ['football', 'cs2', 'basketball', 'hockey']:
        p = ml.analyze_performance(s)
        if "error" not in p:
            print(f"Анализ {s}: ROI={p.get('roi')}%, Accuracy={p.get('accuracy')}% ({p.get('total')} матчей)")
            u = ml.suggest_updates(s, p)
            if u:
                print(f"Обновление {s}: {u}")
                ml.apply_updates(s, u)
        else:
            print(f"Анализ {s}: {p['error']}")

    # Анализ весов баскетбольной модели
    bball_weights = ml.analyze_basketball_weights()
    if bball_weights:
        print(f"[Basketball] Рекомендованные веса: {bball_weights}")
        ml.apply_updates('basketball', bball_weights)
