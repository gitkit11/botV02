import sqlite3
from typing import Dict, List, Optional
import json
import os
import re
import logging

logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(self, db_path: str = 'chimera_predictions.db', signal_engine_path: str = 'signal_engine.py'):
        self.db_path = db_path
        self.signal_engine_path = signal_engine_path
        self.current_cfgs = self._load_current_cfgs()

    def _load_current_cfgs(self) -> Dict:
        cfgs = {"FOOTBALL_CFG": {}, "CS2_CFG": {}}
        try:
            if not os.path.exists(self.signal_engine_path):
                return cfgs
            with open(self.signal_engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for name in cfgs.keys():
                match = re.search(f'{name} = \\{{(.*?)\\}}', content, re.DOTALL)
                if match:
                    cfg_str = "{" + re.sub(r'#.*', '', match.group(1)) + "}"
                    cfgs[name] = eval(cfg_str)
        except Exception as e:
            logger.error(f"Ошибка при загрузке CFG: {e}")
        return cfgs

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
                outcome = pred.get('value_bet_outcome') or pred.get('outcome') or pred.get('recommended_outcome')
                if not outcome: continue

                is_win = (outcome == pred[res_col])
                odds = float(pred.get('value_bet_odds') or pred.get('odds') or pred.get('bookmaker_odds_home') or 1.0)
                kelly = float(pred.get('value_bet_kelly') or pred.get('kelly') or 1.0)
                
                total_bets += 1
                if is_win:
                    wins += 1
                    total_profit += kelly * (odds - 1)
                else:
                    total_profit -= kelly

                ev = float(pred.get('value_bet_ev') or pred.get('ev', 0))
                ev_bucket = int(ev * 100 // 5) * 5
                if ev_bucket not in factor_analysis["ev_groups"]:
                    factor_analysis["ev_groups"][ev_bucket] = {"count": 0, "wins": 0, "profit": 0}
                factor_analysis["ev_groups"][ev_bucket]["count"] += 1
                if is_win: factor_analysis["ev_groups"][ev_bucket]["wins"] += 1
                factor_analysis["ev_groups"][ev_bucket]["profit"] += (kelly * (odds - 1)) if is_win else -kelly

            roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
            accuracy = (wins / total_bets * 100) if total_bets > 0 else 0

            return {"sport": sport, "total": total_bets, "accuracy": round(accuracy, 2), "roi": round(roi, 2), "factors": factor_analysis}
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
        if not updates: return
        try:
            with open(self.signal_engine_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            cfg_start = False
            cfg_name = f"{sport.upper()}_CFG"
            new_lines = []
            for line in lines:
                if cfg_name in line: cfg_start = True
                if cfg_start:
                    for key, val in updates.items():
                        if f'"{key}"' in line:
                            line = re.sub(r'("'+key+'":\s*)([\d\.]+)', r'\g<1>{}'.format(val), line)
                            print(f"MetaLearner: {cfg_name}['{key}'] -> {val}")
                if cfg_start and "}" in line: cfg_start = False
                new_lines.append(line)
            with open(self.signal_engine_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            logger.error(f"Ошибка при обновлении signal_engine.py: {e}")

if __name__ == "__main__":
    ml = MetaLearner()
    for s in ['football', 'cs2']:
        p = ml.analyze_performance(s)
        if "error" not in p:
            print(f"Анализ {s}: ROI={p.get('roi')}%, Accuracy={p.get('accuracy')}% ({p.get('total')} матчей)")
            u = ml.suggest_updates(s, p)
            if u:
                print(f"Обновление {s}: {u}")
                ml.apply_updates(s, u)
        else:
            print(f"Анализ {s}: {p['error']}")
