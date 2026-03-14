import sqlite3
from typing import Dict, List, Optional
import json
import os
import re

# Загрузка конфигурации из signal_engine.py для анализа
# В реальной ситуации лучше импортировать или иметь доступ к актуальным CFG
# Для простоты примера, будем считать, что CFG доступны или передаются.

class MetaLearner:
    def __init__(self, db_path: str = 'chimera_predictions.db', signal_engine_path: str = 'signal_engine.py'):
        self.db_path = db_path
        self.signal_engine_path = signal_engine_path
        self.current_cfgs = self._load_current_cfgs()

    def _load_current_cfgs(self) -> Dict:
        """Загружает текущие FOOTBALL_CFG и CS2_CFG из signal_engine.py."""
        cfgs = {"FOOTBALL_CFG": {}, "CS2_CFG": {}}
        try:
            with open(self.signal_engine_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Ищем FOOTBALL_CFG
            football_cfg_match = re.search(r'FOOTBALL_CFG = (\{.*?\})', content, re.DOTALL)
            if football_cfg_match:
                # Безопасная оценка словаря
                cfgs["FOOTBALL_CFG"] = eval(football_cfg_match.group(1))

            # Ищем CS2_CFG
            cs2_cfg_match = re.search(r'CS2_CFG = (\{.*?\})', content, re.DOTALL)
            if cs2_cfg_match:
                cfgs["CS2_CFG"] = eval(cs2_cfg_match.group(1))

        except Exception as e:
            print(f"Ошибка при загрузке CFG из {self.signal_engine_path}: {e}")
        return cfgs

    def analyze_performance(self, sport: str) -> Dict:
        """Анализирует производительность прогнозов для заданного вида спорта."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        table_name = f"{sport}_predictions"

        # Извлекаем все завершенные прогнозы
        cursor.execute(f"SELECT * FROM {table_name} WHERE result IS NOT NULL")
        predictions = cursor.fetchall()
        
        # Получаем названия колонок
        col_names = [description[0] for description in cursor.description]
        
        conn.close()

        if not predictions:
            return {"roi": 0, "accuracy": 0, "total_predictions": 0, "details": {}}

        total_invested = 0
        total_profit = 0
        correct_predictions = 0

        # Детальный анализ по параметрам (пример: по EV)
        ev_performance = {}

        for pred_tuple in predictions:
            pred = dict(zip(col_names, pred_tuple))
            
            if pred['kelly'] > 0: # Только если была ставка
                invested = pred['kelly'] # Келли - это процент от банка, для простоты считаем как единицу
                total_invested += invested

                is_correct = (pred['outcome'] == pred['result'])
                if is_correct:
                    profit = invested * (pred['odds'] - 1) # Выигрыш
                    correct_predictions += 1
                else:
                    profit = -invested # Проигрыш
                total_profit += profit

                # Анализ по EV-диапазонам
                ev_bucket = int(pred['ev'] // 5) * 5 # Группируем по 5% EV
                if ev_bucket not in ev_performance:
                    ev_performance[ev_bucket] = {'invested': 0, 'profit': 0, 'count': 0, 'correct': 0}
                ev_performance[ev_bucket]['invested'] += invested
                ev_performance[ev_bucket]['profit'] += profit
                ev_performance[ev_bucket]['count'] += 1
                if is_correct: ev_performance[ev_bucket]['correct'] += 1

        roi = (total_profit / total_invested) * 100 if total_invested > 0 else 0
        accuracy = (correct_predictions / len(predictions)) * 100

        # Преобразование ev_performance в более читаемый формат
        for bucket, data in ev_performance.items():
            data['roi'] = (data['profit'] / data['invested']) * 100 if data['invested'] > 0 else 0
            data['accuracy'] = (data['correct'] / data['count']) * 100

        return {
            "roi": round(roi, 2),
            "accuracy": round(accuracy, 2),
            "total_predictions": len(predictions),
            "ev_performance": ev_performance,
            # Здесь можно добавить анализ по другим параметрам: elo_gap, form, map_advantage и т.д.
        }

    def suggest_config_updates(self, sport: str, performance_data: Dict) -> Dict:
        """Предлагает обновления конфигурации на основе анализа производительности."""
        suggested_updates = {}
        current_cfg = self.current_cfgs.get(f"{sport.upper()}_CFG", {})

        if not current_cfg: return {}

        # Пример логики: корректировка min_ev на основе ROI по EV-диапазонам
        ev_performance = performance_data.get('ev_performance', {})
        best_ev_bucket = None
        max_roi = -float('inf')

        for bucket, data in ev_performance.items():
            if data['count'] > 5 and data['roi'] > max_roi: # Учитываем только достаточное количество данных
                max_roi = data['roi']
                best_ev_bucket = bucket
        
        if best_ev_bucket is not None and best_ev_bucket / 100 > current_cfg.get('min_ev', 0):
            # Если лучший ROI достигается при более высоком EV, предлагаем его
            suggested_updates['min_ev'] = round(best_ev_bucket / 100, 2)

        # Здесь можно добавить логику для других параметров (min_elo_gap, min_map_advantage и т.д.)
        # Например, если сигналы с высоким ELO_GAP имеют плохой ROI, можно увеличить min_elo_gap

        return suggested_updates

    def apply_config_updates(self, sport: str, updates: Dict):
        """Применяет предложенные обновления к signal_engine.py."""
        if not updates: return

        cfg_name = f"{sport.upper()}_CFG"
        
        # Создаем резервную копию
        backup_path = self.signal_engine_path + ".bak"
        os.replace(self.signal_engine_path, backup_path)
        
        try:
            with open(backup_path, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
            
            updated_content = content
            for key, value in updates.items():
                # Ищем строку с параметром и обновляем его значение
                # Пример: "min_ev":       0.07,
                # Регулярное выражение для поиска и замены значения
                pattern = r'("{}":\s*)([\d\.]+)(.*)'.format(re.escape(key))
                replacement = r'\g<1>{}{}'.format(value, r'\g<3>')
                updated_content = re.sub(pattern, replacement, updated_content, count=1)

            with open(self.signal_engine_path, 'w', encoding='utf-8') as f_out:
                f_out.write(updated_content)
            print(f"Конфигурация {cfg_name} обновлена в {self.signal_engine_path}")

        except Exception as e:
            print(f"Ошибка при применении обновлений к {self.signal_engine_path}: {e}")
            # Восстанавливаем из бэкапа в случае ошибки
            os.replace(backup_path, self.signal_engine_path)
        finally:
            if os.path.exists(backup_path):
                os.remove(backup_path) # Удаляем бэкап после успешного применения
