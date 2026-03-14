from meta_learner import MetaLearner
import os

# Получаем абсолютный путь к директории, где находится скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))

# Формируем абсолютные пути к файлам
db_path = os.path.join(script_dir, 'chimera_predictions.db')
signal_engine_path = os.path.join(script_dir, 'signal_engine.py')

learner = MetaLearner(db_path=db_path, signal_engine_path=signal_engine_path)
cs2_performance = learner.analyze_performance("cs2")
print("CS2 Performance Analysis:")
print(cs2_performance)

suggested_cs2_updates = learner.suggest_config_updates("cs2", cs2_performance)
print("Suggested CS2 Updates:")
print(suggested_cs2_updates)
