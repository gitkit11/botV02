# -*- coding: utf-8 -*-
"""
config_thresholds.py — Загрузчик порогов из config_thresholds.json
===================================================================
Все числовые пороги хранятся в config_thresholds.json.
Этот файл только загружает их и экспортирует как Python-переменные —
все импорты в системе остаются неизменными:
    from config_thresholds import FOOTBALL_CFG, CS2_CFG, ...

Для изменения порогов вручную — редактируй config_thresholds.json.
MetaLearner пишет туда же атомарно (temp-файл + rename).
"""
import json
import os

_DIR  = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_DIR, "config_thresholds.json")


def _load() -> dict:
    with open(_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_cfg = _load()

FOOTBALL_CFG     = _cfg["FOOTBALL_CFG"]
CS2_CFG          = _cfg["CS2_CFG"]
BASKETBALL_CFG   = _cfg["BASKETBALL_CFG"]
HOCKEY_CFG       = _cfg["HOCKEY_CFG"]
TENNIS_CFG       = _cfg["TENNIS_CFG"]
CHIMERA_WEIGHTS  = _cfg["CHIMERA_WEIGHTS"]
MIN_CHIMERA_SCORE = _cfg["MIN_CHIMERA_SCORE"]
VALUE_BET_CFG    = _cfg["VALUE_BET_CFG"]
EXPRESS_CFG      = _cfg["EXPRESS_CFG"]
