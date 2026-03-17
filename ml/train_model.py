# -*- coding: utf-8 -*-
"""
ml/train_model.py v3 — Two-model approach
==========================================
Модель A: БЕЗ коэффициентов → чистые спортивные данные
Модель B: С коэффициентами  → учится у рынка

Ставим когда: Модель A сильно расходится с рынком (value).
Это единственный способ найти edge над букмекерами.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

from build_features import FEATURE_COLS

# Признаки БЕЗ коэффициентов
SPORT_COLS = [c for c in FEATURE_COLS if c not in ("imp_home", "imp_draw", "imp_away", "overround")]
# Признаки С коэффициентами
MARKET_COLS = FEATURE_COLS[:]


def load_data():
    path = DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError("features.csv не найден.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def split_data(df):
    n = len(df)
    tr, vl, te = df.iloc[:int(n*0.75)], df.iloc[int(n*0.75):int(n*0.875)], df.iloc[int(n*0.875):]
    print(f"Train: {len(tr)} | Val: {len(vl)} | Test: {len(te)}")
    return tr, vl, te


def _train_xgb(X_tr, y_tr, X_vl, y_vl, label=""):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        gamma=2,
        reg_alpha=1.0,
        reg_lambda=3.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=100)
    from sklearn.metrics import accuracy_score
    preds = model.predict(X_vl)
    print(f"  {label} Val Accuracy: {accuracy_score(y_vl, preds)*100:.1f}%")
    return model


def _calibrate(model, X_vl, y_vl):
    """Isotonic regression калибровка (лучше Platt для больших данных)."""
    from sklearn.isotonic import IsotonicRegression
    raw = model.predict_proba(X_vl)
    cals = {}
    for cls_idx in range(3):
        y_bin = (y_vl == cls_idx).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw[:, cls_idx], y_bin)
        cals[cls_idx] = ir
    return cals


def _apply_cal(model, cals, X):
    raw = model.predict_proba(X)
    cal = np.zeros_like(raw)
    for i, ir in cals.items():
        cal[:, i] = ir.predict(raw[:, i])
    row_sums = cal.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cal / row_sums


def simulate_roi_two_model(df_test, proba_sport, proba_market):
    """
    Ставим когда спортивная модель видит value vs рыночная модель.
    Edge = sport_prob > market_prob * threshold
    """
    print(f"\n{'='*55}")
    print("ROI СИМУЛЯЦИЯ — Two-Model Divergence Strategy")
    print(f"{'='*55}")

    outcome_map = {0: "odds_home", 2: "odds_away"}
    results = {}

    for div_thresh in [1.10, 1.15, 1.20]:
        bets = profit = correct = 0
        for i, (_, row) in enumerate(df_test.iterrows()):
            actual = int(row["target"])
            for cls_idx, odds_col in outcome_map.items():
                if odds_col not in row or pd.isna(row[odds_col]):
                    continue
                bm_odds = float(row[odds_col])
                if bm_odds <= 1.0:
                    continue

                sp = float(proba_sport[i][cls_idx])
                mk = float(proba_market[i][cls_idx])

                # Ставим только если наша спортивная модель превышает рыночную на threshold
                if sp > mk * div_thresh:
                    bets += 1
                    if actual == cls_idx:
                        profit += bm_odds - 1
                        correct += 1
                    else:
                        profit -= 1.0

        roi = profit / bets * 100 if bets > 0 else 0
        wr  = correct / bets * 100 if bets > 0 else 0
        mark = "✅" if roi > 0 else "⚠️"
        print(f"Div>{div_thresh:.0%}: {bets:4d} ставок | WR {wr:.1f}% | ROI {roi:+.1f}% {mark}")
        results[f"roi_div{int(div_thresh*100)}"] = round(roi, 2)

    return results


def simulate_roi_ev(df_test, proba_sport):
    """Классический EV подход (только спортивная модель без коэффициентов)."""
    print(f"\n{'='*55}")
    print("ROI — Pure Sports Model (без коэффициентов в признаках)")
    print(f"{'='*55}")
    outcome_map = {0: "odds_home", 2: "odds_away"}

    for ev_thr in [5, 8, 12]:
        bets = profit = correct = 0
        for i, (_, row) in enumerate(df_test.iterrows()):
            actual = int(row["target"])
            for cls_idx, odds_col in outcome_map.items():
                if odds_col not in row or pd.isna(row[odds_col]):
                    continue
                bm_odds = float(row[odds_col])
                if bm_odds <= 1.0: continue
                sp = float(proba_sport[i][cls_idx])
                implied = 1.0 / bm_odds
                ev = (sp - implied) / implied * 100
                if ev >= ev_thr:
                    bets += 1
                    if actual == cls_idx:
                        profit += bm_odds - 1
                        correct += 1
                    else:
                        profit -= 1.0
        roi = profit / bets * 100 if bets > 0 else 0
        wr  = correct / bets * 100 if bets > 0 else 0
        mark = "✅" if roi > 0 else "⚠️"
        print(f"EV>{ev_thr}%: {bets:4d} ставок | WR {wr:.1f}% | ROI {roi:+.1f}% {mark}")


def feature_importance(model, cols, label):
    scores = model.get_booster().get_fscore()
    named = []
    for feat, score in scores.items():
        try:
            idx = int(feat[1:])
            name = cols[idx] if idx < len(cols) else feat
        except (ValueError, IndexError):
            name = feat
        named.append((name, score))
    named.sort(key=lambda x: x[1], reverse=True)
    print(f"\nВажность [{label}]:")
    mx = named[0][1] if named else 1
    for name, score in named[:8]:
        bar = "█" * int(score / mx * 15)
        print(f"  {name:32s} {bar} {score:.0f}")


def retrain_incremental(min_new_rows: int = 30) -> dict:
    """
    Инкрементальное переобучение XGBoost с новыми живыми матчами.

    Алгоритм:
    1. Читаем ml/data/live_matches.csv (матчи записанные check_results_task)
    2. Если новых строк < min_new_rows — пропускаем (недостаточно данных)
    3. Добавляем к all_matches_raw.csv → перестраиваем features.csv
    4. Переобучаем обе XGBoost модели → сохраняем

    Возвращает {"status": "ok"|"skip"|"error", "new_rows": N, ...}
    """
    import sys
    live_path = DATA_DIR / "live_matches.csv"
    raw_path  = DATA_DIR / "all_matches_raw.csv"

    if not live_path.exists():
        return {"status": "skip", "reason": "live_matches.csv не существует"}

    try:
        live_df = pd.read_csv(live_path)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    if len(live_df) < min_new_rows:
        return {"status": "skip", "reason": f"Только {len(live_df)} новых матчей (нужно ≥{min_new_rows})"}

    print(f"[Retrain] {len(live_df)} новых матчей. Начинаю переобучение...")

    try:
        # 1. Добавляем живые матчи к историческим
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path, low_memory=False)
            # Переименовываем live колонки если нужно
            live_for_merge = live_df.rename(columns={
                "FTHG": "FTHG", "FTAG": "FTAG", "FTR": "FTR",
                "B365H": "B365H", "B365D": "B365D", "B365A": "B365A",
            })
            combined = pd.concat([raw_df, live_for_merge], ignore_index=True, sort=False)
        else:
            combined = live_df.copy()

        combined_path = DATA_DIR / "all_matches_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"[Retrain] Всего матчей для обучения: {len(combined)}")

        # 2. Перестраиваем features.csv из combined
        # Временно подменяем путь для build_features
        import sys as _sys, importlib
        _sys.path.insert(0, str(Path(__file__).parent))
        try:
            import build_features as _bf
            # build_features читает all_matches_raw.csv — временно подменяем
            _orig = _bf.DATA_DIR / "all_matches_raw.csv"
            _tmp  = _bf.DATA_DIR / "_tmp_combined.csv"
            combined.to_csv(_tmp, index=False)

            # Читаем и строим признаки
            raw = pd.read_csv(_tmp, low_memory=False)
            raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
            raw = raw.dropna(subset=["FTHG", "FTAG", "FTR"])
            raw["FTHG"] = pd.to_numeric(raw["FTHG"], errors="coerce")
            raw["FTAG"] = pd.to_numeric(raw["FTAG"], errors="coerce")
            raw = raw.dropna(subset=["FTHG", "FTAG"])
            raw["FTHG"] = raw["FTHG"].astype(int)
            raw["FTAG"] = raw["FTAG"].astype(int)

            feat_df = _bf.build_features(raw)
            feat_path = _bf.DATA_DIR / "features.csv"
            feat_df.to_csv(feat_path, index=False)
            print(f"[Retrain] features.csv перестроен: {len(feat_df)} строк")
            _tmp.unlink(missing_ok=True)
        except Exception as _fe:
            print(f"[Retrain] Ошибка build_features: {_fe}")
            return {"status": "error", "reason": f"build_features: {_fe}"}

        # 3. Переобучаем модели
        df = load_data()
        tr, vl, te = split_data(df)

        sport_cols  = [c for c in SPORT_COLS  if c in df.columns]
        market_cols = [c for c in MARKET_COLS if c in df.columns]

        X_tr_s = tr[sport_cols].values;  X_vl_s = vl[sport_cols].values;  X_te_s = te[sport_cols].values
        X_tr_m = tr[market_cols].values; X_vl_m = vl[market_cols].values;  X_te_m = te[market_cols].values
        y_tr, y_vl, y_te = tr["target"].values, vl["target"].values, te["target"].values

        model_sport  = _train_xgb(X_tr_s, y_tr, X_vl_s, y_vl, "Sport")
        model_market = _train_xgb(X_tr_m, y_tr, X_vl_m, y_vl, "Market")
        cal_sport    = _calibrate(model_sport,  X_vl_s, y_vl)
        cal_market   = _calibrate(model_market, X_vl_m, y_vl)

        from sklearn.metrics import accuracy_score
        proba_s = _apply_cal(model_sport,  cal_sport,  X_te_s)
        proba_m = _apply_cal(model_market, cal_market, X_te_m)
        acc_s = accuracy_score(y_te, np.argmax(proba_s, axis=1))
        acc_m = accuracy_score(y_te, np.argmax(proba_m, axis=1))

        # 4. Сохраняем модели
        with open(MODEL_DIR / "football_sport_model.pkl", "wb") as f:
            pickle.dump({"model": model_sport, "cals": cal_sport, "cols": sport_cols}, f)
        with open(MODEL_DIR / "football_market_model.pkl", "wb") as f:
            pickle.dump({"model": model_market, "cals": cal_market, "cols": market_cols}, f)

        meta = {"sport_cols": sport_cols, "market_cols": market_cols,
                "accuracy_sport": round(acc_s, 4), "accuracy_market": round(acc_m, 4),
                "trained_on": len(df), "live_rows_added": len(live_df)}
        with open(MODEL_DIR / "football_xgb_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 5. Очищаем live_matches.csv (данные уже в combined)
        combined.to_csv(raw_path, index=False)
        live_path.unlink(missing_ok=True)

        print(f"[Retrain] Готово! Sport acc={acc_s*100:.1f}% | Market acc={acc_m*100:.1f}%")
        return {
            "status": "ok",
            "new_rows": len(live_df),
            "total_rows": len(df),
            "acc_sport": round(acc_s * 100, 1),
            "acc_market": round(acc_m * 100, 1),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}


if __name__ == "__main__":
    print("Chimera Football XGBoost v3 — Two-Model")
    print("="*55)

    df = load_data()
    print(f"Загружено {len(df)} матчей")
    tr, vl, te = split_data(df)

    sport_cols  = [c for c in SPORT_COLS  if c in df.columns]
    market_cols = [c for c in MARKET_COLS if c in df.columns]

    X_tr_s = tr[sport_cols].values;  X_vl_s = vl[sport_cols].values;  X_te_s = te[sport_cols].values
    X_tr_m = tr[market_cols].values; X_vl_m = vl[market_cols].values;  X_te_m = te[market_cols].values
    y_tr, y_vl, y_te = tr["target"].values, vl["target"].values, te["target"].values

    print("\n--- Модель A: спортивная (без коэффициентов) ---")
    model_sport = _train_xgb(X_tr_s, y_tr, X_vl_s, y_vl, "Sport")

    print("\n--- Модель B: рыночная (с коэффициентами) ---")
    model_market = _train_xgb(X_tr_m, y_tr, X_vl_m, y_vl, "Market")

    print("\nКалибровка...")
    cal_sport  = _calibrate(model_sport,  X_vl_s, y_vl)
    cal_market = _calibrate(model_market, X_vl_m, y_vl)

    proba_sport  = _apply_cal(model_sport,  cal_sport,  X_te_s)
    proba_market = _apply_cal(model_market, cal_market, X_te_m)

    from sklearn.metrics import accuracy_score
    acc_s = accuracy_score(y_te, np.argmax(proba_sport, axis=1))
    acc_m = accuracy_score(y_te, np.argmax(proba_market, axis=1))
    print(f"\nTest Accuracy — Sport: {acc_s*100:.1f}% | Market: {acc_m*100:.1f}%")

    simulate_roi_ev(te, proba_sport)
    roi_metrics = simulate_roi_two_model(te, proba_sport, proba_market)

    feature_importance(model_sport,  sport_cols,  "Sport")
    feature_importance(model_market, market_cols, "Market")

    # Сохраняем обе модели
    with open(MODEL_DIR / "football_sport_model.pkl", "wb") as f:
        pickle.dump({"model": model_sport, "cals": cal_sport, "cols": sport_cols}, f)
    with open(MODEL_DIR / "football_market_model.pkl", "wb") as f:
        pickle.dump({"model": model_market, "cals": cal_market, "cols": market_cols}, f)

    meta = {"sport_cols": sport_cols, "market_cols": market_cols,
            "accuracy_sport": round(acc_s, 4), "accuracy_market": round(acc_m, 4),
            **roi_metrics}
    with open(MODEL_DIR / "football_xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nГотово! Модели сохранены.")
