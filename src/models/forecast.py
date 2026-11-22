"""Backtest rolling-origin, model LightGBM multikelas, dan bobot keputusan untuk macro F1."""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

WEIGHT_GRID = [0.5, 1, 1.5, 2, 3, 4, 6]


def lgb_model(seed: int = 0):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


def macro_f1(y, pred) -> float:
    return f1_score(y, pred, average="macro")


def decide(P: np.ndarray, w) -> np.ndarray:
    """Kelas = argmax(peluang × bobot). Bobot SEDANG tetap 1."""
    return (P * np.asarray(w)).argmax(axis=1)


def tune_weights(P_list, y_list) -> tuple[tuple, float]:
    """Bobot (BAIK, 1, TIDAK SEHAT) yang memaksimalkan rata-rata macro F1 di beberapa backtest."""
    best, best_s = (1, 1, 1), -1
    for wb, wt in itertools.product(WEIGHT_GRID, WEIGHT_GRID):
        s = np.mean([macro_f1(y, decide(P, (wb, 1, wt))) for P, y in zip(P_list, y_list)])
        if s > best_s:
            best, best_s = (wb, 1, wt), s
    return best, best_s


def train_origins(origins, cutoff: pd.Timestamp, horizon: int = 91):
    """Tanggal asal yang seluruh 91 hari targetnya sudah lewat sebelum `cutoff`."""
    return [o for o in origins if o + pd.Timedelta(days=horizon) <= cutoff]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features, seeds=range(3)) -> np.ndarray:
    P = np.zeros((len(test), 3))
    for s in seeds:
        m = lgb_model(s).fit(train[features], train.kelas)
        P[:, m.classes_] += m.predict_proba(test[features])
    return P / len(list(seeds))
