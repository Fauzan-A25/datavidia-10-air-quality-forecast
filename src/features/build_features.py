"""Tabel peramalan langsung: dari satu tanggal asal, setiap stasiun diramal untuk 91 hari ke depan.

Fitur hanya memakai informasi yang tersedia di tanggal asal (kondisi 30 dan 365 hari terakhir) dan informasi kalender
yang sudah diketahui sebelumnya (hari dalam tahun, hari dalam minggu, libur nasional).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import STATIONS

CALENDAR = ["stasiun", "doy_sin", "doy_cos", "hari_minggu", "libur"]
STATE = ["r30_max", "r30_tidak_sehat", "r30_baik", "r365_max", "r365_tidak_sehat"]
HORIZON = ["h"]


def origin_table(ispu: pd.DataFrame, holidays: pd.DataFrame, origin: pd.Timestamp, horizon: int = 91,
                 with_target: bool = True) -> pd.DataFrame:
    origin = pd.Timestamp(origin)
    hist = ispu[(ispu.tanggal <= origin) & (ispu.tanggal > origin - pd.Timedelta(days=365))]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    rows = []
    for st in STATIONS:
        h = hist[hist.station == st]
        h30 = h[h.tanggal > origin - pd.Timedelta(days=30)]
        if len(h30) < 5:
            continue
        state = dict(r30_max=h30["max"].mean(), r30_tidak_sehat=(h30.kelas == 2).mean(), r30_baik=(h30.kelas == 0).mean(),
                     r365_max=h["max"].mean(), r365_tidak_sehat=(h.kelas == 2).mean())
        rows.append(pd.DataFrame({"tanggal": dates, "station": st, "stasiun": int(st[-1]), "asal": origin,
                                  "h": np.arange(1, horizon + 1), **state}))
    if not rows:
        return pd.DataFrame()
    X = pd.concat(rows, ignore_index=True)
    doy = X.tanggal.dt.dayofyear / 365.25
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    X["hari_minggu"] = X.tanggal.dt.dayofweek
    X = X.merge(holidays[["tanggal", "is_holiday_nasional"]].rename(columns={"is_holiday_nasional": "libur"}), on="tanggal", how="left")
    X["libur"] = X.libur.fillna(0)
    if with_target:
        X = X.merge(ispu[["tanggal", "station", "kelas"]], on=["tanggal", "station"], how="inner")
    return X


def training_table(ispu, holidays, origins) -> pd.DataFrame:
    parts = [origin_table(ispu, holidays, o) for o in origins]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)
