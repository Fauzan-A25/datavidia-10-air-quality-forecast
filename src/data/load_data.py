"""Load dan seragamkan data Datavidia 10: ISPU 2010–2025 (format kolom dan tanggal berbeda tiap tahun), cuaca, libur."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"]
MODEL_CLASSES = ["BAIK", "SEDANG", "TIDAK SEHAT"]          # sangat tidak sehat & berbahaya digabung ke tidak sehat
STATIONS = ["DKI1", "DKI2", "DKI3", "DKI4", "DKI5"]
RENAME = {"pm_sepuluh": "pm10", "pm_10": "pm10", "pm_duakomalima": "pm25", "sulfur_dioksida": "so2",
          "karbon_monoksida": "co", "ozon": "o3", "nitrogen_dioksida": "no2", "parameter_pencemar_kritis": "critical",
          "categori": "kategori", "lokasi_spku": "stasiun"}


def find_data_dir(local: str | Path = "../data/raw") -> Path:
    """Folder yang berisi subfolder ISPU/ dan sample_submission.csv (di /kaggle/input, ikuti symlink)."""
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "ISPU" in dirs and "sample_submission.csv" in files:
            return Path(root)
    return Path(local)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Tanggal ISO, dd/mm/yyyy (2018), atau nomor seri Excel (beberapa baris 2022)."""
    s = s.astype(str).str.strip()
    slash = s.str.contains("/")
    iso = pd.to_datetime(s.where(~slash), errors="coerce", format="mixed")
    dmy = pd.to_datetime(s.where(slash), errors="coerce", dayfirst=True, format="mixed")
    num = pd.to_numeric(s, errors="coerce")
    xl = pd.to_datetime("1899-12-30") + pd.to_timedelta(num.fillna(0).astype(int), unit="D")
    return iso.fillna(dmy).fillna(xl.where(num.notna() & (num > 30000))).dt.normalize()


def load_ispu(data_dir: str | Path) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(Path(data_dir) / "ISPU" / "*.csv"))):
        x = pd.read_csv(f).rename(columns=RENAME)
        if "bulan" in x.columns and x.tanggal.astype(str).str.len().max() <= 2:     # 2023–2025: tanggal = hari
            x["tanggal"] = pd.to_datetime(x.periode_data.astype(str).str[:4] + "-" + x.bulan.astype(str) + "-"
                                          + x.tanggal.astype(str), errors="coerce")
        else:
            x["tanggal"] = _parse_dates(x.tanggal)
        frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    d["station"] = d.stasiun.astype(str).str.extract(r"(DKI\s?\d)")[0].str.replace(" ", "")
    for c in ["pm10", "pm25", "so2", "co", "o3", "no2", "max"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["kategori"] = d.kategori.astype(str).str.upper().str.strip()
    d["critical"] = d.critical.astype(str).str.upper().str.replace(",", ".").str.replace(r"PM2\.?5", "PM25", regex=True)
    raw_rows = len(d)
    d = d[d.kategori.isin(CATEGORIES) & d.tanggal.notna() & d.station.isin(STATIONS)]
    d = d.sort_values(["station", "tanggal"]).drop_duplicates(["station", "tanggal"], keep="last")
    d = d[["tanggal", "station", "pm10", "pm25", "so2", "co", "o3", "no2", "max", "critical", "kategori"]].reset_index(drop=True)
    d["kelas"] = d.kategori.map({"BAIK": 0, "SEDANG": 1, "TIDAK SEHAT": 2, "SANGAT TIDAK SEHAT": 2, "BERBAHAYA": 2})
    d.attrs["raw_rows"] = raw_rows
    return d


def load_holidays(data_dir: str | Path) -> pd.DataFrame:
    h = pd.read_csv(Path(data_dir) / "libur-nasional" / "dataset-libur-nasional-dan-weekend.csv", parse_dates=["tanggal"])
    return h[["tanggal", "is_holiday_nasional", "is_weekend"]]


def load_weather(data_dir: str | Path) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(Path(data_dir) / "cuaca-harian" / "*.csv"))):
        x = pd.read_csv(f)
        x = x.rename(columns={"time": "tanggal"})
        x["tanggal"] = pd.to_datetime(x.tanggal)
        x["station"] = Path(f).stem.split("-")[2].upper()
        frames.append(x)
    w = pd.concat(frames, ignore_index=True)
    w.columns = [c.split(" (")[0] for c in w.columns]
    return w.loc[:, ~w.columns.duplicated()]
