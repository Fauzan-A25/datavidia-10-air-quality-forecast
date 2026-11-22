"""Generate notebook riset Datavidia 10 (peramalan kategori ISPU Jakarta, metrik macro F1) dan script runner Kaggle.

Output:
- kaggle-notebook/datavidia-10-air-quality-forecast.ipynb  (notebook bersih, di-push sebagai notebook Kaggle)
- kaggle/run_on_kaggle.py                                   (script kernel terpisah untuk mengambil notebook ber-output)
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NB_NAME = "datavidia-10-air-quality-forecast.ipynb"
FINDINGS = json.loads((ROOT / "_findings.json").read_text()) if (ROOT / "_findings.json").exists() else {}


def T(key: str) -> str:
    return FINDINGS.get(key, "_(temuan diisi setelah run pertama)_")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ════════════════════════════════════════════════════════════════════ Header
md(
    f"""
# Meramal Kategori Kualitas Udara Jakarta Tiga Bulan ke Depan

| | |
| --- | --- |
| **Kompetisi** | [Air Pollution Prediction, Penyisihan Datavidia 10](https://www.kaggle.com/competitions/penyisihan-datavidia-10) (Arkavidia ITB, Kaggle, Januari–Februari 2026) |
| **Tujuan** | Meramal kategori ISPU harian di 5 stasiun DKI Jakarta untuk 1 September – 30 November 2025 (metrik macro F1) |
| **Data** | ISPU harian 2010–Agustus 2025, cuaca harian, NDVI, libur nasional; 455 baris submission |
| **Penulis** | Fauzan Ahsanudin Alfikri · [Kaggle: fauzan2](https://www.kaggle.com/fauzan2) |
| **Tanggal** | Disusun ulang November 2026, menggantikan notebook kompetisi Februari 2026 |
| **Lingkungan** | Dijalankan penuh di Kaggle (CPU) |

**Pertanyaan penelitian.** Dengan data yang berakhir 31 Agustus 2025 (termasuk cuaca), seberapa baik kategori ISPU bisa
diramal untuk 1–91 hari ke depan, dan informasi apa yang masih berguna sejauh itu?

**Ringkasan temuan.**
{T("summary")}

**Alur:** 1. Setup → 2. Memahami data → 3. EDA → 4. Desain validasi → 5. Baseline → 6. Model →
7. Bobot keputusan untuk macro F1 → 8. Interpretasi → 9. Submission → 10. Kesimpulan
"""
)

md(
    """
## 1. Setup

Semua import, parameter, dan path ada di sel ini. Penyeragaman data ISPU ada di `src/data`, tabel peramalan di
`src/features`, dan backtest serta bobot keputusan di `src/models`.
"""
)

code(
    """
import os
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

SRC_ROOT = Path.cwd().parent
for r, _, f in os.walk("/kaggle/input", followlinks=True):
    if Path(r).name == "models" and "forecast.py" in f:
        pkg = Path(r).parent
        if pkg.name == "src":
            SRC_ROOT = pkg.parent
        else:
            SRC_ROOT = Path("/tmp/project")
            shutil.copytree(pkg, SRC_ROOT / "src", dirs_exist_ok=True)
        break
sys.path.insert(0, str(SRC_ROOT))
from src.data.load_data import MODEL_CLASSES, STATIONS, find_data_dir, load_holidays, load_ispu, load_weather
from src.features.build_features import CALENDAR, HORIZON, STATE, origin_table, training_table
from src.models.forecast import decide, fit_predict, macro_f1, train_origins, tune_weights
from src.visualization.plots import PALETTE, pct_axis, save_fig, set_style

# ── Parameter eksperimen
HORIZON_DAYS = 91
ORIGINS = pd.date_range("2014-12-31", "2025-08-31", freq="ME")   # asal bulanan untuk data latih
FINAL_ORIGIN = pd.Timestamp("2025-08-31")

# ── Path
DATA_DIR = find_data_dir(local="../data/raw")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
set_style()
C1, C2, C3, CN = PALETTE["normal"], PALETTE["fraud"], PALETTE["accent"], PALETTE["neutral"]
CLS_COL = {"BAIK": C3, "SEDANG": C1, "TIDAK SEHAT": C2}
T0 = time.time()
print("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)
"""
)

# ════════════════════════════════════════════════════════════════════ Data
md(
    """
## 2. Memahami data

Data ISPU dipublikasikan per tahun dengan format yang berubah-ubah: nama kolom (`pm10` vs `pm_sepuluh`), format tanggal
(ISO, dd/mm/yyyy, nomor seri Excel), dan penulisan nama stasiun. Semua diseragamkan di `load_ispu`. Baris tanpa kategori
yang valid ("TIDAK ADA DATA") dan duplikat dibuang.
"""
)

code(
    """
ispu = load_ispu(DATA_DIR)
hol = load_holidays(DATA_DIR)
weather = load_weather(DATA_DIR)
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"ISPU: {ispu.attrs['raw_rows']:,} baris mentah → {len(ispu):,} baris valid | {ispu.tanggal.min().date()} s.d. {ispu.tanggal.max().date()}")
print(f"Cuaca: {weather.tanggal.min().date()} s.d. {weather.tanggal.max().date()} | Submission: {len(sample)} baris, {sample.id.str[:10].min()} s.d. {sample.id.str[:10].max()}")
cov = ispu.groupby([ispu.tanggal.dt.year, "station"]).size().unstack().fillna(0)
fig, ax = plt.subplots(figsize=(13, 3.6))
sns.heatmap(cov.T, cmap="Blues", annot=True, fmt=".0f", cbar=False, ax=ax, annot_kws={"fontsize": 7})
ax.set(xlabel="tahun", ylabel="", title="Jumlah hari berlabel per stasiun dan tahun")
save_fig(FIG_DIR, "01_coverage")
ispu.kategori.value_counts().to_frame("hari").T
"""
)

md("**Temuan.** " + T("data"))

# ════════════════════════════════════════════════════════════════════ EDA
md("## 3. Exploratory Data Analysis\n\n### 3.1 PM2.5 mengubah skala sejak 2021")

code(
    """
yr = ispu.groupby(ispu.tanggal.dt.year)
crit = pd.crosstab(ispu.tanggal.dt.year, ispu.critical.where(ispu.critical.isin(["PM25", "PM10", "O3", "CO", "SO2", "NO2"]), "lain"), normalize="index")
share = pd.crosstab(ispu.tanggal.dt.year, ispu.kategori, normalize="index")[["BAIK", "SEDANG", "TIDAK SEHAT"]]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.3))
crit[[c for c in ["PM25", "PM10", "O3", "CO", "SO2", "NO2", "lain"] if c in crit]].plot.bar(stacked=True, ax=ax[0], width=0.85, colormap="tab20")
pct_axis(ax[0], 0); ax[0].set(xlabel="tahun", ylabel="porsi hari", title="Parameter pencemar kritis"); ax[0].legend(fontsize=7, ncol=2)
share.plot.bar(stacked=True, ax=ax[1], width=0.85, color=[CLS_COL[c] for c in share.columns])
pct_axis(ax[1], 0); ax[1].set(xlabel="tahun", ylabel="porsi hari", title="Kategori ISPU per tahun"); ax[1].legend(fontsize=8)
save_fig(FIG_DIR, "02_regime")
"""
)

md("**Temuan.** " + T("regime"))

md("### 3.2 Musim: September–November adalah masa peralihan")

code(
    """
rec = ispu[ispu.tanggal.dt.year >= 2021]
mon = pd.crosstab(rec.tanggal.dt.month, rec.kategori, normalize="index")[["BAIK", "SEDANG", "TIDAK SEHAT"]]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.3))
mon.plot.bar(stacked=True, ax=ax[0], width=0.85, color=[CLS_COL[c] for c in mon.columns])
ax[0].axvspan(7.5, 10.5, color="black", alpha=0.08)
pct_axis(ax[0], 0); ax[0].set(xlabel="bulan", ylabel="porsi hari", title="Kategori per bulan, 2021–2025 (arsir = periode test)"); ax[0].legend(fontsize=8)
per = rec[rec.tanggal.dt.month.isin([9, 10, 11])]
yy = pd.crosstab([per.tanggal.dt.year], per.kategori, normalize="index").reindex(columns=["BAIK", "SEDANG", "TIDAK SEHAT"], fill_value=0)
yy.plot.bar(stacked=True, ax=ax[1], width=0.7, color=[CLS_COL[c] for c in yy.columns])
pct_axis(ax[1], 0); ax[1].set(xlabel="tahun", ylabel="porsi hari", title="September–November per tahun"); ax[1].legend(fontsize=8)
save_fig(FIG_DIR, "03_season")
yy.style.format("{:.1%}")
"""
)

md("**Temuan.** " + T("season"))

md(
    """
### 3.3 Cuaca menjelaskan hari buruk, tetapi tidak tersedia untuk periode test

Data cuaca berakhir 31 Agustus 2025, sama dengan ISPU. Jadi cuaca bisa menjelaskan mekanisme, tetapi tidak bisa dipakai
sebagai fitur untuk September–November.
"""
)

code(
    """
wm = rec.merge(weather[["tanggal", "station", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"]], on=["tanggal", "station"], how="inner")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for a, (c, lab, bins) in zip(ax, [("precipitation_sum", "curah hujan (mm)", [-0.1, 0, 2, 5, 10, 20, 300]),
                                  ("wind_speed_10m_max", "angin maksimum (km/jam)", [0, 8, 10, 12, 14, 17, 60]),
                                  ("relative_humidity_2m_mean", "kelembapan rata-rata (%)", [0, 70, 75, 80, 85, 100])]):
    g = wm.groupby(pd.cut(wm[c], bins), observed=True)["max"].agg(["mean", "size"])
    ts = wm.groupby(pd.cut(wm[c], bins), observed=True).kelas.apply(lambda s: (s == 2).mean())
    a.bar([str(i) for i in g.index], ts, color=C2)
    pct_axis(a, 0); a.set(xlabel=lab, ylabel="porsi hari tidak sehat", title=f"Hari tidak sehat menurut {lab.split(' (')[0]}")
    a.tick_params(axis="x", rotation=30)
save_fig(FIG_DIR, "04_weather")
print(wm[["max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"]].corr(method="spearman")["max"].round(3).to_string())
"""
)

md("**Temuan.** " + T("weather"))

md("### 3.4 Berapa lama kondisi hari ini bertahan?")

code(
    """
lags = [1, 3, 7, 14, 30, 60, 91]
ac = {}
for st in STATIONS:
    s = rec[rec.station == st].set_index("tanggal")["max"].asfreq("D")
    an = s - s.groupby(s.index.dayofyear).transform("mean")          # anomali terhadap rata-rata hari-dalam-tahun
    ac[st] = {l: s.autocorr(l) for l in lags} | {f"anomali {l}": an.autocorr(l) for l in lags}
ac = pd.DataFrame(ac).mean(axis=1)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags, [ac[l] for l in lags], marker="o", color=C1, label="nilai indeks")
ax.plot(lags, [ac[f"anomali {l}"] for l in lags], marker="s", color=C2, label="anomali dari pola musiman")
ax.axhline(0, color="black", lw=0.7); ax.set_xscale("log"); ax.set_xticks(lags, lags)
ax.set(xlabel="jarak hari (lag)", ylabel="autokorelasi", title="Autokorelasi indeks maksimum harian (rata-rata 5 stasiun, 2021–2025)"); ax.legend()
save_fig(FIG_DIR, "05_persistence")
print({l: round(ac[f"anomali {l}"], 2) for l in lags})
"""
)

md("**Temuan.** " + T("persistence"))

# ════════════════════════════════════════════════════════════════════ Validation
md(
    """
## 4. Desain validasi

Situasi kompetisi: data berakhir 31 Agustus, lalu 91 hari diramal sekaligus. Backtest meniru situasi itu dari **setiap akhir
bulan** antara Maret 2021 dan Mei 2025 yang datanya cukup (minimal 250 hari-stasiun berlabel dalam 91 hari berikutnya).
Satu backtest hanya berisi sekitar 450 baris dan kelas jarang bisa muncul hanya beberapa kali, sehingga skor satu backtest
sangat bising; rata-rata dari banyak backtest jauh lebih bisa dipercaya. Backtest dengan asal 31 Agustus dilaporkan
terpisah karena paling mirip dengan periode kompetisi.

Untuk tiap backtest, model hanya dilatih dengan tanggal asal bulanan yang seluruh 91 hari targetnya sudah lewat sebelum
tanggal asal backtest. Kategori *sangat tidak sehat* dan *berbahaya* (201 hari dalam 15 tahun) digabung ke *tidak sehat*.
"""
)

code(
    """
TABLE = training_table(ispu, hol, ORIGINS)
EVAL = [o for o in pd.date_range("2021-03-31", "2025-05-31", freq="ME") if (TABLE.asal == o).sum() >= 250]
AUG = [o for o in EVAL if o.month == 8]
BT = {o: TABLE[TABLE.asal == o] for o in EVAL}
print(f"Tabel latih: {len(TABLE):,} baris dari {TABLE.asal.nunique()} tanggal asal | backtest: {len(EVAL)} (asal 31 Agustus: {[o.year for o in AUG]})")
cnt = pd.DataFrame({o.strftime("%Y-%m"): BT[o].kelas.value_counts().reindex([0, 1, 2], fill_value=0).values for o in EVAL}, index=MODEL_CLASSES).T
print(f"Backtest tanpa satu pun hari BAIK: {(cnt.BAIK == 0).sum()} | tanpa hari TIDAK SEHAT: {(cnt['TIDAK SEHAT'] == 0).sum()}")
cnt.describe().loc[["mean", "min", "max"]].round(1)
"""
)

md("**Temuan.** " + T("validation"))

code(
    """
RESULTS, PROBS = {}, {}

def score_preds(name, preds):
    s = pd.Series({o: macro_f1(BT[o].kelas, preds[o]) for o in EVAL})
    RESULTS[name] = {"rata-rata": s.mean(), "std": s.std(), "asal 31 Agustus": s[AUG].mean(), "per_asal": s}
    return f"{name}: rata-rata {s.mean():.3f} ± {s.std():.3f} | asal 31 Agustus {s[AUG].mean():.3f}"
"""
)

# ════════════════════════════════════════════════════════════════════ Baselines
md(
    """
## 5. Baseline

- **SEDANG untuk semua hari:** kelas mayoritas.
- **Klimatologi:** kelas tersering per stasiun dan bulan, dari tahun 2014 sampai tanggal asal.
- **Kondisi 30 hari terakhir:** kelas tersering per stasiun di 30 hari sebelum tanggal asal, dipakai untuk 91 hari.
"""
)

code(
    """
print(score_preds("SEDANG untuk semua", {o: np.ones(len(BT[o]), int) for o in EVAL}))
clim, last30 = {}, {}
for o in EVAL:
    hist = ispu[(ispu.tanggal <= o) & (ispu.tanggal.dt.year >= 2014)]
    mode = hist.groupby([hist.station, hist.tanggal.dt.month]).kelas.agg(lambda s: s.mode()[0])
    clim[o] = np.array([mode.get((s, m), 1) for s, m in zip(BT[o].station, BT[o].tanggal.dt.month)])
    m30 = hist[hist.tanggal > o - pd.Timedelta(days=30)].groupby("station").kelas.agg(lambda s: s.mode()[0])
    last30[o] = BT[o].station.map(m30).fillna(1).astype(int).values
print(score_preds("klimatologi stasiun × bulan", clim))
print(score_preds("kondisi 30 hari terakhir", last30))
"""
)

md("**Temuan.** " + T("baseline"))

# ════════════════════════════════════════════════════════════════════ Models
md(
    """
## 6. Model

LightGBM multikelas dengan dua set fitur:

- **Kalender:** stasiun, hari dalam tahun (sin/cos), hari dalam minggu, libur nasional.
- **Kalender + kondisi di tanggal asal:** ditambah rata-rata indeks dan porsi hari tidak sehat/baik dalam 30 dan 365 hari
  terakhir, serta jarak ramalan (1–91 hari).

Bagian ini memakai argmax (kelas dengan peluang tertinggi).
"""
)

code(
    """
FEATS = {"kalender": CALENDAR, "kalender + kondisi": CALENDAR + STATE + HORIZON}
t = time.time()
for name, F in FEATS.items():
    PROBS[name] = {o: fit_predict(TABLE[TABLE.asal.isin(train_origins(ORIGINS, o, HORIZON_DAYS))], BT[o], F, seeds=range(1)) for o in EVAL}
    print(score_preds(f"LightGBM {name}, argmax", {o: PROBS[name][o].argmax(1) for o in EVAL}), f"({(time.time() - t) / 60:.1f} menit)")
"""
)

md("**Temuan.** " + T("models"))

# ════════════════════════════════════════════════════════════════════ Decision weights
md(
    """
## 7. Bobot keputusan untuk macro F1

Macro F1 memberi bobot sama untuk setiap kelas, jadi kelas jarang (BAIK, TIDAK SEHAT) perlu lebih sering ditebak daripada
yang disarankan argmax. Peluang tiap kelas dikalikan bobot, lalu dipilih yang terbesar. Bobot dipilih dengan
**leave-one-backtest-out**: bobot untuk menilai satu backtest dipilih hanya dari backtest lainnya, sehingga skornya tidak
bocor.
"""
)

code(
    """
W_LOO = {}
for name in FEATS:
    preds = {}
    for o in EVAL:
        others = [z for z in EVAL if z != o]
        w, _ = tune_weights([PROBS[name][z] for z in others], [BT[z].kelas.values for z in others])
        W_LOO[(name, o)] = w
        preds[o] = decide(PROBS[name][o], w)
    print(score_preds(f"LightGBM {name}, bobot macro F1", preds))
res = pd.DataFrame({k: {c: v[c] for c in ["rata-rata", "std", "asal 31 Agustus"]} for k, v in RESULTS.items()}).T.sort_values("rata-rata")
per = pd.DataFrame({k: v["per_asal"] for k, v in RESULTS.items()})[res.index]
fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
ax[0].boxplot(per.values, vert=False, showmeans=True, meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5))
for o in AUG:
    ax[0].scatter(per.loc[o].values, np.arange(1, per.shape[1] + 1), color=C2, s=25, zorder=3)
ax[0].set_yticks(np.arange(1, per.shape[1] + 1), per.columns)
ax[0].scatter([], [], color=C2, s=25, label="asal 31 Agustus"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set(xlabel="macro F1 per backtest", title=f"Skor {len(EVAL)} backtest (berlian = rata-rata)")
best_l = res.index[-1]
gain = per[best_l] - per["SEDANG untuk semua"]
ax[1].bar(range(len(gain)), gain.values, color=[C2 if o.month == 8 else C1 for o in gain.index])
ax[1].axhline(0, color="black", lw=0.7)
ax[1].set_xticks(range(0, len(gain), 6), [o.strftime("%Y-%m") for o in gain.index[::6]], rotation=30)
ax[1].set(ylabel="selisih macro F1", title=f"{best_l}\\nvs SEDANG untuk semua, per backtest")
save_fig(FIG_DIR, "06_model_comparison")
res.style.format("{:.3f}")
"""
)

md("**Temuan.** " + T("weights"))

code(
    """
FINAL_FEATS = "kalender + kondisi"
W_FINAL, w_score = tune_weights([PROBS[FINAL_FEATS][o] for o in EVAL], [BT[o].kelas.values for o in EVAL])
print(f"Model akhir: LightGBM {FINAL_FEATS} | bobot (BAIK, SEDANG, TIDAK SEHAT) dari semua backtest: {W_FINAL} | macro F1 in-sample {w_score:.3f}")
"""
)

# ════════════════════════════════════════════════════════════════════ Interpretation
md("## 8. Interpretasi\n\n### 8.1 Ramalan vs kenyataan di backtest 31 Agustus 2024")

code(
    """
o0 = pd.Timestamp("2024-08-31")
b = BT[o0].copy()
b["pred"] = decide(PROBS[FINAL_FEATS][o0], W_LOO[(FINAL_FEATS, o0)])
fig, axes = plt.subplots(2, 1, figsize=(16, 5.5), sharex=True)
for a, col, title in [(axes[0], "kelas", "aktual"), (axes[1], "pred", "ramalan")]:
    piv = b.pivot_table(index="station", columns="tanggal", values=col)
    a.imshow(piv.values, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap([C3, C1, C2]), vmin=0, vmax=2, interpolation="nearest")
    a.set_yticks(range(len(piv)), piv.index); a.set_title(f"Asal 31 Agustus 2024: {title} (hijau = baik, biru = sedang, oranye = tidak sehat)")
    ticks = np.arange(0, piv.shape[1], 14)
    a.set_xticks(ticks, [d.strftime("%d %b") for d in piv.columns[ticks]])
save_fig(FIG_DIR, "07_timeline")
print(f"Macro F1 backtest ini: {macro_f1(b.kelas, b.pred):.3f}")
pd.DataFrame(confusion_matrix(b.kelas, b.pred, labels=[0, 1, 2]), index=[f"aktual {c}" for c in MODEL_CLASSES], columns=[f"ramal {c}" for c in MODEL_CLASSES])
"""
)

md("**Temuan.** " + T("timeline"))

md("### 8.2 Fitur apa yang dipakai, dan seberapa cepat ramalan memburuk?")

code(
    """
from src.models.forecast import lgb_model
F = FEATS[FINAL_FEATS]
m = lgb_model(0).fit(TABLE[F], TABLE.kelas)
imp = pd.Series(m.booster_.feature_importance("gain"), index=F)
imp = (imp / imp.sum()).sort_values()
rows = []
for o in EVAL:
    p = decide(PROBS[FINAL_FEATS][o], W_LOO[(FINAL_FEATS, o)])
    for lo, hi in [(1, 30), (31, 60), (61, 91)]:
        mk = BT[o].h.between(lo, hi).values
        if mk.sum() < 30:
            continue
        rows.append({"jarak": f"{lo}–{hi} hari", "model": macro_f1(BT[o].kelas.values[mk], p[mk]),
                     "SEDANG untuk semua": macro_f1(BT[o].kelas.values[mk], np.ones(mk.sum()))})
f1_h = pd.DataFrame(rows).groupby("jarak").mean()
fig, ax = plt.subplots(1, 2, figsize=(16, 4.3))
ax[0].barh(imp.index, imp.values, color=C1); pct_axis(ax[0], 0, axis="x"); ax[0].set(xlabel="porsi gain", title="Kepentingan fitur")
f1_h.plot.bar(ax=ax[1], color=[C2, CN], rot=0)
ax[1].set(ylabel=f"macro F1 rata-rata {len(EVAL)} backtest", title="Macro F1 menurut jarak ramalan")
save_fig(FIG_DIR, "08_importance_horizon")
f1_h.round(3)
"""
)

md("**Temuan.** " + T("importance"))

# ════════════════════════════════════════════════════════════════════ Submission
md("## 9. Submission\n\nModel dilatih dengan semua tanggal asal yang targetnya lewat sebelum 31 Agustus 2025, lalu meramal dari asal 31 Agustus 2025 dengan bobot akhir.")

code(
    """
X_final = origin_table(ispu, hol, FINAL_ORIGIN, HORIZON_DAYS, with_target=False)
trn = TABLE[TABLE.asal.isin(train_origins(ORIGINS, FINAL_ORIGIN, HORIZON_DAYS))]
P_final = fit_predict(trn, X_final, FEATS[FINAL_FEATS])
X_final["category"] = np.array(MODEL_CLASSES)[decide(P_final, W_FINAL)]
X_final["id"] = X_final.tanggal.dt.strftime("%Y-%m-%d") + "_" + X_final.station
sub = sample[["id"]].merge(X_final[["id", "category"]], on="id", how="left")
assert sub.category.notna().all() and len(sub) == len(sample)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
res.to_csv(OUT_DIR / "backtest_results.csv")
fig, ax = plt.subplots(figsize=(16, 2.8))
piv = X_final.assign(k=decide(P_final, W_FINAL)).pivot_table(index="station", columns="tanggal", values="k")
ax.imshow(piv.values, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap([C3, C1, C2]), vmin=0, vmax=2, interpolation="nearest")
ax.set_yticks(range(len(piv)), piv.index); ticks = np.arange(0, piv.shape[1], 14)
ax.set_xticks(ticks, [d.strftime("%d %b") for d in piv.columns[ticks]]); ax.set_title("Ramalan September–November 2025")
save_fig(FIG_DIR, "09_forecast")
sub.category.value_counts(normalize=True).to_frame("porsi").style.format("{:.1%}")
"""
)

md("**Temuan.** " + T("submission"))

md("## 10. Kesimpulan\n\n" + T("conclusion") + "\n\n### Jejak lingkungan")

code(
    """
import platform
pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib", "seaborn"]
print(" | ".join([f"python {platform.python_version()}"] + [f"{p} {version(p)}" for p in pkgs]))
print(f"Waktu eksekusi total: {(time.time() - T0) / 60:.1f} menit")
"""
)

nb["cells"] = cells
nb_path = ROOT / "kaggle-notebook" / NB_NAME
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, nb_path)
print("ditulis:", nb_path)

src_files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))}
runner = f'''"""Runner Kaggle untuk {NB_NAME} (di-generate oleh _build_notebook.py; jangan diedit manual)."""
import json, shutil, subprocess, sys
from pathlib import Path

SRC_FILES = {src_files!r}
NOTEBOOK = json.loads({nbf.writes(nb)!r})

proj = Path("/kaggle/working/project")
for rel, text in SRC_FILES.items():
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(proj / "notebooks").mkdir(parents=True, exist_ok=True)
nb_in = proj / "notebooks" / "{NB_NAME}"
nb_in.write_text(json.dumps(NOTEBOOK))
cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_in),
       "--output", "{NB_NAME}", "--output-dir", "/kaggle/working", "--ExecutePreprocessor.timeout=-1"]
ret = subprocess.run(cmd, cwd=proj / "notebooks")
shutil.rmtree(proj, ignore_errors=True)
sys.exit(ret.returncode)
'''
(ROOT / "kaggle").mkdir(exist_ok=True)
(ROOT / "kaggle" / "run_on_kaggle.py").write_text(runner)
print("ditulis:", ROOT / "kaggle" / "run_on_kaggle.py")
