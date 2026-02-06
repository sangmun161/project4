# app_common.py
import os
import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prophet
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# Slack
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# =========================
# Constants
# =========================
BASE_DIR = Path(__file__).resolve().parent

def _collect_csv_candidates(patterns: List[str], search_dirs: List[Path]) -> List[Path]:
    """프로젝트 내 CSV 후보를 자동 탐색합니다."""
    found: List[Path] = []
    for d in search_dirs:
        if not d or not d.exists():
            continue
        for pat in patterns:
            found.extend(d.glob(pat))
    # 중복 제거 + 파일만
    uniq = []
    seen = set()
    for p in found:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        if rp.is_file():
            uniq.append(rp)
            seen.add(rp)
    # 최신 수정 파일 우선
    uniq.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    return uniq

def _relpath_str(p: Path) -> str:
    """가능하면 CWD 기준 상대경로로 표시(대시보드 입력값/표시용)."""
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)

def list_available_pollution_files() -> List[str]:
    """sidebar 선택용: 자동 탐지된 대기 데이터 후보 리스트"""
    search_dirs = [Path.cwd(), BASE_DIR, Path.cwd() / "data", BASE_DIR / "data"]
    patterns = ["pollution_2018_2023_*.parquet", "pollution_*.parquet", "pollution_2018_2023_*.csv", "pollution_*.csv"]
    cands = _collect_csv_candidates(patterns, search_dirs)
    return [_relpath_str(p) for p in cands]

def _guess_default_pollution_path() -> str:
    cands = list_available_pollution_files()
    if cands:
        return cands[0]
    # 최소 fallback(없으면 오류 메시지에서 안내됨)
    return "pollution_2018_2023_3.csv"

DEFAULT_DATA_PATH = os.getenv("POLLUTION_DATA_PATH", _guess_default_pollution_path())
DEFAULT_WEATHER_FC_PATH = ""  # optional

AQI_COLS = ["o3_aqi", "no2_aqi", "co_aqi", "so2_aqi"]
MEAN_COLS = ["o3_mean", "no2_mean", "co_mean", "so2_mean"]
TARGET_OPTIONS = MEAN_COLS

# =========================
# Site Cluster (KMeans = 3)
# =========================

SITE_CLUSTER_LABELS_3 = {
    0: "safe",
    1: "moderate",
    2: "high-risk",
}

CLUSTER_COLORS = {
    "safe": "#2ca02c",
    "moderate": "#ff7f0e",
    "high-risk": "#d62728",
    "Unknown": "#7f7f7f",
}


STATE_RISK_ICON = {"None": "", "Medium": "⚠", "High": "❗"}
SPIKE_RISK_ICON = {"None": "—", "Watch": "🟡 Watch", "Warn": "🔴 Warn"}


SPIKE_ICON_OFFSET = {
    "Watch": (0.05, -0.05),
    "Warn":  (0.06,  0.06),
}

ALERT_STATE_PATH = Path(".cache") / "slack_alert_state.json"
ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": True, "responsive": True}


# =========================
# Helpers
# =========================
def resolve_existing_path(
    path: str, *,
    kind: str = "파일",
    patterns: Optional[List[str]] = None,
    search_dirs: Optional[List[Path]] = None,
) -> Path:
    """상대경로/절대경로 모두 지원하며, 존재하는 경로로 해석합니다.
    존재하지 않으면 후보 파일 목록과 함께 FileNotFoundError를 발생시킵니다.
    """
    if path is None:
        raise FileNotFoundError(f"{kind} 경로가 비어 있습니다.")
    raw = os.path.expandvars(os.path.expanduser(str(path).strip()))
    p = Path(raw)

    # 1) 그대로(절대/상대) 체크
    if p.exists():
        return p

    # 2) 상대경로면 기준 디렉토리들을 붙여가며 탐색
    if not p.is_absolute():
        base_dirs = search_dirs or [Path.cwd(), BASE_DIR, Path.cwd() / "data", BASE_DIR / "data"]
        for bd in base_dirs:
            cand = bd / p
            if cand.exists():
                return cand

    # 3) 후보 파일 목록 생성
    pats = patterns or ["*.csv"]
    base_dirs = search_dirs or [Path.cwd(), BASE_DIR, Path.cwd() / "data", BASE_DIR / "data"]
    cands = _collect_csv_candidates(pats, base_dirs)
    cand_preview = "\n".join([f" - {_relpath_str(x)}" for x in cands[:20]]) if cands else " (후보 없음)"
    dirs_preview = ", ".join(str(d) for d in base_dirs)

    msg = (
        f"{kind}을(를) 찾지 못했습니다. 입력값: {path}\n"
        f"확인 위치: {dirs_preview}\n"
        f"자동 탐지 후보(상위 20개):\n{cand_preview}\n\n"
        "조치: (1) 파일을 프로젝트 폴더에 두거나 (2) sidebar의 경로를 실제 파일명으로 수정하세요."
    )
    raise FileNotFoundError(msg)

def safe_get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    if "date" not in df.columns:
        for cand in ["day", "datetime", "timestamp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    if "site" not in df.columns:
        for cand in ["address", "site_name", "station", "station_name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "site"})
                break
    return df

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def parse_point_geometry(val: str) -> Tuple[Optional[float], Optional[float]]:
    if pd.isna(val):
        return (None, None)
    try:
        obj = json.loads(val)
        if obj.get("type") != "Point":
            return (None, None)
        lon, lat = obj.get("coordinates", [None, None])
        return (float(lat), float(lon))
    except Exception:
        return (None, None)

def coerce_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == "object":
            out[c] = (
                out[c].astype(str)
                .str.replace(",", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    u = pd.to_numeric(u, errors="coerce")
    v = pd.to_numeric(v, errors="coerce")
    return np.sqrt(u * u + v * v)

def fmt_date(d) -> str:
    if pd.isna(d):
        return "-"
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def get_query_params() -> Dict[str, List[str]]:
    try:
        return {k: [v] if isinstance(v, str) else list(v) for k, v in dict(st.query_params).items()}
    except Exception:
        return st.experimental_get_query_params()

def set_query_params_safe(**kwargs):
    qp = get_query_params()
    new_qp = {k: ([v] if isinstance(v, str) else list(v)) for k, v in kwargs.items()}

    def _norm(d):
        out = {}
        for k, v in d.items():
            if v is None:
                out[k] = [""]
            elif isinstance(v, list):
                out[k] = [str(x) for x in v]
            else:
                out[k] = [str(v)]
        return out

    if _norm(qp) == _norm(new_qp):
        return

    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)


# =========================
# Load & Preprocess
# =========================
@st.cache_data(show_spinner=False)
def load_pollution_data(path: str) -> pd.DataFrame:
    p = resolve_existing_path(
        path,
        kind="대기 데이터(CSV)",
        patterns=["pollution_2018_2023_*.csv", "pollution_*.csv", "pollution_2018_2023_*.parquet", "pollution_*.parquet"],
    )
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df = normalize_columns(df)

    if "date" not in df.columns or "site" not in df.columns:
        raise ValueError("필수 컬럼(date, site)을 찾지 못했습니다. 원본 컬럼명을 확인하세요.")

    df["date"] = to_datetime_safe(df["date"])
    df = df.dropna(subset=["date", "site"]).copy()

    if "geometry" in df.columns:
        latlon = df["geometry"].apply(parse_point_geometry)
        df["lat"] = [t[0] for t in latlon]
        df["lon"] = [t[1] for t in latlon]
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    exclude = ["site", "date", "geometry", "state", "county", "city", "region_name"]
    df = coerce_numeric_columns(df, exclude=exclude)

    if "met_wind_u" in df.columns and "met_wind_v" in df.columns:
        df["wind_speed"] = wind_speed(df["met_wind_u"], df["met_wind_v"])

    # site-date 중복 제거
    key_cols = ["site", "date"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in key_cols]
    other_cols = [c for c in df.columns if c not in key_cols + numeric_cols]
    agg = {**{c: "mean" for c in numeric_cols}, **{c: "first" for c in other_cols}}
    df = df.groupby(key_cols, as_index=False).agg(agg)

    return df.sort_values(["site", "date"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_weather_forecast(path: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = resolve_existing_path(
        path,
        kind="미래 기상 예보(CSV)",
        patterns=["*weather*.csv", "*ERA5*.csv", "*.csv"],
    )
    w = pd.read_csv(p)
    w = normalize_columns(w)

    if "date" not in w.columns or "site" not in w.columns:
        raise ValueError("기상 파일에 date, site 컬럼이 필요합니다.")

    w["date"] = to_datetime_safe(w["date"])
    w = w.dropna(subset=["date", "site"]).copy()

    exclude = ["site", "date", "geometry", "state", "county", "city", "region_name"]
    w = coerce_numeric_columns(w, exclude=exclude)

    if "met_wind_u" in w.columns and "met_wind_v" in w.columns:
        w["wind_speed"] = wind_speed(w["met_wind_u"], w["met_wind_v"])

    key_cols = ["site", "date"]
    numeric_cols = w.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in key_cols]
    other_cols = [c for c in w.columns if c not in key_cols + numeric_cols]
    agg = {**{c: "mean" for c in numeric_cols}, **{c: "first" for c in other_cols}}
    w = w.groupby(key_cols, as_index=False).agg(agg)

    return w.sort_values(["site", "date"]).reset_index(drop=True)


# =========================
# Clustering
# =========================
@st.cache_resource(show_spinner=False)
def fit_day_cluster(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    work = df.dropna(subset=AQI_COLS).copy()
    if work.empty:
        raise ValueError(f"AQI 컬럼({AQI_COLS}) 결측으로 day-cluster를 학습할 수 없습니다.")
    X = work[AQI_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Xs)
    return km, scaler

@st.cache_resource(show_spinner=False)
def fit_site_cluster(
    df: pd.DataFrame,
    _day_km: KMeans,
    _day_scaler: StandardScaler,
    n_clusters: int = 3,
    random_state: int = 42,
):

    work = df.dropna(subset=AQI_COLS).copy()
    X_day = _day_scaler.transform(work[AQI_COLS].values)
    work["day_cluster"] = _day_km.predict(X_day)
    work["total_aqi"] = work[AQI_COLS].sum(axis=1)

    site_features = (
        work.groupby("site")
        .agg(
            mean_total_aqi=("total_aqi", "mean"),
            std_total_aqi=("total_aqi", "std"),
            **{f"pct_day_cluster_{i}": ("day_cluster", lambda x, i=i: (x == i).mean()) for i in range(_day_km.n_clusters)},
        )
        .reset_index()
        .fillna(0)
    )

    feat_cols = [c for c in site_features.columns if c != "site"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(site_features[feat_cols].values)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Xs)
    return km, scaler

@st.cache_data(show_spinner=False)
def attach_site_clusters(df: pd.DataFrame) -> pd.DataFrame:
    day_km, day_scaler = fit_day_cluster(df)
    site_km, site_scaler = fit_site_cluster(df, day_km, day_scaler)

    work = df.dropna(subset=AQI_COLS).copy()

    # day-cluster
    X_day = day_scaler.transform(work[AQI_COLS].values)
    work["day_cluster"] = day_km.predict(X_day)
    work["total_aqi"] = work[AQI_COLS].sum(axis=1)

    site_feat = (
        work.groupby("site")
        .agg(
            mean_total_aqi=("total_aqi", "mean"),
            std_total_aqi=("total_aqi", "std"),
            **{
                f"pct_day_cluster_{i}": ("day_cluster", lambda x, i=i: (x == i).mean())
                for i in range(day_km.n_clusters)
            },
        )
        .reset_index()
        .fillna(0)
    )

    feat_cols = [c for c in site_feat.columns if c != "site"]
    Xs = site_scaler.transform(site_feat[feat_cols].values)

    site_feat["cluster_k3"] = site_km.predict(Xs).astype(int)
    site_feat["site_cluster"] = (
        site_feat["cluster_k3"].map(SITE_CLUSTER_LABELS_3).fillna("moderate")
    )

    # ✅ 여기부터가 핵심: 기존 컬럼 충돌 방지
    out = df.copy()

    if "site_cluster" in out.columns:
        out = out.rename(columns={"site_cluster": "site_cluster_src"})  # 원본 보존
    if "cluster_k3" in out.columns:
        out = out.rename(columns={"cluster_k3": "cluster_k3_src"})

    return out.merge(
        site_feat[["site", "cluster_k3", "site_cluster"]],
        on="site",
        how="left",
    )



# =========================
# Threshold Policy
# =========================
@st.cache_data(show_spinner=False)
def compute_threshold_tables(df: pd.DataFrame, target: str, site_q: float, season_q: float):
    hist = df.dropna(subset=[target]).copy()
    if hist.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=["site", "month", "thr_season"])
    hist["month"] = hist["date"].dt.month.astype(int)
    site_thr = hist.groupby("site")[target].quantile(site_q)
    season_thr = hist.groupby(["site", "month"])[target].quantile(season_q).rename("thr_season").reset_index()
    return site_thr, season_thr

def threshold_for(
    site: str,
    date: pd.Timestamp,
    target: str,
    fixed_value: float,
    use_fixed: bool,
    site_thr: pd.Series,
    use_site: bool,
    season_thr_df: pd.DataFrame,
    use_season: bool,
) -> float:
    vals = []
    if use_fixed and np.isfinite(fixed_value):
        vals.append(float(fixed_value))
    if use_site and site in site_thr.index and np.isfinite(site_thr.loc[site]):
        vals.append(float(site_thr.loc[site]))
    if use_season and not season_thr_df.empty:
        m = int(pd.to_datetime(date).month)
        hit = season_thr_df[(season_thr_df["site"] == site) & (season_thr_df["month"] == m)]
        if not hit.empty and np.isfinite(hit["thr_season"].iloc[0]):
            vals.append(float(hit["thr_season"].iloc[0]))
    return float(np.max(vals)) if vals else np.nan

def calc_state_risk_today(y_today: float, thr_today: float, medium_ratio: float = 0.90) -> str:
    if not np.isfinite(y_today) or not np.isfinite(thr_today):
        return "None"
    if y_today >= thr_today:
        return "High"
    if y_today >= (thr_today * medium_ratio):
        return "Medium"
    return "None"


# =========================
# Prophet Forecast + Spike Risk
# =========================
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def fill_exog_climatology(df_site: pd.DataFrame, exog_cols: List[str]) -> pd.DataFrame:
    if not exog_cols:
        return df_site
    out = df_site.copy()
    out["doy"] = out["date"].dt.dayofyear
    for c in exog_cols:
        if c not in out.columns:
            out[c] = np.nan
        clim = out.groupby("doy")[c].mean()
        out[c] = out[c].fillna(out["doy"].map(clim))
        out[c] = out[c].fillna(out[c].mean())
    return out.drop(columns=["doy"])

@st.cache_data(show_spinner=False)
def prophet_predict_site(
    df_site: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    weather_fc: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if not _HAS_PROPHET:
        raise RuntimeError("Prophet 미설치: pip install prophet")

    exog_candidates = ["temp_c", "pressure_pa", "met_rain_mm", "wind_speed", "ndbi_mean", "ndvi_mean"]
    exog_cols = [c for c in exog_candidates if c in df_site.columns]

    hist = df_site[df_site["date"] <= anchor].copy()
    hist = hist.dropna(subset=[target]).copy()
    if hist["date"].nunique() < 60:
        return pd.DataFrame(columns=["date", "y", "yhat", "yhat_lower", "yhat_upper"])

    hist = ensure_columns(hist, exog_cols)
    hist = fill_exog_climatology(hist, exog_cols)

    future_dates = pd.date_range(anchor + pd.Timedelta(days=1), anchor + pd.Timedelta(days=horizon), freq="D")
    fut = pd.DataFrame({"date": future_dates})
    fut = ensure_columns(fut, exog_cols)

    if weather_fc is not None and not weather_fc.empty:
        site_key = str(df_site["site"].iloc[0])
        wf = weather_fc[weather_fc["site"].astype(str) == site_key].copy()
        wf = wf[wf["date"].isin(future_dates)].copy()
        if not wf.empty:
            keep = ["date"] + [c for c in exog_cols if c in wf.columns]
            fut = fut.merge(wf[keep], on="date", how="left", suffixes=("", "_wf"))
            for c in exog_cols:
                if f"{c}_wf" in fut.columns:
                    fut[c] = fut[c].combine_first(fut[f"{c}_wf"])
                    fut = fut.drop(columns=[f"{c}_wf"])

    # 공간지표 고정값 유지
    for c in ["ndbi_mean", "ndvi_mean"]:
        if c in exog_cols:
            const_val = pd.to_numeric(hist[c], errors="coerce").dropna()
            const_val = float(const_val.iloc[-1]) if not const_val.empty else np.nan
            fut[c] = fut[c].fillna(const_val)

    # 남은 exog 결측은 climatology로 채움(hist+fut)
    if exog_cols:
        tmp = pd.concat([hist[["date"] + exog_cols], fut[["date"] + exog_cols]], ignore_index=True)
        tmp = fill_exog_climatology(tmp, exog_cols)
        fut[exog_cols] = tmp.iloc[-len(fut):][exog_cols].values

    train_cols = ["ds", "y"] + exog_cols
    train = hist.rename(columns={"date": "ds", target: "y"})[train_cols]

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    for c in exog_cols:
        m.add_regressor(c)
    m.fit(train)

    all_dates = pd.date_range(hist["date"].min(), anchor + pd.Timedelta(days=horizon), freq="D")
    base = pd.DataFrame({"date": all_dates})
    merge_cols = ["date", target] + exog_cols
    base = base.merge(hist[merge_cols], on="date", how="left")
    base = ensure_columns(base, exog_cols)
    base = fill_exog_climatology(base, exog_cols)

    if not fut.empty and exog_cols:
        base = base.merge(fut[["date"] + exog_cols], on="date", how="left", suffixes=("", "_fut"))
        for c in exog_cols:
            if f"{c}_fut" in base.columns:
                base.loc[base["date"] > anchor, c] = base.loc[base["date"] > anchor, f"{c}_fut"]
                base = base.drop(columns=[f"{c}_fut"])

    pred_in = base.rename(columns={"date": "ds"})[["ds"] + exog_cols]
    fc = m.predict(pred_in)

    out = pd.DataFrame({
        "date": pd.to_datetime(pred_in["ds"]),
        "y": pd.to_numeric(base[target], errors="coerce"),
        "yhat": fc["yhat"].values,
        "yhat_lower": fc["yhat_lower"].values,
        "yhat_upper": fc["yhat_upper"].values,
    })
    return out

@st.cache_data(show_spinner=False)
def compute_spike_risk_all_sites(
    df: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    fixed_value: float,
    use_fixed: bool,
    use_site: bool,
    site_q: float,
    use_season: bool,
    season_q: float,
    warn_days: int,
    watch_days: int,
    weather_fc: Optional[pd.DataFrame],
    medium_ratio: float,
) -> pd.DataFrame:
    site_thr, season_thr_df = compute_threshold_tables(df, target=target, site_q=site_q, season_q=season_q)
    sites = df["site"].astype(str).unique().tolist()
    rows = []

    for site in sites:
        df_site = df[df["site"].astype(str) == str(site)].copy().sort_values("date")

        today_row = df_site[df_site["date"] == anchor]
        if today_row.empty:
            today_row = df_site[df_site["date"] <= anchor].tail(1)
        if today_row.empty:
            continue

        y_today = float(pd.to_numeric(today_row[target].iloc[0], errors="coerce")) if target in today_row.columns else np.nan
        thr_today = threshold_for(site, anchor, target, fixed_value, use_fixed, site_thr, use_site, season_thr_df, use_season)
        state_risk = calc_state_risk_today(y_today, thr_today, medium_ratio=medium_ratio)

        spike_level = "None"
        exceed_days = 0
        max_upper = np.nan
        max_thr = np.nan
        exceed_tplus = ""
        exceed_first_tplus = np.nan

        try:
            pred = prophet_predict_site(df_site, target, anchor, horizon, interval_width, weather_fc)
            if not pred.empty:
                fut = pred[(pred["date"] > anchor) & (pred["date"] <= anchor + pd.Timedelta(days=horizon))].copy()
                if not fut.empty:
                    fut["thr"] = [
                        threshold_for(site, d_, target, fixed_value, use_fixed, site_thr, use_site, season_thr_df, use_season)
                        for d_ in fut["date"]
                    ]
                    fut["exceed"] = (fut["yhat_upper"] > fut["thr"])
                    exceed_rows = fut[fut["exceed"] == True].copy()
                    exceed_days = int(exceed_rows.shape[0])

                    if exceed_days > 0:
                        offsets = sorted({int((d - anchor).days) for d in exceed_rows["date"].tolist()})
                        exceed_first_tplus = float(offsets[0]) if offsets else np.nan
                        exceed_tplus = ",".join([f"t+{o}" for o in offsets[:7]])

                    max_upper = float(np.nanmax(fut["yhat_upper"].values))
                    max_thr = float(np.nanmax(fut["thr"].values))

                    if exceed_days >= warn_days:
                        spike_level = "Warn"
                    elif exceed_days >= watch_days:
                        spike_level = "Watch"
                    else:
                        spike_level = "None"
        except Exception:
            spike_level = "None"

        lat = float(pd.to_numeric(today_row["lat"].iloc[0], errors="coerce")) if "lat" in today_row.columns else np.nan
        lon = float(pd.to_numeric(today_row["lon"].iloc[0], errors="coerce")) if "lon" in today_row.columns else np.nan
        state = str(today_row["state"].iloc[0]) if "state" in today_row.columns and pd.notna(today_row["state"].iloc[0]) else ""
        county = str(today_row["county"].iloc[0]) if "county" in today_row.columns and pd.notna(today_row["county"].iloc[0]) else ""
        city = str(today_row["city"].iloc[0]) if "city" in today_row.columns and pd.notna(today_row["city"].iloc[0]) else ""
        cluster_3 = (
            str(today_row["site_cluster"].iloc[0]).lower().strip()
            if "site_cluster" in today_row.columns and pd.notna(today_row["site_cluster"].iloc[0])
            else "moderate"
        )


        margin = (max_upper - max_thr) if (np.isfinite(max_upper) and np.isfinite(max_thr)) else np.nan

        rows.append({
            "site": site,
            "site_cluster": cluster_3,
            "lat": lat,
            "lon": lon,
            "state": state,
            "county": county,
            "city": city,
            "today_y": y_today,
            "today_thr": thr_today,
            "state_risk": state_risk,
            "spike_exceed_days": exceed_days,
            "spike_risk": spike_level,
            "max_yhat_upper_7d": max_upper,
            "max_thr_7d": max_thr,
            "exceed_tplus": exceed_tplus,
            "exceed_first_tplus": exceed_first_tplus,
            "exceed_margin": margin,
        })

    return pd.DataFrame(rows)


# =========================
# Slack Alert (Cooldown)
# =========================
def load_alert_state() -> Dict[str, str]:
    if ALERT_STATE_PATH.exists():
        try:
            return json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_alert_state(state: Dict[str, str]):
    ALERT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def can_send(site: str, alert_key: str, cooldown_hours: int, state: Dict[str, str]) -> bool:
    k = f"{site}::{alert_key}"
    last = state.get(k)
    if not last:
        return True
    try:
        last_dt = dt.datetime.fromisoformat(last)
    except Exception:
        return True
    return (dt.datetime.now() - last_dt) >= dt.timedelta(hours=cooldown_hours)

def mark_sent(site: str, alert_key: str, state: Dict[str, str]):
    k = f"{site}::{alert_key}"
    state[k] = dt.datetime.now().isoformat(timespec="seconds")

def send_slack(webhook_url: str, text: str):
    if not _HAS_REQUESTS:
        return False, "requests 미설치(pip install requests)"
    if not webhook_url:
        return False, "Slack webhook URL이 비어있습니다."
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        if 200 <= r.status_code < 300:
            return True, "OK"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)

def has_prophet() -> bool:
    return _HAS_PROPHET

def has_requests() -> bool:
    return _HAS_REQUESTS

# app_common.py 맨 아래쪽에 추가

@st.cache_data(show_spinner=True)
def precompute_all_prophet(
    df_all: pd.DataFrame,
    targets: list,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    weather_fc: Optional[pd.DataFrame],
):
    """
    모든 관측소 × 모든 target에 대해 Prophet 예측을 미리 수행
    """
    out = {}

    sites = df_all["site"].astype(str).unique().tolist()

    for site in sites:
        df_site = (
            df_all[df_all["site"].astype(str) == site]
            .copy()
            .sort_values("date")
        )

        for target in targets:
            try:
                pred = prophet_predict_site(
                    df_site=df_site,
                    target=target,
                    anchor=anchor,
                    horizon=horizon,
                    interval_width=interval_width,
                    weather_fc=weather_fc,
                )
            except Exception:
                pred = pd.DataFrame()

            out[(site, target)] = pred

    return out