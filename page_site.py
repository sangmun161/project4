# ==============================
# page_site.py (PART 1 / 2)
# ==============================
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# ------------------------------
# Optional deps
# ------------------------------
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

from ui_components import apply_plotly_card_style
from app_common import (
    PLOTLY_CFG,
    set_query_params_safe,
    normalize_columns,
    to_datetime_safe,
)

# ------------------------------
# Constants
# ------------------------------
TARGETS = {"O3": "o3_mean", "NO2": "no2_mean", "CO": "co_mean", "SO2": "so2_mean"}

MODEL_PATHS = {
    "O3": "models/model_o3.joblib",
    "NO2": "models/model_no2.joblib",
    "CO": "models/model_co.joblib",
    "SO2": "models/model_so2.joblib",
}

FUTURE_DATA_PATH_DEFAULT = "future_input_2024_01_01_to_07_all_sites_MODELREADY.csv"
FUTURE_DATA_PATTERNS = ["future_input_*MODELREADY*.csv", "future_input_*.csv"]

SPIKE_THRESHOLD = 0.30
P90_Q = 0.90


# ------------------------------
# Dummy model (fallback)
# ------------------------------
class DummySpikeModel:
    _is_dummy = True

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_name_ = feature_names or []

    def predict_proba(self, X):
        n = 0 if X is None else len(X)
        # always negative
        return np.column_stack([np.ones(n), np.zeros(n)])


def _get_model_features(model) -> List[str]:
    feats = getattr(model, "feature_name_", None)
    if feats is None:
        feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return []
    return list(feats)


def _resolve_existing_path(path_str: str, patterns: List[str]) -> Path:
    base_file = Path(__file__).resolve().parent
    search_dirs = [
        Path.cwd(),
        base_file,
        base_file.parent,
        base_file / "data",
        base_file.parent / "data",
        Path.cwd() / "data",
    ]

    if path_str:
        p = Path(path_str)
        if p.is_file():
            return p
        for d in search_dirs:
            cand = d / path_str
            if cand.is_file():
                return cand

    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            hits = sorted([p for p in d.glob(pat) if p.is_file()])
            if hits:
                return hits[0]

    raise FileNotFoundError(
        f"미래 입력 데이터(CSV)를 찾지 못했습니다. 입력='{path_str}'. "
        f"탐색 폴더={[str(x) for x in search_dirs]}"
    )


@st.cache_data(show_spinner=False)
def load_future(path_str: str) -> pd.DataFrame:
    p = _resolve_existing_path(path_str, FUTURE_DATA_PATTERNS)
    df = pd.read_csv(p)
    df = normalize_columns(df)
    if "date" not in df.columns or "site" not in df.columns:
        raise ValueError("미래 입력 CSV에 date, site 컬럼이 필요합니다.")
    df["date"] = to_datetime_safe(df["date"])
    df = df.dropna(subset=["date", "site"]).copy()
    return df.sort_values(["site", "date"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_spike_models_safe() -> Tuple[Dict[str, object], List[str]]:
    models: Dict[str, object] = {}
    errors: List[str] = []
    for k, path in MODEL_PATHS.items():
        try:
            models[k] = joblib.load(path)
        except Exception as e:
            models[k] = DummySpikeModel()
            errors.append(f"{k}: {type(e).__name__}")
    return models, errors


# ✅ SHAP 캐시 키는 model_key만 hash, 모델 객체는 해싱 제외
@st.cache_resource(show_spinner=False)
def get_shap_explainer(model_key: str, _model):
    if not _HAS_SHAP:
        return None
    if getattr(_model, "_is_dummy", False):
        return None
    try:
        return shap.TreeExplainer(_model)  # type: ignore
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_openai_client():
    if not _HAS_OPENAI:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.")
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)  # type: ignore


def get_openai_model_name() -> str:
    return st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")


def ensure_required_features(
    df_future_site: pd.DataFrame,
    df_site_hist: pd.DataFrame,
    required_features: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    미래 입력(df_future_site)에 모델이 요구하는 피처(required_features)가 없으면
    - 미래 입력 내 alias(대소문자/정규화)로 채우거나
    - 과거 df_site_hist의 마지막 유효값으로 상수 채우거나
    - 없으면 0으로 채움
    """
    out = df_future_site.copy()
    filled_info: Dict[str, str] = {}

    fut_map = {c.lower(): c for c in out.columns}
    hist_map = {c.lower(): c for c in df_site_hist.columns}

    def _last_nonnull_value(df_: pd.DataFrame, col_: str) -> float:
        s = pd.to_numeric(df_[col_], errors="coerce")
        if s.notna().any():
            return float(s.dropna().iloc[-1])
        return 0.0

    for feat in required_features:
        if feat in out.columns:
            continue

        key = feat.lower()

        if key in fut_map:
            src = fut_map[key]
            out[feat] = out[src]
            filled_info[feat] = f"alias_from_future:{src}"
            continue

        if key in hist_map:
            src = hist_map[key]
            out[feat] = _last_nonnull_value(df_site_hist, src)
            filled_info[feat] = f"constant_from_hist:{src}"
            continue

        out[feat] = 0.0
        filled_info[feat] = "default:0"

    # numeric + fillna
    for feat in required_features:
        out[feat] = pd.to_numeric(out[feat], errors="coerce")
        if out[feat].isna().all():
            out[feat] = 0.0
        else:
            out[feat] = out[feat].fillna(out[feat].median())

    return out, filled_info


WEATHER_FEATURE_MEANING = {
    "wind_speed": ("저풍속", "고풍속"),
    "pressure_pa": ("고기압 정체", "기압 혼합"),
    "temp_c": ("고온", "저온"),
}
# SHAP 조건 → 실제 기상 컬럼 매핑
SHAP_REASON_TO_COLUMN = {
    "풍속 조건": ("wind_speed", "풍속 (m/s)"),
    "기온 조건": ("temp_c", "기온 (°C)"),
    "기압 조건": ("pressure_pa", "기압 (Pa)"),
    "습도 조건": ("humidity", "습도 (%)"),
    "일사 조건": ("solar_radiation", "일사량"),
}



def explain_weather_keyword(model_key: str, model, X_row: pd.DataFrame) -> Optional[str]:
    """
    SHAP에서 |값| 큰 피처들 중 WEATHER_FEATURE_MEANING에 해당하는 게 있으면
    방향(+, -)에 따라 문구 반환
    """
    explainer = get_shap_explainer(model_key, model)
    if explainer is None:
        return None
    try:
        sv = explainer(X_row, check_additivity=False)
        ranked = sorted(
            zip(X_row.columns, sv.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for feat, val in ranked:
            fkey = str(feat).lower()
            if fkey in WEATHER_FEATURE_MEANING:
                return WEATHER_FEATURE_MEANING[fkey][0 if val > 0 else 1]
    except Exception:
        return None
    return None

# ===============================
# Event-level SHAP explanation
# ===============================
WEATHER_FEATURE_GROUPS = {
    "wind": "풍속 조건",
    "temp": "기온 조건",
    "pressure": "기압 조건",
    "humidity": "습도 조건",
    "solar": "일사 조건",
}
# ===============================
# Spatial context summarizer
# ===============================
def summarize_spatial_context(row: pd.Series) -> dict:
    # 도시화 수준
    if row["impervious_pct"] > 60:
        urban_level = "고도시화"
    elif row["impervious_pct"] > 40:
        urban_level = "중간 도시화"
    else:
        urban_level = "저도시화"

    # 도시 토지피복
    if row["urban_landcover_pct"] > 0.9:
        urban_cover = "도시 토지피복 지배"
    else:
        urban_cover = "혼합 토지피복"

    # 고도
    if row["elevation_mean"] > 500:
        elevation = "고지대"
    elif row["elevation_mean"] > 100:
        elevation = "중간 고도"
    else:
        elevation = "저지대"

    # 인공 구조 / 녹지 (컬럼 없으면 중립 처리)
    ndbi = row.get("NDBI_mean", np.nan)
    ndvi = row.get("NDVI_mean", np.nan)

    if pd.notna(ndbi):
        built_env = "인공 구조 우세" if ndbi > 0 else "자연·혼합 구조"
    else:
        built_env = "구조 정보 부족"

    if pd.notna(ndvi):
        green_env = "녹지 부족" if ndvi < 0.2 else "중간 이상 녹지"
    else:
        green_env = "녹지 정보 부족"

    return {
        "urban_level": urban_level,
        "urban_cover": urban_cover,
        "elevation": elevation,
        "built_env": built_env,
        "green_env": green_env,
    }


def explain_weather_keyword_event(
    model_key: str,
    model,
    X_all: pd.DataFrame,
    dates: pd.Series,
    center_date: pd.Timestamp,
    window: int = 1,
) -> Optional[str]:
    """
    하루가 아닌 이벤트(±window일) 기준 SHAP 평균으로
    '모델 판단에 기여한 기상 그룹'을 반환
    """
    explainer = get_shap_explainer(model_key, model)
    if explainer is None:
        return None

    mask = (
        (dates >= center_date - pd.Timedelta(days=window)) &
        (dates <= center_date + pd.Timedelta(days=window))
    )
    X_evt = X_all.loc[mask]

    if X_evt.empty:
        return None

    try:
        sv = explainer(X_evt, check_additivity=False)
        mean_shap = np.mean(sv.values, axis=0)
    except Exception:
        return None

    ranked = sorted(
        zip(X_evt.columns, mean_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for feat, _ in ranked:
        fkey = feat.lower()
        for group_key, group_name in WEATHER_FEATURE_GROUPS.items():
            if group_key in fkey:
                return group_name

    return None


def _top_right_quick_widgets(anchor_date: pd.Timestamp):
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        st.toggle("🌙", key="ui_dark_mode", help="다크/라이트 모드")
    with c2:
        st.toggle("🔕", key="mute_alerts", help="알림 음소거")
    with c3:
        if st.button("⟳", use_container_width=True, help="즉시 새로고침"):
            st.rerun()
    st.caption(f"🕒 {now} | 기준일: {anchor_date.date()}")

# ==============================
# page_site.py (PART 2 / 2)
# ==============================

def build_ts_figure(
    pred: pd.DataFrame,
    df_all: pd.DataFrame,
    y_col: str,
    anchor: pd.Timestamp,
    spike_days: List[pd.Timestamp],
    show_5months: bool,
    view_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> go.Figure:
    fig = go.Figure()
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])


    hist = pred[pred["date"] <= anchor]

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["y"],
            name="Observed",
            line=dict(color="gray"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred["date"],
            y=pred["yhat"],
            name="Prophet",
            line=dict(color="#1f77b4", dash="dash"),
        )
    )

    if y_col in df_all.columns:
        fig.add_hline(
            y=float(df_all[y_col].quantile(P90_Q)),
            line=dict(color="orange", dash="dot"),
        )

    for d in spike_days:
        d0 = pd.to_datetime(d)
        fig.add_vrect(
            x0=d0,
            x1=d0 + pd.Timedelta(days=1),
            fillcolor="rgba(255,0,0,0.25)",
            line_width=0,
        )

    if view_range is not None:
        x0, x1 = view_range
        fig.update_xaxes(range=[x0, x1])

    fig.update_layout(height=255, margin=dict(l=8, r=8, t=20, b=8), uirevision=True)
    return apply_plotly_card_style(fig)


REPORT_SCHEMA_KEYS = [
    ("요약", "executive_summary"),
    ("현재 리스크 상태", "current_risk_status"),
    ("핵심 원인", "key_drivers"),
    ("7일 전망", "seven_day_outlook"),
    ("권고 조치", "recommended_actions"),
    ("비고 및 한계", "notes_limitations"),
]


@st.cache_data(show_spinner=False)
def get_ts_figure_cached(
    site: str,
    y_col: str,
    show_5months: bool,
    view_start_iso: str,
    view_end_iso: str,
    anchor_iso: str,
    spike_days_iso: Tuple[str, ...],
):
    """
    Plotly figure 캐싱:
    - 해시 가능한 값(문자열/tuple)만 받는다.
    """
    pred = st.session_state["ALL_PROPHET"].get((site, y_col))
    if pred is None:
        return None

    df_all = st.session_state["DF_ALL"]
    anchor = pd.to_datetime(anchor_iso)
    view_range = (pd.to_datetime(view_start_iso), pd.to_datetime(view_end_iso))
    spike_days = [pd.to_datetime(x) for x in spike_days_iso]

    fig = build_ts_figure(
        pred=pred,
        df_all=df_all,
        y_col=y_col,
        anchor=anchor,
        spike_days=spike_days,
        show_5months=show_5months,
        view_range=view_range,
    )
    return fig


def build_llm_payload(
    site: str,
    anchor: pd.Timestamp,
    horizon: int,
    pollutant_summaries: List[dict],
    style_hint: str,
    spatial_context: dict,
) -> List[dict]:
    sys = (
        "너는 대기환경 운영관리자 보조 AI다. "
        "아래 입력을 기반으로 운영자용 보고서를 반드시 한국어로 작성하라. "
        "출력은 반드시 JSON 형식이어야 하며,"
        "모든 value는 한국어 문장으로 작성하라."
        "영어 사용은 고유명사(단위, 기호 등)를 제외하고 금지한다."
        "관측소의 공간적 특성(spatial_context)을 기상 요인과 함께 반드시 종합적으로 해석해라."
    )
    user = {
        "site": site,
        "anchor_date": str(anchor.date()),
        "horizon_days": horizon,
        "pollutant_summaries": pollutant_summaries,
        "spatial_context": spatial_context,
        "style_hint": style_hint,
        "required_json_keys": [k for _, k in REPORT_SCHEMA_KEYS],
    }
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def generate_report_from_llm(messages: List[dict]) -> str:
    client = get_openai_client()
    model = get_openai_model_name()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def safe_parse_report_json(text: str) -> Dict[str, str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {k: str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {"executive_summary": text}


def render_page2(
    df_all: pd.DataFrame,
    site: str,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    weather_fc,
    thr_config: dict,
    ALL_PROPHET: dict,
):
    # ------------------------------
    # CSS (원본 유지)
    # ------------------------------
    st.markdown(
        """
        <style>
        div.block-container { padding-top: 0.8rem; padding-bottom: 1.5rem; }
        section.main > div { padding-top: 0.8rem !important; }
        header { margin-bottom: 0rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 캐시 함수들이 접근할 수 있도록 저장
    st.session_state["DF_ALL"] = df_all
    st.session_state["ALL_PROPHET"] = ALL_PROPHET

    # ------------------------------
    # Top header (원본 유지)
    # ------------------------------
    P2_TOP_RIGHT_BTN_MARGIN_PX = 20

    h1, h2 = st.columns([8, 2], gap="small")
    with h1:
        P2_TITLE_FONT_PX = 70
        st.markdown(
            f"""
            <div style="
              font-size:{P2_TITLE_FONT_PX}px;
              font-weight:800;
              margin-top:0px;
              margin-bottom:0px;
            ">
                {site}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            f"<div style='margin-top:{P2_TOP_RIGHT_BTN_MARGIN_PX}px;'></div>",
            unsafe_allow_html=True,
        )
        _top_right_quick_widgets(anchor)

        if st.button("관측소 선택", use_container_width=True, key="btn_back_overview"):
            set_query_params_safe(page="overview")
            st.rerun()

    # ------------------------------
    # Data
    # ------------------------------
    df_site = df_all[df_all["site"].astype(str) == str(site)].sort_values("date").copy()
    # 공간 요약용: 관측소 대표 1행 (공간 변수는 시간에 따라 변하지 않음)
    if not df_site.empty and "site-cluster" in df_site.columns:
        site_cluster = str(df_site.iloc[-1]["site-cluster"])
    else:
        site_cluster = "moderate"

    if df_site.empty:
        spatial_context = {}
    else:
        spatial_row = df_site.iloc[-1]
        spatial_context = summarize_spatial_context(spatial_row)

    # ------------------------------
    # (A) SPIKE_DF 요약 (표시용) - 토글 없음, 항상 표시
    # ------------------------------
    spike_df = st.session_state.get("SPIKE_DF")
    site_spike_row = (
        spike_df[spike_df["site"].astype(str) == str(site)]
        if spike_df is not None
        else pd.DataFrame()
    )
    if not site_spike_row.empty:
        n_spike_summary = int(site_spike_row.iloc[0].get("spike_days", 0))
        priority_summary = str(site_spike_row.iloc[0].get("priority", "LOW"))
    else:
        n_spike_summary = 0
        priority_summary = "LOW"

    # ------------------------------
    # (B) 상세 스파이크/SHAP (미래입력+모델 있으면 자동, 없으면 graceful)
    # ✅ 사이드바 토글(spike_enabled) 절대 사용 안 함
    # ------------------------------
    df_future_site_filled = pd.DataFrame()
    models: Dict[str, object] = {k: DummySpikeModel() for k in TARGETS.keys()}
    spike_detail_available = False
    detail_warn: Optional[str] = None

    try:
        models, model_errors = load_spike_models_safe()
        df_future = load_future(st.session_state.get("future_input_path", FUTURE_DATA_PATH_DEFAULT))
        df_future_site = (
            df_future[df_future["site"].astype(str) == str(site)]
            .copy()
            .sort_values("date")
        )

        required_union: List[str] = []
        for m in models.values():
            required_union += _get_model_features(m)

        # unique preserve order
        seen = set()
        required_union = [x for x in required_union if not (x in seen or seen.add(x))]

        if not df_future_site.empty and required_union:
            df_future_site_filled, _ = ensure_required_features(df_future_site, df_site, required_union)
            spike_detail_available = True
        else:
            spike_detail_available = False
            detail_warn = "미래 입력 데이터(관측소)가 없거나, 모델 피처를 확인할 수 없어 원인 분석을 생략합니다."

        if model_errors:
            # 모델 일부만 실패해도, 성공한 모델은 쓸 수 있으니 warning만
            detail_warn = (detail_warn + " / " if detail_warn else "") + f"모델 일부 로드 실패: {', '.join(model_errors)}"

    except FileNotFoundError as e:
        spike_detail_available = False
        detail_warn = f"미래 입력 CSV가 없어 스파이크/원인 분석을 비활성화합니다. ({e})"
    except Exception as e:
        spike_detail_available = False
        detail_warn = f"미래 입력/모델 처리 실패 → 원인 분석 비활성화. ({type(e).__name__}: {e})"

    if detail_warn:
        st.caption(detail_warn)

    # ------------------------------
    # View controls
    # ------------------------------
    min_d = pd.to_datetime(df_site["date"].min())
    max_d_hist = pd.to_datetime(df_site["date"].max())

    # ✅ site의 prophet 예측 끝 날짜(없으면 과거 max)
    pred_end = max_d_hist
    for (_site, _y), _pred in ALL_PROPHET.items():
        if str(_site) == str(site):
            try:
                pred_end = max(pred_end, pd.to_datetime(_pred["date"].max()))
            except Exception:
                pass

    # ✅ 2페이지 첫 진입: 전체 기간(2018 ~ pred_end) 보이게
    default_start = min_d
    default_end = pred_end

    show_5months = st.toggle("최근 5개월 시계열 보기", value=False)

    # ✅ date_input 자체도 pred_end까지 선택 가능해야 함
    dr = st.date_input(
        "📆 날짜 범위 선택 (그래프 이동)",
        value=(default_start.date(), default_end.date()),
        min_value=min_d.date(),
        max_value=default_end.date(),  # ⭐ 여기 중요
    )

    if isinstance(dr, tuple) and len(dr) == 2:
        view_start = pd.to_datetime(dr[0])
        view_end = pd.to_datetime(dr[1])
    else:
        single = pd.to_datetime(dr)
        view_start = single - pd.Timedelta(days=7)
        view_end = single + pd.Timedelta(days=7)

    # ✅ clamp도 pred_end 기준으로 해야 미래가 안 잘림
    view_start = max(view_start, min_d)
    view_end = min(view_end, default_end)

    if view_start > view_end:
        view_start, view_end = view_end, view_start

    # ✅ 최근 5개월 토글: 과거 5개월 ~ pred_end(= 1/7까지)
    if show_5months:
        view_start = max(min_d, anchor - pd.DateOffset(months=5))
        view_end = default_end

    view_range = (pd.to_datetime(view_start), pd.to_datetime(view_end))

    # ------------------------------
    # 상단 요약(표시용): SPIKE_DF 기반
    # ------------------------------
    st.info(
        f"**요약:** 본 관측소는 현재 **{site_cluster.upper()}** 상태이며, "
        f"향후 {horizon}일 중 **{n_spike_summary}번** 스파이크 위험이 예측됩니다 "
        f"(우선순위: **{priority_summary}**)"
    )


    # ------------------------------
    # Main layout (원본 유지: 좌 2x2 / 우 AI 보고서)
    # ------------------------------
    left, right = st.columns([2, 1], gap="large")

    # ========== LEFT: 2x2 charts ==========
    with left:
        items = list(TARGETS.items())
        for row in [items[:2], items[2:]]:
            c1, c2 = st.columns(2, gap="medium")

            for (label, y_col), col in zip(row, [c1, c2]):
                with col:
                    st.markdown(f"**{label}**")

                    pred = ALL_PROPHET.get((str(site), y_col))
                    if pred is None:
                        st.info("예측 데이터 없음")
                        continue

                    # --- spike shading days
                    spike_days: List[pd.Timestamp] = []
                    top_risk_days: List[pd.Timestamp] = []  # ✅ 임계치 못 넘어도 “위험 상위” 제공
                    top_risk_probs: List[float] = []

                    if spike_detail_available:
                        model = models.get(label, DummySpikeModel())
                        feats = _get_model_features(model)

                        if feats and (not getattr(model, "_is_dummy", False)) and (not df_future_site_filled.empty):
                            X = df_future_site_filled[feats]
                            probs = model.predict_proba(X)[:, 1]

                            # 임계치 넘는 날(스파이크)
                            spike_days = df_future_site_filled.loc[probs >= SPIKE_THRESHOLD, "date"].tolist()

                            # ✅ 임계치 미만이어도 “위험 상위 3일” 뽑기 (원인분석 비어있지 않게)
                            order = np.argsort(-probs)  # desc
                            k = min(3, len(order))
                            if k > 0:
                                idxs = order[:k]
                                top_risk_days = [pd.to_datetime(df_future_site_filled.iloc[i]["date"]) for i in idxs]
                                top_risk_probs = [float(probs[i]) for i in idxs]

                    fig = get_ts_figure_cached(
                        site=str(site),
                        y_col=y_col,
                        show_5months=show_5months,
                        view_start_iso=str(view_range[0]),
                        view_end_iso=str(view_range[1]),
                        anchor_iso=str(anchor),
                        spike_days_iso=tuple([str(pd.to_datetime(d)) for d in spike_days]),
                    )

                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
                    else:
                        fig2 = build_ts_figure(
                            pred=pred,
                            df_all=df_all,
                            y_col=y_col,
                            anchor=anchor,
                            spike_days=spike_days,
                            show_5months=show_5months,
                            view_range=view_range,
                        )
                        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)

                    # --- Cause analysis
                    with st.expander("🔍 모델 판단에 기여한 요인", expanded=False):
                        if not spike_detail_available:
                            st.caption("미래 입력/모델이 없어서 원인 분석이 비활성화되었습니다.")
                        else:
                            model = models.get(label, DummySpikeModel())
                            feats = _get_model_features(model)

                            if not feats or getattr(model, "_is_dummy", False):
                                st.caption("모델 피처를 확인할 수 없어 원인 분석이 불가합니다.")
                            else:
                                X_all = df_future_site_filled[feats]
                                date_series = df_future_site_filled["date"]

                                # 1) 스파이크 발생일 기준
                                if spike_days:
                                    st.markdown("**스파이크 발생 구간 기준(이벤트) 판단 요인:**")
                                    for d in spike_days[:3]:
                                        reason = explain_weather_keyword_event(
                                            model_key=label,
                                            model=model,
                                            X_all=X_all,
                                            dates=date_series,
                                            center_date=pd.to_datetime(d),
                                            window=1,
                                        )
                                        st.markdown(
                                            f"- **{pd.to_datetime(d).strftime('%Y-%m-%d')}**: "
                                            f"{reason or '기상 요인 영향 미약'}"
                                        )
                                        # ===============================
                                        # 📈 SHAP 결과 기반 미래 기상 그래프 (7일)
                                        # ===============================
                                        if reason in SHAP_REASON_TO_COLUMN:
                                            col, label_kr = SHAP_REASON_TO_COLUMN[reason]

                                            if col in df_future_site_filled.columns:
                                                with st.expander(f"📈 {label_kr} 7일 예보", expanded=False):
                                                    df_view = (
                                                        df_future_site_filled
                                                        .set_index("date")[[col]]
                                                        .rename(columns={col: label_kr})
                                                    )

                                                    st.line_chart(df_view)
                                            else:
                                                st.caption(f"{label_kr} 데이터가 없습니다.")

                                # 2) 스파이크 없으면 위험 상위일 기준
                                else:
                                    if not top_risk_days:
                                        st.markdown("스파이크 위험이 매우 낮거나 예측 확률을 계산할 수 없습니다.")
                                    else:
                                        st.markdown("**스파이크는 없으나, 위험 상위 구간 기준 판단 요인:**")
                                        for d, p in zip(top_risk_days, top_risk_probs):
                                            reason = explain_weather_keyword_event(
                                                model_key=label,
                                                model=model,
                                                X_all=X_all,
                                                dates=date_series,
                                                center_date=pd.to_datetime(d),
                                                window=1,
                                            )
                                            st.markdown(
                                                f"- **{pd.to_datetime(d).strftime('%Y-%m-%d')}** "
                                                f"(prob={p:.3f}): {reason or '기상 요인 영향 미약'}"
                                            )


    # ========== RIGHT: AI report (원본 유지) ==========
    with right:
        st.markdown("### 🧠 AI 최종 보고서")
        if not _HAS_OPENAI:
            st.warning("openai 패키지가 없어 보고서 기능이 비활성화됩니다. (대시보드 핵심 기능에는 영향 없음)")

        report_style = st.selectbox("보고서 톤", ["운영자용(간결)", "팀 발표용(조금 자세히)"], index=0)
        style_hint = (
            "각 섹션은 3~6줄 내로 간결하게. 의사결정에 필요한 숫자/날짜 우선."
            if report_style == "운영자용(간결)"
            else "각 섹션은 5~10줄. 근거(날짜/확률/원인)를 1문장 더 포함."
        )

        if "site_report_json" not in st.session_state:
            st.session_state["site_report_json"] = None  # type: ignore

        if st.button("🧠 보고서 생성", use_container_width=True):
            pollutant_summaries: List[dict] = []

            for label in TARGETS:
                model = models.get(label, DummySpikeModel())
                feats = _get_model_features(model)

                if spike_detail_available and feats and (not getattr(model, "_is_dummy", False)) and (not df_future_site_filled.empty):
                    X = df_future_site_filled[feats]
                    probs = model.predict_proba(X)[:, 1]
                    order = np.argsort(-probs)
                    k = min(3, len(order))
                    top_idxs = order[:k] if k > 0 else []

                    spike_mask = probs >= SPIKE_THRESHOLD
                    spike_dates = df_future_site_filled.loc[spike_mask, "date"]

                    examples = []
                    for i in top_idxs:
                        d = pd.to_datetime(df_future_site_filled.iloc[i]["date"])
                        X_row = X[df_future_site_filled["date"] == d]
                        reason = explain_weather_keyword_event(
                            model_key = label,
                            model = model,
                            X_all = X,
                            dates = df_future_site_filled["date"],
                            center_date=d,
                            window=1) if not X_row.empty else None
                        examples.append(f"{d:%Y-%m-%d} (prob={float(probs[i]):.3f}) - driver={reason or 'N/A'}")

                    pollutant_summaries.append(
                        {
                            "label": label,
                            "n_spike": int(spike_mask.sum()),
                            "max_prob": float(np.max(probs)) if len(probs) else 0.0,
                            "mean_prob": float(np.mean(probs)) if len(probs) else 0.0,
                            "spike_examples": examples,
                        }
                    )
                else:
                    pollutant_summaries.append(
                        {
                            "label": label,
                            "n_spike": 0,
                            "max_prob": 0.0,
                            "mean_prob": 0.0,
                            "spike_examples": [],
                        }
                    )

            messages = build_llm_payload(site, anchor, horizon, pollutant_summaries, style_hint, spatial_context)
            try:
                with st.spinner("GPT가 보고서를 작성 중입니다..."):
                    report_text = generate_report_from_llm(messages)
                st.session_state["site_report_json"] = safe_parse_report_json(report_text)
            except Exception as e:
                st.error(f"보고서 생성 실패: {e}")

        report_json: Optional[Dict[str, str]] = st.session_state.get("site_report_json")
        if report_json:
            with st.expander("📌 보고서 요약", expanded=True):
                for idx, (title_kr, key) in enumerate(REPORT_SCHEMA_KEYS, start=1):
                    st.markdown(f"### {idx}. {title_kr}")
                    content = (report_json.get(key) or "").strip()
                    st.markdown(content if content else "_(내용 없음)_")
        else:
            st.caption("버튼을 누르면 이 칸에 보고서가 생성됩니다.")
