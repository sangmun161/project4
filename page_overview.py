import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

from ui_components import card_end, kpi_card
from app_common import (
    CLUSTER_COLORS,
    PLOTLY_CFG,
    set_query_params_safe,
)

# ===============================
# [조절포인트] Page1 우측상단 버튼 위치(위에서 얼마나 내릴지)
# ===============================
P1_TOP_RIGHT_BTN_MARGIN_PX = 35  # ✅ 이 숫자만 바꾸면 "자세히 보기" 위치가 바뀝니다.

# ===============================
# [조절포인트] 지도/도넛 높이(아래로 더 늘리기)
# ===============================
MAP_HEIGHT_PX = 900
DONUT_HEIGHT_PX = 455

# ===============================
# [조절포인트] 변화율(4박스) 레이아웃 간격
# ===============================
DELTA_ROW1_MB_PX = 10          # ✅ (O3↔CO, NO2↔SO2) 1행 박스 아래 여백
DELTA_ROW2_MT_PX = -6          # ✅ 2행 박스 위 여백(음수면 위로 당겨져 간격 감소)
DELTA_TO_DONUT_MT_PX = -8      # ✅ SO2/CO 행 ↔ 도넛 사이 간격(음수면 도넛이 위로 당겨짐)


def _inject_delta_color_css():
    st.markdown(
        """
        <style>
          /* 전역 *{color:... !important} 보다 class가 우선 */
          .delta-pos { color: #EF4444 !important; }   /* + : 빨강 */
          .delta-neg { color: #3B82F6 !important; }   /* - : 파랑 */
          .delta-zero { color: #94A3B8 !important; }  /* 0 : 중립 */
          .delta-na { color: #94A3B8 !important; }    /* N/A : 중립 */
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hover_text(row: pd.Series) -> str:
    state = row.get("state", "")
    county = row.get("county", "")
    city = row.get("city", "")

    return (
        f"<b>{row.get('site','-')}</b><br>"
        f"Cluster: {row.get('cluster_3name','-')}<br>"
        f"Spike Days (7d): {int(row.get('spike_days', 0))}<br>"
        f"Priority: {row.get('priority','-')}<br>"
        f"{state} {county} {city}"
    )


def build_map_figure(snap: pd.DataFrame, selected_site: str | None = None) -> go.Figure:
    s = snap.dropna(subset=["lat", "lon"]).copy()
    fig = go.Figure()

    # ✅ 데이터가 없을 때도 동일 높이 적용
    if s.empty:
        fig.update_layout(height=MAP_HEIGHT_PX)
        return fig

    if selected_site:
        sel = s[s["site"].astype(str) == str(selected_site)]
        if not sel.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=sel["lat"],
                    lon=sel["lon"],
                    mode="markers",
                    marker=dict(
                        size=sel["spike_days"].clip(1, 10) * 2.2 + 20,
                        color="black",
                        opacity=0.95,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    for cl in ["Stable", "Risk", "High-risk"]:
        sub = s[s["cluster_3name"] == cl]
        if sub.empty:
            continue

        sizes = sub["spike_days"].clip(1, 10) * 2 + 10

        fig.add_trace(
            go.Scattermapbox(
                lat=sub["lat"],
                lon=sub["lon"],
                mode="markers",
                name=cl,
                marker=dict(
                    size=sizes,
                    color=CLUSTER_COLORS.get(cl, "#999999"),
                    opacity=0.85,
                ),
                customdata=sub[["site"]].values,
                hovertext=sub.apply(_hover_text, axis=1),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=float(np.nanmedian(s["lat"])),
                lon=float(np.nanmedian(s["lon"])),
            ),
            zoom=4,
        ),
        height=MAP_HEIGHT_PX,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            borderwidth=0,
            font=dict(size=12),
        ),
    )
    return fig


AQI_COLS = {"O3": "o3_aqi", "NO2": "no2_aqi", "CO": "co_aqi", "SO2": "so2_aqi"}
MEAN_COLS = {"O3": "o3_mean", "NO2": "no2_mean", "CO": "co_mean", "SO2": "so2_mean"}


def build_aqi_donut(df_all: pd.DataFrame, site: str):
    df_s = df_all[df_all["site"].astype(str) == str(site)]
    if df_s.empty:
        return None

    row = df_s.sort_values("date").tail(1).iloc[0]
    labels, values = [], []
    for label, col in AQI_COLS.items():
        v = row.get(col, np.nan)
        if pd.notna(v) and v > 0:
            labels.append(label)
            values.append(float(v))

    if not values:
        return None

    fig = go.Figure(
        go.Pie(labels=labels, values=values, hole=0.6, textinfo="label+percent")
    )
    fig.update_layout(
        title=dict(
            text="오염지표 구성 비율 (AQI)",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        height=DONUT_HEIGHT_PX,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


def compute_day_over_day_change(df_all: pd.DataFrame, site: str) -> dict:
    df_s = df_all[df_all["site"].astype(str) == str(site)].sort_values("date").tail(2)
    if len(df_s) < 2:
        return {}

    prev, curr = df_s.iloc[0], df_s.iloc[1]
    changes = {}
    for label, col in MEAN_COLS.items():
        v0, v1 = prev.get(col), curr.get(col)
        if pd.isna(v0) or pd.isna(v1) or v0 == 0:
            changes[label] = None
        else:
            changes[label] = (v1 - v0) / v0 * 100
    return changes


def _render_delta_box(label: str, pct, dark: bool, mt_px: int = 0, mb_px: int = 0):
    bg = "#0F172A" if dark else "#FFFFFF"
    border = "rgba(255,255,255,0.16)" if dark else "rgba(17,24,39,0.14)"
    sub = "#9CA3AF" if dark else "#6B7280"

    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        val = "N/A"
        arrow = ""
        cls = "delta-na"
    else:
        if pct > 0:
            arrow = "▲"
            cls = "delta-pos"
        elif pct < 0:
            arrow = "▼"
            cls = "delta-neg"
        else:
            arrow = ""
            cls = "delta-zero"
        val = f"{pct:+.1f}%"

    st.markdown(
        f"""
        <div style="
          background:{bg};
          border:1px solid {border};
          border-radius:14px;
          padding:10px 12px;
          line-height:1.1;
          margin-top:{mt_px}px;
          margin-bottom:{mb_px}px;
        ">
          <div style="font-size:12px;color:{sub} !important;font-weight:700;">{label}</div>
          <div style="margin-top:6px;font-size:18px;font-weight:900;">
            <span class="{cls}">{arrow} {val}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _top_right_quick_widgets(anchor_date: pd.Timestamp):
    # 현재시간(Asia/Seoul)
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")

    # 한 줄 구성(좁게)
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        st.toggle("🌙", key="ui_dark_mode", help="다크/라이트 모드")
    with c2:
        st.toggle("🔕", key="mute_alerts", help="알림 음소거")
    with c3:
        if st.button("⟳", use_container_width=True, help="즉시 새로고침"):
            st.rerun()

    st.caption(f"🕒 {now} | 기준일: {anchor_date.date()}")


def render_page1(df_all: pd.DataFrame, spike_df: pd.DataFrame, map_show_state: bool):
    base = df_all.sort_values("date").groupby("site", as_index=False).tail(1).copy()
    snap = base.merge(spike_df, on="site", how="left")
    snap["spike_days"] = snap["spike_days"].fillna(0).astype(int)
    snap["priority"] = snap["priority"].fillna("LOW")

    PRIORITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
    site_label_map = {
        row["site"]: f"{PRIORITY_EMOJI.get(row['priority'], '⚪')} {row['site']}"
        for _, row in snap.iterrows()
    }
    site_list = list(site_label_map.keys())

    if "p1_site_sel" not in st.session_state:
        st.session_state["p1_site_sel"] = site_list[0]

    selected_site = st.session_state["p1_site_sel"]
    anchor = pd.to_datetime(df_all["date"].max())

    # ✅ delta color CSS 1회 주입
    _inject_delta_color_css()

    # ✅ 상단 헤더 + 우측 상단 컨트롤
    h1, h2 = st.columns([8, 2], gap="small")
    with h1:
        P1_TITLE_FONT_PX = 50
        P1_TITLE_MARGIN_TOP_PX = 50
        P1_TITLE_MARGIN_BOTTOM_PX = 0

        st.markdown(
            f"""
            <div style="
              font-size:{P1_TITLE_FONT_PX}px;
              font-weight:800;
              margin-top:{P1_TITLE_MARGIN_TOP_PX}px;
              margin-bottom:{P1_TITLE_MARGIN_BOTTOM_PX}px;
            ">
              대기질 현황 요약 및 지도
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Spike Days × Cluster 기반 관측소 우선순위 시각화")
    with h2:
        st.markdown(
            f"<div style='margin-top:{P1_TOP_RIGHT_BTN_MARGIN_PX}px;'></div>",
            unsafe_allow_html=True,
        )

        _top_right_quick_widgets(anchor)

        if st.button("자세히 보기", use_container_width=True):
            set_query_params_safe(page="site", site=selected_site)
            st.rerun()

    # KPI
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("❗즉시 대응", str((snap["priority"] == "HIGH").sum()))
    with k2:
        kpi_card("🟠우선 검토", str((snap["priority"] == "MEDIUM").sum()))
    with k3:
        kpi_card("🟢일반 모니터링", str((snap["priority"] == "LOW").sum()))

    left, right = st.columns([3.2, 1.2], gap="large")

    with left:
        fig = build_map_figure(snap, selected_site)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            selection_mode="points",
            on_select="rerun",
            config=PLOTLY_CFG,
        )

        if event and getattr(event, "selection", None):
            pts = event.selection.get("points", [])
            if pts:
                cd = pts[0].get("customdata")
                if cd:
                    st.session_state["p1_site_sel"] = cd[0]

        card_end()

    with right:
        sel = st.selectbox(
            "관측소 선택",
            options=site_list,
            format_func=lambda x: site_label_map[x],
            key="p1_site_sel",
        )
        row = snap[snap["site"] == sel].iloc[0]

        st.write(f"**{row['site']}**")
        st.write(f"- 종합오염지표: **{row['cluster_3name']}**")
        st.write(f"- 예측위험지수: **{row['priority']}**")
        st.write(f"- 스파이크 탐지 수: **{row['spike_days']}개**")

        st.markdown("**전일 대비 변화율 (%)**")
        changes = compute_day_over_day_change(df_all, sel)
        dark = bool(st.session_state.get("ui_dark_mode", False))

        r1c1, r1c2 = st.columns(2, gap="small")
        r2c1, r2c2 = st.columns(2, gap="small")

        with r1c1:
            _render_delta_box("O3", changes.get("O3"), dark, mt_px=0, mb_px=DELTA_ROW1_MB_PX)
        with r1c2:
            _render_delta_box("NO2", changes.get("NO2"), dark, mt_px=0, mb_px=DELTA_ROW1_MB_PX)

        with r2c1:
            _render_delta_box("CO", changes.get("CO"), dark, mt_px=DELTA_ROW2_MT_PX, mb_px=0)
        with r2c2:
            _render_delta_box("SO2", changes.get("SO2"), dark, mt_px=DELTA_ROW2_MT_PX, mb_px=0)

        # ✅ 4박스 ↔ 도넛 간격 조절
        st.markdown(
            f"<div style='margin-top:{DELTA_TO_DONUT_MT_PX}px;'></div>",
            unsafe_allow_html=True,
        )

        donut = build_aqi_donut(df_all, sel)
        if donut:
            st.plotly_chart(donut, use_container_width=True, config=PLOTLY_CFG)

        card_end()
