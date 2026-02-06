# ui_components.py
import streamlit as st
import plotly.graph_objects as go


def inject_dashboard_css(sidebar_bg: str = "#0B1F3A", dark_mode: bool = False):
    # ---------- palette ----------
    if dark_mode:
        app_bg = "#0B1220"
        main_bg = "#0B1220"
        card_bg = "#111827"
        main_text = "#E5E7EB"
        caption_text = "#9CA3AF"
        border = "rgba(255,255,255,0.12)"
        shadow = "0 10px 26px rgba(0, 0, 0, 0.35)"
        sidebar_text = "#F9FAFB"

        # main widgets
        main_input_bg = "#0F172A"
        main_input_text = "#E5E7EB"
        main_input_border = "rgba(255,255,255,0.18)"
        main_placeholder = "#9CA3AF"
        pop_bg = "#0F172A"
        pop_border = "rgba(255,255,255,0.18)"

        # buttons
        btn_bg = "#1F2937"
        btn_text = "#F9FAFB"
        btn_border = "rgba(255,255,255,0.18)"
        btn_bg_hover = "#243244"
    else:
        app_bg = "#F6F7FB"
        main_bg = "#F6F7FB"
        card_bg = "#FFFFFF"
        main_text = "#111827"
        caption_text = "#6B7280"
        border = "rgba(17,24,39,0.10)"
        shadow = "0 10px 26px rgba(17, 24, 39, 0.08)"
        sidebar_text = "#F9FAFB"

        # main widgets
        main_input_bg = "#FFFFFF"
        main_input_text = "#111827"
        main_input_border = "rgba(17,24,39,0.18)"
        main_placeholder = "#6B7280"
        pop_bg = "#FFFFFF"
        pop_border = "rgba(17,24,39,0.18)"

        # buttons
        btn_bg = "#FFFFFF"
        btn_text = "#111827"
        btn_border = "rgba(17,24,39,0.16)"
        btn_bg_hover = "#F3F4F6"

    st.markdown(
        f"""
        <style>
        /* Base */
        .stApp {{ background: {app_bg}; }}
        footer {{visibility: hidden;}}

        /* Sidebar background */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebarContent"] {{
            background: {sidebar_bg} !important;
            background-color: {sidebar_bg} !important;
        }}

        /* Sidebar label text */
        section[data-testid="stSidebar"] * {{
            color: {sidebar_text} !important;
        }}

        /* ✅ Sidebar inputs: ALWAYS white bg + black text */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {{
            background: #FFFFFF !important;
            color: #111827 !important;
            border: 1px solid rgba(17,24,39,0.18) !important;
        }}
        section[data-testid="stSidebar"] input::placeholder,
        section[data-testid="stSidebar"] textarea::placeholder {{
            color: #6B7280 !important;
        }}

        /* Sidebar select */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #FFFFFF !important;
            border: 1px solid rgba(17,24,39,0.18) !important;
        }}
        section[data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: #111827 !important;
        }}

        /* Main */
        [data-testid="stAppViewContainer"] [data-testid="stMain"] {{
            background: {main_bg} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] [data-testid="stMain"] * {{
            color: {main_text} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{
            color: {caption_text} !important;
        }}

        /* ✅ Main inputs (text/date input 포함) */
        [data-testid="stAppViewContainer"] [data-testid="stMain"] input,
        [data-testid="stAppViewContainer"] [data-testid="stMain"] textarea {{
            background: {main_input_bg} !important;
            color: {main_input_text} !important;
            border: 1px solid {main_input_border} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMain"] input::placeholder,
        [data-testid="stAppViewContainer"] [data-testid="stMain"] textarea::placeholder {{
            color: {main_placeholder} !important;
        }}

        /* ✅ Main selectbox / multiselect */
        [data-testid="stAppViewContainer"] [data-testid="stMain"] [data-baseweb="select"] > div {{
            background: {main_input_bg} !important;
            border: 1px solid {main_input_border} !important;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stMain"] [data-baseweb="select"] * {{
            color: {main_input_text} !important;
        }}

        /* ✅ Main date_input */
        [data-testid="stAppViewContainer"] [data-testid="stMain"] [data-baseweb="datepicker"] input {{
            background: {main_input_bg} !important;
            color: {main_input_text} !important;
            border: 1px solid {main_input_border} !important;
        }}

        /* ✅ Dropdown / calendar popover */
        .stApp div[role="listbox"] {{
            background: {pop_bg} !important;
            border: 1px solid {pop_border} !important;
        }}
        .stApp li[role="option"] {{
            color: {main_input_text} !important;
        }}
        .stApp li[role="option"]:hover {{
            background: rgba(148,163,184,0.15) !important;
        }}
        .stApp [data-baseweb="calendar"] {{
            background: {pop_bg} !important;
            border: 1px solid {pop_border} !important;
            color: {main_input_text} !important;
        }}

        /* ✅ Buttons: 구조가 달라도 전부 잡기 (⟳ 포함)
           - '>' 제거하고 descendant selector로 강제 적용
        */
        .stApp [data-testid="stButton"] button,
        .stApp [data-testid="stFormSubmitButton"] button,
        .stApp [data-testid="stDownloadButton"] button {{
          background-color: {btn_bg} !important;
          color: {btn_text} !important;
          border: 1px solid {btn_border} !important;
          box-shadow: none !important;
        }}

        /* ✅ 버튼 내부 텍스트/아이콘까지 강제 */
        .stApp [data-testid="stButton"] button *,
        .stApp [data-testid="stFormSubmitButton"] button *,
        .stApp [data-testid="stDownloadButton"] button * {{
          color: {btn_text} !important;
        }}

        /* hover/focus/active 상태도 동일 */
        .stApp [data-testid="stButton"] button:hover,
        .stApp [data-testid="stButton"] button:active,
        .stApp [data-testid="stButton"] button:focus,
        .stApp [data-testid="stButton"] button:focus-visible,
        .stApp [data-testid="stFormSubmitButton"] button:hover,
        .stApp [data-testid="stFormSubmitButton"] button:active,
        .stApp [data-testid="stFormSubmitButton"] button:focus,
        .stApp [data-testid="stFormSubmitButton"] button:focus-visible,
        .stApp [data-testid="stDownloadButton"] button:hover,
        .stApp [data-testid="stDownloadButton"] button:active,
        .stApp [data-testid="stDownloadButton"] button:focus,
        .stApp [data-testid="stDownloadButton"] button:focus-visible {{
          background-color: {btn_bg_hover} !important;
          color: {btn_text} !important;
          border: 1px solid {btn_border} !important;
          box-shadow: none !important;
        }}

        .stApp [data-testid="stButton"] button:hover *,
        .stApp [data-testid="stButton"] button:focus-visible *,
        .stApp [data-testid="stFormSubmitButton"] button:hover *,
        .stApp [data-testid="stFormSubmitButton"] button:focus-visible *,
        .stApp [data-testid="stDownloadButton"] button:hover *,
        .stApp [data-testid="stDownloadButton"] button:focus-visible * {{
          color: {btn_text} !important;
        }}

        /* Cards */
        .card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: {shadow};
            border: 1px solid {border};
            border-left: 4px solid rgba(229,231,235,0.9);
            margin-bottom: 14px;
        }}
        .card.high {{ border-left-color: #EF4444; }}
        .card.medium {{ border-left-color: #F59E0B; }}
        .card.low {{ border-left-color: #10B981; }}

        .card-title {{
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .card-subtitle {{
            font-size: 12px;
            margin-bottom: 10px;
        }}

        /* KPI */
        .kpi-wrap {{
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:12px;
        }}
        .kpi-label {{
            font-size:12px;
            font-weight:700;
        }}
        .kpi-value {{
            font-size:28px;
            font-weight:900;
            line-height:1.15;
        }}
        .kpi-delta {{
            font-size:12px;
            font-weight:800;
            color:#10B981 !important;
        }}
        .kpi-delta.neg {{
            color:#EF4444 !important;
        }}

        .js-plotly-plot, .plotly, .plot-container {{
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_start(title: str, subtitle: str = "", variant: str = "neutral"):
    sub_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    cls = f"card {variant}" if variant in ["high", "medium", "low"] else "card"
    st.markdown(
        f'<div class="{cls}"><div class="card-title">{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", variant: str = "neutral"):
    delta_html = ""
    if delta:
        cls = "kpi-delta neg" if delta.strip().startswith("-") else "kpi-delta"
        delta_html = f'<div class="{cls}">{delta}</div>'

    cls = f"card {variant}" if variant in ["high", "medium", "low"] else "card"

    st.markdown(
        f"""
        <div class="{cls}">
          <div class="kpi-wrap">
            <div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
            </div>
            <div>{delta_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_card_style(fig: go.Figure) -> go.Figure:
    dark_mode = bool(st.session_state.get("ui_dark_mode", False))
    font_color = "#E5E7EB" if dark_mode else "#111827"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=font_color),
    )
    return fig
# ui_components.py
import streamlit as st
import plotly.graph_objects as go



def card_start(title: str, subtitle: str = "", variant: str = "neutral"):
    sub_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    cls = f"card {variant}" if variant in ["high", "medium", "low"] else "card"
    st.markdown(
        f'<div class="{cls}"><div class="card-title">{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", variant: str = "neutral"):
    delta_html = ""
    if delta:
        cls = "kpi-delta neg" if delta.strip().startswith("-") else "kpi-delta"
        delta_html = f'<div class="{cls}">{delta}</div>'

    cls = f"card {variant}" if variant in ["high", "medium", "low"] else "card"

    st.markdown(
        f"""
        <div class="{cls}">
          <div class="kpi-wrap">
            <div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
            </div>
            <div>{delta_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_card_style(fig: go.Figure) -> go.Figure:
    dark_mode = bool(st.session_state.get("ui_dark_mode", False))
    font_color = "#E5E7EB" if dark_mode else "#111827"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=font_color),
    )
    return fig
