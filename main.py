import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 끝
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(), errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])  # 관측값이 있는 날만 남김
    df["연도"] = df["날짜"].dt.year

    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 해 제외
    return yearly.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온 추세로 미래(와 과거)의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("조건을 만족하는 연도가 부족해 회귀 직선을 만들 수 없습니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# --- 연도 선택 & 예상 기온 ---
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2050)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; color:gray;">{year}년 서울 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.1f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 요약 정보 ---
c1, c2, c3 = st.columns(3)
c1.metric("상관계수 (r)", f"{r:.3f}")
c2.metric("10년당 변화", f"{slope * 10:+.2f} ℃")
c3.metric("결정계수 (R²)", f"{r ** 2:.3f}")

st.info(
    f"회귀 직선은 **{n_years}개** 연도의 자료로 만들었습니다 "
    f"(시작 연도: **{start_year}년**, 끝 연도: **{end_year}년**)."
)

# --- 그래프 ---
line_x = np.array([1900, 2100])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#d62728", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, color="#ff7f0e", symbol="star",
                    line=dict(color="black", width=1)),
        hovertemplate="%{x}년<br>예상 %{y:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"회귀식: 기온 = {slope:.4f} × 연도 + ({intercept:.2f}) · "
    f"{LAST_YEAR}년 이후 자료와 관측일 {MIN_DAYS}일 미만인 해는 제외했습니다. "
    "범위를 벗어난 연도의 값은 직선을 그대로 늘린 단순 추정입니다."
)
