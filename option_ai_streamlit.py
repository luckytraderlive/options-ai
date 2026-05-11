import pandas as pd
import streamlit as st
import plotly.express as px

from options_engine import fetch_and_rank_options

st.set_page_config(page_title="Options AI Analyzer", layout="wide")

st.title("Options AI Analyzer")
st.caption("Введите тикер — система сама загрузит базовый актив, option chain, Black-Scholes и AI-рейтинг контрактов.")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    direction = st.selectbox("Direction", ["bullish", "bearish"], index=0)
    mode = st.selectbox("Risk mode", ["conservative", "balanced", "aggressive"], index=1)
    max_exp = st.slider("Max expirations", 1, 12, 8)
    run = st.button("Run analysis", type="primary")

if run or ticker:
    try:
        snapshot, df = fetch_and_rank_options(
            ticker=ticker,
            direction=direction,
            mode=mode,
            max_expirations=max_exp,
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"{snapshot.price:.2f}")
        c2.metric("Trend", snapshot.trend)
        c3.metric("RSI 14", f"{snapshot.rsi_14:.1f}")
        c4.metric("ATR 14", f"{snapshot.atr_14:.2f}")
        c5.metric("Hist Vol 30D", f"{snapshot.hist_vol_30:.1%}")

        best = df.iloc[0]
        st.subheader("Лучший контракт")
        st.success(
            f"{best['contractSymbol']} | {best['side'].upper()} | Strike {best['strike']} | "
            f"Expiration {best['expiration']} | Score {best['ai_score']}"
        )

        st.subheader("Top ranked options")
        show_cols = [
            "ai_score", "contractSymbol", "side", "expiration", "dte", "strike",
            "bid", "ask", "mid", "spread_pct", "volume", "openInterest",
            "impliedVolatility", "bs_price", "bs_delta", "bs_theta", "edge_vs_bs",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].head(100), use_container_width=True)

        st.subheader("Score by strike")
        chart_df = df.head(150).copy()
        fig = px.scatter(
            chart_df,
            x="strike",
            y="ai_score",
            color="expiration",
            size="openInterest",
            hover_data=["contractSymbol", "side", "dte", "mid", "bs_delta"],
            title="AI score by strike and expiration",
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranked CSV",
            data=csv,
            file_name=f"{ticker}_options_ranked.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))
        st.info("Проверь тикер и наличие опционов. Некоторые тикеры не имеют option chain в Yahoo Finance.")