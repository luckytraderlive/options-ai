import time
import pandas as pd
import streamlit as st
import plotly.express as px

from options_engine_v4 import scan_tickers, get_underlying_snapshot, payoff_curve


st.set_page_config(page_title="Options AI v4", layout="wide")

st.title("Options AI v4")
st.caption("Smart cache + anti-rate-limit + retry engine + async scanner + better ranking.")

with st.sidebar:
    strategy = st.selectbox(
        "Strategy",
        ["Long Call", "Long Put", "Bull Call Debit Spread"],
        index=0
    )

    direction = st.selectbox(
        "Direction",
        ["bullish", "bearish"],
        index=0
    )

    risk_per_trade = st.number_input(
        "Risk per trade ($)",
        min_value=50,
        max_value=100000,
        value=500,
        step=50
    )

    max_expirations = st.slider(
        "Max expirations",
        min_value=1,
        max_value=8,
        value=3
    )

    workers = st.slider(
        "Scanner speed",
        min_value=1,
        max_value=6,
        value=3,
        help="Higher = faster, but higher chance of Yahoo rate limit."
    )

    tickers_input = st.text_area(
        "Tickers",
        value="SPY,QQQ,NVDA,TSLA,META,AMD,AAPL,MSFT"
    )

    run = st.button("Run Scanner", type="primary")


@st.cache_data(ttl=600, show_spinner=False)
def cached_scan(tickers, strategy, direction, risk_per_trade, max_expirations, workers):
    return scan_tickers(
        tickers=tickers,
        strategy=strategy,
        direction=direction,
        risk_per_trade=risk_per_trade,
        max_expirations=max_expirations,
        workers=workers
    )


if run:
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

    with st.spinner("Scanning tickers with smart cache and retry engine..."):
        df = cached_scan(
            tickers,
            strategy,
            direction,
            risk_per_trade,
            max_expirations,
            workers
        )

    if df.empty:
        st.warning("No setups found.")
    else:
        clean = df.drop(columns=["error"], errors="ignore")

        st.subheader("Best setups")
        st.dataframe(clean, use_container_width=True)

        valid = clean[clean["ai_score"] > 0].copy()

        if not valid.empty:
            top = valid.iloc[0]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Top ticker", top["ticker"])
            c2.metric("Strategy", top["strategy"])
            c3.metric("AI Score", f"{top['ai_score']:.2f}")
            c4.metric("Max Loss", f"${top['max_loss']}")
            c5.metric("Contracts", int(top["contracts"]))

            st.success(
                f"TOP SETUP: {top['ticker']} | {top['strategy']} | "
                f"{top['contract']} | Exp {top['expiration']} | Score {top['ai_score']}"
            )

            fig = px.bar(
                valid,
                x="ticker",
                y="ai_score",
                color="strategy",
                hover_data=[c for c in ["contract","expiration","max_loss","max_profit","prob_itm","risk_reward"] if c in valid.columns],
                title="AI Score by Ticker"
            )
            st.plotly_chart(fig, use_container_width=True)

            try:
                snap = get_underlying_snapshot(top["ticker"])
                payoff = payoff_curve(top, snap.price)

                fig2 = px.line(
                    payoff,
                    x="underlying_at_expiration",
                    y="payoff_per_share",
                    title=f"Payoff Chart: {top['ticker']} {top['strategy']}"
                )
                fig2.add_hline(y=0, line_dash="dash")
                fig2.add_vline(x=snap.price, line_dash="dash", annotation_text="Current")
                fig2.add_vline(x=float(payoff["breakeven"].iloc[0]), line_dash="dot", annotation_text="Breakeven")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"Payoff chart unavailable: {e}")

            csv = clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="options_ai_v4_scan.csv",
                mime="text/csv"
            )

        errors = df[df.get("error").notna()] if "error" in df.columns else pd.DataFrame()
        if not errors.empty:
            with st.expander("Errors / rate limit notes"):
                st.dataframe(errors, use_container_width=True)

else:
    st.info("Choose strategy, tickers, and press Run Scanner.")

st.markdown("---")
st.caption(
    "Data source: Yahoo Finance via yfinance with local SQLite cache. "
    "For production-grade options data, connect IBKR, Polygon, Tradier, or ThetaData."
)