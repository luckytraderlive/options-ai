"""
Options AI Engine
-----------------
Pulls underlying data and option chain via yfinance, calculates Black-Scholes values,
technical indicators, and ranks option contracts.

Usage:
    python options_engine.py --ticker AAPL
    python options_engine.py --ticker AAPL --csv-output AAPL_options_ranked.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


OptionSide = Literal["call", "put"]
Direction = Literal["bullish", "bearish"]


@dataclass
class UnderlyingSnapshot:
    ticker: str
    price: float
    trend: str
    rsi_14: float
    atr_14: float
    hist_vol_30: float


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    side: OptionSide,
) -> float:
    """Return theoretical Black-Scholes price."""
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return np.nan

    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * math.sqrt(time_to_expiry)
    )
    d2 = d1 - volatility * math.sqrt(time_to_expiry)

    if side == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)

    return strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    side: OptionSide,
) -> dict:
    """Return delta, gamma, theta, vega."""
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return {"bs_delta": np.nan, "bs_gamma": np.nan, "bs_theta": np.nan, "bs_vega": np.nan}

    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * math.sqrt(time_to_expiry)
    )
    d2 = d1 - volatility * math.sqrt(time_to_expiry)

    delta = norm.cdf(d1) if side == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))
    vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100

    if side == "call":
        theta = (
            -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
            - rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
            + rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2)
        ) / 365

    return {"bs_delta": delta, "bs_gamma": gamma, "bs_theta": theta, "bs_vega": vega}


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return float(value.iloc[-1]) if not value.empty else np.nan


def atr(data: pd.DataFrame, period: int = 14) -> float:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


def get_underlying_snapshot(ticker: str) -> UnderlyingSnapshot:
    t = yf.Ticker(ticker)
    hist = t.history(period="1y", interval="1d", auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No price history found for {ticker}")

    price = float(hist["Close"].iloc[-1])
    ema20 = hist["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = hist["Close"].ewm(span=50).mean().iloc[-1]
    trend = "bullish" if ema20 > ema50 else "bearish"

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hist_vol_30 = float(log_returns.tail(30).std() * math.sqrt(252))

    return UnderlyingSnapshot(
        ticker=ticker.upper(),
        price=price,
        trend=trend,
        rsi_14=rsi(hist["Close"], 14),
        atr_14=atr(hist, 14),
        hist_vol_30=hist_vol_30,
    )


def _score_contract(row: pd.Series, direction: Direction, mode: str) -> float:
    side = row["side"]
    delta = row.get("delta_used", np.nan)
    iv = row.get("impliedVolatility", np.nan)
    dte = row.get("dte", np.nan)
    spread_pct = row.get("spread_pct", np.nan)
    volume = row.get("volume", 0) or 0
    oi = row.get("openInterest", 0) or 0
    theta = row.get("theta_used", np.nan)
    moneyness = row.get("moneyness", np.nan)

    if direction == "bullish" and side != "call":
        return 0
    if direction == "bearish" and side != "put":
        return 0

    target_delta = {"conservative": 0.55, "balanced": 0.40, "aggressive": 0.25}.get(mode, 0.40)
    target_dte = {"conservative": 45, "balanced": 35, "aggressive": 21}.get(mode, 35)

    delta_abs = abs(delta) if pd.notna(delta) else 0
    delta_score = max(0, 100 - abs(delta_abs - target_delta) * 250)

    dte_score = max(0, 100 - abs(dte - target_dte) * 2.2) if pd.notna(dte) else 0
    liquidity_score = min(100, np.log1p(max(volume, 0)) * 13 + np.log1p(max(oi, 0)) * 9)
    spread_score = max(0, 100 - (spread_pct or 100) * 800)
    theta_score = max(0, 100 - abs(theta if pd.notna(theta) else 0) * 600)
    iv_score = max(0, 100 - (iv if pd.notna(iv) else 1) * 80)
    moneyness_score = max(0, 100 - abs((moneyness if pd.notna(moneyness) else 1) - 1) * 450)

    return round(
        liquidity_score * 0.20
        + spread_score * 0.15
        + delta_score * 0.25
        + dte_score * 0.15
        + theta_score * 0.10
        + iv_score * 0.05
        + moneyness_score * 0.10,
        2,
    )


def fetch_and_rank_options(
    ticker: str,
    direction: Direction = "bullish",
    mode: str = "balanced",
    max_expirations: int = 8,
    risk_free_rate: float = 0.045,
) -> tuple[UnderlyingSnapshot, pd.DataFrame]:
    ticker = ticker.upper().strip()
    snapshot = get_underlying_snapshot(ticker)
    t = yf.Ticker(ticker)

    expirations = list(t.options or [])[:max_expirations]
    if not expirations:
        raise ValueError(f"No option expirations found for {ticker}")

    frames = []
    now = datetime.utcnow().date()

    for exp in expirations:
        chain = t.option_chain(exp)

        for side_name, df in [("call", chain.calls.copy()), ("put", chain.puts.copy())]:
            if df.empty:
                continue

            df["side"] = side_name
            df["expiration"] = exp
            df["dte"] = (pd.to_datetime(exp).date() - now).days
            df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
            df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)
            df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.nan)
            df["underlying_price"] = snapshot.price
            df["moneyness"] = df["strike"] / snapshot.price
            df["time_to_expiry"] = df["dte"] / 365

            bs_prices = []
            bs_delta = []
            bs_gamma = []
            bs_theta = []
            bs_vega = []

            for _, row in df.iterrows():
                iv = row.get("impliedVolatility", np.nan)
                iv = float(iv) if pd.notna(iv) and iv > 0 else snapshot.hist_vol_30
                bs_prices.append(
                    black_scholes_price(
                        snapshot.price, float(row["strike"]), max(float(row["time_to_expiry"]), 1/365),
                        risk_free_rate, iv, side_name
                    )
                )
                greeks = black_scholes_greeks(
                    snapshot.price, float(row["strike"]), max(float(row["time_to_expiry"]), 1/365),
                    risk_free_rate, iv, side_name
                )
                bs_delta.append(greeks["bs_delta"])
                bs_gamma.append(greeks["bs_gamma"])
                bs_theta.append(greeks["bs_theta"])
                bs_vega.append(greeks["bs_vega"])

            df["bs_price"] = bs_prices
            df["bs_delta"] = bs_delta
            df["bs_gamma"] = bs_gamma
            df["bs_theta"] = bs_theta
            df["bs_vega"] = bs_vega

            df["delta_used"] = df["bs_delta"]
            df["theta_used"] = df["bs_theta"]
            df["edge_vs_bs"] = np.where(df["mid"] > 0, (df["bs_price"] - df["mid"]) / df["mid"], np.nan)

            frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    result["ai_score"] = result.apply(lambda r: _score_contract(r, direction, mode), axis=1)

    useful_cols = [
        "contractSymbol", "side", "expiration", "dte", "strike",
        "underlying_price", "lastPrice", "bid", "ask", "mid", "spread_pct",
        "volume", "openInterest", "impliedVolatility",
        "bs_price", "bs_delta", "bs_gamma", "bs_theta", "bs_vega",
        "edge_vs_bs", "moneyness", "ai_score",
    ]

    result = result[[c for c in useful_cols if c in result.columns]].copy()
    result = result.sort_values("ai_score", ascending=False).reset_index(drop=True)
    return snapshot, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--direction", default="bullish", choices=["bullish", "bearish"])
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()

    snapshot, ranked = fetch_and_rank_options(args.ticker, args.direction, args.mode)

    print(f"Ticker: {snapshot.ticker}")
    print(f"Price: {snapshot.price:.2f}")
    print(f"Trend: {snapshot.trend}")
    print(f"RSI 14: {snapshot.rsi_14:.2f}")
    print(f"ATR 14: {snapshot.atr_14:.2f}")
    print(f"Historical Vol 30D: {snapshot.hist_vol_30:.2%}")
    print()
    print(ranked.head(20).to_string(index=False))

    if args.csv_output:
        ranked.to_csv(args.csv_output, index=False)
        print(f"\nSaved: {args.csv_output}")


if __name__ == "__main__":
    main()