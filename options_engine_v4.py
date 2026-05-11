from __future__ import annotations

import math
import time
import sqlite3
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed


Direction = Literal["bullish", "bearish"]
Strategy = Literal["Long Call", "Long Put", "Bull Call Debit Spread"]


CACHE_DB = Path("options_cache.sqlite3")


@dataclass
class UnderlyingSnapshot:
    ticker: str
    price: float
    trend: str
    rsi_14: float
    atr_14: float
    hist_vol_30: float
    iv_rank_est: float


def _cache_key(prefix: str, *parts) -> str:
    raw = "|".join([prefix] + [str(x) for x in parts])
    return hashlib.sha256(raw.encode()).hexdigest()


def _init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL,
            created_at REAL NOT NULL,
            ttl INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def cache_get(key: str):
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    row = conn.execute(
        "SELECT value, created_at, ttl FROM cache WHERE key=?",
        (key,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    value, created_at, ttl = row
    if time.time() - created_at > ttl:
        return None

    return pickle.loads(value)


def cache_set(key: str, value, ttl: int):
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "REPLACE INTO cache(key, value, created_at, ttl) VALUES (?, ?, ?, ?)",
        (key, pickle.dumps(value), time.time(), ttl)
    )
    conn.commit()
    conn.close()


def retry_call(fn, retries=3, base_sleep=1.5):
    last_error = None

    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            msg = str(e).lower()

            if "too many requests" in msg or "rate limited" in msg or "429" in msg:
                sleep = base_sleep * (2 ** attempt) + np.random.uniform(0.5, 2.0)
            else:
                sleep = base_sleep + np.random.uniform(0.2, 1.0)

            time.sleep(sleep)

    raise last_error


def black_scholes_price(spot, strike, t, r, vol, side):
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return np.nan

    d1 = (math.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)

    if side == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)

    return strike * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_delta(spot, strike, t, r, vol, side):
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return np.nan

    d1 = (math.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
    return norm.cdf(d1) if side == "call" else norm.cdf(d1) - 1


def probability_itm(spot, strike, t, r, vol, side):
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return np.nan

    d2 = (math.log(spot / strike) + (r - 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
    return float(norm.cdf(d2)) if side == "call" else float(norm.cdf(-d2))


def expected_move(spot, vol, dte):
    if spot <= 0 or vol <= 0 or dte <= 0:
        return np.nan
    return float(spot * vol * math.sqrt(dte / 365))


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return float(value.iloc[-1]) if not value.empty else np.nan


def atr(data, period=14):
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


def estimate_iv_rank_from_history(hist):
    closes = hist["Close"].dropna()
    log_returns = np.log(closes / closes.shift(1)).dropna()
    rolling_vol = log_returns.rolling(30).std() * math.sqrt(252)

    if rolling_vol.dropna().empty:
        return np.nan

    current = rolling_vol.iloc[-1]
    low = rolling_vol.min()
    high = rolling_vol.max()

    if pd.isna(current) or pd.isna(low) or pd.isna(high) or high == low:
        return np.nan

    return float((current - low) / (high - low) * 100)


def get_underlying_snapshot(ticker: str, ttl=1800) -> UnderlyingSnapshot:
    key = _cache_key("snapshot", ticker)
    cached = cache_get(key)

    if cached:
        return cached

    def fetch():
        tk = yf.Ticker(ticker)
        return tk.history(period="1y", interval="1d", auto_adjust=True)

    hist = retry_call(fetch)

    if hist.empty:
        raise ValueError(f"No price history found for {ticker}")

    price = float(hist["Close"].iloc[-1])
    ema20 = hist["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = hist["Close"].ewm(span=50).mean().iloc[-1]
    trend = "bullish" if ema20 > ema50 else "bearish"

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hist_vol_30 = float(log_returns.tail(30).std() * math.sqrt(252))

    snap = UnderlyingSnapshot(
        ticker=ticker.upper(),
        price=price,
        trend=trend,
        rsi_14=rsi(hist["Close"]),
        atr_14=atr(hist),
        hist_vol_30=hist_vol_30,
        iv_rank_est=estimate_iv_rank_from_history(hist),
    )

    cache_set(key, snap, ttl)
    return snap


def get_yahoo_chain(ticker: str, max_expirations=3, ttl=900):
    key = _cache_key("chain", ticker, max_expirations)
    cached = cache_get(key)

    if cached:
        return cached

    tk = yf.Ticker(ticker)

    def get_expirations():
        return list(tk.options or [])[:max_expirations]

    expirations = retry_call(get_expirations)

    if not expirations:
        raise ValueError(f"No expirations found for {ticker}")

    rows = []

    for exp in expirations:
        def fetch_chain():
            return tk.option_chain(exp)

        chain = retry_call(fetch_chain)

        for side, df in [("call", chain.calls.copy()), ("put", chain.puts.copy())]:
            if df.empty:
                continue

            df["side"] = side
            df["expiration"] = exp
            rows.append(df)

        time.sleep(0.35)

    if not rows:
        raise ValueError(f"No option chain found for {ticker}")

    result = pd.concat(rows, ignore_index=True)
    cache_set(key, result, ttl)
    return result


def enrich_options(ticker, direction="bullish", max_expirations=3, risk_free_rate=0.045):
    snap = get_underlying_snapshot(ticker)
    df = get_yahoo_chain(ticker, max_expirations=max_expirations).copy()

    now = datetime.utcnow().date()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["dte"] = df["expiration"].apply(lambda x: max((x - now).days, 1))
    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
    df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.nan)
    df["underlying_price"] = snap.price
    df["moneyness"] = df["strike"] / snap.price

    rows = []

    for _, row in df.iterrows():
        side = row["side"]
        iv = row.get("impliedVolatility", np.nan)
        iv = float(iv) if pd.notna(iv) and iv > 0 else snap.hist_vol_30
        t = max(row["dte"] / 365, 1 / 365)

        bs_price = black_scholes_price(snap.price, row["strike"], t, risk_free_rate, iv, side)
        delta = black_scholes_delta(snap.price, row["strike"], t, risk_free_rate, iv, side)
        prob = probability_itm(snap.price, row["strike"], t, risk_free_rate, iv, side)
        em = expected_move(snap.price, iv, row["dte"])

        rows.append({
            **row.to_dict(),
            "bs_price": bs_price,
            "bs_delta": delta,
            "prob_itm": prob,
            "expected_move": em,
            "expected_move_low": snap.price - em,
            "expected_move_high": snap.price + em,
            "iv_rank_est": snap.iv_rank_est,
        })

    out = pd.DataFrame(rows)
    out["ai_score"] = out.apply(lambda r: rank_single_option(r, direction, snap), axis=1)
    return snap, out.sort_values("ai_score", ascending=False).reset_index(drop=True)


def rank_single_option(row, direction, snap):
    side = row["side"]

    if direction == "bullish" and side != "call":
        return 0
    if direction == "bearish" and side != "put":
        return 0

    oi = row.get("openInterest", 0) or 0
    vol = row.get("volume", 0) or 0
    spread_pct = row.get("spread_pct", 1)
    delta = abs(row.get("bs_delta", 0) or 0)
    prob = row.get("prob_itm", 0) or 0
    dte = row.get("dte", 0) or 0
    iv_rank = snap.iv_rank_est if pd.notna(snap.iv_rank_est) else 50

    liquidity = min(100, np.log1p(oi) * 10 + np.log1p(vol) * 12)
    spread_score = max(0, 100 - spread_pct * 900)
    delta_score = max(0, 100 - abs(delta - 0.40) * 230)
    prob_score = max(0, 100 - abs(prob - 0.42) * 200)
    dte_score = max(0, 100 - abs(dte - 35) * 2.0)
    iv_score = max(0, 100 - iv_rank)

    trend_bonus = 10 if snap.trend == direction else -5

    return round(
        liquidity * 0.22 +
        spread_score * 0.18 +
        delta_score * 0.20 +
        prob_score * 0.14 +
        dte_score * 0.12 +
        iv_score * 0.09 +
        trend_bonus,
        2
    )


def build_long_option(ticker, direction, risk_per_trade, max_expirations=3):
    snap, df = enrich_options(ticker, direction, max_expirations)
    side = "call" if direction == "bullish" else "put"

    candidates = df[(df["side"] == side) & (df["ask"] > 0)].copy()

    if candidates.empty:
        return None

    best = candidates.sort_values("ai_score", ascending=False).iloc[0]
    premium = float(best["ask"]) * 100
    contracts = max(int(risk_per_trade // premium), 1)

    return {
        "ticker": ticker,
        "strategy": "Long Call" if side == "call" else "Long Put",
        "contract": best["contractSymbol"],
        "expiration": str(best["expiration"]),
        "strike": float(best["strike"]),
        "premium": round(float(best["ask"]), 2),
        "contracts": contracts,
        "max_loss": round(premium * contracts, 2),
        "max_profit": "Unlimited" if side == "call" else "High",
        "prob_itm": round(float(best["prob_itm"]), 4),
        "expected_move": round(float(best["expected_move"]), 2),
        "iv_rank_est": round(float(best["iv_rank_est"]), 2) if pd.notna(best["iv_rank_est"]) else None,
        "ai_score": float(best["ai_score"]),
        "source": "Yahoo cached",
    }


def build_bull_call_debit_spread(ticker, risk_per_trade, max_expirations=3):
    snap, df = enrich_options(ticker, "bullish", max_expirations)

    calls = df[(df["side"] == "call") & (df["ask"] > 0) & (df["bid"] >= 0)].copy()
    calls = calls.sort_values(["expiration", "strike"])

    spreads = []

    for exp, group in calls.groupby("expiration"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - 1):
            long_leg = group.iloc[i]

            for j in range(i + 1, min(i + 5, len(group))):
                short_leg = group.iloc[j]
                width = short_leg["strike"] - long_leg["strike"]

                if width <= 0:
                    continue

                debit = (long_leg["ask"] - short_leg["bid"]) * 100

                if debit <= 5:
                    continue

                max_profit = width * 100 - debit

                if max_profit <= 0:
                    continue

                rr = max_profit / debit
                contracts = max(int(risk_per_trade // debit), 1)

                score = (
                    long_leg["ai_score"] * 0.55 +
                    min(100, rr * 35) * 0.25 +
                    max(0, 100 - abs(long_leg["prob_itm"] - 0.50) * 150) * 0.20
                )

                spreads.append({
                    "ticker": ticker,
                    "strategy": "Bull Call Debit Spread",
                    "contract": f"{long_leg['strike']} / {short_leg['strike']}",
                    "long_leg": long_leg["contractSymbol"],
                    "short_leg": short_leg["contractSymbol"],
                    "expiration": str(exp),
                    "strike": float(long_leg["strike"]),
                    "short_strike": float(short_leg["strike"]),
                    "premium": round(debit / 100, 2),
                    "contracts": contracts,
                    "max_loss": round(debit * contracts, 2),
                    "max_profit": round(max_profit * contracts, 2),
                    "risk_reward": round(rr, 2),
                    "prob_itm": round(float(long_leg["prob_itm"]), 4),
                    "expected_move": round(float(long_leg["expected_move"]), 2),
                    "iv_rank_est": round(float(long_leg["iv_rank_est"]), 2) if pd.notna(long_leg["iv_rank_est"]) else None,
                    "ai_score": round(float(score), 2),
                    "source": "Yahoo cached",
                })

    if not spreads:
        return None

    return sorted(spreads, key=lambda x: x["ai_score"], reverse=True)[0]


def scan_tickers(tickers, strategy, direction, risk_per_trade, max_expirations=3, workers=3):
    results = []

    def run_one(ticker):
        ticker = ticker.strip().upper()

        if not ticker:
            return None

        if strategy == "Bull Call Debit Spread":
            return build_bull_call_debit_spread(ticker, risk_per_trade, max_expirations)

        if strategy == "Long Put":
            return build_long_option(ticker, "bearish", risk_per_trade, max_expirations)

        return build_long_option(ticker, direction, risk_per_trade, max_expirations)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, t): t for t in tickers}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                results.append({
                    "ticker": futures[future],
                    "strategy": strategy,
                    "error": str(e),
                    "ai_score": 0,
                })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("ai_score", ascending=False).reset_index(drop=True)


def payoff_curve(strategy_row, spot):
    strategy = strategy_row["strategy"]
    prices = np.linspace(spot * 0.65, spot * 1.35, 160)

    if strategy == "Long Call":
        strike = float(strategy_row["strike"])
        premium = float(strategy_row["premium"])
        payoff = np.maximum(prices - strike, 0) - premium
        breakeven = strike + premium

    elif strategy == "Long Put":
        strike = float(strategy_row["strike"])
        premium = float(strategy_row["premium"])
        payoff = np.maximum(strike - prices, 0) - premium
        breakeven = strike - premium

    else:
        long_strike = float(strategy_row["strike"])
        short_strike = float(strategy_row["short_strike"])
        debit = float(strategy_row["premium"])
        payoff = np.maximum(prices - long_strike, 0) - np.maximum(prices - short_strike, 0) - debit
        breakeven = long_strike + debit

    return pd.DataFrame({
        "underlying_at_expiration": prices,
        "payoff_per_share": payoff,
        "breakeven": breakeven,
    })