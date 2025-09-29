# =============================
# File: daily_feature_logger.py
# =============================

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import kurtosis, skew
from supabase import create_client
from datetime import timezone

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))

def get_ml_tracked_pairs():
    res = supabase.table("hn_unique_asset_pairs") \
        .select("asset_pair, exchange_name") \
        .eq("is_active", True) \
        .eq("is_ml_trained", True) \
        .execute()
    return [(r['asset_pair'], r['exchange_name']) for r in res.data]

def fetch_market_data(asset_pair, exchange, start_ts, end_ts):
    res = supabase.table("hn_market_data") \
        .select("*") \
        .eq("asset_pair", asset_pair) \
        .eq("exchange_name", exchange) \
        .gte("timestamp", start_ts.isoformat()) \
        .lte("timestamp", end_ts.isoformat()) \
        .order("timestamp") \
        .execute()
    data = pd.DataFrame(res.data)
    if data.empty:
        raise Exception("No market data found.")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def fetch_strategy(asset_pair, exchange):
    res = supabase.table("hn_strategies") \
        .select("*") \
        .eq("asset_pair", asset_pair) \
        .eq("exchange_name", exchange) \
        .eq("strategy_type", "rsi") \
        .eq("is_active", True) \
        .limit(1) \
        .execute()
    return res.data[0] if res.data else None

def fetch_signals(strategy_id, start_ts, end_ts):
    res = supabase.table("hn_trading_signals") \
        .select("*") \
        .eq("strategy_id", strategy_id) \
        .eq("executed", False) \
        .gte("timestamp", start_ts.isoformat()) \
        .lte("timestamp", end_ts.isoformat()) \
        .order("timestamp") \
        .execute()
    return pd.DataFrame(res.data)

def compute_features(df):
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna()

    return {
        "avg_volume": df["volume"].mean(),
        "volatility": df["close"].std(),
        "mean_close": df["close"].mean(),
        "median_close": df["close"].median(),
        "rsi_mean": df["close"].rolling(window=14).mean().mean(),
        "rsi_min": df["close"].rolling(window=14).mean().min(),
        "rsi_max": df["close"].rolling(window=14).mean().max(),
        "close_skewness": skew(df["close"]),
        "close_kurtosis": kurtosis(df["close"])
    }

def compute_performance(signals_df, asset_pair, exchange_name, strategy_id, start_ts, end_ts):
    if signals_df.empty:
        return {
            "total_profit": 0, "profit_pct": 0,
            "sharpe_ratio": 0, "win_rate_pct": 0,
            "max_drawdown_pct": 0
        }

    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

    filtered = signals_df[
        (signals_df["asset_pair"] == asset_pair) &
        (signals_df["exchange_name"] == exchange_name) &
        (signals_df["strategy_id"] == strategy_id) &
        (signals_df["timestamp"] >= start_ts) &
        (signals_df["timestamp"] <= end_ts)
    ].sort_values("timestamp").reset_index(drop=True)


    if filtered.empty:
        return {
            "total_profit": 0, "profit_pct": 0,
            "sharpe_ratio": 0, "win_rate_pct": 0,
            "max_drawdown_pct": 0
        }

    profits = []
    buy_price = None

    for _, row in filtered.iterrows():
        if row["signal_type"] == "BUY":
            buy_price = row["price"]
        elif row["signal_type"] == "SELL" and buy_price is not None:
            sell_price = row["price"]
            profit = sell_price - buy_price
            profits.append(profit)
            buy_price = None  # reset for next round-trip

    if not profits:
        return {
            "total_profit": 0, "profit_pct": 0,
            "sharpe_ratio": 0, "win_rate_pct": 0,
            "max_drawdown_pct": 0
        }

    profits = pd.Series(profits)
    total_profit = profits.sum()
    profit_pct = (total_profit / 1000) * 100  # Example: 1000 USDT base
    sharpe_ratio = profits.mean() / (profits.std() + 1e-9) * np.sqrt(252)
    win_rate = (profits > 0).sum() / len(profits) * 100
    drawdowns = profits.cumsum().cummax() - profits.cumsum()
    max_drawdown = drawdowns.max()

    return {
        "total_profit": float(total_profit),
        "profit_pct": float(profit_pct),
        "sharpe_ratio": float(sharpe_ratio),
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_drawdown)
    }


def log_training_row(asset_pair, exchange):
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=1)

    market_df = fetch_market_data(asset_pair, exchange, start_ts, end_ts)
    strategy = fetch_strategy(asset_pair, exchange)
    if not strategy:
        print("No active RSI strategy found.")
        return

    strategy_id = strategy["id"]
    signals_df = fetch_signals(strategy_id, start_ts, end_ts)

    features = compute_features(market_df)
    metrics = compute_performance(signals_df, asset_pair, exchange, strategy_id, start_ts, end_ts)

    row = {
        "asset_pair": asset_pair,
        "exchange_name": exchange,
        **features,
        "rsi_period": strategy["rsi_period"],
        "overbought_level": strategy["overbought_level"],
        "oversold_level": strategy["oversold_level"],
        "stop_loss_pct": strategy["stop_loss_pct"],
        "profit_target_pct": strategy["profit_target_pct"],
        **metrics
    }

    supabase.table("hn_ml_training_rsi").insert(row).execute()
    print("Training row inserted:", row)

def main():
    pairs = get_ml_tracked_pairs()
    if not pairs:
        print("No ML-tracked asset pairs found.")
        return
    for asset_pair, exchange in pairs:
        try:
            print(f"\n🔍 Logging for {asset_pair} on {exchange}...")
            log_training_row(asset_pair, exchange)
        except Exception as e:
            print(f"⚠️ Error logging {asset_pair} on {exchange}: {e}")

if __name__ == "__main__":
    main()