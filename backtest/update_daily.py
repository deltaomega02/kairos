#!/usr/bin/env python3
"""일봉 데이터 증분 업데이트 — V11 1D 필터 용."""
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = "D"
INTERVAL_MS = 86_400_000


def fetch_kline(symbol, start_ts, end_ts):
    all_candles = []
    current_start = start_ts
    while current_start < end_ts:
        params = {
            "category": "linear", "symbol": symbol, "interval": INTERVAL,
            "start": current_start,
            "end": min(current_start + 999 * INTERVAL_MS, end_ts),
            "limit": 1000,
        }
        try:
            resp = requests.get("https://api.bybit.com/v5/market/kline", params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"  ! error: {e}"); time.sleep(2); continue

        if data.get("retCode") != 0:
            print(f"  ! bybit error: {data.get('retMsg')}"); break
        rows = data.get("result", {}).get("list", [])
        if not rows: break
        for row in rows:
            ts = int(row[0])
            if ts < current_start or ts >= end_ts: continue
            all_candles.append({"timestamp": ts, "open": float(row[1]), "high": float(row[2]),
                                "low": float(row[3]), "close": float(row[4]), "volume": float(row[5])})
        newest = max(int(r[0]) for r in rows)
        if newest + INTERVAL_MS <= current_start: break
        current_start = newest + INTERVAL_MS
        time.sleep(0.15)
    if not all_candles: return pd.DataFrame()
    return pd.DataFrame(all_candles).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"[일봉 업데이트] {datetime.utcfromtimestamp(now_ts/1000)}")
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_D.csv")
        existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if existing.empty:
            print(f"  ! {symbol}_D 없음"); continue
        last_ts = int(existing["timestamp"].iloc[-1])
        start_ts = last_ts + INTERVAL_MS
        if start_ts >= now_ts:
            print(f"  ✓ {symbol}_D 최신")
            continue
        print(f"  ↓ {symbol}_D {datetime.utcfromtimestamp(start_ts/1000)} ~ ...", end=" ", flush=True)
        new_df = fetch_kline(symbol, start_ts, now_ts)
        if new_df.empty:
            print("없음"); continue
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        merged.to_csv(path, index=False)
        last_new = datetime.utcfromtimestamp(int(merged['timestamp'].iloc[-1])/1000)
        print(f"+{len(new_df)}개 (총 {len(merged)}개 → {last_new})")


if __name__ == "__main__":
    main()
