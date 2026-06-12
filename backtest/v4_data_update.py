#!/usr/bin/env python3
"""v4 데이터 증분 업데이트 — 기존 CSV 끝 이후부터 오늘까지 fetch."""
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVALS = [("15", 900000), ("60", 3600000), ("240", 14400000)]


def fetch_kline(symbol: str, interval: str, interval_ms: int,
                start_ts: int, end_ts: int) -> pd.DataFrame:
    all_candles = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": current_start,
            "end": min(current_start + 999 * interval_ms, end_ts),
            "limit": 1000,
        }
        try:
            resp = requests.get(
                "https://api.bybit.com/v5/market/kline",
                params=params, timeout=15,
            )
            data = resp.json()
        except Exception as e:
            print(f"  ! error: {e}")
            time.sleep(2)
            continue

        if data.get("retCode") != 0:
            print(f"  ! bybit error: {data.get('retMsg')}")
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        for row in rows:
            ts = int(row[0])
            if ts < current_start or ts >= end_ts:
                continue
            all_candles.append({
                "timestamp": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        timestamps = [int(r[0]) for r in rows]
        newest = max(timestamps)
        if newest + interval_ms <= current_start:
            break
        current_start = newest + interval_ms
        time.sleep(0.15)

    if not all_candles:
        return pd.DataFrame()
    return pd.DataFrame(all_candles).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    all_data = []
    current_end = end_ts

    while current_end > start_ts:
        params = {
            "category": "linear", "symbol": symbol,
            "endTime": current_end, "limit": 200,
        }
        try:
            resp = requests.get(
                "https://api.bybit.com/v5/market/funding/history",
                params=params, timeout=15,
            )
            data = resp.json()
        except Exception as e:
            print(f"  ! funding error: {e}")
            time.sleep(2)
            continue

        if data.get("retCode") != 0:
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            if ts < start_ts:
                break
            all_data.append({"timestamp": ts, "funding_rate": float(row["fundingRate"])})

        oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
        if oldest <= start_ts:
            break
        current_end = oldest - 1
        time.sleep(0.15)

    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def update_kline_file(symbol: str, interval: str, interval_ms: int, now_ts: int):
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}_long.csv")
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    if existing.empty:
        print(f"  ! {symbol}_{interval} 기존 파일 없음 - 스킵")
        return

    last_ts = int(existing["timestamp"].iloc[-1])
    start_ts = last_ts + interval_ms
    if start_ts >= now_ts:
        print(f"  ✓ {symbol}_{interval} 최신 (마지막: {datetime.utcfromtimestamp(last_ts/1000)})")
        return

    print(f"  ↓ {symbol}_{interval} {datetime.utcfromtimestamp(start_ts/1000)} ~ ...", end=" ", flush=True)
    new_df = fetch_kline(symbol, interval, interval_ms, start_ts, now_ts)
    if new_df.empty:
        print("신규 데이터 없음")
        return

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(path, index=False)
    last_new = datetime.utcfromtimestamp(int(merged['timestamp'].iloc[-1])/1000)
    print(f"+{len(new_df)}개 ({len(merged)}개 → {last_new})")


def update_funding_file(symbol: str, now_ts: int):
    path = os.path.join(DATA_DIR, f"{symbol}_funding_long.csv")
    existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    if existing.empty:
        print(f"  ! {symbol}_funding 없음")
        return

    last_ts = int(existing["timestamp"].iloc[-1])
    if last_ts + 8 * 3600000 >= now_ts:
        print(f"  ✓ {symbol}_funding 최신 (마지막: {datetime.utcfromtimestamp(last_ts/1000)})")
        return

    print(f"  ↓ {symbol}_funding 증분 fetch...", end=" ", flush=True)
    new_df = fetch_funding(symbol, last_ts + 1, now_ts)
    if new_df.empty:
        print("신규 없음")
        return
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    merged.to_csv(path, index=False)
    print(f"+{len(new_df)}개 ({len(merged)}개)")


def main():
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"[데이터 증분 업데이트] 현재: {datetime.utcfromtimestamp(now_ts/1000)}")
    print(f"데이터 폴더: {DATA_DIR}\n")

    for symbol in SYMBOLS:
        print(f"▶ {symbol}")
        for interval, ms in INTERVALS:
            update_kline_file(symbol, interval, ms, now_ts)
        update_funding_file(symbol, now_ts)
        print()

    print("✓ 업데이트 완료")


if __name__ == "__main__":
    main()
