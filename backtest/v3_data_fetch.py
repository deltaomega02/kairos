#!/usr/bin/env python3
"""
백테스팅 v3.0 데이터 수집
==========================
- 15분봉 (멀티 TF 필터용)
- 펀딩 레이트 히스토리
- 기존 1H/4H 데이터 재사용
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LONG_START = "2022-01-01"
LONG_END = "2026-04-11"


def fetch_kline(symbol: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Bybit v5 API에서 OHLCV"""
    all_candles = []
    interval_ms_map = {"5": 300000, "15": 900000, "60": 3600000, "240": 14400000}
    interval_ms = interval_ms_map.get(interval, 3600000)
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

        for retry in range(5):
            try:
                resp = requests.get(
                    "https://api.bybit.com/v5/market/kline",
                    params=params, timeout=10,
                )
                data = resp.json()
                if data.get("retCode") == 0:
                    break
                if data.get("retCode") == 10006:
                    time.sleep(2 ** retry)
                    continue
                break
            except Exception:
                time.sleep(2 ** retry)
        else:
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
        current_start = newest + interval_ms
        time.sleep(0.15)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding_rate(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Bybit 펀딩 히스토리"""
    all_data = []
    current_end = end_ts

    while current_end > start_ts:
        params = {
            "category": "linear",
            "symbol": symbol,
            "endTime": current_end,
            "limit": 200,
        }

        for retry in range(5):
            try:
                resp = requests.get(
                    "https://api.bybit.com/v5/market/funding/history",
                    params=params, timeout=10,
                )
                data = resp.json()
                if data.get("retCode") == 0:
                    break
                time.sleep(2 ** retry)
            except Exception:
                time.sleep(2 ** retry)
        else:
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            if ts < start_ts:
                break
            all_data.append({
                "timestamp": ts,
                "funding_rate": float(row["fundingRate"]),
            })

        if not rows:
            break

        oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
        if oldest <= start_ts:
            break
        current_end = oldest - 1
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    print("=" * 60)
    print("백테스팅 v3.0 데이터 수집")
    print(f"기간: {LONG_START} ~ {LONG_END}")
    print("=" * 60)

    start_ts = int(datetime.strptime(LONG_START, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(LONG_END, "%Y-%m-%d").timestamp() * 1000)

    # 15분봉 다운로드
    print("\n[1] 15분봉 다운로드...")
    for symbol in SYMBOLS:
        cache = os.path.join(DATA_DIR, f"{symbol}_15_long.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache)
            if len(df) > 100:
                print(f"  ✓ 캐시: {symbol} 15m ({len(df)}개)")
                continue
        print(f"  ↓ {symbol} 15m...")
        df = fetch_kline(symbol, "15", start_ts, end_ts)
        if len(df) > 0:
            df.to_csv(cache, index=False)
            print(f"  ✓ 저장: {symbol} 15m ({len(df)}개)")

    # 펀딩 히스토리
    print("\n[2] 펀딩 레이트 히스토리...")
    for symbol in SYMBOLS:
        cache = os.path.join(DATA_DIR, f"{symbol}_funding_long.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache)
            if len(df) > 100:
                print(f"  ✓ 캐시: {symbol} 펀딩 ({len(df)}개)")
                continue
        print(f"  ↓ {symbol} 펀딩...")
        df = fetch_funding_rate(symbol, start_ts, end_ts)
        if len(df) > 0:
            df.to_csv(cache, index=False)
            print(f"  ✓ 저장: {symbol} 펀딩 ({len(df)}개)")

    print("\n✓ 데이터 수집 완료")


if __name__ == "__main__":
    main()
