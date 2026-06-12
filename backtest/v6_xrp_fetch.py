#!/usr/bin/env python3
"""XRP 데이터 다운로드 — 완전 out-of-sample 검증용"""
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
START = "2022-01-01"
END = "2026-04-15"

INTERVALS = [("15", 900000), ("60", 3600000), ("240", 14400000)]


def fetch_kline(symbol, interval, interval_ms, start_ts, end_ts):
    all_c = []
    cur = start_ts
    while cur < end_ts:
        params = {"category": "linear", "symbol": symbol, "interval": interval,
                  "start": cur, "end": min(cur + 999 * interval_ms, end_ts), "limit": 1000}
        try:
            r = requests.get("https://api.bybit.com/v5/market/kline",
                             params=params, timeout=15).json()
        except Exception as e:
            print(f"  ! {e}")
            time.sleep(2); continue
        if r.get("retCode") != 0:
            print(f"  ! {r.get('retMsg')}"); break
        rows = r.get("result", {}).get("list", [])
        if not rows: break
        for row in rows:
            ts = int(row[0])
            if ts < cur or ts >= end_ts: continue
            all_c.append({"timestamp": ts, "open": float(row[1]), "high": float(row[2]),
                          "low": float(row[3]), "close": float(row[4]), "volume": float(row[5])})
        newest = max(int(r_[0]) for r_ in rows)
        if newest + interval_ms <= cur: break
        cur = newest + interval_ms
        time.sleep(0.12)
    if not all_c: return pd.DataFrame()
    return pd.DataFrame(all_c).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding(symbol, start_ts, end_ts):
    all_d = []
    cur_end = end_ts
    while cur_end > start_ts:
        try:
            r = requests.get("https://api.bybit.com/v5/market/funding/history",
                             params={"category": "linear", "symbol": symbol,
                                     "endTime": cur_end, "limit": 200}, timeout=15).json()
        except Exception: time.sleep(2); continue
        if r.get("retCode") != 0: break
        rows = r.get("result", {}).get("list", [])
        if not rows: break
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            if ts < start_ts: break
            all_d.append({"timestamp": ts, "funding_rate": float(row["fundingRate"])})
        oldest = min(int(r_["fundingRateTimestamp"]) for r_ in rows)
        if oldest <= start_ts: break
        cur_end = oldest - 1
        time.sleep(0.12)
    if not all_d: return pd.DataFrame()
    return pd.DataFrame(all_d).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    sym = "XRPUSDT"
    start_ts = int(datetime.strptime(START, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(END, "%Y-%m-%d").timestamp() * 1000)

    print(f"[{sym} 데이터 다운로드] {START} ~ {END}")

    for iv, ms in INTERVALS:
        path = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
        print(f"  ↓ {sym}_{iv}...", end=" ", flush=True)
        df = fetch_kline(sym, iv, ms, start_ts, end_ts)
        if df.empty:
            print("실패"); continue
        df.to_csv(path, index=False)
        print(f"{len(df)}개 ({datetime.utcfromtimestamp(df['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')} ~ "
              f"{datetime.utcfromtimestamp(df['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')})")

    # 펀딩
    path = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
    print(f"  ↓ {sym}_funding...", end=" ", flush=True)
    df = fetch_funding(sym, start_ts, end_ts)
    if not df.empty:
        df.to_csv(path, index=False)
        print(f"{len(df)}개")
    else:
        print("실패")

    print("\n✓ XRP 데이터 완료")


if __name__ == "__main__":
    main()
