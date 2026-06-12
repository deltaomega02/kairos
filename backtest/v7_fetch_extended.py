#!/usr/bin/env python3
"""
v7 확장 데이터 다운로드 (2020-2026, 최대 6년)
===============================================
사용자 피드백: "4년은 짧다. 10년 또는 최대 가용"
Bybit USDT Perp 런칭 시점:
- BTCUSDT: 2020-03
- ETHUSDT: 2020-03
- XRPUSDT: 2020-08
- SOLUSDT: 2021-10

저장: data/ 폴더에 *_long.csv로 덮어쓰기
(기존 4년 데이터를 6년으로 확장)
"""
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
START = "2020-01-01"
END = "2026-04-15"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVALS = [("60", 3600000), ("240", 14400000)]  # 1H, 4H만 (15m은 시간 오래)


def fetch_kline(symbol, interval, interval_ms, start_ts, end_ts):
    all_c = []
    cur = start_ts
    empty_count = 0
    while cur < end_ts:
        params = {
            "category": "linear", "symbol": symbol, "interval": interval,
            "start": cur, "end": min(cur + 999 * interval_ms, end_ts),
            "limit": 1000,
        }
        try:
            r = requests.get("https://api.bybit.com/v5/market/kline",
                             params=params, timeout=15).json()
        except Exception as e:
            print(f"  ! {e}")
            time.sleep(2); continue
        if r.get("retCode") != 0:
            print(f"  ! {r.get('retMsg')}")
            break
        rows = r.get("result", {}).get("list", [])
        if not rows:
            empty_count += 1
            if empty_count > 3:  # 3번 연속 빈 응답 = 데이터 시작 전
                break
            cur += 1000 * interval_ms  # 다음 구간으로
            continue
        empty_count = 0

        for row in rows:
            ts = int(row[0])
            if ts < cur or ts >= end_ts:
                continue
            all_c.append({
                "timestamp": ts,
                "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]),
                "volume": float(row[5]),
            })
        newest = max(int(r_[0]) for r_ in rows)
        if newest + interval_ms <= cur:
            break
        cur = newest + interval_ms
        time.sleep(0.12)

    if not all_c:
        return pd.DataFrame()
    return pd.DataFrame(all_c).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_funding(symbol, start_ts, end_ts):
    all_d = []
    cur_end = end_ts
    empty_count = 0
    while cur_end > start_ts:
        try:
            r = requests.get("https://api.bybit.com/v5/market/funding/history",
                             params={"category": "linear", "symbol": symbol,
                                     "endTime": cur_end, "limit": 200}, timeout=15).json()
        except Exception:
            time.sleep(2); continue
        if r.get("retCode") != 0:
            break
        rows = r.get("result", {}).get("list", [])
        if not rows:
            empty_count += 1
            if empty_count > 3:
                break
            cur_end -= 200 * 8 * 3600 * 1000
            continue
        empty_count = 0
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            if ts < start_ts:
                break
            all_d.append({"timestamp": ts, "funding_rate": float(row["fundingRate"])})
        oldest = min(int(r_["fundingRateTimestamp"]) for r_ in rows)
        if oldest <= start_ts:
            break
        cur_end = oldest - 1
        time.sleep(0.12)
    if not all_d:
        return pd.DataFrame()
    return pd.DataFrame(all_d).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    start_ts = int(datetime.strptime(START, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(END, "%Y-%m-%d").timestamp() * 1000)

    print("="*80)
    print(f"v7 확장 데이터 다운로드 — {START} ~ {END}")
    print("="*80)

    for sym in SYMBOLS:
        print(f"\n▶ {sym}")
        for iv, ms in INTERVALS:
            # 기존 파일을 _long_v7.csv로 다운로드 후 교체
            path = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            backup_path = path.replace(".csv", "_v6_backup.csv")

            # 기존 파일 백업
            if os.path.exists(path) and not os.path.exists(backup_path):
                import shutil
                shutil.copy(path, backup_path)
                print(f"  (기존 v6 데이터 백업: {os.path.basename(backup_path)})")

            print(f"  ↓ {sym}_{iv}...", end=" ", flush=True)
            t0 = time.time()
            df = fetch_kline(sym, iv, ms, start_ts, end_ts)
            elapsed = time.time() - t0
            if df.empty:
                print("실패")
                continue
            df.to_csv(path, index=False)
            first = datetime.utcfromtimestamp(df['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')
            last = datetime.utcfromtimestamp(df['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')
            print(f"{len(df)}개 ({first} ~ {last}) [{elapsed:.0f}s]")

        # 펀딩
        path = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        backup = path.replace(".csv", "_v6_backup.csv")
        if os.path.exists(path) and not os.path.exists(backup):
            import shutil
            shutil.copy(path, backup)

        print(f"  ↓ {sym}_funding...", end=" ", flush=True)
        t0 = time.time()
        df = fetch_funding(sym, start_ts, end_ts)
        if df.empty:
            print("실패")
        else:
            df.to_csv(path, index=False)
            print(f"{len(df)}개 [{time.time()-t0:.0f}s]")

    print("\n✓ 확장 데이터 다운로드 완료")


if __name__ == "__main__":
    main()
