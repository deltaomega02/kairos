#!/usr/bin/env python3
"""V14 Sanity Check — 실전 V12/V13 거래를 tick 데이터로 재현.

목적: 백테스트 엔진 신뢰도 검증.
방법: 운영자 실전 거래 (CSV)의 진입가/청산가를 tick에서 정확히 찾고,
      실제 결과(PnL)와 시뮬 결과 비교.
일치하면 V13 vs V14 비교 백테스트 신뢰 가능.
"""

import gzip
import os
import sys
from datetime import datetime, timezone, timedelta
import pandas as pd

TICK_DIR = "/Users/sue/Projects/HERMES/backtest/tick_data"

# 운영자 실전 BTC 거래 (CSV에서 추출, KST→UTC 변환)
LIVE_TRADES_BTC = [
    {
        "id": 27,
        "entry_kst": "2026-04-20 16:21:20",
        "exit_kst":  "2026-04-20 18:39:00",  # 약 2.3시간 SL
        "direction": "SHORT",
        "entry_price": 74494.30,
        "exit_price": 75143.20,
        "qty": 0.011,
        "expected_pnl": -8.04,
    },
    {
        "id": 30,
        "entry_kst": "2026-04-21 17:47:51",
        "exit_kst":  "2026-04-22 02:33:00",
        "direction": "LONG",
        "entry_price": 75785.10,
        "exit_price": 76786.60,
        "qty": 0.005,
        "expected_pnl": 4.61,
    },
    {
        "id": 31,
        "entry_kst": "2026-04-21 21:43:43",
        "exit_kst":  "2026-04-21 22:25:00",
        "direction": "LONG",
        "entry_price": 76432.52,
        "exit_price": 75826.90,
        "qty": 0.005,
        "expected_pnl": -3.45,
    },
    {
        "id": 37,
        "entry_kst": "2026-04-24 12:32:18",
        "exit_kst":  "2026-04-24 16:00:00",
        "direction": "LONG",
        "entry_price": 78253.70,
        "exit_price": 77523.00,
        "qty": 0.01,
        "expected_pnl": -8.16,
    },
]


def load_tick_day(date_str: str) -> pd.DataFrame:
    """날짜별 BTC tick 데이터 로드 (gzip)."""
    path = os.path.join(TICK_DIR, f"BTCUSDT{date_str}.csv.gz")
    if not os.path.exists(path):
        return pd.DataFrame()
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    df = df[["timestamp", "side", "size", "price"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df.dropna()


def kst_to_utc_ts(kst_str: str) -> float:
    """KST 문자열 → UTC unix timestamp (초)."""
    kst = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S")
    utc = kst - timedelta(hours=9)
    return utc.replace(tzinfo=timezone.utc).timestamp()


def find_price_near(ticks: pd.DataFrame, target_ts: float, window: float = 30) -> dict:
    """대상 timestamp 근처 ±window초 tick에서 평균/최대/최소가 추출."""
    mask = (ticks["timestamp"] >= target_ts - window) & (ticks["timestamp"] <= target_ts + window)
    near = ticks[mask]
    if len(near) == 0:
        return {"count": 0}
    return {
        "count": len(near),
        "avg": near["price"].mean(),
        "min": near["price"].min(),
        "max": near["price"].max(),
        "first": near.iloc[0]["price"],
        "last": near.iloc[-1]["price"],
    }


def main():
    print("=" * 70)
    print("V14 Sanity Check — 실전 BTC 거래 tick 재현")
    print("=" * 70)
    print()

    # 모든 날짜 데이터 로드
    dates = ["2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23",
             "2026-04-24", "2026-04-25"]
    daily_ticks = {}
    for d in dates:
        df = load_tick_day(d)
        if len(df) > 0:
            daily_ticks[d] = df
            print(f"  {d}: {len(df):>9,} ticks (가격 ${df['price'].min():,.0f}~${df['price'].max():,.0f})")
    print()

    # 각 거래 검증
    results = []
    for trade in LIVE_TRADES_BTC:
        entry_ts = kst_to_utc_ts(trade["entry_kst"])
        exit_ts = kst_to_utc_ts(trade["exit_kst"])

        entry_date = trade["entry_kst"][:10]
        exit_date = trade["exit_kst"][:10]

        # 진입 시점 tick 검색
        if entry_date in daily_ticks:
            entry_check = find_price_near(daily_ticks[entry_date], entry_ts, window=60)
        else:
            entry_check = {"count": 0}

        # 청산 시점 tick 검색
        if exit_date in daily_ticks:
            exit_check = find_price_near(daily_ticks[exit_date], exit_ts, window=300)
        else:
            exit_check = {"count": 0}

        results.append({
            "id": trade["id"],
            "kst": trade["entry_kst"],
            "dir": trade["direction"],
            "expected_entry": trade["entry_price"],
            "tick_entry": entry_check.get("avg", 0),
            "expected_exit": trade["exit_price"],
            "tick_exit": exit_check.get("avg", 0),
            "expected_pnl": trade["expected_pnl"],
            "entry_ticks": entry_check.get("count", 0),
            "exit_ticks": exit_check.get("count", 0),
        })

    # 출력
    print("=" * 70)
    print("실전 vs Tick 매칭 결과")
    print("=" * 70)
    print(f"{'ID':>3} {'KST':>16} {'DIR':>5} | {'실제진입':>10} {'Tick평균':>10} {'갭':>7} | {'실제청산':>10} {'Tick평균':>10} {'갭':>7}")
    print("-" * 100)

    for r in results:
        entry_gap = (r["tick_entry"] - r["expected_entry"]) if r["tick_entry"] else 0
        exit_gap = (r["tick_exit"] - r["expected_exit"]) if r["tick_exit"] else 0
        entry_gap_pct = (entry_gap / r["expected_entry"] * 100) if r["expected_entry"] else 0
        exit_gap_pct = (exit_gap / r["expected_exit"] * 100) if r["expected_exit"] else 0

        if r["entry_ticks"] == 0:
            tick_e_str = f"{'(no data)':>10}"
        else:
            tick_e_str = f"${r['tick_entry']:>9,.0f}"

        if r["exit_ticks"] == 0:
            tick_x_str = f"{'(no data)':>10}"
        else:
            tick_x_str = f"${r['tick_exit']:>9,.0f}"

        print(f"#{r['id']:>2} {r['kst'][5:16]:>16} {r['dir']:>5} | "
              f"${r['expected_entry']:>9,.0f} {tick_e_str} {entry_gap_pct:+6.2f}% | "
              f"${r['expected_exit']:>9,.0f} {tick_x_str} {exit_gap_pct:+6.2f}%")

    print()
    print("=" * 70)
    print("판정")
    print("=" * 70)

    # 진입가 매칭 정확도
    matched_entries = [r for r in results if r["entry_ticks"] > 0]
    if matched_entries:
        avg_gap = sum(abs(r["tick_entry"] - r["expected_entry"]) / r["expected_entry"] * 100
                      for r in matched_entries) / len(matched_entries)
        print(f"진입가 매칭: {len(matched_entries)}/{len(results)} 거래 / 평균 갭 {avg_gap:.3f}%")
        if avg_gap < 0.1:
            print("  → ✅ 매칭 매우 정확 (0.1% 미만) — Tick 데이터 신뢰 가능")
        elif avg_gap < 0.5:
            print("  → 🟡 매칭 양호 (0.1~0.5%) — 슬리피지 범위 내")
        else:
            print("  → 🔴 매칭 부정확 (0.5%+) — 시간/데이터 검토 필요")

    matched_exits = [r for r in results if r["exit_ticks"] > 0]
    if matched_exits:
        avg_gap = sum(abs(r["tick_exit"] - r["expected_exit"]) / r["expected_exit"] * 100
                      for r in matched_exits) / len(matched_exits)
        print(f"청산가 매칭: {len(matched_exits)}/{len(results)} 거래 / 평균 갭 {avg_gap:.3f}%")


if __name__ == "__main__":
    main()
