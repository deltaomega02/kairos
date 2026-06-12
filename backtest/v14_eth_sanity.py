#!/usr/bin/env python3
"""V14 ETH Sanity — V12 ETH 실전 거래 13건을 tick으로 재현."""

import gzip
import os
from datetime import datetime, timezone, timedelta
import pandas as pd

TICK_DIR = "/Users/sue/Projects/HERMES/backtest/tick_data"

# V12 시기 ETH 실전 거래 (CSV 기반)
LIVE_TRADES_ETH = [
    {"id": 10, "entry_kst": "2026-04-13 05:43:18", "exit_kst": "2026-04-13 06:25:00",
     "direction": "SHORT", "entry_price": 2197.87, "exit_price": 2214.71, "pnl": -7.71},
    {"id": 13, "entry_kst": "2026-04-13 23:44:32", "exit_kst": "2026-04-14 12:30:00",
     "direction": "SHORT", "entry_price": 2203.45, "exit_price": 2225.75, "pnl": -7.18},
    {"id": 16, "entry_kst": "2026-04-16 03:55:20", "exit_kst": "2026-04-16 09:30:00",
     "direction": "LONG", "entry_price": 2329.85, "exit_price": 2356.31, "pnl": 7.16},
    {"id": 18, "entry_kst": "2026-04-16 19:03:55", "exit_kst": "2026-04-17 06:10:00",
     "direction": "LONG", "entry_price": 2360.76, "exit_price": 2331.77, "pnl": -7.62},
    {"id": 21, "entry_kst": "2026-04-16 22:54:20", "exit_kst": "2026-04-17 02:30:00",
     "direction": "SHORT", "entry_price": 2334.92, "exit_price": 2307.97, "pnl": 8.78},
    {"id": 22, "entry_kst": "2026-04-17 04:59:19", "exit_kst": "2026-04-17 08:30:00",
     "direction": "SHORT", "entry_price": 2327.82, "exit_price": 2360.80, "pnl": -6.40},
    {"id": 24, "entry_kst": "2026-04-18 17:23:01", "exit_kst": "2026-04-19 04:30:00",
     "direction": "LONG", "entry_price": 2418.91, "exit_price": 2379.23, "pnl": -7.50},
    {"id": 31, "entry_kst": "2026-04-21 21:43:21", "exit_kst": "2026-04-21 22:25:00",
     "direction": "LONG", "entry_price": 2319.22, "exit_price": 2296.98, "pnl": -8.67},
    {"id": 33, "entry_kst": "2026-04-22 02:20:05", "exit_kst": "2026-04-22 05:35:00",
     "direction": "SHORT", "entry_price": 2311.94, "exit_price": 2285.49, "pnl": 6.89},
    {"id": 34, "entry_kst": "2026-04-22 11:10:51", "exit_kst": "2026-04-22 19:25:00",
     "direction": "SHORT", "entry_price": 2303.62, "exit_price": 2332.97, "pnl": -7.65},
    {"id": 35, "entry_kst": "2026-04-22 14:19:22", "exit_kst": "2026-04-22 14:37:00",
     "direction": "LONG", "entry_price": 2361.74, "exit_price": 2389.58, "pnl": 5.80},
    {"id": 36, "entry_kst": "2026-04-22 22:31:45", "exit_kst": "2026-04-23 03:00:00",
     "direction": "LONG", "entry_price": 2386.48, "exit_price": 2414.58, "pnl": 5.86},
    {"id": 37, "entry_kst": "2026-04-23 09:09:20", "exit_kst": "2026-04-23 14:15:00",
     "direction": "LONG", "entry_price": 2392.29, "exit_price": 2368.01, "pnl": -7.77},
]


def load_tick_day(symbol: str, date: str) -> pd.DataFrame:
    path = os.path.join(TICK_DIR, f"{symbol}{date}.csv.gz")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
        return df[["timestamp", "side", "size", "price"]].copy()
    except Exception as e:
        print(f"  로드 실패 {date}: {e}")
        return pd.DataFrame()


def kst_to_utc_ts(kst_str: str) -> float:
    kst = datetime.strptime(kst_str, "%Y-%m-%d %H:%M:%S")
    utc = kst - timedelta(hours=9)
    return utc.replace(tzinfo=timezone.utc).timestamp()


def find_exact_price(ticks: pd.DataFrame, target_ts: float, target_price: float, window: float = 120):
    """타겟 가격에 가장 근접한 tick 찾기 (±window 초 내)."""
    mask = (ticks["timestamp"] >= target_ts - window) & (ticks["timestamp"] <= target_ts + window)
    near = ticks[mask]
    if len(near) == 0:
        return None
    # 가격 차이 절댓값 최소
    diffs = (near["price"] - target_price).abs()
    idx = diffs.idxmin()
    closest = near.loc[idx]
    return {
        "actual_ts": closest["timestamp"],
        "actual_price": closest["price"],
        "delta_sec": closest["timestamp"] - target_ts,
        "delta_price": closest["price"] - target_price,
    }


def main():
    print("=" * 80)
    print("V14 ETH Sanity — V12 ETH 실전 거래 13건 tick 재현")
    print("=" * 80)
    print()

    # 모든 ETH 데이터 로드
    dates = sorted(set([t["entry_kst"][:10] for t in LIVE_TRADES_ETH] +
                       [t["exit_kst"][:10] for t in LIVE_TRADES_ETH]))

    daily = {}
    for d in dates:
        df = load_tick_day("ETHUSDT", d)
        if len(df) > 0:
            daily[d] = df
            print(f"  {d}: {len(df):>9,} ticks (${df['price'].min():.2f}~${df['price'].max():.2f})")

    if not daily:
        print("ETH tick 데이터 없음 - 다운로드 대기 중일 수 있음")
        return

    print()
    print("=" * 80)
    print(f"{'ID':>3} {'KST':>16} {'DIR':>5} | {'실제진입':>9} {'tick근접':>9} Δsec Δprc% | "
          f"{'실제청산':>9} {'tick근접':>9} Δsec Δprc%")
    print("-" * 100)

    entry_gaps = []
    exit_gaps = []
    pnl_recreated = []

    for t in LIVE_TRADES_ETH:
        entry_date = t["entry_kst"][:10]
        exit_date = t["exit_kst"][:10]
        entry_ts = kst_to_utc_ts(t["entry_kst"])
        exit_ts = kst_to_utc_ts(t["exit_kst"])

        e_check = None
        x_check = None

        if entry_date in daily:
            e_check = find_exact_price(daily[entry_date], entry_ts, t["entry_price"], 300)
        if exit_date in daily:
            x_check = find_exact_price(daily[exit_date], exit_ts, t["exit_price"], 1800)

        if e_check:
            e_str = f"${e_check['actual_price']:>8.2f}"
            e_dt = f"{e_check['delta_sec']:+5.0f}"
            e_pct = (e_check['delta_price'] / t['entry_price']) * 100
            e_pct_str = f"{e_pct:+.3f}"
            entry_gaps.append(abs(e_pct))
        else:
            e_str = "(no data)"
            e_dt = "-"
            e_pct_str = "-"

        if x_check:
            x_str = f"${x_check['actual_price']:>8.2f}"
            x_dt = f"{x_check['delta_sec']:+5.0f}"
            x_pct = (x_check['delta_price'] / t['exit_price']) * 100
            x_pct_str = f"{x_pct:+.3f}"
            exit_gaps.append(abs(x_pct))
        else:
            x_str = "(no data)"
            x_dt = "-"
            x_pct_str = "-"

        print(f"#{t['id']:>2} {t['entry_kst'][5:16]:>16} {t['direction']:>5} | "
              f"${t['entry_price']:>8.2f} {e_str:>9} {e_dt:>4} {e_pct_str:>6} | "
              f"${t['exit_price']:>8.2f} {x_str:>9} {x_dt:>4} {x_pct_str:>6}")

    print()
    print("=" * 80)
    print("종합 매칭 정확도")
    print("=" * 80)

    if entry_gaps:
        print(f"진입가 매칭: 평균 {sum(entry_gaps)/len(entry_gaps):.3f}% / "
              f"최대 {max(entry_gaps):.3f}% / 최소 {min(entry_gaps):.3f}%")
    if exit_gaps:
        print(f"청산가 매칭: 평균 {sum(exit_gaps)/len(exit_gaps):.3f}% / "
              f"최대 {max(exit_gaps):.3f}% / 최소 {min(exit_gaps):.3f}%")

    # 가격 매칭이 0.05% 이하인 경우 = tick에 그 가격이 실제로 존재
    near_perfect_entry = sum(1 for g in entry_gaps if g < 0.05)
    near_perfect_exit = sum(1 for g in exit_gaps if g < 0.05)
    print()
    print(f"진입가 0.05% 이내 매칭: {near_perfect_entry}/{len(entry_gaps)}")
    print(f"청산가 0.05% 이내 매칭: {near_perfect_exit}/{len(exit_gaps)}")

    if near_perfect_entry / max(len(entry_gaps), 1) >= 0.7:
        print("→ ✅ Tick 데이터 신뢰 가능 (대부분 진입가 정확 매칭)")
    elif near_perfect_entry / max(len(entry_gaps), 1) >= 0.4:
        print("→ 🟡 Tick 부분 신뢰 (절반 정도 매칭)")
    else:
        print("→ 🔴 Tick 매칭 부족 — 시간 정확도/실전 시간 검토 필요")


if __name__ == "__main__":
    main()
