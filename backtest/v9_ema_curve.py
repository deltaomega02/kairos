#!/usr/bin/env python3
"""
v9 EMA Curve Test — EMA 1부터 큰 값까지 체계적 스윕
단일 파라미터로 EMA fast/slow 조합만 변경, 나머지 v7 유지.
"""
import os
import sys
import json
import time
from datetime import datetime

import pandas as pd
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS
from v9_mega_sweep import _load_data, SYMBOLS, SEED, SLIP, DAILY_COST

_DATA = None


def _worker_init():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def v7_params():
    return {
        "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
        "pullback_ema_dist_pct": 1.5,
        "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
        "entry_score_threshold": 40,
        "adx_enter_trending": 30, "adx_exit_trending": 20,
        "atr_high_vol_percentile": 85,
        "orderbook_imbalance_min": 0.55,
        "funding_bias_threshold": 0.0005,
    }


def run_one(args):
    ema_f, ema_s = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()

    params = {**DEFAULT_PARAMS, **v7_params(),
              "ema_fast": ema_f, "ema_slow": ema_s}
    try:
        r = run_shared_backtest(
            _DATA, params, SEED,
            use_funding=True,
            trailing_activation=1.2, trailing_distance=0.1,
            block_sol_long=True,
            skip_years=(2023,),
            daily_cost_usd=DAILY_COST,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=SLIP,
            max_simultaneous=3,
            risk_per_trade=0.015,
            max_leverage=7,
            enabled_symbols=SYMBOLS,
        )
        return {
            "ema_fast": ema_f, "ema_slow": ema_s,
            "profit": r["net_profit"], "dd": r["max_dd"],
            "trades": r["total_trades"], "wr": r["win_rate"],
            "ruined": r["ruined"], "final": r["final_balance"],
        }
    except Exception as e:
        return {"ema_fast": ema_f, "ema_slow": ema_s,
                "error": str(e), "ruined": True,
                "profit": 0, "dd": 0, "trades": 0, "wr": 0}


def main():
    # Wide sweep: ema_fast 1-15, ema_slow 5-35, enforce ema_slow > ema_fast + 2
    tasks = []
    for f in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        for s in [5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 21, 25, 30, 35]:
            if s > f + 2:  # reasonable gap
                tasks.append((f, s))
    print(f"Tasks: {len(tasks)}", flush=True)

    t0 = time.time()
    results = []
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(run_one, tasks), 1):
            results.append(res)
            if i % 30 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)

    # Save
    out = os.path.join("~/Projects/HERMES_백테스팅/v9", "v9_ema_curve.json")
    with open(out, "w") as f:
        json.dump(results, f, default=str, indent=2)

    # Print as table: rows=ema_fast, cols=ema_slow
    fasts = sorted(set(r["ema_fast"] for r in results))
    slows = sorted(set(r["ema_slow"] for r in results))

    print("\n" + "=" * 140)
    print("📊 EMA (fast/slow) × 6년 순이익 (v7 나머지 설정 유지)")
    print("=" * 140)
    print(f"{'fast↓/slow→':<12}", end="")
    for s in slows:
        print(f"{s:>10}", end="")
    print()
    print("-" * 140)
    for f in fasts:
        print(f"{f:<12}", end="")
        for s in slows:
            match = [r for r in results if r["ema_fast"] == f and r["ema_slow"] == s]
            if not match:
                print(f"{'—':>10}", end="")
                continue
            r = match[0]
            if r.get("ruined") or r.get("error"):
                print(f"{'RUIN':>10}", end="")
            elif r["profit"] > 1e6:
                print(f"{r['profit']/1e6:>9.1f}M", end="")
            elif r["profit"] > 1e3:
                print(f"{r['profit']/1e3:>9.0f}k", end="")
            else:
                print(f"{r['profit']:>10.0f}", end="")
        print()

    # Also print DD table
    print("\n" + "=" * 140)
    print("📉 EMA (fast/slow) × 최대 DD (%)")
    print("=" * 140)
    print(f"{'fast↓/slow→':<12}", end="")
    for s in slows:
        print(f"{s:>10}", end="")
    print()
    print("-" * 140)
    for f in fasts:
        print(f"{f:<12}", end="")
        for s in slows:
            match = [r for r in results if r["ema_fast"] == f and r["ema_slow"] == s]
            if not match:
                print(f"{'—':>10}", end="")
                continue
            r = match[0]
            if r.get("ruined") or r.get("error"):
                print(f"{'RUIN':>10}", end="")
            else:
                print(f"{r['dd']:>9.1f}%", end="")
        print()

    # Top 10
    alive = [r for r in results if not r.get("ruined")]
    alive.sort(key=lambda x: x["profit"], reverse=True)
    print("\n" + "=" * 140)
    print("🏆 TOP 10 EMA 조합 (6년 profit 기준)")
    print("=" * 140)
    for i, r in enumerate(alive[:10], 1):
        print(f"  {i:2}. EMA {r['ema_fast']:>2}/{r['ema_slow']:<2}  "
              f"profit=${r['profit']:>12,.0f}  dd={r['dd']:>5.1f}%  "
              f"trades={r['trades']:>4}  wr={r['wr']:>5.1f}%")


if __name__ == "__main__":
    main()
