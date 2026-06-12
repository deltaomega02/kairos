#!/usr/bin/env python3
"""
v10 Multi-EMA Sweep
===================
모든 다중 EMA 변형을 체계적으로 테스트.

Phases:
  1. Single feature sweep (each multi-EMA mode alone)
  2. Best-of-each combined
  3. Walk-forward + Monte Carlo on top candidates
  4. Head-to-head: v7, v8, 모든 multi-EMA 승자들

Evaluation on:
  - 6년 Full
  - 실전 기간 (2026-04-09 ~ 04-18)
  - 최근 2주, 2026 YTD, 각 연도
"""
import os, sys, json, time, random
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SYMBOLS, SEED, SLIP, DAILY_COST
from comprehensive_backtest import DEFAULT_PARAMS
from v10_multi_ema_engine import run_shared_backtest_v10

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v10"
os.makedirs(RESULTS_DIR, exist_ok=True)

_DATA = None


def _worker_init():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def filter_data(data, start, end):
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and "timestamp" in v.columns:
            m = (v["timestamp"] >= start_ts) & (v["timestamp"] < end_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


# v8 baseline params (EMA 3/15)
V8_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 3, "ema_slow": 15,
    "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
    "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5,
    "adx_enter_trending": 30, "adx_exit_trending": 20,
    "atr_high_vol_percentile": 85,
    "orderbook_imbalance_min": 0.55,
    "funding_bias_threshold": 0.0005,
    "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
    "regime_debounce_bars": 1,
}
V8_ENGINE_KW = {
    "trailing_activation": 1.2, "trailing_distance": 0.1,
    "skip_years": (2023,),
    "daily_cost_usd": DAILY_COST,
    "slippage_pct": SLIP,
    "max_simultaneous": 3, "risk_per_trade": 0.015, "max_leverage": 7,
    "enabled_symbols": SYMBOLS,
    "block_sol_long": True,
}


def base_kw(**extras):
    kw = dict(V8_ENGINE_KW)
    kw.update(extras)
    return kw


def run_one(args):
    label, mode_kwargs, period_name, start, end = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    data = filter_data(_DATA, start, end)
    kw = base_kw(**mode_kwargs)
    try:
        r = run_shared_backtest_v10(data, V8_PARAMS, SEED, **kw)
        trades = r.get("_trades", [])
        long_w = sum(1 for t in trades if t["direction"] == "LONG" and t["pnl"] > 0)
        long_t = sum(1 for t in trades if t["direction"] == "LONG")
        short_w = sum(1 for t in trades if t["direction"] == "SHORT" and t["pnl"] > 0)
        short_t = sum(1 for t in trades if t["direction"] == "SHORT")
        return {
            "label": label, "period": period_name,
            "profit": r["net_profit"], "dd": r["max_dd"],
            "trades": r["total_trades"], "wr": r["win_rate"],
            "ruined": r["ruined"],
            "long": {"t": long_t, "w": long_w, "wr": round(long_w/max(long_t,1)*100,1)},
            "short": {"t": short_t, "w": short_w, "wr": round(short_w/max(short_t,1)*100,1)},
            "_trades": trades if r["total_trades"] < 50 else None,  # keep short-period trades
        }
    except Exception as e:
        return {"label": label, "period": period_name, "error": str(e),
                "ruined": True, "profit": 0, "dd": 0, "trades": 0, "wr": 0,
                "long": {}, "short": {}}


# ================================================================
# Variant definitions
# ================================================================

# Baselines
VARIANTS = []
VARIANTS.append(("V7 (5/18 no-filter)", {
    # override params via custom: we'll handle via EMA override in V8_PARAMS... actually this needs special
}))
# To compare v7 we'll override PARAMS on the fly. Use a different approach.

# We'll keep V8_PARAMS as baseline and vary.
# v7 baseline requires ema_fast=5, ema_slow=18 — will add as special variant.

# Triple EMA (v8 base + 3rd EMA)
for med in [5, 7, 8, 10, 13]:
    VARIANTS.append((f"Triple(3/{med}/15)", {
        "triple_ema_enable": True,
        "ema_medium_period": med,
    }))

# 1D filter, direction mode
for period in [10, 14, 20, 30, 50]:
    VARIANTS.append((f"1D-dir EMA{period}", {
        "d1_filter_enable": True,
        "d1_ema_period": period,
        "d1_filter_mode": "direction",
    }))

# 1D filter, price above mode
for period in [10, 14, 20, 30, 50]:
    VARIANTS.append((f"1D-above EMA{period}", {
        "d1_filter_enable": True,
        "d1_ema_period": period,
        "d1_filter_mode": "price_above_ema",
    }))

# 4H long EMA filter, direction
for period in [30, 50, 80, 100]:
    VARIANTS.append((f"4H-dir EMA{period}", {
        "h4_long_ema_enable": True,
        "h4_long_ema_period": period,
        "h4_filter_mode": "direction",
    }))

# 4H long EMA filter, price above
for period in [30, 50, 80, 100]:
    VARIANTS.append((f"4H-above EMA{period}", {
        "h4_long_ema_enable": True,
        "h4_long_ema_period": period,
        "h4_filter_mode": "price_above_ema",
    }))

# Ribbon
for ribbon, label in [
    ([3, 5, 8, 13, 21], "Ribbon5 (3-21)"),
    ([3, 8, 13, 21, 34], "Ribbon5 (3-34)"),
    ([5, 8, 13, 21, 34, 55], "Ribbon6 (5-55)"),
    ([3, 5, 8, 13], "Ribbon4 (3-13)"),
    ([3, 5, 8, 13, 21, 34, 55], "Ribbon7 (3-55)"),
]:
    VARIANTS.append((label, {
        "ribbon_enable": True,
        "ribbon_periods": ribbon,
    }))

# Per-direction EMA (various combos)
pd_combos = [
    ((3, 15), (5, 18), "PD long3/15 short5/18"),
    ((3, 15), (5, 21), "PD long3/15 short5/21"),
    ((3, 15), (3, 12), "PD long3/15 short3/12"),
    ((3, 12), (3, 15), "PD long3/12 short3/15"),
    ((5, 18), (3, 15), "PD long5/18 short3/15"),
    ((3, 15), (2, 12), "PD long3/15 short2/12"),
    ((2, 15), (3, 15), "PD long2/15 short3/15"),
]
for (fl, sl), (fs, ss), lbl in pd_combos:
    VARIANTS.append((lbl, {
        "per_direction_enable": True,
        "ema_fast_long": fl, "ema_slow_long": sl,
        "ema_fast_short": fs, "ema_slow_short": ss,
    }))

# Combinations (Phase 2 — will add after single-feature results)

# Periods for eval
PERIODS = [
    ("Full 6y", "2020-03-25", "2026-04-20"),
    ("실전기간", "2026-04-09", "2026-04-20"),
    ("최근2주", "2026-04-05", "2026-04-20"),
    ("2026 YTD", "2026-01-01", "2026-04-20"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2024 Bull", "2024-01-01", "2025-01-01"),
    ("2023 저변동", "2023-01-01", "2024-01-01"),
    ("2022 크래시", "2022-01-01", "2023-01-01"),
    ("2021 대상승", "2021-01-01", "2022-01-01"),
    ("2020 COVID", "2020-03-25", "2021-01-01"),
]


def main():
    # Build task list
    tasks = []
    for lbl, kw in VARIANTS:
        for pl, start, end in PERIODS:
            tasks.append((lbl, kw, pl, start, end))

    # Also add v7 and v8 baselines as reference
    # Using a special trick: override params via kwargs is not directly supported,
    # so we'll run those separately below.

    print(f"Variants: {len(VARIANTS)}, Periods: {len(PERIODS)}, Total tasks: {len(tasks)}", flush=True)

    t0 = time.time()
    results = []
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, r in enumerate(p.imap_unordered(run_one, tasks), 1):
            results.append(r)
            if i % 30 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)

    # Also run v7 and v8 reference
    print("\n[Reference] v7 & v8 baselines", flush=True)
    data = _load_data()
    refs = []
    for lbl, ema_f, ema_s in [("V7 ref (5/18)", 5, 18), ("V8 ref (3/15)", 3, 15)]:
        ref_params = {**V8_PARAMS, "ema_fast": ema_f, "ema_slow": ema_s}
        for pl, start, end in PERIODS:
            d = filter_data(data, start, end)
            r = run_shared_backtest_v10(d, ref_params, SEED, **base_kw())
            trades = r.get("_trades", [])
            long_t = sum(1 for t in trades if t["direction"] == "LONG")
            long_w = sum(1 for t in trades if t["direction"] == "LONG" and t["pnl"] > 0)
            short_t = sum(1 for t in trades if t["direction"] == "SHORT")
            short_w = sum(1 for t in trades if t["direction"] == "SHORT" and t["pnl"] > 0)
            refs.append({
                "label": lbl, "period": pl,
                "profit": r["net_profit"], "dd": r["max_dd"],
                "trades": r["total_trades"], "wr": r["win_rate"],
                "ruined": r["ruined"],
                "long": {"t": long_t, "w": long_w, "wr": round(long_w/max(long_t,1)*100,1)},
                "short": {"t": short_t, "w": short_w, "wr": round(short_w/max(short_t,1)*100,1)},
            })
    results = refs + results

    # Save
    path = os.path.join(RESULTS_DIR, "v10_sweep.json")
    # Strip _trades from saved
    for r in results:
        if "_trades" in r:
            r.pop("_trades")
    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, default=str, indent=2)
    print(f"\nSaved: {path}", flush=True)

    # ================== Display ==================
    # Pivot: rows=variant, cols=period. Show 6y profit + DD, others profit only.
    variants = list(dict.fromkeys([r["label"] for r in results]))
    period_names = [p[0] for p in PERIODS]
    by_var = {}
    for r in results:
        by_var.setdefault(r["label"], {})[r["period"]] = r

    # Sort variants by 6y profit descending
    def six_y_profit(v):
        r = by_var.get(v, {}).get("Full 6y", {})
        if r.get("ruined"):
            return -1e12
        return r.get("profit", 0)

    variants.sort(key=six_y_profit, reverse=True)

    print("\n" + "=" * 180)
    print("📊 v10 Multi-EMA Sweep — 6년 기준 순위, 각 셀: profit / DD%")
    print("=" * 180)
    header = f"{'Variant':<35}"
    for p in period_names:
        header += f"{p[:10]:>14}"
    print(header)
    print("-" * 180)
    for v in variants:
        row = f"{v[:35]:<35}"
        for p in period_names:
            r = by_var.get(v, {}).get(p, {})
            if r.get("ruined"):
                cell = "RUIN"
            else:
                prof = r.get("profit", 0)
                dd = r.get("dd", 0)
                if abs(prof) >= 1e9:
                    ps = f"{prof/1e9:+.1f}B"
                elif abs(prof) >= 1e6:
                    ps = f"{prof/1e6:+.2f}M"
                elif abs(prof) >= 1e3:
                    ps = f"{prof/1e3:+.1f}k"
                else:
                    ps = f"{prof:+.0f}"
                cell = f"{ps}/{dd:.0f}%"
            row += f"{cell:>14}"
        print(row)

    # Top 15 on Full 6y
    print("\n" + "=" * 120)
    print("🏆 TOP 15 Variants by 6Y Profit")
    print("=" * 120)
    for i, v in enumerate(variants[:15], 1):
        r = by_var.get(v, {}).get("Full 6y", {})
        live = by_var.get(v, {}).get("실전기간", {})
        last2w = by_var.get(v, {}).get("최근2주", {})
        print(f"  {i:2}. {v[:35]:<35}  "
              f"6y: ${r.get('profit',0):>12,.0f}/{r.get('dd',0):>4.1f}%  "
              f"trades={r.get('trades',0):>4}  wr={r.get('wr',0):>4.1f}%  "
              f"실전=${live.get('profit',0):>+6.1f}  2w=${last2w.get('profit',0):>+6.1f}")


if __name__ == "__main__":
    main()
