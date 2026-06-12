#!/usr/bin/env python3
"""
v11 Sweep — 남아있는 모든 개선 아이디어 + 조합
"""
import os, sys, json, time, random
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SEED, SLIP, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v11"
os.makedirs(RESULTS_DIR, exist_ok=True)

V8_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 3, "ema_slow": 15,
    "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
    "entry_score_threshold": 40, "pullback_ema_dist_pct": 1.5,
    "adx_enter_trending": 30, "adx_exit_trending": 20,
    "atr_high_vol_percentile": 85,
    "orderbook_imbalance_min": 0.55,
    "funding_bias_threshold": 0.0005,
    "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
    "regime_debounce_bars": 1,
}

# v9 baseline (V8 + 1D EMA10)
V9_BASE = {
    "d1_filter_enable": True, "d1_ema_period": 10, "d1_mode": "direction",
}

BASE_KW = {
    "trailing_activation": 1.2, "trailing_distance": 0.1,
    "skip_years": (2023,), "daily_cost_usd": DAILY_COST, "slippage_pct": SLIP,
    "max_simultaneous": 3, "risk_per_trade": 0.015, "max_leverage": 7,
    "enabled_symbols": SYMBOLS, "block_sol_long": True,
}

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


def run_one(args):
    label, v11_kw, period_name, start, end = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    data = filter_data(_DATA, start, end) if start else _DATA
    kw = {**BASE_KW, **V9_BASE, **v11_kw}
    try:
        r = run_shared_backtest_v11(data, V8_PARAMS, SEED, **kw)
        trades = r.get("_trades", [])
        long_t = sum(1 for t in trades if t["direction"] == "LONG")
        long_w = sum(1 for t in trades if t["direction"] == "LONG" and t["pnl"] > 0)
        short_t = sum(1 for t in trades if t["direction"] == "SHORT")
        short_w = sum(1 for t in trades if t["direction"] == "SHORT" and t["pnl"] > 0)
        return {
            "label": label, "period": period_name,
            "profit": r["net_profit"], "dd": r["max_dd"],
            "trades": r["total_trades"], "wr": r["win_rate"],
            "ruined": r["ruined"],
            "long_n": long_t, "long_w": long_w,
            "short_n": short_t, "short_w": short_w,
        }
    except Exception as e:
        return {"label": label, "period": period_name, "error": str(e),
                "ruined": True, "profit": 0, "dd": 0, "trades": 0, "wr": 0}


# ===== Variants =====
VARIANTS = [
    ("V9 baseline", {}),
]

# 1. ATR Adaptive SL — various combinations
for atr_thr in [0.5, 0.6, 0.8, 1.0]:
    for sl_mult in [1.75, 2.0, 2.5, 3.0]:
        VARIANTS.append((f"V9+ATR adp (thr{atr_thr} mult{sl_mult})", {
            "atr_adaptive_sl": True,
            "atr_low_threshold": atr_thr, "atr_low_sl_mult": sl_mult,
        }))

# 2. Correlation filter
for look in [2, 4, 8]:
    for thr in [1.0, 2.0, 3.0, 5.0]:
        VARIANTS.append((f"V9+Corr L{look}h T{thr}%", {
            "corr_filter_enable": True,
            "corr_lookback_hours": look, "corr_threshold_pct": thr,
        }))

# 3. Time filter
for blocks, label in [
    ([0, 8, 16], "펀딩시"),
    ([0, 1, 8, 9, 16, 17], "펀딩±1h"),
    ([22, 23, 0, 1, 2], "심야"),
    ([0, 8, 16, 22, 23], "펀딩+심야"),
]:
    VARIANTS.append((f"V9+Time {label}", {
        "time_filter_enable": True, "block_hours": blocks,
    }))

# 4. Asymmetric pullback
for lp, sp in [(1.0, 2.0), (1.5, 2.0), (2.0, 1.0), (2.0, 1.5), (1.0, 1.0), (2.5, 1.5)]:
    VARIANTS.append((f"V9+AsymPB L{lp}/S{sp}", {
        "asym_pullback_enable": True,
        "long_pullback_dist": lp, "short_pullback_dist": sp,
    }))

# 5. Partial TP
for ratio, mid in [(0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)]:
    VARIANTS.append((f"V9+PartTP {ratio}@{mid}", {
        "partial_tp_enable": True,
        "partial_tp_ratio": ratio, "partial_tp_pct": mid,
    }))

# 6. 1D filter mode variants
for mode in ["sma", "price_above_ema"]:
    for period in [5, 10, 15, 20]:
        VARIANTS.append((f"V9 1D-{mode}-{period}", {
            "d1_mode": mode, "d1_ema_period": period,
        }))

# 7. 1D ADX filter
for adx_thr in [15, 20, 25, 30]:
    VARIANTS.append((f"V9+1D ADX>={adx_thr}", {
        "d1_adx_enable": True, "d1_adx_threshold": adx_thr,
    }))

# 8. 1D MACD confirmation
VARIANTS.append(("V9+1D MACD", {"d1_macd_enable": True}))

# 9. Vol regime size (고변동 시 50% 축소)
for pct in [0.3, 0.5, 0.7]:
    VARIANTS.append((f"V9+VolRegime {pct}x", {
        "vol_regime_size_enable": True, "vol_regime_size_pct": pct,
    }))

# 10. Combined top candidates (will expand after single-feature results)
# Best guesses:
COMBO_VARIANTS = [
    ("V9+ATR+Corr", {
        "atr_adaptive_sl": True, "atr_low_threshold": 0.8, "atr_low_sl_mult": 2.0,
        "corr_filter_enable": True, "corr_lookback_hours": 4, "corr_threshold_pct": 2.0,
    }),
    ("V9+ATR+Time", {
        "atr_adaptive_sl": True, "atr_low_threshold": 0.8, "atr_low_sl_mult": 2.0,
        "time_filter_enable": True, "block_hours": [0, 8, 16],
    }),
    ("V9+Corr+Time", {
        "corr_filter_enable": True, "corr_lookback_hours": 4, "corr_threshold_pct": 2.0,
        "time_filter_enable": True, "block_hours": [0, 8, 16],
    }),
    ("V9+Corr+ATR+Time", {
        "atr_adaptive_sl": True, "atr_low_threshold": 0.8, "atr_low_sl_mult": 2.0,
        "corr_filter_enable": True, "corr_lookback_hours": 4, "corr_threshold_pct": 2.0,
        "time_filter_enable": True, "block_hours": [0, 8, 16],
    }),
    ("V9+AsymPB+ATR", {
        "asym_pullback_enable": True, "long_pullback_dist": 1.0, "short_pullback_dist": 2.0,
        "atr_adaptive_sl": True, "atr_low_threshold": 0.8, "atr_low_sl_mult": 2.0,
    }),
    ("V9+1D ADX20+MACD", {
        "d1_adx_enable": True, "d1_adx_threshold": 20,
        "d1_macd_enable": True,
    }),
    ("V9+ATR+Corr+Time+AsymPB", {
        "atr_adaptive_sl": True, "atr_low_threshold": 0.8, "atr_low_sl_mult": 2.0,
        "corr_filter_enable": True, "corr_lookback_hours": 4, "corr_threshold_pct": 2.0,
        "time_filter_enable": True, "block_hours": [0, 8, 16],
        "asym_pullback_enable": True, "long_pullback_dist": 1.0, "short_pullback_dist": 2.0,
    }),
]
VARIANTS.extend(COMBO_VARIANTS)

PERIODS = [
    ("Full 6y", "2020-03-25", "2026-04-20"),
    ("실전10d", "2026-04-09", "2026-04-20"),
    ("최근2w", "2026-04-05", "2026-04-20"),
    ("2026 YTD", "2026-01-01", "2026-04-20"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2022 crash", "2022-01-01", "2023-01-01"),
    ("2021 Bull", "2021-01-01", "2022-01-01"),
    ("2020 COVID", "2020-03-25", "2021-01-01"),
]


def main():
    tasks = [(lbl, kw, pl, s, e) for lbl, kw in VARIANTS for pl, s, e in PERIODS]
    print(f"Variants: {len(VARIANTS)}, Total tasks: {len(tasks)}", flush=True)

    t0 = time.time()
    results = []
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, r in enumerate(p.imap_unordered(run_one, tasks), 1):
            results.append(r)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)

    by_var = {}
    for r in results:
        by_var.setdefault(r["label"], {})[r["period"]] = r

    # Save
    path = os.path.join(RESULTS_DIR, "v11_sweep.json")
    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "by_variant": by_var},
                  f, default=str, indent=2)
    print(f"Saved: {path}", flush=True)

    # Sort by 6y profit
    variants = sorted(by_var.keys(),
                       key=lambda v: (by_var[v].get("Full 6y", {}).get("profit", 0)
                                      if not by_var[v].get("Full 6y", {}).get("ruined") else -1e12),
                       reverse=True)
    period_labels = [p[0] for p in PERIODS]

    print("\n" + "=" * 200)
    print("v11 Sweep — 6y profit 기준 순위")
    print("=" * 200)
    header = f"{'Variant':<42}"
    for p in period_labels:
        header += f"{p[:10]:>13}"
    print(header)
    print("-" * 200)
    for v in variants[:40]:
        row = f"{v[:42]:<42}"
        for p in period_labels:
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
            row += f"{cell:>13}"
        print(row)

    # Top 15 with context
    print("\n🏆 TOP 15 (6y profit) — V9 baseline 대비")
    v9_profit = by_var.get("V9 baseline", {}).get("Full 6y", {}).get("profit", 1)
    for i, v in enumerate(variants[:15], 1):
        r = by_var.get(v, {}).get("Full 6y", {})
        live = by_var.get(v, {}).get("실전10d", {})
        last2w = by_var.get(v, {}).get("최근2w", {})
        ratio = (r.get("profit", 0) / v9_profit) if v9_profit > 0 else 0
        print(f"  {i:2}. {v[:42]:<42}  "
              f"6y: ${r.get('profit',0):>+13,.0f}/{r.get('dd',0):>4.1f}%  "
              f"vs V9 {ratio:.2f}x  "
              f"실전 ${live.get('profit',0):+.1f}  2w ${last2w.get('profit',0):+.1f}")


if __name__ == "__main__":
    main()
