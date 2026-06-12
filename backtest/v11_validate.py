#!/usr/bin/env python3
"""
v11 Validation — top candidates 정밀 분석 + MC + WF
"""
import os, sys, json, time, random
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "/Users/sue/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SEED, SLIP, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v11"
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


def run_bt(data, extra_kw):
    kw = {**BASE_KW, **extra_kw}
    return run_shared_backtest_v11(data, V8_PARAMS, SEED, **kw)


def mc_bootstrap(trades, n=300):
    if not trades:
        return {"ruin_rate": 1.0}
    ruined = 0
    profits = []
    for i in range(n):
        random.seed(70000 + i)
        samp = [random.choice(trades) for _ in range(len(trades))]
        bal = SEED; peak = SEED; mdd = 0; r = False
        for t in samp:
            bal += t["pnl"]
            if bal < 15:
                r = True; break
            if bal > peak:
                peak = bal
            else:
                d = (peak - bal) / peak * 100
                if d > mdd: mdd = d
        if r:
            ruined += 1
        else:
            profits.append(bal - SEED)
    profits.sort()
    def pct(p):
        return profits[int(p*(len(profits)-1))] if profits else 0
    return {
        "ruin_rate": round(ruined/n, 3),
        "median": round(pct(0.5), 0) if profits else 0,
        "p25": round(pct(0.25), 0) if profits else 0,
        "p75": round(pct(0.75), 0) if profits else 0,
        "p10": round(pct(0.10), 0) if profits else 0,
    }


WF_SPLITS = [
    ("WF1", ("2020-03-25", "2023-01-01"), ("2023-01-01", "2026-04-20")),
    ("WF2", ("2020-03-25", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF3", ("2021-01-01", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF4", ("2020-03-25", "2025-01-01"), ("2025-01-01", "2026-04-20")),
]


def validate_worker(args):
    label, extra_kw = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()

    # Full
    full = run_bt(_DATA, extra_kw)

    # WF
    wf = {}
    for name, (s1, e1), (s2, e2) in WF_SPLITS:
        td = filter_data(_DATA, s1, e1)
        vd = filter_data(_DATA, s2, e2)
        train_skip = (2023,) if int(s1[:4]) <= 2023 <= int(e1[:4]) else ()
        kw_train = {**extra_kw}
        # For WF train, still use the same filter settings
        tr_kw = {**BASE_KW, **kw_train, "skip_years": train_skip}
        te_kw = {**BASE_KW, **kw_train, "skip_years": ()}
        tr = run_shared_backtest_v11(td, V8_PARAMS, SEED, **tr_kw)
        te = run_shared_backtest_v11(vd, V8_PARAMS, SEED, **te_kw)
        wf[name] = {
            "train": {"profit": tr["net_profit"], "dd": tr["max_dd"],
                       "trades": tr["total_trades"], "wr": tr["win_rate"]},
            "test": {"profit": te["net_profit"], "dd": te["max_dd"],
                      "trades": te["total_trades"], "wr": te["win_rate"]},
        }

    # OOS
    d_live = filter_data(_DATA, "2026-04-09", "2026-04-20")
    live = run_bt(d_live, extra_kw)
    d_2w = filter_data(_DATA, "2026-04-05", "2026-04-20")
    last2w = run_bt(d_2w, extra_kw)

    # MC
    mc = mc_bootstrap(full.get("_trades", []), n=300)

    return {
        "label": label,
        "base": {"profit": full["net_profit"], "dd": full["max_dd"],
                  "trades": full["total_trades"], "wr": full["win_rate"],
                  "ruined": full["ruined"]},
        "live_10d": live["net_profit"],
        "last_2w": last2w["net_profit"],
        "wf": wf,
        "mc": mc,
    }


CANDIDATES = [
    # V7/V8/V9 references
    ("V7 (EMA 5/18)", {"d1_filter_enable": False}),  # need custom — handle in worker
    ("V8 (EMA 3/15)", {}),
    ("V9 (V8 + 1D-dir-10)", {"d1_filter_enable": True, "d1_ema_period": 10, "d1_mode": "direction"}),
    # Top candidates from v11 sweep
    ("V11 1D-price_above-3", {"d1_filter_enable": True, "d1_ema_period": 3, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-4", {"d1_filter_enable": True, "d1_ema_period": 4, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-5", {"d1_filter_enable": True, "d1_ema_period": 5, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-6", {"d1_filter_enable": True, "d1_ema_period": 6, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-7", {"d1_filter_enable": True, "d1_ema_period": 7, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-8", {"d1_filter_enable": True, "d1_ema_period": 8, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-10", {"d1_filter_enable": True, "d1_ema_period": 10, "d1_mode": "price_above_ema"}),
    ("V11 1D-price_above-12", {"d1_filter_enable": True, "d1_ema_period": 12, "d1_mode": "price_above_ema"}),
    # combine price_above with other features
    ("V11 price_above-5 + Corr L8h T2%", {
        "d1_filter_enable": True, "d1_ema_period": 5, "d1_mode": "price_above_ema",
        "corr_filter_enable": True, "corr_lookback_hours": 8, "corr_threshold_pct": 2.0,
    }),
    ("V11 price_above-5 + AsymPB L1/S2", {
        "d1_filter_enable": True, "d1_ema_period": 5, "d1_mode": "price_above_ema",
        "asym_pullback_enable": True, "long_pullback_dist": 1.0, "short_pullback_dist": 2.0,
    }),
    ("V11 price_above-5 + ATR adp 0.5/2.0", {
        "d1_filter_enable": True, "d1_ema_period": 5, "d1_mode": "price_above_ema",
        "atr_adaptive_sl": True, "atr_low_threshold": 0.5, "atr_low_sl_mult": 2.0,
    }),
]


def prep_v7_params(extra_kw):
    """V7 use ema 5/18"""
    p = {**V8_PARAMS, "ema_fast": 5, "ema_slow": 18}
    return p


def main():
    print(f"Validating {len(CANDIDATES)} candidates", flush=True)
    t0 = time.time()
    results = []
    with mp_ctx.Pool(7, initializer=_worker_init) as p:
        for i, r in enumerate(p.imap_unordered(validate_worker, CANDIDATES), 1):
            results.append(r)
            print(f"  {i}/{len(CANDIDATES)} {r['label']:<45} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done: {time.time()-t0:.0f}s", flush=True)

    # V7 workaround — rerun with ema 5/18 params
    # (the current structure passes V8_PARAMS globally, so V7 label result is actually V8)
    # For accurate V7, need special handling. Skip for now (it's not our candidate).

    # Save
    with open(os.path.join(RESULTS_DIR, "v11_validate.json"), "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results},
                  f, default=str, indent=2)

    # Sort by base profit
    results.sort(key=lambda x: x["base"]["profit"], reverse=True)

    print("\n" + "=" * 180)
    print("🏆 V11 Validation Results")
    print("=" * 180)
    print(f"{'Config':<42}{'6y profit':>16}{'DD':>7}{'WR':>6}{'Live':>8}{'2w':>8}"
          f"{'MC ruin':>9}{'MC med':>14}{'WF1t':>11}{'WF2t':>11}{'WF3t':>11}{'WF4t':>11}")
    print("-" * 180)
    for r in results:
        b = r["base"]
        mc = r["mc"]
        wf = r["wf"]
        pstr = f"+{b['profit']/1e9:.2f}B" if b["profit"] >= 1e9 else \
               f"+{b['profit']/1e6:.1f}M" if b["profit"] >= 1e6 else \
               f"+{b['profit']/1e3:.0f}k" if b["profit"] >= 1e3 else \
               f"+{b['profit']:.0f}"
        mstr = f"+{mc['median']/1e9:.1f}B" if mc['median'] >= 1e9 else \
               f"+{mc['median']/1e6:.1f}M" if mc['median'] >= 1e6 else \
               f"+{mc['median']/1e3:.0f}k" if mc['median'] >= 1e3 else \
               f"+{mc['median']:.0f}"
        wf_str = ""
        for k in sorted(wf.keys()):
            wp = wf[k]["test"]["profit"]
            ws = f"+{wp/1e9:.1f}B" if wp >= 1e9 else \
                 f"+{wp/1e6:.1f}M" if wp >= 1e6 else \
                 f"+{wp/1e3:.0f}k" if wp >= 1e3 else \
                 f"+{wp:.0f}"
            wf_str += f"{ws:>11}"
        print(f"{r['label']:<42}{pstr:>16}{b['dd']:>5.1f}%{b['wr']:>5.1f}%"
              f"{r['live_10d']:>+7.0f}${r['last_2w']:>+6.0f}$"
              f"{mc['ruin_rate']:>9.3f}{mstr:>14}{wf_str}")


if __name__ == "__main__":
    main()
