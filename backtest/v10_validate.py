#!/usr/bin/env python3
"""
v10 Validation — MC + Walk-Forward on top multi-EMA variants
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
V7_PARAMS = {**V8_PARAMS, "ema_fast": 5, "ema_slow": 18}


def base_kw():
    return {
        "trailing_activation": 1.2, "trailing_distance": 0.1,
        "skip_years": (2023,),
        "daily_cost_usd": DAILY_COST,
        "slippage_pct": SLIP,
        "max_simultaneous": 3, "risk_per_trade": 0.015, "max_leverage": 7,
        "enabled_symbols": SYMBOLS,
        "block_sol_long": True,
    }


# Candidates to validate (most promising)
CANDIDATES = [
    # Baselines
    ("V7 (EMA 5/18)", V7_PARAMS, {}),
    ("V8 (EMA 3/15)", V8_PARAMS, {}),
    # 1D filters (top)
    ("1D-dir EMA10", V8_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction"}),
    ("1D-above EMA10", V8_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "price_above_ema"}),
    ("1D-dir EMA14", V8_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 14, "d1_filter_mode": "direction"}),
    ("1D-dir EMA20", V8_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 20, "d1_filter_mode": "direction"}),
    ("1D-dir EMA30", V8_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 30, "d1_filter_mode": "direction"}),
    # 4H filters (top)
    ("4H-dir EMA30", V8_PARAMS, {"h4_long_ema_enable": True, "h4_long_ema_period": 30, "h4_filter_mode": "direction"}),
    ("4H-dir EMA50", V8_PARAMS, {"h4_long_ema_enable": True, "h4_long_ema_period": 50, "h4_filter_mode": "direction"}),
    # 1D + 4H combined
    ("1D-10 + 4H-30", V8_PARAMS, {
        "d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction",
        "h4_long_ema_enable": True, "h4_long_ema_period": 30, "h4_filter_mode": "direction",
    }),
    ("1D-20 + 4H-50", V8_PARAMS, {
        "d1_filter_enable": True, "d1_ema_period": 20, "d1_filter_mode": "direction",
        "h4_long_ema_enable": True, "h4_long_ema_period": 50, "h4_filter_mode": "direction",
    }),
    # 1D filter + Triple EMA
    ("1D-10 + Triple(3/8/15)", V8_PARAMS, {
        "d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction",
        "triple_ema_enable": True, "ema_medium_period": 8,
    }),
    ("1D-10 + Triple(3/13/15)", V8_PARAMS, {
        "d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction",
        "triple_ema_enable": True, "ema_medium_period": 13,
    }),
    # 1D on V7 base (see if v7 also benefits)
    ("V7 + 1D-dir EMA10", V7_PARAMS, {"d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction"}),
]


WF_SPLITS = [
    ("WF1 train20-22/test23-26", ("2020-03-25", "2023-01-01"), ("2023-01-01", "2026-04-20")),
    ("WF2 train20-23/test24-26", ("2020-03-25", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF3 train21-23/test24-26", ("2021-01-01", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF4 train20-24/test25-26", ("2020-03-25", "2025-01-01"), ("2025-01-01", "2026-04-20")),
]


def run_bt_and_trades(data, params, kw_extras):
    try:
        r = run_shared_backtest_v10(data, params, SEED, **base_kw(), **kw_extras)
        return r
    except Exception as e:
        return {"error": str(e), "ruined": True, "net_profit": 0, "max_dd": 0,
                "total_trades": 0, "win_rate": 0, "_trades": []}


def mc_bootstrap(trades, n=300):
    if not trades:
        return {"ruin_rate": 1.0, "n_iter": 0}
    ruined = 0
    profits = []
    for i in range(n):
        random.seed(50000 + i)
        samp = [random.choice(trades) for _ in range(len(trades))]
        bal = SEED
        peak = SEED
        mdd = 0
        r = False
        for t in samp:
            bal += t["pnl"]
            if bal < 15:
                r = True
                break
            if bal > peak:
                peak = bal
            else:
                d = (peak - bal) / peak * 100
                if d > mdd:
                    mdd = d
        if r:
            ruined += 1
        else:
            profits.append(bal - SEED)
    profits.sort()
    def pct(p):
        return profits[int(p * (len(profits)-1))] if profits else 0
    return {
        "ruin_rate": round(ruined / n, 3),
        "n_iter": n,
        "median": round(pct(0.5), 0) if profits else 0,
        "p25": round(pct(0.25), 0) if profits else 0,
        "p75": round(pct(0.75), 0) if profits else 0,
        "p10": round(pct(0.10), 0) if profits else 0,
        "p90": round(pct(0.90), 0) if profits else 0,
    }


def validate_worker(args):
    label, params, kw_extras = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()

    # Full 6y
    full = run_bt_and_trades(_DATA, params, kw_extras)

    # WF
    wf_results = {}
    for name, (s1, e1), (s2, e2) in WF_SPLITS:
        td = filter_data(_DATA, s1, e1)
        vd = filter_data(_DATA, s2, e2)
        train_skip_year = 2023 if (int(s1[:4]) <= 2023 <= int(e1[:4])) else None
        kw_train = dict(kw_extras)
        if train_skip_year is None:
            # Only override if our base skip includes 2023 but WF doesn't need it
            pass
        tr = run_bt_and_trades(td, params, kw_extras)
        te = run_bt_and_trades(vd, params, kw_extras)
        wf_results[name] = {"train": {"profit": tr["net_profit"], "dd": tr["max_dd"],
                                       "trades": tr["total_trades"], "wr": tr["win_rate"],
                                       "ruined": tr["ruined"]},
                             "test": {"profit": te["net_profit"], "dd": te["max_dd"],
                                      "trades": te["total_trades"], "wr": te["win_rate"],
                                      "ruined": te["ruined"]}}

    # OOS recent
    d_live = filter_data(_DATA, "2026-04-09", "2026-04-20")
    live = run_bt_and_trades(d_live, params, kw_extras)
    d_2w = filter_data(_DATA, "2026-04-05", "2026-04-20")
    two_w = run_bt_and_trades(d_2w, params, kw_extras)

    # MC
    mc = mc_bootstrap(full.get("_trades", []))

    return {
        "label": label,
        "base": {"profit": full["net_profit"], "dd": full["max_dd"],
                 "trades": full["total_trades"], "wr": full["win_rate"],
                 "ruined": full["ruined"]},
        "live_10d": {"profit": live["net_profit"], "dd": live["max_dd"],
                     "trades": live["total_trades"], "wr": live["win_rate"]},
        "last_2w": {"profit": two_w["net_profit"], "dd": two_w["max_dd"],
                    "trades": two_w["total_trades"], "wr": two_w["win_rate"]},
        "wf": wf_results,
        "mc": mc,
    }


def main():
    print(f"Validating {len(CANDIDATES)} configs", flush=True)
    t0 = time.time()
    results = []
    with mp_ctx.Pool(7, initializer=_worker_init) as p:  # fewer workers (each does 6 backtests)
        for i, r in enumerate(p.imap_unordered(validate_worker, CANDIDATES), 1):
            results.append(r)
            print(f"  {i}/{len(CANDIDATES)} {r['label']:<30} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, "v10_validate.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"Saved: {path}", flush=True)

    # Print
    results.sort(key=lambda x: x["base"]["profit"], reverse=True)
    print("\n" + "=" * 160)
    print("🏆 Multi-EMA Validation Results")
    print("=" * 160)
    print(f"{'Config':<32}{'6y profit':>14}{'DD':>7}{'WR':>6}{'Live10d':>9}{'2w':>8}"
          f"{'MC ruin':>9}{'MC med':>14}{'WF1':>11}{'WF2':>11}{'WF3':>11}{'WF4':>11}")
    print("-" * 160)
    for r in results:
        b = r["base"]
        mc = r["mc"]
        wf = r["wf"]
        wf_str = ""
        for k in sorted(wf.keys()):
            p = wf[k]["test"]["profit"]
            if abs(p) >= 1e9:
                s = f"{p/1e9:+.1f}B"
            elif abs(p) >= 1e6:
                s = f"{p/1e6:+.1f}M"
            elif abs(p) >= 1e3:
                s = f"{p/1e3:+.1f}k"
            else:
                s = f"{p:+.0f}"
            wf_str += f"{s:>11}"
        if b["profit"] >= 1e9:
            pstr = f"+{b['profit']/1e9:.1f}B"
        elif b["profit"] >= 1e6:
            pstr = f"+{b['profit']/1e6:.1f}M"
        elif b["profit"] >= 1e3:
            pstr = f"+{b['profit']/1e3:.0f}k"
        else:
            pstr = f"+{b['profit']:.0f}"
        mstr = ""
        if mc["median"] >= 1e9:
            mstr = f"{mc['median']/1e9:+.1f}B"
        elif mc["median"] >= 1e6:
            mstr = f"{mc['median']/1e6:+.1f}M"
        elif mc["median"] >= 1e3:
            mstr = f"{mc['median']/1e3:+.0f}k"
        else:
            mstr = f"{mc['median']:+.0f}"
        print(f"{r['label']:<32}{pstr:>14}{b['dd']:>5.1f}%{b['wr']:>5.1f}%"
              f"{r['live_10d']['profit']:>+7.0f}${r['last_2w']['profit']:>+6.0f}$"
              f"{mc['ruin_rate']:>9.3f}{mstr:>14}{wf_str}")


if __name__ == "__main__":
    main()
