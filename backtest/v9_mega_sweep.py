#!/usr/bin/env python3
"""
v9 — Mega Parameter Sweep (Opus 4.7)
=====================================
Full re-exploration of HERMES parameter space.
6-year data (2020-03 ~ 2026-04-18, fresh).
Parallel execution, 14 workers.

Phases:
  0. Sanity (baseline v7 match)
  1. Single-param sensitivity (~250 configs)
  2. Random Latin-hypercube-ish sample of main space (~8000 configs)
  3. Refined neighborhood around top-20 (~2000 configs)
  4. Structural dims (EMA pair × max_sim × max_lev × risk) (~400 configs)

Metrics saved per config:
  - net_profit, max_dd, total_trades, win_rate, ruined
  - yearly breakdown
  - Calmar (profit / max_dd), log_profit, WR_vs_backtest
  - by_direction (LONG/SHORT win rates)

Engine: v4_shared_engine.run_shared_backtest
Seed $580, slippage 0.05%, 2023 skip (수동 oversight), SOL LONG blocked.
"""
import os
import sys
import json
import time
import random
import hashlib
from datetime import datetime
import multiprocessing as mp
from multiprocessing import cpu_count
from itertools import product

# macOS: force fork (spawn breaks with top-level test scripts that don't guard __main__)
mp_ctx = mp.get_context("fork")

import pandas as pd

sys.path.insert(0, "/Users/sue/Projects/HERMES/backtest")
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v9"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 580.0
SLIP = 0.05
DAILY_COST = 1150 / 1470  # ≈ $0.782/day
SKIP_YEARS = (2023,)       # Claude oversight scenario

# v7 baseline (reference point)
V7_BASELINE = {
    "ema_fast": 5, "ema_slow": 18,
    "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
    "entry_score_threshold": 40, "pullback_ema_dist_pct": 1.5,
    "adx_enter_trending": 30, "adx_exit_trending": 20,
    "atr_high_vol_percentile": 85,
    "orderbook_imbalance_min": 0.55,
    "trailing_activation": 1.2, "trailing_distance": 0.1,
    "risk_per_trade": 0.015, "max_leverage": 7,
    "max_simultaneous": 3,
    "funding_bias_threshold": 0.0005,
    "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
    "use_funding": True,
}

# Global data — loaded per worker
_DATA = None


def _load_data():
    """Load 6-year data for 4 coins."""
    data = {}
    for sym in SYMBOLS:
        for iv in ["60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
        f = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        if os.path.exists(f):
            data[f"{sym}_funding"] = pd.read_csv(f)
    return data


def _worker_init():
    """Called once per worker — with fork, data is already COW-shared from parent."""
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def config_hash(cfg: dict) -> str:
    """Stable hash for a config dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:10]


def run_one(cfg: dict) -> dict:
    """Execute one backtest. cfg merges params, engine kwargs."""
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    data = _DATA

    params = {**DEFAULT_PARAMS}
    # Merge param-domain keys
    for k in ["ema_fast", "ema_slow", "rsi_period", "rsi_oversold", "rsi_overbought",
              "pullback_ema_dist_pct", "sl_atr_mult", "tp_rr_ratio",
              "entry_score_threshold", "adx_enter_trending", "adx_exit_trending",
              "atr_high_vol_percentile", "orderbook_imbalance_min",
              "funding_bias_threshold"]:
        if k in cfg:
            params[k] = cfg[k]

    try:
        r = run_shared_backtest(
            data, params, SEED,
            use_funding=cfg.get("use_funding", True),
            trailing_activation=cfg["trailing_activation"],
            trailing_distance=cfg["trailing_distance"],
            block_sol_long=cfg.get("block_sol_long", True),
            skip_years=tuple(cfg.get("skip_years", SKIP_YEARS)),
            daily_cost_usd=DAILY_COST,
            ruin_threshold=15.0,
            use_cooldown=cfg.get("use_cooldown", False),
            slippage_pct=SLIP,
            max_simultaneous=cfg["max_simultaneous"],
            risk_per_trade=cfg["risk_per_trade"],
            max_leverage=cfg["max_leverage"],
            enabled_symbols=cfg.get("enabled_symbols", SYMBOLS),
        )

        trades = r.get("_trades", [])
        long_w, long_t, short_w, short_t = 0, 0, 0, 0
        for t in trades:
            if t["direction"] == "LONG":
                long_t += 1
                if t["pnl"] > 0:
                    long_w += 1
            else:
                short_t += 1
                if t["pnl"] > 0:
                    short_w += 1

        dd = r["max_dd"]
        profit = r["net_profit"]
        # Calmar: profit / max_dd (higher is better; only meaningful if not ruined)
        if r["ruined"]:
            calmar = -1e9
        elif dd <= 0:
            calmar = profit * 100  # no dd, use profit itself
        else:
            calmar = profit / dd

        # Composite score: Calmar but penalize if ruined, if dd >60%, if trades<500
        score = calmar
        if r["ruined"]:
            score = -1e9
        elif r["total_trades"] < 500:
            score *= 0.5
        elif dd > 60:
            score *= 0.7

        return {
            "cfg_hash": config_hash(cfg),
            "cfg": cfg,
            "profit": profit,
            "dd": dd,
            "trades": r["total_trades"],
            "wr": r["win_rate"],
            "ruined": r["ruined"],
            "final_bal": r["final_balance"],
            "yearly": r["yearly"],
            "long": {"t": long_t, "w": long_w, "wr": round(long_w/max(long_t,1)*100, 1)},
            "short": {"t": short_t, "w": short_w, "wr": round(short_w/max(short_t,1)*100, 1)},
            "calmar": round(calmar, 2),
            "score": round(score, 2),
        }
    except Exception as e:
        return {
            "cfg_hash": config_hash(cfg),
            "cfg": cfg,
            "error": str(e),
            "ruined": True,
            "score": -1e9,
            "profit": 0, "dd": 0, "trades": 0, "wr": 0, "final_bal": 0,
            "yearly": {}, "long": {}, "short": {}, "calmar": 0,
        }


# ================================================================
# Phase 1: Single-param sensitivity
# ================================================================

PARAM_RANGES = {
    "ema_pair": [(3, 12), (3, 15), (5, 13), (5, 15), (5, 18), (5, 21), (5, 26),
                 (7, 15), (7, 18), (7, 21), (7, 26), (9, 21), (9, 26), (12, 26)],
    "sl_atr_mult": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    "tp_rr_ratio": [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
    "entry_score_threshold": [25, 30, 35, 40, 45, 50, 55],
    "pullback_ema_dist_pct": [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0],
    "adx_enter_trending": [22, 25, 28, 30, 32, 35, 38],
    "adx_exit_trending": [15, 18, 20, 22, 25],
    "atr_high_vol_percentile": [70, 75, 80, 85, 90, 95],
    "orderbook_imbalance_min": [0.50, 0.52, 0.55, 0.58, 0.60, 0.65],
    "trailing_activation": [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8],
    "trailing_distance": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30],
    "risk_per_trade": [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025],
    "max_leverage": [4, 5, 6, 7, 8, 10, 12],
    "max_simultaneous": [2, 3, 4, 5],
    "use_funding": [True, False],
}


def build_base_cfg():
    """Base config from v7 baseline, for sensitivity sweeps."""
    return {
        "ema_fast": V7_BASELINE["ema_fast"],
        "ema_slow": V7_BASELINE["ema_slow"],
        "sl_atr_mult": V7_BASELINE["sl_atr_mult"],
        "tp_rr_ratio": V7_BASELINE["tp_rr_ratio"],
        "entry_score_threshold": V7_BASELINE["entry_score_threshold"],
        "pullback_ema_dist_pct": V7_BASELINE["pullback_ema_dist_pct"],
        "adx_enter_trending": V7_BASELINE["adx_enter_trending"],
        "adx_exit_trending": V7_BASELINE["adx_exit_trending"],
        "atr_high_vol_percentile": V7_BASELINE["atr_high_vol_percentile"],
        "orderbook_imbalance_min": V7_BASELINE["orderbook_imbalance_min"],
        "trailing_activation": V7_BASELINE["trailing_activation"],
        "trailing_distance": V7_BASELINE["trailing_distance"],
        "risk_per_trade": V7_BASELINE["risk_per_trade"],
        "max_leverage": V7_BASELINE["max_leverage"],
        "max_simultaneous": V7_BASELINE["max_simultaneous"],
        "use_funding": V7_BASELINE["use_funding"],
        "funding_bias_threshold": V7_BASELINE["funding_bias_threshold"],
        "rsi_period": V7_BASELINE["rsi_period"],
        "rsi_oversold": V7_BASELINE["rsi_oversold"],
        "rsi_overbought": V7_BASELINE["rsi_overbought"],
    }


def phase1_sensitivity():
    """Each param varied; others at v7 values."""
    configs = []
    base = build_base_cfg()

    # Baseline itself
    configs.append({**base, "_phase": 1, "_dim": "baseline"})

    for dim, vals in PARAM_RANGES.items():
        for v in vals:
            cfg = {**base}
            if dim == "ema_pair":
                cfg["ema_fast"], cfg["ema_slow"] = v
            else:
                cfg[dim] = v
            cfg["_phase"] = 1
            cfg["_dim"] = dim
            cfg["_val"] = v if dim != "ema_pair" else f"{v[0]}/{v[1]}"
            configs.append(cfg)
    return configs


# ================================================================
# Phase 2: Random search over big space
# ================================================================

def phase2_random(n=6000, seed=42):
    """Random sample of full space."""
    random.seed(seed)
    base = build_base_cfg()
    configs = []

    for i in range(n):
        cfg = {**base}
        ema_f, ema_s = random.choice(PARAM_RANGES["ema_pair"])
        cfg["ema_fast"] = ema_f
        cfg["ema_slow"] = ema_s
        cfg["sl_atr_mult"] = random.choice(PARAM_RANGES["sl_atr_mult"])
        cfg["tp_rr_ratio"] = random.choice(PARAM_RANGES["tp_rr_ratio"])
        cfg["entry_score_threshold"] = random.choice(PARAM_RANGES["entry_score_threshold"])
        cfg["pullback_ema_dist_pct"] = random.choice(PARAM_RANGES["pullback_ema_dist_pct"])
        cfg["adx_enter_trending"] = random.choice(PARAM_RANGES["adx_enter_trending"])
        cfg["adx_exit_trending"] = random.choice(PARAM_RANGES["adx_exit_trending"])
        cfg["atr_high_vol_percentile"] = random.choice(PARAM_RANGES["atr_high_vol_percentile"])
        cfg["orderbook_imbalance_min"] = random.choice(PARAM_RANGES["orderbook_imbalance_min"])
        cfg["trailing_activation"] = random.choice(PARAM_RANGES["trailing_activation"])
        cfg["trailing_distance"] = random.choice(PARAM_RANGES["trailing_distance"])
        cfg["risk_per_trade"] = random.choice(PARAM_RANGES["risk_per_trade"])
        cfg["max_leverage"] = random.choice(PARAM_RANGES["max_leverage"])
        cfg["max_simultaneous"] = random.choice(PARAM_RANGES["max_simultaneous"])
        cfg["use_funding"] = random.choice(PARAM_RANGES["use_funding"])
        # Enforce ADX exit < enter
        if cfg["adx_exit_trending"] >= cfg["adx_enter_trending"]:
            cfg["adx_exit_trending"] = cfg["adx_enter_trending"] - 5
        # Enforce ema_fast < ema_slow
        if cfg["ema_fast"] >= cfg["ema_slow"]:
            cfg["ema_slow"] = cfg["ema_fast"] + 10
        cfg["_phase"] = 2
        cfg["_idx"] = i
        configs.append(cfg)
    return configs


# ================================================================
# Runner
# ================================================================

def run_batch(configs, label, workers=14, save_path=None, save_every=200):
    print(f"\n{'='*80}")
    print(f"Phase: {label} | configs: {len(configs)} | workers: {workers}")
    print(f"{'='*80}")
    t0 = time.time()
    # Pre-load in parent so fork copies ready data to workers
    global _DATA
    if _DATA is None:
        _DATA = _load_data()

    chunksize = max(1, len(configs) // (workers * 8))
    results = []
    with mp_ctx.Pool(workers, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, configs, chunksize=chunksize), 1):
            results.append(r)
            if i % 50 == 0 or i == len(configs):
                el = time.time() - t0
                rate = i / el
                eta = (len(configs) - i) / rate if rate else 0
                # Top-3 current
                top = sorted([r for r in results if not r.get("ruined")],
                             key=lambda x: x["score"], reverse=True)[:3]
                top_str = " | ".join([f"${r['profit']:,.0f}/{r['dd']:.0f}%" for r in top])
                print(f"  [{i:5}/{len(configs):5}] {el:6.0f}s rate={rate:.1f}/s eta={eta:.0f}s | top3: {top_str}", flush=True)
            if save_path and i % save_every == 0:
                _save_incremental(results, save_path, label, t0)

    if save_path:
        _save_incremental(results, save_path, label, t0)
    el = time.time() - t0
    print(f"  Completed: {el:.0f}s ({el/60:.1f}min)")
    return results


def _save_incremental(results, path, label, t0):
    try:
        out = {
            "label": label,
            "n_total": len(results),
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  (save warn: {e})")


# ================================================================
# Phase 3: Refined grid around top-20
# ================================================================

def phase3_refined(top_configs, n_per=5):
    """Perturb top configs with small deltas."""
    configs = []
    for j, top in enumerate(top_configs):
        cfg_base = top["cfg"]
        # For each continuous dim, ± small delta
        deltas = {
            "sl_atr_mult": [-0.25, -0.1, 0, 0.1, 0.25],
            "tp_rr_ratio": [-1.0, -0.5, 0, 0.5, 1.0],
            "trailing_activation": [-0.2, -0.1, 0, 0.1, 0.2],
            "trailing_distance": [-0.03, -0.01, 0, 0.01, 0.03],
            "entry_score_threshold": [-5, -3, 0, 3, 5],
            "pullback_ema_dist_pct": [-0.5, -0.25, 0, 0.25, 0.5],
            "adx_enter_trending": [-3, -1, 0, 1, 3],
        }
        # Random combinations
        import random as _r
        _r.seed(1000 + j)
        for _ in range(n_per):
            c = {**cfg_base}
            for dim, ds in deltas.items():
                delta = _r.choice(ds)
                c[dim] = max(0.01, cfg_base[dim] + delta)
            c["_phase"] = 3
            c["_from_hash"] = top["cfg_hash"]
            configs.append(c)
    return configs


# ================================================================
# Main
# ================================================================

def main():
    print("="*80)
    print("HERMES v9 — MEGA SWEEP (Opus 4.7)")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)

    data = _load_data()
    for sym in SYMBOLS:
        df = data.get(f"{sym}_60")
        if df is not None and len(df):
            first = datetime.utcfromtimestamp(df['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')
            last = datetime.utcfromtimestamp(df['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')
            print(f"  {sym}: {len(df):>6} rows | {first} → {last}")

    t_all = time.time()
    all_results = {"phase1": [], "phase2": [], "phase3": []}

    # Phase 1
    p1_configs = phase1_sensitivity()
    p1_results = run_batch(p1_configs, "Phase1 sensitivity", workers=14,
                           save_path=os.path.join(RESULTS_DIR, "v9_phase1.json"))
    all_results["phase1"] = p1_results

    # Phase 2 — random search
    p2_configs = phase2_random(n=10000, seed=42)
    p2_results = run_batch(p2_configs, "Phase2 random(10000)", workers=14,
                           save_path=os.path.join(RESULTS_DIR, "v9_phase2.json"))
    all_results["phase2"] = p2_results

    # Phase 3 — refined around top-40 from P1+P2
    pool_results = [r for r in (p1_results + p2_results) if not r.get("ruined")]
    pool_results.sort(key=lambda x: x["score"], reverse=True)
    top_for_refine = pool_results[:40]
    p3_configs = phase3_refined(top_for_refine, n_per=100)
    p3_results = run_batch(p3_configs, f"Phase3 refined(n={len(p3_configs)})", workers=14,
                           save_path=os.path.join(RESULTS_DIR, "v9_phase3.json"))
    all_results["phase3"] = p3_results

    # Combined summary
    all_pool = [r for r in (p1_results + p2_results + p3_results) if not r.get("ruined")]
    all_pool.sort(key=lambda x: x["score"], reverse=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_sec": round(time.time() - t_all, 1),
        "counts": {
            "phase1": len(p1_results),
            "phase2": len(p2_results),
            "phase3": len(p3_results),
            "survivors": len(all_pool),
        },
        "top_50_by_score": all_pool[:50],
    }

    with open(os.path.join(RESULTS_DIR, "v9_summary.json"), "w") as f:
        json.dump(summary, f, default=str, indent=2)

    print(f"\n{'='*80}")
    print(f"DONE — total {time.time()-t_all:.0f}s ({(time.time()-t_all)/60:.1f}min)")
    print(f"Survivors: {len(all_pool)}")
    print(f"\nTop 10 by composite score:")
    for i, r in enumerate(all_pool[:10], 1):
        c = r["cfg"]
        print(f"  {i:2}. score={r['score']:>10,.0f} profit=${r['profit']:>9,.0f} dd={r['dd']:>5.1f}% "
              f"trades={r['trades']:>4} wr={r['wr']:>4.1f}% | "
              f"EMA{c['ema_fast']}/{c['ema_slow']} sl{c['sl_atr_mult']} tp{c['tp_rr_ratio']} "
              f"trail{c['trailing_activation']}/{c['trailing_distance']} "
              f"adx{c['adx_enter_trending']} max_sim{c['max_simultaneous']} lev{c['max_leverage']} "
              f"risk{c['risk_per_trade']*100:.1f}%")


if __name__ == "__main__":
    main()
