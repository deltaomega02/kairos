#!/usr/bin/env python3
"""
v9 Validation — Walk-Forward + Monte Carlo + Out-of-Sample + Realism tiers
==========================================================================
Takes top candidates from sweep, validates rigorously, produces final ranking.

  - Walk-Forward: 4 train/test splits
  - Monte Carlo: trade-level bootstrap (300 resamples)
  - OOS recent: 2026-04-10 ~ 2026-04-20
  - Realism tiers: A (realistic), B (aggressive), C (mathematically best)
  - Composite score: Calmar × WF_pass × OOS_sign × realism_multiplier

Output: ~/Projects/HERMES_백테스팅/v9/v9_final_ranking.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

import pandas as pd

import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS
from v9_mega_sweep import (_load_data, build_base_cfg,
                           SYMBOLS, SEED, SLIP, DAILY_COST)

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v9"

_DATA = None


def _worker_init():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def filter_data(data, start_date, end_date):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and "timestamp" in v.columns:
            m = (v["timestamp"] >= start_ts) & (v["timestamp"] < end_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


WF_SPLITS = [
    ("WF1", ("2020-03-25", "2023-01-01"), ("2023-01-01", "2026-04-20")),
    ("WF2", ("2020-03-25", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF3", ("2021-01-01", "2024-01-01"), ("2024-01-01", "2026-04-20")),
    ("WF4", ("2020-03-25", "2025-01-01"), ("2025-01-01", "2026-04-20")),
]


def _run_bt(data, cfg, skip_years=(), return_trades=False):
    params = {**DEFAULT_PARAMS}
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
            skip_years=skip_years,
            daily_cost_usd=DAILY_COST,
            ruin_threshold=15.0,
            use_cooldown=cfg.get("use_cooldown", False),
            slippage_pct=SLIP,
            max_simultaneous=cfg["max_simultaneous"],
            risk_per_trade=cfg["risk_per_trade"],
            max_leverage=cfg["max_leverage"],
            enabled_symbols=cfg.get("enabled_symbols", SYMBOLS),
        )
        out = {"profit": r["net_profit"], "dd": r["max_dd"],
               "trades": r["total_trades"], "wr": r["win_rate"],
               "ruined": r["ruined"], "final_bal": r["final_balance"]}
        if return_trades:
            out["trades_list"] = r.get("_trades", [])
        return out
    except Exception as e:
        return {"error": str(e), "ruined": True, "profit": 0, "dd": 0, "trades": 0,
                "wr": 0, "final_bal": 0, "trades_list": [] if return_trades else None}


def wf_worker(cfg):
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    out = {}
    for name, (s1, e1), (s2, e2) in WF_SPLITS:
        td = filter_data(_DATA, s1, e1)
        vd = filter_data(_DATA, s2, e2)
        # Train can optionally skip 2023 (if in range), test doesn't skip
        train_skip = (2023,) if int(s1[:4]) <= 2023 <= int(e1[:4]) else ()
        tr = _run_bt(td, cfg, skip_years=train_skip)
        te = _run_bt(vd, cfg, skip_years=())
        out[name] = {"train": tr, "test": te}
    return {"cfg_hash": cfg.get("cfg_hash", "?"), "wf": out}


def oos_worker(cfg):
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    d = filter_data(_DATA, "2026-04-10", "2026-04-20")
    return {"cfg_hash": cfg.get("cfg_hash", "?"),
            "oos": _run_bt(d, cfg, skip_years=())}


def mc_bootstrap(trades, n_iter=300, seed_base=42):
    """Trade-level bootstrap. Returns stats dict."""
    if not trades:
        return {"ruin_rate": 1.0, "n": 0}
    n = len(trades)
    results = []
    ruined = 0
    for i in range(n_iter):
        random.seed(seed_base + i)
        sampled = [random.choice(trades) for _ in range(n)]
        bal = SEED
        peak = SEED
        mdd = 0
        r = False
        for t in sampled:
            bal += t["pnl"]
            if bal < 15:
                r = True
                break
            if bal > peak:
                peak = bal
            else:
                dd = (peak - bal) / peak * 100
                if dd > mdd:
                    mdd = dd
        if r:
            ruined += 1
            results.append({"profit": bal - SEED, "dd": mdd, "ruined": True})
        else:
            results.append({"profit": bal - SEED, "dd": mdd, "ruined": False})

    alive = [r for r in results if not r["ruined"]]
    if not alive:
        return {"ruin_rate": 1.0, "n_iter": n_iter}
    profits = sorted(r["profit"] for r in alive)
    def pct(p):
        return profits[int(p * (len(profits)-1))]
    return {
        "ruin_rate": round(ruined / n_iter, 3),
        "n_iter": n_iter,
        "median": round(pct(0.5), 0),
        "p25": round(pct(0.25), 0),
        "p75": round(pct(0.75), 0),
        "p10": round(pct(0.10), 0),
        "p90": round(pct(0.90), 0),
        "worst": round(min(profits), 0),
        "best": round(max(profits), 0),
    }


def mc_worker(cfg):
    """Run base bt to get trades, then bootstrap."""
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    base = _run_bt(_DATA, cfg, skip_years=(2023,), return_trades=True)
    trades = base.get("trades_list", []) or []
    mc = mc_bootstrap(trades, n_iter=300)
    return {"cfg_hash": cfg.get("cfg_hash", "?"),
            "base": {k: v for k, v in base.items() if k != "trades_list"},
            "mc": mc}


def realism_tier(cfg, base_profit, base_dd):
    """A: realistic; B: aggressive; C: mathematical only."""
    # Realistic: final balance < $10M AND DD < 45%
    final_bal = SEED + base_profit
    if base_dd < 45 and final_bal < 10_000_000 and cfg["max_leverage"] <= 10 and cfg["risk_per_trade"] <= 0.02:
        return "A"
    if base_dd < 55 and final_bal < 50_000_000 and cfg["max_leverage"] <= 12 and cfg["risk_per_trade"] <= 0.025:
        return "B"
    return "C"


def load_candidates():
    """Combine top from all phases. Dedupe by cfg_hash. Return top N by score."""
    all_results = []
    for phase in ("phase1", "phase2", "phase3"):
        path = os.path.join(RESULTS_DIR, f"v9_{phase}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for r in d.get("results", []):
                if not r.get("ruined") and not r.get("error"):
                    all_results.append(r)
    # Dedupe
    seen = set()
    uniq = []
    for r in all_results:
        h = r.get("cfg_hash")
        if h and h not in seen:
            seen.add(h)
            uniq.append(r)
    # Sort by score, take top 100
    uniq.sort(key=lambda x: x.get("score", 0), reverse=True)
    return uniq


def main():
    print("=" * 80, flush=True)
    print("HERMES v9 — VALIDATION PHASE", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 80, flush=True)

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} unique candidates from sweep", flush=True)

    # Take top 100 by base score; diversify to also include top-100 by Calmar,
    # and top configs at DD < 40% to ensure realistic candidates make it in.
    by_score = candidates[:80]
    by_calmar = sorted(candidates, key=lambda x: x.get("calmar", 0), reverse=True)[:40]
    low_dd = [c for c in candidates if (c.get("dd") or 99) < 40][:40]
    mid_dd = [c for c in candidates if 40 <= (c.get("dd") or 99) < 50][:30]

    merged_ids = set()
    pool = []
    for lst in [by_score, by_calmar, low_dd, mid_dd]:
        for c in lst:
            h = c.get("cfg_hash")
            if h and h not in merged_ids:
                merged_ids.add(h)
                pool.append(c)

    # Add v7 baseline
    v7_cfg = build_base_cfg()
    v7_cfg["cfg_hash"] = "v7_baseline"
    pool.insert(0, {"cfg_hash": "v7_baseline", "cfg": v7_cfg,
                    "profit": 221393, "dd": 51.7, "calmar": 4282})
    print(f"Validation pool: {len(pool)} configs", flush=True)

    # Extract cfg dicts and ensure hash is present inside cfg
    cfg_list = []
    for r in pool:
        c = r.get("cfg", r)
        c = {**c}
        c["cfg_hash"] = r.get("cfg_hash", "?")
        cfg_list.append(c)

    # ------ WF ------
    print(f"\n[WF] {len(cfg_list)} configs × 4 splits", flush=True)
    t0 = time.time()
    wf_by_hash = {}
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(wf_worker, cfg_list), 1):
            wf_by_hash[res["cfg_hash"]] = res["wf"]
            if i % 20 == 0 or i == len(cfg_list):
                print(f"  WF {i}/{len(cfg_list)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"  WF done: {time.time()-t0:.0f}s", flush=True)

    # ------ OOS ------
    print(f"\n[OOS] {len(cfg_list)} configs", flush=True)
    t1 = time.time()
    oos_by_hash = {}
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(oos_worker, cfg_list), 1):
            oos_by_hash[res["cfg_hash"]] = res["oos"]
            if i % 30 == 0 or i == len(cfg_list):
                print(f"  OOS {i}/{len(cfg_list)} elapsed={time.time()-t1:.0f}s", flush=True)
    print(f"  OOS done: {time.time()-t1:.0f}s", flush=True)

    # ------ Merge results + compute composite ------
    merged = []
    for r in pool:
        h = r.get("cfg_hash")
        c = r.get("cfg", r)
        wf = wf_by_hash.get(h, {})
        oos = oos_by_hash.get(h, {})

        # WF test profits
        wf_test = [wf[k]["test"] for k in wf.keys() if "test" in wf[k]]
        wf_profits = [x.get("profit", 0) for x in wf_test]
        wf_dds = [x.get("dd", 0) for x in wf_test]
        wf_ruined = any(x.get("ruined") for x in wf_test)
        wf_all_pos = all(p > 0 for p in wf_profits) if wf_profits else False
        wf_min_profit = min(wf_profits) if wf_profits else 0
        wf_max_dd = max(wf_dds) if wf_dds else 0

        tier = realism_tier(c, r.get("profit", 0), r.get("dd", 99))

        merged.append({
            "cfg_hash": h,
            "cfg": c,
            "tier": tier,
            "base_profit": r.get("profit"),
            "base_dd": r.get("dd"),
            "base_trades": r.get("trades"),
            "base_wr": r.get("wr"),
            "base_calmar": r.get("calmar"),
            "base_long_wr": r.get("long", {}).get("wr"),
            "base_short_wr": r.get("short", {}).get("wr"),
            "wf": wf,
            "wf_all_pos": wf_all_pos,
            "wf_min_profit": round(wf_min_profit, 0),
            "wf_max_dd": round(wf_max_dd, 1),
            "wf_ruined": wf_ruined,
            "oos": oos,
        })

    # Composite scoring (different angles)
    def composite_realistic(m):
        if m["tier"] == "C" or m.get("wf_ruined"):
            return -1e9
        calmar = m["base_calmar"] or 0
        mult = 1.0
        if m["wf_all_pos"]:
            mult *= 1.5
        if m["wf_min_profit"] < 0:
            mult *= 0.5
        oos_p = m["oos"].get("profit", 0)
        if oos_p > 0:
            mult *= 1.3
        elif m["oos"].get("ruined"):
            mult *= 0.5
        # prefer tier A
        if m["tier"] == "A":
            mult *= 1.2
        return round(calmar * mult, 2)

    def composite_max_return(m):
        # Pure return, only bankruptcy filter
        if m.get("wf_ruined"):
            return -1e9
        if (m["base_dd"] or 0) >= 75:
            return -1e9
        return round(m["base_profit"] or 0, 2)

    def composite_wf_robust(m):
        # Best WF consistency
        if not m["wf_all_pos"]:
            return -1e9
        if m.get("wf_ruined"):
            return -1e9
        # Score = geometric mean of WF test profits × base Calmar weight
        wf_profits = [m["wf"][k]["test"].get("profit", 0) for k in m["wf"].keys()]
        if not all(p > 0 for p in wf_profits):
            return -1e9
        prod = 1.0
        for p in wf_profits:
            prod *= max(1, p)
        geomean = prod ** (1/len(wf_profits))
        return round(geomean, 2)

    for m in merged:
        m["score_realistic"] = composite_realistic(m)
        m["score_maxret"] = composite_max_return(m)
        m["score_wfrobust"] = composite_wf_robust(m)

    # ------ Run MC on top-10 per ranking ------
    mc_targets = set()
    for key in ("score_realistic", "score_maxret", "score_wfrobust"):
        ranked = sorted(merged, key=lambda x: x[key], reverse=True)[:10]
        for m in ranked:
            mc_targets.add(m["cfg_hash"])
    # Always include v7 baseline
    mc_targets.add("v7_baseline")

    mc_cfgs = []
    for m in merged:
        if m["cfg_hash"] in mc_targets:
            c = {**m["cfg"], "cfg_hash": m["cfg_hash"]}
            mc_cfgs.append(c)
    print(f"\n[MC] {len(mc_cfgs)} configs × 300 bootstrap", flush=True)
    t2 = time.time()
    mc_by_hash = {}
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(mc_worker, mc_cfgs), 1):
            mc_by_hash[res["cfg_hash"]] = res
            if i % 5 == 0 or i == len(mc_cfgs):
                print(f"  MC {i}/{len(mc_cfgs)} elapsed={time.time()-t2:.0f}s", flush=True)
    print(f"  MC done: {time.time()-t2:.0f}s", flush=True)
    for m in merged:
        if m["cfg_hash"] in mc_by_hash:
            m["mc"] = mc_by_hash[m["cfg_hash"]]["mc"]

    # ------ Save + Print rankings ------
    out = {
        "timestamp": datetime.now().isoformat(),
        "n_candidates": len(merged),
        "rankings": {
            "realistic": sorted(merged, key=lambda x: x["score_realistic"], reverse=True)[:30],
            "max_return": sorted(merged, key=lambda x: x["score_maxret"], reverse=True)[:30],
            "wf_robust": sorted(merged, key=lambda x: x["score_wfrobust"], reverse=True)[:30],
        },
        "all_merged": merged,
    }
    path = os.path.join(RESULTS_DIR, "v9_final_ranking.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nSaved: {path}", flush=True)

    def _print_ranking(title, key, n=15):
        print(f"\n{'='*80}\n{title}\n{'='*80}", flush=True)
        ranked = sorted(merged, key=lambda x: x[key], reverse=True)[:n]
        for i, m in enumerate(ranked, 1):
            c = m["cfg"]
            wf_summary = ""
            for k in sorted(m["wf"].keys()):
                t = m["wf"][k]["test"]
                wf_summary += f" {k[-1]}=${t.get('profit',0):,.0f}"
            oos_p = m["oos"].get("profit", 0)
            mc = m.get("mc", {})
            mc_str = ""
            if mc:
                mc_str = f" MC: ruin={mc.get('ruin_rate','?')} median=${mc.get('median','?'):,.0f}"
            print(f"{i:3}. [{m['tier']}] {m['cfg_hash'][:8]} base=${m['base_profit']:>10,.0f}/{m['base_dd']:>4.1f}% "
                  f"| EMA{c['ema_fast']}/{c['ema_slow']} sl{c['sl_atr_mult']} tp{c['tp_rr_ratio']} "
                  f"tr{c['trailing_activation']}/{c['trailing_distance']} "
                  f"adx{c['adx_enter_trending']}/{c['adx_exit_trending']} atr{c['atr_high_vol_percentile']} "
                  f"sim{c['max_simultaneous']} lev{c['max_leverage']} risk{c['risk_per_trade']*100:.1f}% "
                  f"ob{c.get('orderbook_imbalance_min',0.55)} fund={c.get('use_funding',True)}", flush=True)
            print(f"      WF:{wf_summary} | OOS=${oos_p:.0f}{mc_str}", flush=True)

    _print_ranking("🏆 RANKING 1 — REALISTIC (tier A/B only, WF + OOS weighted)", "score_realistic")
    _print_ranking("💰 RANKING 2 — MAX RETURN (pure profit, DD < 75%)", "score_maxret")
    _print_ranking("🛡️ RANKING 3 — WF ROBUST (all 4 WF tests positive)", "score_wfrobust")


if __name__ == "__main__":
    main()
