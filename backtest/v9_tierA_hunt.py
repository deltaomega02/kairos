#!/usr/bin/env python3
"""
v9 Tier-A Hunt — find the best REALISTIC HERMES that beats v7 in MC robustness.

Filter from full sweep: tier A/B by params.
Run MC on all of them.
Rank by MC median.
"""
import os
import sys
import json
import time
import random

import pandas as pd
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS
from v9_mega_sweep import (_load_data, build_base_cfg,
                           SYMBOLS, SEED, SLIP, DAILY_COST)
from v9_validate import _run_bt, mc_bootstrap, mc_worker

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v9"
_DATA = None


def _worker_init():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def param_tier(cfg):
    risk = cfg["risk_per_trade"]
    lev = cfg["max_leverage"]
    sim = cfg["max_simultaneous"]
    if risk <= 0.015 and lev <= 7 and sim <= 3:
        return "A"
    if risk <= 0.020 and lev <= 10 and sim <= 4:
        return "B"
    return "C"


def load_all_alive():
    """Every non-ruined config from phases 1-3."""
    all_results = []
    for phase in ("phase1", "phase2", "phase3"):
        path = os.path.join(RESULTS_DIR, f"v9_{phase}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for r in d.get("results", []):
                if not r.get("ruined") and not r.get("error"):
                    all_results.append(r)
    seen = set()
    uniq = []
    for r in all_results:
        h = r.get("cfg_hash")
        if h and h not in seen:
            seen.add(h)
            uniq.append(r)
    return uniq


def main():
    print(f"Loading all alive configs from phases 1-3", flush=True)
    all_alive = load_all_alive()
    print(f"Total unique alive: {len(all_alive)}", flush=True)

    # Annotate with tier
    for r in all_alive:
        r["ptier"] = param_tier(r["cfg"])

    tier_A = [r for r in all_alive if r["ptier"] == "A"]
    tier_B = [r for r in all_alive if r["ptier"] == "B"]
    tier_C = [r for r in all_alive if r["ptier"] == "C"]
    print(f"  Tier A (≤ v7 aggression): {len(tier_A)}", flush=True)
    print(f"  Tier B (moderate): {len(tier_B)}", flush=True)
    print(f"  Tier C (aggressive): {len(tier_C)}", flush=True)

    # Sort each tier by profit
    tier_A.sort(key=lambda x: x["profit"], reverse=True)
    tier_B.sort(key=lambda x: x["profit"], reverse=True)

    # Pick top 80 from each tier A & B
    candidates = tier_A[:80] + tier_B[:80]
    # Inject v7 baseline
    v7 = build_base_cfg()
    v7_res = {"cfg": v7, "cfg_hash": "v7_baseline", "profit": 221393, "dd": 51.7,
              "ptier": "A"}
    candidates.insert(0, v7_res)

    print(f"\nCandidates for MC: {len(candidates)}", flush=True)

    # Ensure cfg_hash in cfg
    cfg_list = []
    for r in candidates:
        c = {**r["cfg"]}
        c["cfg_hash"] = r.get("cfg_hash", "?")
        cfg_list.append(c)

    # MC in parallel
    t0 = time.time()
    mc_by_hash = {}
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(mc_worker, cfg_list), 1):
            mc_by_hash[res["cfg_hash"]] = res
            if i % 20 == 0 or i == len(cfg_list):
                print(f"  MC {i}/{len(cfg_list)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"MC done in {time.time()-t0:.0f}s", flush=True)

    # Enrich candidates with MC
    for r in candidates:
        h = r.get("cfg_hash")
        mc = mc_by_hash.get(h, {})
        r["mc"] = mc.get("mc", {})
        r["base_from_mc"] = mc.get("base", {})

    # Ranking: Tier A by MC median, Tier B by MC median, combined by (MC_median - ruin_penalty)
    def mc_med(r):
        return r.get("mc", {}).get("median", 0) or 0

    def mc_ruin(r):
        return r.get("mc", {}).get("ruin_rate", 1) or 1

    # Realistic score: MC median × (1 - ruin_rate)
    def score(r):
        med = mc_med(r)
        ruin = mc_ruin(r)
        return med * (1 - ruin)

    tier_A_sorted = sorted([r for r in candidates if r["ptier"] == "A"],
                           key=score, reverse=True)
    tier_B_sorted = sorted([r for r in candidates if r["ptier"] == "B"],
                           key=score, reverse=True)
    combined = sorted(candidates, key=score, reverse=True)

    def pshow(title, lst, n=15):
        print(f"\n{'='*100}\n{title}\n{'='*100}")
        for i, r in enumerate(lst[:n], 1):
            c = r["cfg"]
            mc = r.get("mc", {})
            ruin = mc.get("ruin_rate", "?")
            med = mc.get("median", 0)
            p25 = mc.get("p25", 0)
            p75 = mc.get("p75", 0)
            print(f"{i:2}. [{r['ptier']}] base=${r.get('profit',0):>10,.0f}/{r.get('dd',0):>4.1f}% "
                  f"ruin={ruin} mc_p25=${p25:>10,.0f} mc_med=${med:>10,.0f} mc_p75=${p75:>10,.0f}")
            print(f"    EMA{c['ema_fast']}/{c['ema_slow']} sl{c['sl_atr_mult']} tp{c['tp_rr_ratio']} "
                  f"tr{c['trailing_activation']}/{c['trailing_distance']} "
                  f"adx{c['adx_enter_trending']}/{c['adx_exit_trending']} atr{c['atr_high_vol_percentile']} "
                  f"sim{c['max_simultaneous']} lev{c['max_leverage']} risk{c['risk_per_trade']*100:.1f}% "
                  f"ob{c.get('orderbook_imbalance_min',0.55)} pb{c.get('pullback_ema_dist_pct',1.5)} "
                  f"es{c.get('entry_score_threshold',40)} fund={c.get('use_funding',True)}")

    pshow("🔒 TIER A — Risk≤1.5% Lev≤7 Sim≤3 — ranked by MC median × (1-ruin)", tier_A_sorted)
    pshow("⚖️ TIER B — Risk≤2.0% Lev≤10 Sim≤4 — ranked by MC median × (1-ruin)", tier_B_sorted)
    pshow("🏆 COMBINED — all tier A+B", combined, n=20)

    # Find configs with LOW MC ruin — truly robust
    robust = [r for r in candidates if mc_ruin(r) < 0.5]
    robust.sort(key=mc_med, reverse=True)
    pshow("✅ TRULY ROBUST — MC ruin<50%, ranked by MC median", robust, n=15)

    low_ruin = [r for r in candidates if mc_ruin(r) < 0.7]
    low_ruin.sort(key=mc_med, reverse=True)
    pshow("⚠️ PASSABLE ROBUST — MC ruin<70%, ranked by MC median", low_ruin, n=15)

    # Save
    out = {
        "timestamp": time.time(),
        "tier_A_top30": tier_A_sorted[:30],
        "tier_B_top30": tier_B_sorted[:30],
        "combined_top30": combined[:30],
        "robust_top30": robust[:30],
        "low_ruin_top30": low_ruin[:30],
    }
    path = os.path.join(RESULTS_DIR, "v9_tierA_hunt.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
