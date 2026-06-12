#!/usr/bin/env python3
"""
v9 Realistic Re-rank — applies MC robustness filter, param-based tiers.
"""
import json
import os
import sys

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v9"

with open(os.path.join(RESULTS_DIR, "v9_final_ranking.json")) as f:
    data = json.load(f)

merged = data["all_merged"]
print(f"Total validated configs: {len(merged)}")


def param_tier(cfg):
    """Tier based on parameter aggressiveness, not outcome."""
    risk = cfg["risk_per_trade"]
    lev = cfg["max_leverage"]
    sim = cfg["max_simultaneous"]
    # Tier A: matches v7 or less aggressive
    if risk <= 0.015 and lev <= 7 and sim <= 3:
        return "A"
    # Tier B: moderate aggression
    if risk <= 0.020 and lev <= 10 and sim <= 4:
        return "B"
    return "C"


def classify(m):
    c = m["cfg"]
    t = param_tier(c)
    mc = m.get("mc", {}) or {}
    ruin = mc.get("ruin_rate", None)
    mc_med = mc.get("median", 0)
    wf_profits = [m["wf"][k]["test"].get("profit", 0) for k in m["wf"].keys()]
    wf_all_pos = all(p > 0 for p in wf_profits) if wf_profits else False
    wf_min = min(wf_profits) if wf_profits else 0
    oos_p = m.get("oos", {}).get("profit", 0)
    return {
        "tier": t, "ruin": ruin, "mc_median": mc_med,
        "wf_all_pos": wf_all_pos, "wf_min": wf_min, "oos_p": oos_p,
    }


# Annotate
for m in merged:
    info = classify(m)
    m["ptier"] = info["tier"]
    m["ruin"] = info["ruin"]
    m["mc_median"] = info["mc_median"]
    m["wf_all_pos_fl"] = info["wf_all_pos"]
    m["wf_min_fl"] = info["wf_min"]
    m["oos_p_fl"] = info["oos_p"]

# Ranking 1 — Robust: MC ruin < 0.5 AND WF all positive
robust = [m for m in merged if m["ruin"] is not None
          and m["ruin"] < 0.5
          and m["wf_all_pos_fl"]]
robust.sort(key=lambda x: x["mc_median"], reverse=True)

# Ranking 2 — Param realistic (tier A or B) AND MC ruin < 0.8
realistic = [m for m in merged
             if m["ptier"] in ("A", "B")
             and m["ruin"] is not None
             and m["ruin"] < 0.8]
realistic.sort(key=lambda x: x["mc_median"], reverse=True)

# Ranking 3 — Best MC median (any params)
best_mc = sorted([m for m in merged if m["ruin"] is not None],
                 key=lambda x: x["mc_median"], reverse=True)

# Ranking 4 — Conservative: tier A strict, ranked by MC median
conservative = [m for m in merged
                if m["ptier"] == "A"
                and m["ruin"] is not None]
conservative.sort(key=lambda x: x["mc_median"], reverse=True)


def print_ranking(title, lst, n=15):
    print(f"\n{'='*100}\n{title} (found {len(lst)})\n{'='*100}")
    for i, m in enumerate(lst[:n], 1):
        c = m["cfg"]
        wf = " ".join([f"{k[-1]}=${m['wf'][k]['test'].get('profit',0):,.0f}"
                        for k in sorted(m["wf"].keys())])
        oos_p = m["oos"].get("profit", 0)
        ruin = m["ruin"] if m["ruin"] is not None else "?"
        mc_med = m["mc_median"]
        print(f"{i:2}. [{m['ptier']}] base=${m['base_profit']:>12,.0f}/{m['base_dd']:>4.1f}% "
              f"ruin={ruin} mc_med=${mc_med:>11,.0f} oos=${oos_p:.0f}")
        print(f"    EMA{c['ema_fast']}/{c['ema_slow']} sl{c['sl_atr_mult']} tp{c['tp_rr_ratio']} "
              f"tr{c['trailing_activation']}/{c['trailing_distance']} "
              f"adx{c['adx_enter_trending']}/{c['adx_exit_trending']} atr{c['atr_high_vol_percentile']} "
              f"sim{c['max_simultaneous']} lev{c['max_leverage']} risk{c['risk_per_trade']*100:.1f}% "
              f"ob{c.get('orderbook_imbalance_min',0.55)} pb{c.get('pullback_ema_dist_pct',1.5)} "
              f"es{c.get('entry_score_threshold',40)} fund={c.get('use_funding',True)}")
        print(f"    WF:{wf}")


print_ranking("🛡️ ROBUST — MC ruin<0.5 + WF all positive", robust)
print_ranking("✅ PARAM REALISTIC — tier A/B + MC ruin<0.8", realistic, n=10)
print_ranking("💰 BEST MC MEDIAN (any params, ruin computed)", best_mc, n=10)
print_ranking("🔒 CONSERVATIVE — tier A (≤v7 aggression)", conservative, n=10)

# Specifically: v7 vs best improvements
v7 = [m for m in merged if m["cfg_hash"] == "v7_baseline"]
if v7:
    v7 = v7[0]
    print(f"\n{'='*100}\nV7 REFERENCE\n{'='*100}")
    print(f"  base=${v7['base_profit']:,} dd={v7['base_dd']}% trades={v7['base_trades']}")
    print(f"  ruin={v7['ruin']} mc_median=${v7['mc_median']:,}")
    print(f"  WF:  " + " ".join([f"{k[-1]}=${v7['wf'][k]['test'].get('profit',0):,.0f}"
                                  for k in sorted(v7['wf'].keys())]))
    print(f"  OOS=${v7['oos'].get('profit', 0):.1f}")

# Save re-ranking to JSON
out_path = os.path.join(RESULTS_DIR, "v9_realistic_ranking.json")
with open(out_path, "w") as f:
    json.dump({
        "robust_top20": robust[:20],
        "realistic_top20": realistic[:20],
        "best_mc_top20": best_mc[:20],
        "conservative_top20": conservative[:20],
        "v7_baseline": v7 if v7 else None,
    }, f, default=str, indent=2)
print(f"\nSaved: {out_path}")
