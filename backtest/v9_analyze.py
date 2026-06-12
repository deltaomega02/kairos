#!/usr/bin/env python3
"""
v9 Analysis — quick introspection of sweep results.
Usage:
  python3 v9_analyze.py [phase1|phase2|phase3|summary|validation] [N=20]
"""
import os
import sys
import json
from collections import Counter, defaultdict

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v9"


def load(phase):
    f = os.path.join(RESULTS_DIR, f"v9_{phase}.json")
    if not os.path.exists(f):
        print(f"NOT FOUND: {f}")
        return None
    with open(f) as fp:
        return json.load(fp)


def show_top(results, n=20, sort_by="score"):
    alive = [r for r in results if not r.get("ruined") and not r.get("error")]
    alive.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    print(f"\n=== Top {n} by {sort_by} ({len(alive)} alive / {len(results)} total) ===")
    for i, r in enumerate(alive[:n], 1):
        c = r["cfg"]
        lng = r.get("long", {})
        sht = r.get("short", {})
        print(f"{i:3}. ${r['profit']:>11,.0f} dd={r['dd']:>5.1f}% tr={r['trades']:>4} "
              f"wr={r['wr']:>4.1f}% L:{lng.get('wr',0):4.1f}% S:{sht.get('wr',0):4.1f}% "
              f"| EMA{c['ema_fast']}/{c['ema_slow']} sl{c['sl_atr_mult']} tp{c['tp_rr_ratio']} "
              f"trl{c['trailing_activation']}/{c['trailing_distance']} "
              f"adx{c['adx_enter_trending']}/{c['adx_exit_trending']} "
              f"atr{c['atr_high_vol_percentile']} "
              f"sim{c['max_simultaneous']} lev{c['max_leverage']} "
              f"risk{c['risk_per_trade']*100:.1f}% "
              f"ob{c.get('orderbook_imbalance_min',0.55)} "
              f"pb{c.get('pullback_ema_dist_pct',1.5)} "
              f"es{c.get('entry_score_threshold',40)}")


def stability_near(results, top_n=1):
    """For top-K configs, count nearby configs with similar performance (robustness)."""
    alive = sorted([r for r in results if not r.get("ruined")],
                   key=lambda x: x.get("score", 0), reverse=True)
    if not alive:
        return
    print("\n=== Parameter frequencies in top 50 ===")
    top50 = alive[:50]
    fields = ["ema_fast", "ema_slow", "sl_atr_mult", "tp_rr_ratio",
              "entry_score_threshold", "pullback_ema_dist_pct",
              "adx_enter_trending", "adx_exit_trending",
              "atr_high_vol_percentile",
              "trailing_activation", "trailing_distance",
              "risk_per_trade", "max_leverage", "max_simultaneous",
              "orderbook_imbalance_min", "use_funding"]
    for f in fields:
        vals = [r["cfg"].get(f) for r in top50]
        counter = Counter(vals)
        most = counter.most_common(5)
        print(f"  {f:30} {most}")


def profit_dd_scatter(results, bucket_dd=10):
    """Bucket by DD and show profit distribution."""
    alive = [r for r in results if not r.get("ruined")]
    buckets = defaultdict(list)
    for r in alive:
        b = int(r["dd"] // bucket_dd) * bucket_dd
        buckets[b].append(r["profit"])
    print("\n=== Profit by DD bucket ===")
    print(f"  {'DD range':<15} {'count':>6} {'median $':>12} {'max $':>12}")
    for k in sorted(buckets.keys()):
        profits = sorted(buckets[k])
        med = profits[len(profits)//2]
        mx = profits[-1]
        print(f"  {k:>2}-{k+bucket_dd:<10} {len(profits):>6} {med:>12,.0f} {mx:>12,.0f}")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "phase2"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    data = load(phase)
    if not data:
        # Try loading all 3 phases and combine
        combined = []
        for p in ("phase1", "phase2", "phase3"):
            d = load(p)
            if d and "results" in d:
                combined.extend(d["results"])
        if combined:
            print(f"Combined {len(combined)} results from all phases")
            show_top(combined, n, "score")
            print("\n=== Sort by raw profit ===")
            show_top(combined, n, "profit")
            profit_dd_scatter(combined)
            stability_near(combined)
        return

    results = data.get("results", [])
    if not results and "top_50_by_score" in data:
        results = data["top_50_by_score"]
    elif not results and "ranked" in data:
        results = data["ranked"]

    show_top(results, n, "score")
    print("\n=== Sort by raw profit ===")
    show_top(results, n, "profit")
    print("\n=== Sort by Calmar (profit/DD) ===")
    show_top(results, n, "calmar")
    profit_dd_scatter(results)
    stability_near(results)


if __name__ == "__main__":
    main()
