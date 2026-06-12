#!/usr/bin/env python3
"""
EMA 1/12 등 초단기 EMA 조합을 실전 기간 + MC 로 검증.
"""
import os, sys, json, time, random
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


def v7_params(ema_f, ema_s):
    return {
        **DEFAULT_PARAMS,
        "ema_fast": ema_f, "ema_slow": ema_s,
        "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
        "pullback_ema_dist_pct": 1.5,
        "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
        "entry_score_threshold": 40,
        "adx_enter_trending": 30, "adx_exit_trending": 20,
        "atr_high_vol_percentile": 85,
        "orderbook_imbalance_min": 0.55,
        "funding_bias_threshold": 0.0005,
    }


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


def run_bt(data, ema_f, ema_s, skip_years=()):
    params = v7_params(ema_f, ema_s)
    try:
        r = run_shared_backtest(
            data, params, SEED,
            use_funding=True,
            trailing_activation=1.2, trailing_distance=0.1,
            block_sol_long=True,
            skip_years=skip_years,
            daily_cost_usd=DAILY_COST, ruin_threshold=15.0,
            use_cooldown=False, slippage_pct=SLIP,
            max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
            enabled_symbols=SYMBOLS,
        )
        return {"profit": r["net_profit"], "dd": r["max_dd"],
                "trades": r["total_trades"], "wr": r["win_rate"],
                "ruined": r["ruined"],
                "_trades": r.get("_trades", [])}
    except Exception as e:
        return {"error": str(e), "ruined": True, "profit": 0, "dd": 0,
                "trades": 0, "wr": 0, "_trades": []}


def mc_bootstrap(trades, n=300):
    if not trades:
        return {"ruin_rate": 1.0}
    ruined = 0
    profits = []
    dds = []
    for i in range(n):
        random.seed(10000 + i)
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
            dds.append(mdd)
    profits.sort()
    def pct(p):
        return profits[int(p * (len(profits)-1))] if profits else 0
    return {
        "ruin_rate": round(ruined / n, 3),
        "median": round(pct(0.5), 0) if profits else 0,
        "p25": round(pct(0.25), 0) if profits else 0,
        "p75": round(pct(0.75), 0) if profits else 0,
    }


def worker(args):
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    label, ema_f, ema_s, period_name, start, end, skip = args
    data = filter_data(_DATA, start, end)
    r = run_bt(data, ema_f, ema_s, skip_years=skip)
    out = {"label": label, "ema": f"{ema_f}/{ema_s}", "period": period_name,
           "profit": r["profit"], "dd": r["dd"], "trades": r["trades"],
           "wr": r["wr"], "ruined": r["ruined"]}
    if period_name == "Full 6y":
        # Include MC
        mc = mc_bootstrap(r["_trades"], n=300)
        out["mc"] = mc
    return out


def main():
    # Candidates to compare
    cands = [
        ("A v7", 5, 18),
        ("B EMA 3/15", 3, 15),
        ("F EMA 1/12", 1, 12),
        ("G EMA 1/15", 1, 15),
        ("H EMA 1/18", 1, 18),
        ("I EMA 1/10", 1, 10),
        ("J EMA 2/15", 2, 15),
        ("K EMA 2/12", 2, 12),
    ]
    periods = [
        ("Full 6y", "2020-03-25", "2026-04-20", (2023,)),
        ("2026 YTD", "2026-01-01", "2026-04-20", ()),
        ("실전 04-09~18", "2026-04-09", "2026-04-20", ()),
        ("최근 2주", "2026-04-05", "2026-04-20", ()),
        ("2025", "2025-01-01", "2026-01-01", ()),
        ("2024", "2024-01-01", "2025-01-01", ()),
        ("2023 저변동", "2023-01-01", "2024-01-01", ()),
        ("2022 크래시", "2022-01-01", "2023-01-01", ()),
        ("2021 Bull", "2021-01-01", "2022-01-01", ()),
        ("2020 COVID", "2020-03-25", "2021-01-01", ()),
    ]
    tasks = [(l, f, s, p[0], p[1], p[2], p[3])
             for l, f, s in cands for p in periods]
    print(f"Tasks: {len(tasks)}", flush=True)

    t0 = time.time()
    results = []
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, r in enumerate(p.imap_unordered(worker, tasks), 1):
            results.append(r)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} elapsed={time.time()-t0:.0f}s", flush=True)

    # Pivot: rows=candidate, cols=period
    by_cand = {}
    for r in results:
        by_cand.setdefault(r["label"], {})[r["period"]] = r

    # Print table
    cand_labels = [c[0] for c in cands]
    period_labels = [p[0] for p in periods]

    print("\n" + "=" * 160)
    print("📊 EMA 후보 × 기간 비교 (profit / DD%)")
    print("=" * 160)
    print(f"{'Period':<18}", end="")
    for c in cand_labels:
        print(f"{c:>18}", end="")
    print()
    print("-" * 160)
    for p in period_labels:
        print(f"{p:<18}", end="")
        for c in cand_labels:
            r = by_cand.get(c, {}).get(p, {})
            if r.get("ruined"):
                cell = "RUIN"
            else:
                prof = r.get("profit", 0)
                dd = r.get("dd", 0)
                if abs(prof) >= 1e6:
                    ps = f"{prof/1e6:+.1f}M"
                elif abs(prof) >= 1e3:
                    ps = f"{prof/1e3:+.1f}k"
                else:
                    ps = f"{prof:+.0f}"
                cell = f"{ps}/{dd:.0f}%"
            print(f"{cell:>18}", end="")
        print()

    # MC summary
    print("\n" + "=" * 120)
    print("🎲 Monte Carlo (6y trades bootstrap, 300 iter) — 파산률 + 중앙값")
    print("=" * 120)
    for c in cand_labels:
        r = by_cand.get(c, {}).get("Full 6y", {})
        mc = r.get("mc", {})
        print(f"  {c:<18} ruin={mc.get('ruin_rate','?'):<6} "
              f"p25=${mc.get('p25',0):>12,.0f} median=${mc.get('median',0):>12,.0f} "
              f"p75=${mc.get('p75',0):>12,.0f}")

    # Save
    with open("~/Projects/HERMES_백테스팅/v9/v9_ema_live_mc.json", "w") as f:
        json.dump(by_cand, f, default=str, indent=2)


if __name__ == "__main__":
    main()
