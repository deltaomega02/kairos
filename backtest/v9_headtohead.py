#!/usr/bin/env python3
"""
v9 Head-to-Head — 최종 후보들을 다양한 기간에서 직접 비교
=========================================================
Configs:
  A. v7 현재 운영 (EMA 5/18)
  B. v7 + EMA 3/15 교체 (순위 1)
  C. v7 + EMA 3/12
  D. 순위 2 다중 변경 (EMA 3/12 sl2.0 tp8.0 tr1.0/0.05)
  E. 순위 3 초보수 (EMA 3/15 sim2 lev4 tp10)

Periods:
  1. 6년 full
  2. 2026 YTD (01-01 ~ 04-18)
  3. 실전 기간 (04-09 ~ 04-18)
  4. 2025 full year
  5. 2024 ETF bull
  6. 2023 저변동 지옥
  7. 2022 크래시
  8. 2021 대상승장
  9. 2020 COVID
  10. BTC 피크→크래시 (2021-11 ~ 2022-06)
  11. 2026-04 (한 달)
"""
import os
import sys
import json
import time
from datetime import datetime

import pandas as pd
import multiprocessing as mp
mp_ctx = mp.get_context("fork")

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS
from v9_mega_sweep import _load_data, SYMBOLS, SEED, SLIP, DAILY_COST

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v9"
_DATA = None


def _worker_init():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()


def base_cfg():
    """v7 baseline as full kwargs dict."""
    return {
        "ema_fast": 5, "ema_slow": 18,
        "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
        "pullback_ema_dist_pct": 1.5,
        "sl_atr_mult": 1.5, "tp_rr_ratio": 6.0,
        "entry_score_threshold": 40,
        "adx_enter_trending": 30, "adx_exit_trending": 20,
        "atr_high_vol_percentile": 85,
        "orderbook_imbalance_min": 0.55,
        "funding_bias_threshold": 0.0005,
        "trailing_activation": 1.2, "trailing_distance": 0.1,
        "risk_per_trade": 0.015,
        "max_leverage": 7,
        "max_simultaneous": 3,
        "use_funding": True,
    }


CANDIDATES = [
    ("A. v7 현재 (EMA 5/18)", {}),
    ("B. v7 + EMA 3/15 (순위 1)", {"ema_fast": 3, "ema_slow": 15}),
    ("C. v7 + EMA 3/12", {"ema_fast": 3, "ema_slow": 12}),
    ("D. 순위 2 (3/12+sl2.0+tp8.0+tr1.0/0.05)",
     {"ema_fast": 3, "ema_slow": 12, "sl_atr_mult": 2.0, "tp_rr_ratio": 8.0,
      "trailing_activation": 1.0, "trailing_distance": 0.05}),
    ("E. 순위 3 초보수 (3/15+sim2+lev4+tp10+tr1.0/0.08)",
     {"ema_fast": 3, "ema_slow": 15, "sl_atr_mult": 1.75, "tp_rr_ratio": 10.0,
      "trailing_activation": 1.0, "trailing_distance": 0.08,
      "max_simultaneous": 2, "max_leverage": 4}),
]


PERIODS = [
    ("6년 Full (2020-03~2026-04-18)", "2020-03-25", "2026-04-20", (2023,)),
    ("2026 YTD (01-01~04-18)", "2026-01-01", "2026-04-20", ()),
    ("실전기간 (2026-04-09~04-18)", "2026-04-09", "2026-04-20", ()),
    ("최근 2주 (2026-04-05~04-18)", "2026-04-05", "2026-04-20", ()),
    ("2025 전체", "2025-01-01", "2026-01-01", ()),
    ("2024 ETF Bull", "2024-01-01", "2025-01-01", ()),
    ("2023 저변동", "2023-01-01", "2024-01-01", ()),
    ("2022 크래시", "2022-01-01", "2023-01-01", ()),
    ("2021 대상승", "2021-01-01", "2022-01-01", ()),
    ("2020 COVID", "2020-03-25", "2021-01-01", ()),
    ("BTC 피크→크래시", "2021-11-01", "2022-07-01", ()),
    ("FTX 붕괴", "2022-10-15", "2022-12-15", ()),
    ("Luna 붕괴", "2022-04-15", "2022-06-15", ()),
    ("2026-04 (한 달)", "2026-04-01", "2026-04-20", ()),
    ("2026-03 (직전 한 달)", "2026-03-01", "2026-04-01", ()),
]


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


def run_one(args):
    cand_label, overrides, period_label, start, end, skip = args
    global _DATA
    if _DATA is None:
        _DATA = _load_data()

    cfg = {**base_cfg(), **overrides}
    data = filter_data(_DATA, start, end)

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
            use_funding=cfg["use_funding"],
            trailing_activation=cfg["trailing_activation"],
            trailing_distance=cfg["trailing_distance"],
            block_sol_long=True,
            skip_years=skip,
            daily_cost_usd=DAILY_COST,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=SLIP,
            max_simultaneous=cfg["max_simultaneous"],
            risk_per_trade=cfg["risk_per_trade"],
            max_leverage=cfg["max_leverage"],
            enabled_symbols=SYMBOLS,
        )
        trades = r.get("_trades", [])
        long_w = sum(1 for t in trades if t["direction"] == "LONG" and t["pnl"] > 0)
        long_t = sum(1 for t in trades if t["direction"] == "LONG")
        short_w = sum(1 for t in trades if t["direction"] == "SHORT" and t["pnl"] > 0)
        short_t = sum(1 for t in trades if t["direction"] == "SHORT")
        # Trade-level summary for short periods (live period inspection)
        trade_list = []
        if len(trades) <= 40:
            for t in trades:
                trade_list.append({
                    "ts": datetime.utcfromtimestamp(t["timestamp"]/1000).isoformat(),
                    "sym": t["symbol"], "dir": t["direction"],
                    "entry": round(t["entry_price"], 2),
                    "exit": round(t["exit_price"], 2),
                    "pnl": round(t["pnl"], 2),
                    "reason": t["reason"],
                })
        return {
            "candidate": cand_label,
            "period": period_label,
            "profit": r["net_profit"],
            "dd": r["max_dd"],
            "trades": r["total_trades"],
            "wr": r["win_rate"],
            "ruined": r["ruined"],
            "final_bal": r["final_balance"],
            "long_wr": round(long_w / max(long_t, 1) * 100, 1),
            "long_n": long_t,
            "short_wr": round(short_w / max(short_t, 1) * 100, 1),
            "short_n": short_t,
            "trade_list": trade_list,
        }
    except Exception as e:
        return {
            "candidate": cand_label, "period": period_label,
            "error": str(e), "ruined": True,
            "profit": 0, "dd": 0, "trades": 0, "wr": 0,
        }


def main():
    print(f"Starting head-to-head at {datetime.now().isoformat()}", flush=True)
    # Build task list
    tasks = []
    for cand_label, overrides in CANDIDATES:
        for period_label, start, end, skip in PERIODS:
            tasks.append((cand_label, overrides, period_label, start, end, skip))
    print(f"Tasks: {len(tasks)}", flush=True)

    t0 = time.time()
    results = []
    with mp_ctx.Pool(14, initializer=_worker_init) as p:
        for i, res in enumerate(p.imap_unordered(run_one, tasks), 1):
            results.append(res)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)

    # Organize results: nested dict[candidate][period]
    by_cand = {}
    for r in results:
        c = r["candidate"]
        p = r["period"]
        by_cand.setdefault(c, {})[p] = r

    # Save
    out_path = os.path.join(RESULTS_DIR, "v9_headtohead.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "by_candidate": by_cand,
        }, f, default=str, indent=2)
    print(f"Saved: {out_path}", flush=True)

    # ======= PRINT TABLES =======
    cand_labels = [c[0] for c in CANDIDATES]
    period_labels = [p[0] for p in PERIODS]

    # Master table: period × candidate, profit/DD
    print("\n" + "=" * 120)
    print(" 📊 종합 비교표 — 각 셀: profit / DD / trades / win rate")
    print("=" * 120)
    print(f"{'기간':<32}", end="")
    for c in cand_labels:
        short_c = c.split(".")[0].strip()  # "A", "B", "C", "D", "E"
        print(f"{short_c:>19}", end="")
    print()
    print("-" * 120)
    for pl in period_labels:
        print(f"{pl[:32]:<32}", end="")
        for c in cand_labels:
            r = by_cand.get(c, {}).get(pl, {})
            if r.get("ruined") or r.get("error"):
                cell = "RUINED"
            elif r.get("trades", 0) == 0:
                cell = "— 거래없음 —"
            else:
                prof = r["profit"]
                dd = r["dd"]
                tr = r["trades"]
                wr = r["wr"]
                if abs(prof) >= 1e9:
                    ps = f"{prof/1e9:+.1f}B"
                elif abs(prof) >= 1e6:
                    ps = f"{prof/1e6:+.1f}M"
                elif abs(prof) >= 1e3:
                    ps = f"{prof/1e3:+.1f}k"
                else:
                    ps = f"{prof:+.0f}"
                cell = f"${ps}/{dd:.0f}%/{tr}/{wr:.0f}%"
            print(f"{cell:>19}", end="")
        print()

    # Legend
    print("\n범례:")
    for c, _ in CANDIDATES:
        print(f"  {c}")

    # ======= LIVE PERIOD DETAILED TRADE-BY-TRADE =======
    print("\n" + "=" * 120)
    print(" 🎯 실전 기간 (2026-04-09~04-18) — 각 설정이 실제 뽑았을 trades")
    print("=" * 120)
    for c in cand_labels:
        r = by_cand.get(c, {}).get("실전기간 (2026-04-09~04-18)", {})
        print(f"\n{c}")
        if r.get("ruined") or r.get("error"):
            print("  RUINED or ERROR")
            continue
        print(f"  요약: profit=${r['profit']:+.2f}, dd={r['dd']:.1f}%, trades={r['trades']}, wr={r['wr']:.1f}%")
        trades = r.get("trade_list", [])
        if not trades:
            print("  (거래 없음)")
        else:
            for t in trades:
                print(f"  {t['ts'][:16]} {t['sym']:<8} {t['dir']:<5} entry=${t['entry']:>9} exit=${t['exit']:>9} pnl=${t['pnl']:>+7.2f} {t['reason']}")


if __name__ == "__main__":
    main()
