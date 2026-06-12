#!/usr/bin/env python3
"""
실전 거래 (2026-04-09~18, 25건) vs V7/V8/V9/V11 백테스트 비교
==============================================================
사용자의 실제 거래 결과와 각 버전이 같은 기간에서 뽑았을 trades 비교.

핵심: 이게 가장 강력한 엔진 검증. 실전 v7 vs 백테 v7 일치하면 엔진 신뢰 확립.
"""
import os, sys, json
from datetime import datetime
from collections import defaultdict
import pandas as pd

sys.path.insert(0, "/Users/sue/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SEED, SLIP, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v11"

# 실전 거래 (사용자 기록 기준, KST)
ACTUAL_TRADES = [
    # (dt_kst, coin, direction, pnl_usd, pct)
    ("2026-04-09 22:47", "BTC", "LONG",   -7.47, -1.31),
    ("2026-04-10 07:30", "BTC", "LONG",   +3.27, +0.57),
    ("2026-04-11 00:28", "ETH", "LONG",   +6.83, +1.20),
    ("2026-04-11 00:40", "BTC", "LONG",   +3.49, +0.61),
    ("2026-04-12 01:40", "ETH", "LONG",  +10.96, +1.92),
    ("2026-04-12 03:41", "BTC", "LONG",  +11.71, +2.05),
    ("2026-04-12 08:10", "BTC", "LONG",   -7.77, -1.36),
    ("2026-04-12 10:35", "ETH", "LONG",   -8.94, -1.56),
    ("2026-04-13 05:43", "BTC", "SHORT",  -7.14, -1.25),
    ("2026-04-13 05:43", "ETH", "SHORT",  -7.71, -1.35),
    ("2026-04-13 23:43", "BTC", "SHORT",  -7.32, -1.28),
    ("2026-04-13 23:44", "ETH", "SHORT",  -7.18, -1.26),
    ("2026-04-14 22:43", "BTC", "LONG",   +6.72, +1.18),
    ("2026-04-15 03:50", "BTC", "LONG",   -7.76, -1.36),
    ("2026-04-16 03:55", "ETH", "LONG",   +7.16, +1.25),
    ("2026-04-16 04:36", "BTC", "LONG",   +7.40, +1.29),
    ("2026-04-16 19:03", "ETH", "LONG",   -7.62, -1.33),
    ("2026-04-16 22:27", "XRP", "LONG",   +8.32, +1.46),
    ("2026-04-16 22:52", "BTC", "LONG",   -8.42, -1.47),
    ("2026-04-16 22:54", "ETH", "SHORT",  +8.78, +1.54),
    ("2026-04-17 04:59", "ETH", "SHORT",  -6.40, -1.12),
    ("2026-04-17 18:15", "BTC", "LONG",   +6.14, +1.08),
    ("2026-04-18 17:23", "ETH", "LONG",   -7.50, -1.31),
    ("2026-04-18 17:34", "XRP", "LONG",   -8.68, -1.52),
    ("2026-04-18 19:08", "BTC", "LONG",   -8.47, -1.48),
    ("2026-04-19 16:10", "SOL", "SHORT",  +9.81, +1.69),
    ("2026-04-20 16:21", "BTC", "SHORT",  -8.04, -1.39),
    ("2026-04-20 16:21", "SOL", "SHORT",  -6.37, -1.10),
]

# 버전별 설정
V7 = {"params": {**DEFAULT_PARAMS, "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
                  "tp_rr_ratio": 6.0, "entry_score_threshold": 40, "pullback_ema_dist_pct": 1.5,
                  "adx_enter_trending": 30},
      "kw": {}}
V8 = {"params": {**DEFAULT_PARAMS, "ema_fast": 3, "ema_slow": 15, "sl_atr_mult": 1.5,
                  "tp_rr_ratio": 6.0, "entry_score_threshold": 40, "pullback_ema_dist_pct": 1.5,
                  "adx_enter_trending": 30},
      "kw": {}}
V9 = {"params": V8["params"],
      "kw": {"d1_filter_enable": True, "d1_ema_period": 10, "d1_mode": "direction"}}
V11 = {"params": V8["params"],
       "kw": {"d1_filter_enable": True, "d1_ema_period": 2, "d1_mode": "price_above_ema"}}


def filter_d(data, s, e):
    s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
    e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and 'timestamp' in v.columns:
            m = (v['timestamp']>=s_ts) & (v['timestamp']<e_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


def run_version(data, ver, start, end):
    """version을 start~end 기간에서 실행 (warmup은 start 이전 데이터 자동 포함)"""
    d = filter_d(data, start, end)
    base = dict(trailing_activation=1.2, trailing_distance=0.1,
                skip_years=(), daily_cost_usd=DAILY_COST, slippage_pct=SLIP,
                max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                enabled_symbols=SYMBOLS, block_sol_long=True)
    return run_shared_backtest_v11(d, ver["params"], SEED, **base, **ver["kw"])


def filter_trades_by_window(trades, start_ts, end_ts):
    return [t for t in trades if start_ts <= t["timestamp"] < end_ts]


def kst_to_utc_ts(dt_kst_str):
    """KST 문자열 → UTC ms timestamp"""
    dt = datetime.strptime(dt_kst_str, "%Y-%m-%d %H:%M")
    # KST = UTC+9
    utc_ts = int((dt.timestamp() - 9*3600) * 1000)
    return utc_ts


def compare_day_by_day(actual, backtest_trades):
    """날짜별 집계 비교"""
    act_by_day = defaultdict(lambda: {"n": 0, "pnl": 0, "wins": 0})
    for dt_str, coin, direction, pnl, pct in actual:
        d = dt_str.split(" ")[0]
        act_by_day[d]["n"] += 1
        act_by_day[d]["pnl"] += pnl
        if pnl > 0:
            act_by_day[d]["wins"] += 1

    bt_by_day = defaultdict(lambda: {"n": 0, "pnl": 0, "wins": 0, "trades": []})
    for t in backtest_trades:
        ts_utc = t["timestamp"]
        dt_kst = datetime.utcfromtimestamp(ts_utc/1000 + 9*3600)
        d = dt_kst.strftime("%Y-%m-%d")
        bt_by_day[d]["n"] += 1
        bt_by_day[d]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            bt_by_day[d]["wins"] += 1
        bt_by_day[d]["trades"].append({
            "kst": dt_kst.strftime("%H:%M"),
            "coin": t["symbol"][:3],
            "dir": t["direction"],
            "pnl": t["pnl"],
            "reason": t["reason"],
        })
    return dict(act_by_day), dict(bt_by_day)


def main():
    data = _load_data()
    # 충분한 warmup: 2026-03-01부터 시작
    start = "2026-03-01"
    end = "2026-04-21"

    # Window for live period
    live_start = "2026-04-09"
    live_end = "2026-04-21"
    live_start_ts = int(datetime.strptime(live_start, '%Y-%m-%d').timestamp()*1000)
    live_end_ts = int(datetime.strptime(live_end, '%Y-%m-%d').timestamp()*1000)

    results = {}
    for name, ver in [("V7", V7), ("V8", V8), ("V9", V9), ("V11", V11)]:
        r = run_version(data, ver, start, end)
        all_trades = r.get("_trades", [])
        # 04-09 ~ 04-18 창만
        window_trades = filter_trades_by_window(all_trades, live_start_ts, live_end_ts)
        results[name] = {
            "all_trades_n": len(all_trades),
            "window_trades": window_trades,
            "window_n": len(window_trades),
            "window_pnl": sum(t["pnl"] for t in window_trades),
            "window_wins": sum(1 for t in window_trades if t["pnl"] > 0),
        }

    # 실전 통계
    actual_n = len(ACTUAL_TRADES)
    actual_pnl = sum(t[3] for t in ACTUAL_TRADES)
    actual_wins = sum(1 for t in ACTUAL_TRADES if t[3] > 0)
    actual_wr = actual_wins / actual_n * 100

    # ============= PRINT =============
    print("=" * 100)
    print("🎯 실전 2026-04-09~18 vs 각 버전 백테스트 비교")
    print("=" * 100)
    print(f"\n실전 (v7, 2026-04-09 13:47 UTC 시작):")
    print(f"  거래: {actual_n}건  승: {actual_wins}  패: {actual_n-actual_wins}  "
          f"승률: {actual_wr:.1f}%  PnL: ${actual_pnl:+.2f}")

    print(f"\n{'Version':<8}{'거래수':>7}{'승률':>8}{'PnL':>12}{'vs Actual':>14}")
    print("-" * 55)
    for name in ["V7", "V8", "V9", "V11"]:
        r = results[name]
        wr = r["window_wins"] / max(r["window_n"], 1) * 100
        diff_pnl = r["window_pnl"] - actual_pnl
        print(f"{name:<8}{r['window_n']:>7}{wr:>7.1f}%{r['window_pnl']:>+11.2f}${diff_pnl:>+12.2f}$")
    print(f"{'Actual':<8}{actual_n:>7}{actual_wr:>7.1f}%{actual_pnl:>+11.2f}$    (기준)")

    # Day-by-day
    print("\n" + "=" * 100)
    print("📅 날짜별 비교")
    print("=" * 100)
    act_day, _ = compare_day_by_day(ACTUAL_TRADES, [])
    days = sorted(act_day.keys())
    print(f"  {'Date':<12}{'Actual':>16}", end="")
    for name in ["V7", "V8", "V9", "V11"]:
        print(f"{name:>14}", end="")
    print()
    print("-" * 100)

    bt_days = {}
    for name in ["V7", "V8", "V9", "V11"]:
        _, bt = compare_day_by_day(ACTUAL_TRADES, results[name]["window_trades"])
        bt_days[name] = bt

    for d in days:
        a = act_day[d]
        print(f"  {d:<12}{a['n']:>3}건 ${a['pnl']:>+6.1f}  ", end="")
        for name in ["V7", "V8", "V9", "V11"]:
            b = bt_days[name].get(d, {"n": 0, "pnl": 0})
            print(f"  {b['n']:>2}건 ${b['pnl']:>+6.1f}", end="")
        print()

    # V11 trade-by-trade
    print("\n" + "=" * 100)
    print("🔍 V11 Backtest Trades (04-09~18 구간)")
    print("=" * 100)
    for t in results["V11"]["window_trades"]:
        dt_kst = datetime.utcfromtimestamp(t["timestamp"]/1000 + 9*3600).strftime("%m-%d %H:%M KST")
        print(f"  {dt_kst}  {t['symbol'][:3]} {t['direction']:<5}  "
              f"${t['entry_price']:>10.2f}→${t['exit_price']:>10.2f}  ${t['pnl']:>+7.2f}  {t['reason']}")

    # Actual
    print("\n" + "=" * 100)
    print("🔍 Actual Trades (실전)")
    print("=" * 100)
    cumulative = 0
    for dt_kst, coin, direction, pnl, pct in ACTUAL_TRADES:
        cumulative += pnl
        print(f"  {dt_kst}  {coin:<3} {direction:<5}  {pct:>+6.2f}%  ${pnl:>+6.2f}  누적: ${cumulative:>+7.2f}")

    # Summary
    print("\n" + "=" * 100)
    print("📊 핵심 관찰")
    print("=" * 100)
    v11 = results["V11"]
    v9 = results["V9"]
    v8 = results["V8"]
    v7 = results["V7"]
    gap = v11["window_pnl"] - actual_pnl
    print(f"""
  실전 (v7 운영): ${actual_pnl:+.2f} ({actual_n}거래, 승률 {actual_wr:.1f}%)
  V7 백테:        ${v7['window_pnl']:+.2f} ({v7['window_n']}거래)
  V8 백테:        ${v8['window_pnl']:+.2f} ({v8['window_n']}거래)
  V9 백테:        ${v9['window_pnl']:+.2f} ({v9['window_n']}거래)
  V11 백테:       ${v11['window_pnl']:+.2f} ({v11['window_n']}거래)

  실전 vs V7 백테 차이: ${v7['window_pnl'] - actual_pnl:+.2f} (실전이 얼마나 다른가?)
  V7 → V11 개선 폭: ${v11['window_pnl'] - v7['window_pnl']:+.2f}
  실전 → V11 가상 개선: ${gap:+.2f}
    """)

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "actual": {"n": actual_n, "wins": actual_wins, "wr": actual_wr, "pnl": actual_pnl},
        "backtest": {n: {"n": r["window_n"], "pnl": r["window_pnl"],
                         "wins": r["window_wins"]}
                     for n, r in results.items()},
        "v11_window_trades": [
            {"ts_kst": datetime.utcfromtimestamp(t["timestamp"]/1000 + 9*3600).isoformat(),
             "symbol": t["symbol"], "direction": t["direction"],
             "entry": round(t["entry_price"], 2), "exit": round(t["exit_price"], 2),
             "pnl": round(t["pnl"], 2), "reason": t["reason"]}
            for t in results["V11"]["window_trades"]
        ],
    }
    with open(os.path.join(RESULTS_DIR, "v11_live_vs_backtest.json"), "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"Saved: {RESULTS_DIR}/v11_live_vs_backtest.json")


if __name__ == "__main__":
    main()
