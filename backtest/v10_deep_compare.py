#!/usr/bin/env python3
"""
v10 Deep Compare — V8 vs V9 (V8 + 1D EMA10 filter) 심층 분석

내용:
  1. 전체 거래 리스트 비교 (양쪽 전부 추출)
  2. 연도별/월별/코인별/방향별 breakdown
  3. 1D 필터가 구체적으로 뭘 차단했는지 (rejected signals 분석)
  4. 04-18 실전 기간 trade-by-trade 비교
  5. Equity curve (월별 잔고)
  6. Drawdown timeline (피크-트러프)
  7. 하루 단위 실전 10일 비교
  8. 스트레스 구간별 성과
  9. Monte Carlo 분포 상세 (percentile별)
"""
import os, sys, json, time, random
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SYMBOLS, SEED, SLIP, DAILY_COST
from comprehensive_backtest import DEFAULT_PARAMS
from v10_multi_ema_engine import run_shared_backtest_v10

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v10"

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


def run(data, v9=False, period=None):
    if period:
        d = filter_data(data, *period)
    else:
        d = data
    kw = base_kw()
    if v9:
        kw.update({"d1_filter_enable": True, "d1_ema_period": 10,
                   "d1_filter_mode": "direction"})
    return run_shared_backtest_v10(d, V8_PARAMS, SEED, **kw)


def build_equity_curve(trades):
    """거래 리스트 → 시간순 누적 잔고 (daily granularity)"""
    sorted_trades = sorted(trades, key=lambda t: t["timestamp"])
    if not sorted_trades:
        return []
    bal = SEED
    peak = SEED
    curve = [{"ts": sorted_trades[0]["timestamp"] - 86400000, "bal": bal,
              "dd": 0, "peak": SEED}]
    for t in sorted_trades:
        bal += t["pnl"]
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        curve.append({"ts": t["timestamp"], "bal": round(bal, 2),
                      "dd": round(dd, 2), "peak": round(peak, 2)})
    return curve


def compute_monthly_bal(curve):
    """Equity curve → 월별 마감 잔고"""
    by_month = {}
    for pt in curve:
        dt = datetime.utcfromtimestamp(pt["ts"] / 1000)
        month = f"{dt.year:04d}-{dt.month:02d}"
        by_month[month] = pt["bal"]  # 마지막 값이 기록됨 (in order)
    return by_month


def segmenting(trades):
    """거래 → 다양한 차원 집계"""
    by_year = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0, "longs": 0, "long_w": 0, "shorts": 0, "short_w": 0})
    by_coin = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0})
    by_dir = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0})
    by_reason = defaultdict(lambda: {"n": 0, "pnl": 0})
    by_month = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0})
    by_coin_dir = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0})
    by_year_dir = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0})

    for t in trades:
        y = t["year"]
        c = t["symbol"]
        d = t["direction"]
        r = t["reason"]
        dt = datetime.utcfromtimestamp(t["timestamp"] / 1000)
        month = f"{dt.year:04d}-{dt.month:02d}"
        pnl = t["pnl"]
        w = 1 if pnl > 0 else 0

        by_year[y]["n"] += 1; by_year[y]["w"] += w; by_year[y]["pnl"] += pnl
        if d == "LONG":
            by_year[y]["longs"] += 1; by_year[y]["long_w"] += w
        else:
            by_year[y]["shorts"] += 1; by_year[y]["short_w"] += w

        by_coin[c]["n"] += 1; by_coin[c]["w"] += w; by_coin[c]["pnl"] += pnl
        by_dir[d]["n"] += 1; by_dir[d]["w"] += w; by_dir[d]["pnl"] += pnl
        by_reason[r]["n"] += 1; by_reason[r]["pnl"] += pnl
        by_month[month]["n"] += 1; by_month[month]["w"] += w; by_month[month]["pnl"] += pnl
        by_coin_dir[f"{c[:3]}-{d}"]["n"] += 1
        by_coin_dir[f"{c[:3]}-{d}"]["w"] += w
        by_coin_dir[f"{c[:3]}-{d}"]["pnl"] += pnl
        by_year_dir[f"{y}-{d}"]["n"] += 1
        by_year_dir[f"{y}-{d}"]["w"] += w
        by_year_dir[f"{y}-{d}"]["pnl"] += pnl

    return dict(by_year), dict(by_coin), dict(by_dir), dict(by_reason), dict(by_month), dict(by_coin_dir), dict(by_year_dir)


def find_blocked_signals(data, params=V8_PARAMS):
    """
    V8의 모든 거래 중, 1D 필터가 block했을 거래 찾기.
    방식: V8 전체 trades 얻고, 각 거래에 대해 진입 시점의 1D EMA10 방향 계산.
    """
    # V8 full run
    kw = base_kw()
    r8 = run_shared_backtest_v10(data, params, SEED, **kw)
    trades_v8 = r8["_trades"]

    # V9 full run
    kw9 = base_kw()
    kw9.update({"d1_filter_enable": True, "d1_ema_period": 10, "d1_filter_mode": "direction"})
    r9 = run_shared_backtest_v10(data, params, SEED, **kw9)
    trades_v9 = r9["_trades"]

    # V9에 없는 거래는 "1D 필터가 blocked"한 거래일 것 (또는 타이밍 차이로 다른 거래 들어간 경우)
    # 근사: 같은 symbol + direction + 비슷한 timestamp (±2h) 거래는 "match"로 간주
    def key(t):
        return (t["symbol"], t["direction"], t["timestamp"] // (3600 * 1000))

    v9_keys = set(key(t) for t in trades_v9)
    blocked = [t for t in trades_v8 if key(t) not in v9_keys]
    kept = [t for t in trades_v8 if key(t) in v9_keys]

    # Analyze blocked
    blocked_pnl = sum(t["pnl"] for t in blocked)
    blocked_w = sum(1 for t in blocked if t["pnl"] > 0)
    blocked_l = sum(1 for t in blocked if t["pnl"] < 0)
    kept_pnl = sum(t["pnl"] for t in kept)
    kept_w = sum(1 for t in kept if t["pnl"] > 0)

    return {
        "blocked_trades": blocked,
        "kept_trades": kept,
        "blocked_n": len(blocked),
        "blocked_w": blocked_w,
        "blocked_l": blocked_l,
        "blocked_pnl": round(blocked_pnl, 2),
        "blocked_wr": round(blocked_w / max(len(blocked), 1) * 100, 1),
        "kept_n": len(kept),
        "kept_w": kept_w,
        "kept_pnl": round(kept_pnl, 2),
        "kept_wr": round(kept_w / max(len(kept), 1) * 100, 1),
        "v8_total": len(trades_v8),
        "v9_total": len(trades_v9),
    }


def drawdown_events(curve, min_dd_pct=10):
    """Peak-trough 이벤트 추출 (DD > threshold)"""
    events = []
    current_peak = SEED
    current_peak_ts = curve[0]["ts"]
    peak_bal = SEED
    max_dd_reached = 0
    in_dd = False

    for pt in curve:
        bal = pt["bal"]
        ts = pt["ts"]
        if bal > peak_bal:
            # recovered or new peak
            if in_dd and max_dd_reached > min_dd_pct:
                dt_peak = datetime.utcfromtimestamp(current_peak_ts/1000).strftime("%Y-%m-%d")
                dt_rec = datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
                events.append({
                    "peak_date": dt_peak, "recovery_date": dt_rec,
                    "peak_bal": round(peak_bal, 2),
                    "trough_bal": round(peak_bal * (1 - max_dd_reached/100), 2),
                    "dd_pct": round(max_dd_reached, 1),
                    "days_to_recover": (datetime.utcfromtimestamp(ts/1000) -
                                         datetime.utcfromtimestamp(current_peak_ts/1000)).days,
                })
            peak_bal = bal
            current_peak_ts = ts
            max_dd_reached = 0
            in_dd = False
        else:
            dd = (peak_bal - bal) / peak_bal * 100
            if dd > max_dd_reached:
                max_dd_reached = dd
                in_dd = True
    return events


def main():
    print(f"Starting deep compare at {datetime.now().isoformat()}", flush=True)
    data = _load_data()

    # ===== Full 6y runs =====
    print("\n[1] Running V8 (6y)...", flush=True)
    t0 = time.time()
    r_v8 = run(data, v9=False)
    print(f"  V8: profit=${r_v8['net_profit']:,.0f} DD={r_v8['max_dd']}% "
          f"trades={r_v8['total_trades']} WR={r_v8['win_rate']}% "
          f"in {time.time()-t0:.0f}s", flush=True)

    print("[2] Running V9 (V8 + 1D EMA10 filter)...", flush=True)
    t0 = time.time()
    r_v9 = run(data, v9=True)
    print(f"  V9: profit=${r_v9['net_profit']:,.0f} DD={r_v9['max_dd']}% "
          f"trades={r_v9['total_trades']} WR={r_v9['win_rate']}% "
          f"in {time.time()-t0:.0f}s", flush=True)

    trades_v8 = r_v8["_trades"]
    trades_v9 = r_v9["_trades"]

    # ===== Build equity curves =====
    print("\n[3] Building equity curves...", flush=True)
    curve_v8 = build_equity_curve(trades_v8)
    curve_v9 = build_equity_curve(trades_v9)
    monthly_v8 = compute_monthly_bal(curve_v8)
    monthly_v9 = compute_monthly_bal(curve_v9)

    # ===== Segmenting =====
    print("[4] Segmenting trades...", flush=True)
    segs_v8 = segmenting(trades_v8)
    segs_v9 = segmenting(trades_v9)

    # ===== Filter rejection analysis =====
    print("[5] Filter rejection analysis...", flush=True)
    reject = find_blocked_signals(data)

    # ===== Drawdown events =====
    print("[6] Drawdown events...", flush=True)
    dd_v8 = drawdown_events(curve_v8, min_dd_pct=15)
    dd_v9 = drawdown_events(curve_v9, min_dd_pct=15)

    # ===== Live period trade comparison =====
    print("[7] Live period 04-09~18 comparison...", flush=True)
    r_v8_live = run(data, v9=False, period=("2026-04-09", "2026-04-20"))
    r_v9_live = run(data, v9=True, period=("2026-04-09", "2026-04-20"))

    # ===== Save all =====
    out = {
        "timestamp": datetime.now().isoformat(),
        "v8_summary": {
            "profit": r_v8["net_profit"], "dd": r_v8["max_dd"],
            "trades": r_v8["total_trades"], "wr": r_v8["win_rate"],
            "final_bal": r_v8["final_balance"],
        },
        "v9_summary": {
            "profit": r_v9["net_profit"], "dd": r_v9["max_dd"],
            "trades": r_v9["total_trades"], "wr": r_v9["win_rate"],
            "final_bal": r_v9["final_balance"],
        },
        "monthly_v8": monthly_v8,
        "monthly_v9": monthly_v9,
        "segments_v8": {
            "by_year": segs_v8[0], "by_coin": segs_v8[1],
            "by_direction": segs_v8[2], "by_reason": segs_v8[3],
            "by_coin_direction": segs_v8[5], "by_year_direction": segs_v8[6],
        },
        "segments_v9": {
            "by_year": segs_v9[0], "by_coin": segs_v9[1],
            "by_direction": segs_v9[2], "by_reason": segs_v9[3],
            "by_coin_direction": segs_v9[5], "by_year_direction": segs_v9[6],
        },
        "filter_rejection": {
            "v8_total": reject["v8_total"], "v9_total": reject["v9_total"],
            "blocked_n": reject["blocked_n"], "blocked_pnl": reject["blocked_pnl"],
            "blocked_wr": reject["blocked_wr"], "blocked_w": reject["blocked_w"],
            "blocked_l": reject["blocked_l"],
            "kept_n": reject["kept_n"], "kept_pnl": reject["kept_pnl"],
            "kept_wr": reject["kept_wr"], "kept_w": reject["kept_w"],
        },
        "drawdown_events_v8": dd_v8,
        "drawdown_events_v9": dd_v9,
        "live_10d_v8_trades": r_v8_live.get("_trades", [])[:30],
        "live_10d_v9_trades": r_v9_live.get("_trades", [])[:30],
    }
    path = os.path.join(RESULTS_DIR, "v10_deep_compare.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"Saved: {path}\n", flush=True)

    # ===== PRINT =====
    def pprint_seg(name, v8_seg, v9_seg, fmt="{:<15}{:>10}{:>10}{:>10}{:>10}"):
        print(f"\n===== {name} =====")
        keys = sorted(set(v8_seg.keys()) | set(v9_seg.keys()))
        print(fmt.format("Key", "V8 n/pnl", "V9 n/pnl", "Δ n", "Δ pnl"))
        for k in keys:
            v8 = v8_seg.get(k, {"n": 0, "pnl": 0})
            v9 = v9_seg.get(k, {"n": 0, "pnl": 0})
            print(f"  {str(k):<13}  V8:{v8['n']:>4}/${v8['pnl']:>11,.0f}  "
                  f"V9:{v9['n']:>4}/${v9['pnl']:>11,.0f}  "
                  f"Δn={v9['n']-v8['n']:>+4}  Δpnl=${v9['pnl']-v8['pnl']:>+12,.0f}")

    pprint_seg("YEAR", segs_v8[0], segs_v9[0])
    pprint_seg("COIN", segs_v8[1], segs_v9[1])
    pprint_seg("DIRECTION", segs_v8[2], segs_v9[2])
    pprint_seg("REASON", segs_v8[3], segs_v9[3])
    pprint_seg("COIN×DIR", segs_v8[5], segs_v9[5])

    # Year-Direction breakdown
    print("\n===== YEAR × DIRECTION (상세) =====")
    year_dir_v8 = segs_v8[6]
    year_dir_v9 = segs_v9[6]
    years = sorted(set(str(y).split("-")[0] for y in year_dir_v8))
    for y in years:
        for d in ("LONG", "SHORT"):
            k = f"{y}-{d}"
            # Handle y as str or int
            k_int = int(y) if y.isdigit() else y
            k_try = f"{k_int}-{d}"
            v8 = year_dir_v8.get(k_try, year_dir_v8.get(k, {"n": 0, "w": 0, "pnl": 0}))
            v9 = year_dir_v9.get(k_try, year_dir_v9.get(k, {"n": 0, "w": 0, "pnl": 0}))
            wr_v8 = round(v8['w']/max(v8['n'],1)*100, 1)
            wr_v9 = round(v9['w']/max(v9['n'],1)*100, 1)
            print(f"  {y} {d:<5}  V8: n={v8['n']:>4} WR={wr_v8:>4.1f}% PnL=${v8['pnl']:>12,.0f}   "
                  f"V9: n={v9['n']:>4} WR={wr_v9:>4.1f}% PnL=${v9['pnl']:>12,.0f}")

    # Filter rejection
    print("\n" + "=" * 80)
    print(f"🔍 1D 필터 차단 분석 (V8→V9 전환 시)")
    print("=" * 80)
    print(f"  V8 총 거래: {reject['v8_total']:>5}")
    print(f"  V9 총 거래: {reject['v9_total']:>5}")
    print(f"  차단된 거래: {reject['blocked_n']:>5}  ({reject['blocked_n']/reject['v8_total']*100:.1f}%)")
    print(f"    차단 승: {reject['blocked_w']}  패: {reject['blocked_l']}  승률: {reject['blocked_wr']}%")
    print(f"    차단 PnL: ${reject['blocked_pnl']:+,.0f} "
          f"({'이게 손실이면 V9 이득' if reject['blocked_pnl'] < 0 else '이득 놓친 거'})")
    print(f"  유지된 거래: {reject['kept_n']}  승률: {reject['kept_wr']}%  PnL: ${reject['kept_pnl']:+,.0f}")

    # Drawdown events
    print("\n" + "=" * 80)
    print(f"📉 주요 Drawdown 이벤트 (> 15%)")
    print("=" * 80)
    print(f"V8: {len(dd_v8)} 건")
    for e in dd_v8[:10]:
        print(f"  {e['peak_date']} → {e['recovery_date']}  DD {e['dd_pct']}% "
              f"(${e['peak_bal']:,.0f} → ${e['trough_bal']:,.0f})  "
              f"회복 {e['days_to_recover']}일")
    print(f"\nV9: {len(dd_v9)} 건")
    for e in dd_v9[:10]:
        print(f"  {e['peak_date']} → {e['recovery_date']}  DD {e['dd_pct']}% "
              f"(${e['peak_bal']:,.0f} → ${e['trough_bal']:,.0f})  "
              f"회복 {e['days_to_recover']}일")

    # Monthly balance (2025-2026)
    print("\n" + "=" * 80)
    print(f"📊 월별 잔고 추이 (2025 - 2026)")
    print("=" * 80)
    print(f"  {'Month':<10}{'V8 Bal':>15}{'V9 Bal':>18}{'V9/V8':>8}")
    months = sorted(set(monthly_v8.keys()) | set(monthly_v9.keys()))
    for m in months:
        if not m.startswith(("2025", "2026")):
            continue
        v8 = monthly_v8.get(m, 0)
        v9 = monthly_v9.get(m, 0)
        ratio = v9 / v8 if v8 > 0 else 0
        if v8 > 1e6:
            vs_str = f"${v8/1e6:,.1f}M"
        else:
            vs_str = f"${v8:,.0f}"
        if v9 > 1e6:
            v9_str = f"${v9/1e6:,.1f}M"
        else:
            v9_str = f"${v9:,.0f}"
        print(f"  {m:<10}{vs_str:>15}{v9_str:>18}{ratio:>7.1f}x")

    # Live period specific
    print("\n" + "=" * 80)
    print(f"🎯 실전 기간 04-09 ~ 04-18 trade-by-trade")
    print("=" * 80)
    print(f"V8 ({len(out['live_10d_v8_trades'])}거래):")
    for t in out["live_10d_v8_trades"]:
        dt = datetime.utcfromtimestamp(t["timestamp"]/1000).strftime("%m-%d %H:%M")
        print(f"  {dt}  {t['symbol'][:3]} {t['direction']:<5} "
              f"${t['entry_price']:>9.2f}→${t['exit_price']:>9.2f}  "
              f"${t['pnl']:>+6.2f}  {t['reason']}")
    print(f"\nV9 ({len(out['live_10d_v9_trades'])}거래):")
    for t in out["live_10d_v9_trades"]:
        dt = datetime.utcfromtimestamp(t["timestamp"]/1000).strftime("%m-%d %H:%M")
        print(f"  {dt}  {t['symbol'][:3]} {t['direction']:<5} "
              f"${t['entry_price']:>9.2f}→${t['exit_price']:>9.2f}  "
              f"${t['pnl']:>+6.2f}  {t['reason']}")


if __name__ == "__main__":
    main()
