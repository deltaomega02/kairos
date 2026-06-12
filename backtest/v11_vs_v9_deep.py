#!/usr/bin/env python3
"""
V9 vs V11 초심층 비교 — 모든 각도에서 analysis
"""
import os, sys, json, time, random
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, SEED, SLIP, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v11"

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


def filter_d(data, s, e):
    s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
    e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
    out = {}
    for k,v in data.items():
        if isinstance(v, pd.DataFrame) and 'timestamp' in v.columns:
            m = (v['timestamp']>=s_ts) & (v['timestamp']<e_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


def run(data, mode_kw, skip=()):
    base = dict(trailing_activation=1.2, trailing_distance=0.1,
                daily_cost_usd=DAILY_COST, slippage_pct=SLIP,
                max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                enabled_symbols=SYMBOLS, block_sol_long=True, skip_years=skip)
    return run_shared_backtest_v11(data, V8_PARAMS, SEED, **base, **mode_kw)


V9_KW = dict(d1_filter_enable=True, d1_ema_period=10, d1_mode="direction")
V11_KW = dict(d1_filter_enable=True, d1_ema_period=2, d1_mode="price_above_ema")


def equity_curve(trades):
    """누적 잔고 timeline"""
    sorted_t = sorted(trades, key=lambda t: t["timestamp"])
    if not sorted_t:
        return []
    bal = SEED
    peak = SEED
    curve = []
    for t in sorted_t:
        bal += t["pnl"]
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        curve.append({"ts": t["timestamp"], "bal": bal, "peak": peak, "dd": dd})
    return curve


def monthly_bal(curve):
    by_month = {}
    for pt in curve:
        dt = datetime.utcfromtimestamp(pt["ts"]/1000)
        m = f"{dt.year:04d}-{dt.month:02d}"
        by_month[m] = pt["bal"]
    return by_month


def segments(trades):
    by_year = defaultdict(lambda: {"n":0,"w":0,"pnl":0,"long_n":0,"long_w":0,"short_n":0,"short_w":0})
    by_coin = defaultdict(lambda: {"n":0,"w":0,"pnl":0})
    by_dir = defaultdict(lambda: {"n":0,"w":0,"pnl":0})
    by_reason = defaultdict(lambda: {"n":0,"pnl":0})
    by_month = defaultdict(lambda: {"n":0,"w":0,"pnl":0})
    by_coin_dir = defaultdict(lambda: {"n":0,"w":0,"pnl":0})
    by_hour = defaultdict(lambda: {"n":0,"w":0,"pnl":0})

    for t in trades:
        y = t["year"]
        c = t["symbol"]
        d = t["direction"]
        r = t["reason"]
        dt = datetime.utcfromtimestamp(t["timestamp"]/1000)
        m = f"{dt.year:04d}-{dt.month:02d}"
        h = dt.hour
        pnl = t["pnl"]
        w = 1 if pnl > 0 else 0

        by_year[y]["n"]+=1; by_year[y]["w"]+=w; by_year[y]["pnl"]+=pnl
        if d == "LONG":
            by_year[y]["long_n"]+=1; by_year[y]["long_w"]+=w
        else:
            by_year[y]["short_n"]+=1; by_year[y]["short_w"]+=w
        by_coin[c]["n"]+=1; by_coin[c]["w"]+=w; by_coin[c]["pnl"]+=pnl
        by_dir[d]["n"]+=1; by_dir[d]["w"]+=w; by_dir[d]["pnl"]+=pnl
        by_reason[r]["n"]+=1; by_reason[r]["pnl"]+=pnl
        by_month[m]["n"]+=1; by_month[m]["w"]+=w; by_month[m]["pnl"]+=pnl
        by_coin_dir[f"{c[:3]}-{d}"]["n"]+=1; by_coin_dir[f"{c[:3]}-{d}"]["w"]+=w
        by_coin_dir[f"{c[:3]}-{d}"]["pnl"]+=pnl
        by_hour[h]["n"]+=1; by_hour[h]["w"]+=w; by_hour[h]["pnl"]+=pnl

    return dict(by_year), dict(by_coin), dict(by_dir), dict(by_reason), dict(by_month), dict(by_coin_dir), dict(by_hour)


def find_diff(trades_a, trades_b):
    """B에만 있는 거래, A에만 있는 거래 (symbol + direction + 비슷한 시간 매칭)"""
    def key(t):
        return (t["symbol"], t["direction"], t["timestamp"] // (3600 * 1000))
    a_keys = set(key(t) for t in trades_a)
    b_keys = set(key(t) for t in trades_b)
    only_a = [t for t in trades_a if key(t) not in b_keys]
    only_b = [t for t in trades_b if key(t) not in a_keys]
    return only_a, only_b


def dd_events(curve, threshold=10):
    events = []
    peak_bal = SEED
    peak_ts = curve[0]["ts"] if curve else 0
    max_dd = 0
    in_dd = False
    for pt in curve:
        bal = pt["bal"]
        ts = pt["ts"]
        if bal > peak_bal:
            if in_dd and max_dd > threshold:
                dt_p = datetime.utcfromtimestamp(peak_ts/1000).strftime("%Y-%m-%d")
                dt_r = datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
                events.append({
                    "peak_date": dt_p, "recovery_date": dt_r,
                    "dd_pct": round(max_dd, 1),
                    "peak_bal": round(peak_bal, 2),
                    "trough_bal": round(peak_bal * (1 - max_dd/100), 2),
                    "days": (datetime.utcfromtimestamp(ts/1000) - datetime.utcfromtimestamp(peak_ts/1000)).days,
                })
            peak_bal = bal
            peak_ts = ts
            max_dd = 0
            in_dd = False
        else:
            dd = (peak_bal - bal) / peak_bal * 100
            if dd > max_dd:
                max_dd = dd
                in_dd = True
    return events


def main():
    print("v9 vs v11 deep compare\n", flush=True)
    data = _load_data()

    print("[1] Full 6y run both...", flush=True)
    r9 = run(data, V9_KW, skip=(2023,))
    r11 = run(data, V11_KW, skip=(2023,))
    print(f"  v9:  ${r9['net_profit']:,.0f}  DD {r9['max_dd']}%  trades {r9['total_trades']}  WR {r9['win_rate']}%")
    print(f"  v11: ${r11['net_profit']:,.0f}  DD {r11['max_dd']}%  trades {r11['total_trades']}  WR {r11['win_rate']}%")

    t9 = r9["_trades"]
    t11 = r11["_trades"]

    print("\n[2] Equity curves + monthly balance...", flush=True)
    c9 = equity_curve(t9)
    c11 = equity_curve(t11)
    m9 = monthly_bal(c9)
    m11 = monthly_bal(c11)

    print("\n[3] Segmentation...", flush=True)
    s9 = segments(t9)
    s11 = segments(t11)

    print("[4] Trade diff analysis...", flush=True)
    only9, only11 = find_diff(t9, t11)
    only9_pnl = sum(t["pnl"] for t in only9)
    only11_pnl = sum(t["pnl"] for t in only11)
    only9_w = sum(1 for t in only9 if t["pnl"] > 0)
    only11_w = sum(1 for t in only11 if t["pnl"] > 0)
    print(f"  V9만 잡은 거래: {len(only9)} (승률 {only9_w/max(len(only9),1)*100:.1f}%, PnL ${only9_pnl:,.0f})")
    print(f"  V11만 잡은 거래: {len(only11)} (승률 {only11_w/max(len(only11),1)*100:.1f}%, PnL ${only11_pnl:,.0f})")

    print("\n[5] Drawdown events...", flush=True)
    dd9 = dd_events(c9, threshold=12)
    dd11 = dd_events(c11, threshold=12)

    print("[6] Live 10d trades...", flush=True)
    r9_live = run(filter_d(data, "2026-04-09", "2026-04-20"), V9_KW, skip=())
    r11_live = run(filter_d(data, "2026-04-09", "2026-04-20"), V11_KW, skip=())

    # ========= PRINT =========
    print("\n" + "="*130)
    print(" 📊 YEAR-BY-YEAR")
    print("="*130)
    print(f"  {'Year':<6}  {'V9 n':>5} {'V9 WR':>6} {'V9 PnL':>14}   {'V11 n':>5} {'V11 WR':>7} {'V11 PnL':>14}   {'V11/V9':>8}")
    years = sorted(set(list(s9[0].keys()) + list(s11[0].keys())))
    for y in years:
        a = s9[0].get(y, {"n":0,"w":0,"pnl":0})
        b = s11[0].get(y, {"n":0,"w":0,"pnl":0})
        wr_a = a["w"]/max(a["n"],1)*100
        wr_b = b["w"]/max(b["n"],1)*100
        ratio = b["pnl"]/a["pnl"] if a["pnl"] > 0 else 0
        print(f"  {y:<6}  {a['n']:>5} {wr_a:>5.1f}% ${a['pnl']:>13,.0f}   {b['n']:>5} {wr_b:>6.1f}% ${b['pnl']:>13,.0f}   {ratio:>7.1f}x")

    print("\n" + "="*130)
    print(" 📊 YEAR × DIRECTION")
    print("="*130)
    for y in years:
        a = s9[0].get(y, {"long_n":0,"long_w":0,"short_n":0,"short_w":0})
        b = s11[0].get(y, {"long_n":0,"long_w":0,"short_n":0,"short_w":0})
        l_a = a["long_w"]/max(a["long_n"],1)*100
        l_b = b["long_w"]/max(b["long_n"],1)*100
        s_a = a["short_w"]/max(a["short_n"],1)*100
        s_b = b["short_w"]/max(b["short_n"],1)*100
        print(f"  {y} LONG   V9: {a['long_n']:>4} ({l_a:>4.1f}%)   V11: {b['long_n']:>4} ({l_b:>4.1f}%)    Δ승률 {l_b-l_a:+.1f}%p")
        print(f"  {y} SHORT  V9: {a['short_n']:>4} ({s_a:>4.1f}%)   V11: {b['short_n']:>4} ({s_b:>4.1f}%)    Δ승률 {s_b-s_a:+.1f}%p")

    print("\n" + "="*130)
    print(" 📊 COIN × DIRECTION")
    print("="*130)
    for k in sorted(s9[5].keys()):
        a = s9[5].get(k, {"n":0,"w":0,"pnl":0})
        b = s11[5].get(k, {"n":0,"w":0,"pnl":0})
        wr_a = a["w"]/max(a["n"],1)*100
        wr_b = b["w"]/max(b["n"],1)*100
        print(f"  {k:<12}  V9:{a['n']:>4}  WR {wr_a:>4.1f}%  ${a['pnl']:>12,.0f}    V11:{b['n']:>4}  WR {wr_b:>4.1f}%  ${b['pnl']:>12,.0f}")

    print("\n" + "="*130)
    print(" 📊 REASON (청산 사유)")
    print("="*130)
    for k in ["SL", "TP", "TRAILING", "END"]:
        a = s9[3].get(k, {"n":0,"pnl":0})
        b = s11[3].get(k, {"n":0,"pnl":0})
        print(f"  {k:<10}  V9: {a['n']:>5} trades  ${a['pnl']:>14,.0f}    V11: {b['n']:>5} trades  ${b['pnl']:>14,.0f}")

    print("\n" + "="*130)
    print(" 🔍 차단/추가 거래 분석 (V9와 V11이 뽑는 거래 차이)")
    print("="*130)
    print(f"  V9만 잡은 거래 ({len(only9)}건): 승률 {only9_w/max(len(only9),1)*100:.1f}%, PnL ${only9_pnl:+,.0f}")
    print(f"  V11만 잡은 거래 ({len(only11)}건): 승률 {only11_w/max(len(only11),1)*100:.1f}%, PnL ${only11_pnl:+,.0f}")
    print(f"  공통 거래: {len(t9) - len(only9)}건")

    print("\n" + "="*130)
    print(" 📉 드로다운 이벤트 (> 12%)")
    print("="*130)
    print(f"V9: {len(dd9)}건")
    for e in dd9[:8]:
        print(f"  {e['peak_date']} → {e['recovery_date']}  DD {e['dd_pct']}% (${e['peak_bal']:,.0f} → ${e['trough_bal']:,.0f})  회복 {e['days']}일")
    print(f"\nV11: {len(dd11)}건")
    for e in dd11[:8]:
        print(f"  {e['peak_date']} → {e['recovery_date']}  DD {e['dd_pct']}% (${e['peak_bal']:,.0f} → ${e['trough_bal']:,.0f})  회복 {e['days']}일")

    print("\n" + "="*130)
    print(" 💰 월별 잔고 추이")
    print("="*130)
    print(f"  {'Month':<10} {'V9 Bal':>15} {'V11 Bal':>18} {'V11/V9':>10}")
    months = sorted(set(list(m9.keys()) + list(m11.keys())))
    for mm in months[::3]:  # 3개월마다
        a = m9.get(mm, 0)
        b = m11.get(mm, 0)
        ratio = b/a if a > 0 else 0
        def _fmt(v):
            if v >= 1e9: return f"${v/1e9:,.2f}B"
            if v >= 1e6: return f"${v/1e6:,.1f}M"
            if v >= 1e3: return f"${v/1e3:,.1f}k"
            return f"${v:,.0f}"
        print(f"  {mm:<10} {_fmt(a):>15} {_fmt(b):>18} {ratio:>8.1f}x")

    print("\n" + "="*130)
    print(" 🎯 실전 기간 04-09~04-18 (v9 live vs v11 backtest)")
    print("="*130)
    print(f"\nV9 ({r9_live['total_trades']}거래, PnL ${r9_live['net_profit']:+.1f}):")
    for t in r9_live.get("_trades", []):
        dt = datetime.utcfromtimestamp(t['timestamp']/1000).strftime("%m-%d %H:%M")
        print(f"  {dt}  {t['symbol'][:3]} {t['direction']:<5}  ${t['entry_price']:>9.2f}→${t['exit_price']:>9.2f}  ${t['pnl']:>+6.2f}  {t['reason']}")
    print(f"\nV11 ({r11_live['total_trades']}거래, PnL ${r11_live['net_profit']:+.1f}):")
    for t in r11_live.get("_trades", []):
        dt = datetime.utcfromtimestamp(t['timestamp']/1000).strftime("%m-%d %H:%M")
        print(f"  {dt}  {t['symbol'][:3]} {t['direction']:<5}  ${t['entry_price']:>9.2f}→${t['exit_price']:>9.2f}  ${t['pnl']:>+6.2f}  {t['reason']}")

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "v9_summary": {"profit": r9["net_profit"], "dd": r9["max_dd"], "trades": r9["total_trades"], "wr": r9["win_rate"]},
        "v11_summary": {"profit": r11["net_profit"], "dd": r11["max_dd"], "trades": r11["total_trades"], "wr": r11["win_rate"]},
        "monthly_v9": m9, "monthly_v11": m11,
        "segments_v9": {"by_year": s9[0], "by_coin": s9[1], "by_direction": s9[2],
                         "by_reason": s9[3], "by_coin_dir": s9[5], "by_hour": s9[6]},
        "segments_v11": {"by_year": s11[0], "by_coin": s11[1], "by_direction": s11[2],
                          "by_reason": s11[3], "by_coin_dir": s11[5], "by_hour": s11[6]},
        "only9_n": len(only9), "only9_pnl": only9_pnl, "only9_wr": only9_w/max(len(only9),1)*100,
        "only11_n": len(only11), "only11_pnl": only11_pnl, "only11_wr": only11_w/max(len(only11),1)*100,
        "dd_events_v9": dd9, "dd_events_v11": dd11,
        "live_v9_trades": r9_live.get("_trades", []),
        "live_v11_trades": r11_live.get("_trades", []),
    }
    with open(os.path.join(RESULTS_DIR, "v9_vs_v11_deep.json"), "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/v9_vs_v11_deep.json", flush=True)


if __name__ == "__main__":
    main()
