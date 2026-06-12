#!/usr/bin/env python3
"""
v6 최종 비교 — 현재 v5 vs 풀 패키지
=====================================
두 시나리오를 완전히 상세하게 비교.

현재 v5: 3코인, 2pos, trailing 1.2/0.1, lev5, TP4, ADX35, risk 1.5%
풀 패키지: 4코인 (XRP 양방), 3pos, trailing 1.2/0.1, lev7, TP6, ADX30, risk 1.5%

출력:
- 총 거래, 승률, 평균 수익/손실
- 연도별 + 월별 성과
- 방향별 (LONG vs SHORT)
- 코인별
- 청산 사유별 (SL/TP/TRAILING)
- 최대 연승/연패
- 손실 월 비율
- Calmar 비율
- 연평균 수익률
"""
import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict
from statistics import mean, median

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
SAFE_PARAMS = {
    **BASE_PARAMS,
    "tp_rr_ratio": 6.0,
    "adx_enter_trending": 30,
}
SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470


def load_data():
    data = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        for iv in ["15", "60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
        f = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        if os.path.exists(f):
            data[f"{sym}_funding"] = pd.read_csv(f)
    return data


def run_and_analyze(data, label, **kw):
    r = run_shared_backtest(
        data, kw.get("params", BASE_PARAMS), SEED,
        use_funding=True,
        trailing_activation=1.2, trailing_distance=0.1,
        block_sol_long=True,
        skip_years=(2023,),
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        slippage_pct=SLIP,
        max_simultaneous=kw.get("max_simultaneous", 2),
        max_leverage=kw.get("max_leverage", 5),
        enabled_symbols=kw.get("enabled_symbols", ["BTCUSDT","ETHUSDT","SOLUSDT"]),
        blocked_directions=kw.get("blocked_directions", {}),
    )

    trades = r.get("_trades", [])
    if not trades:
        return {"label": label, "empty": True}

    # ==== 통계 추출 ====
    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100

    total_pnl = sum(t["pnl"] for t in trades)
    total_fee = sum(t.get("fee", 0) for t in trades)
    avg_win = sum(t["pnl"] for t in wins) / win_count if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / loss_count if losses else 0
    max_win = max((t["pnl"] for t in wins), default=0)
    max_loss = min((t["pnl"] for t in losses), default=0)
    profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses else 0

    # 최대 연승/연패
    max_con_win = max_con_loss = 0
    cw = cl = 0
    for t in trades:
        if t["pnl"] > 0:
            cw += 1; cl = 0
            max_con_win = max(max_con_win, cw)
        else:
            cl += 1; cw = 0
            max_con_loss = max(max_con_loss, cl)

    # 연도별
    yearly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        y = datetime.utcfromtimestamp(t["timestamp"]/1000).year
        yearly[y]["trades"] += 1
        if t["pnl"] > 0: yearly[y]["wins"] += 1
        yearly[y]["pnl"] += t["pnl"]

    # 월별
    monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        m = datetime.utcfromtimestamp(t["timestamp"]/1000).strftime("%Y-%m")
        monthly[m]["trades"] += 1
        if t["pnl"] > 0: monthly[m]["wins"] += 1
        monthly[m]["pnl"] += t["pnl"]
    loss_months = sum(1 for m in monthly.values() if m["pnl"] < 0)
    total_months = len(monthly)

    # 방향별
    by_dir = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        d = t.get("direction", "?")
        by_dir[d]["trades"] += 1
        if t["pnl"] > 0: by_dir[d]["wins"] += 1
        by_dir[d]["pnl"] += t["pnl"]

    # 코인별
    by_coin = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        s = t.get("symbol", "?")
        by_coin[s]["trades"] += 1
        if t["pnl"] > 0: by_coin[s]["wins"] += 1
        by_coin[s]["pnl"] += t["pnl"]

    # 청산 사유
    by_reason = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        rn = t.get("reason", "?")
        by_reason[rn]["trades"] += 1
        if t["pnl"] > 0: by_reason[rn]["wins"] += 1
        by_reason[rn]["pnl"] += t["pnl"]

    # 기간 (추정)
    first_ts = trades[0]["timestamp"]
    last_ts = trades[-1]["timestamp"]
    days = (last_ts - first_ts) / 86400000
    years = days / 365.25

    # 연평균 수익률 (복리)
    final_balance = r["final_balance"]
    annual_return = (final_balance / SEED) ** (1/years) - 1 if years > 0 else 0

    return {
        "label": label,
        "total_trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 2),
        "final_balance": final_balance,
        "net_profit": r["net_profit"],
        "net_pct": r["net_pct"],
        "multiplier": round(final_balance / SEED, 2),
        "max_dd": r["max_dd"],
        "calmar": round(r["net_profit"] / r["max_dd"], 0) if r["max_dd"] > 0 else 0,
        "annual_return_pct": round(annual_return * 100, 1),
        "total_fee": round(total_fee, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_consec_win": max_con_win,
        "max_consec_loss": max_con_loss,
        "loss_months": loss_months,
        "total_months": total_months,
        "loss_month_ratio": round(loss_months / total_months * 100, 1),
        "years_tested": round(years, 2),
        "yearly": dict(yearly),
        "monthly": dict(monthly),
        "by_direction": dict(by_dir),
        "by_coin": dict(by_coin),
        "by_reason": dict(by_reason),
    }


def print_comparison(a, b):
    print("\n" + "=" * 110)
    print("종합 비교")
    print("=" * 110)
    metrics = [
        ("총 거래 수", f"{a['total_trades']:,}", f"{b['total_trades']:,}",
         lambda: f"{(b['total_trades']-a['total_trades'])/a['total_trades']*100:+.1f}%"),
        ("승률", f"{a['win_rate']}%", f"{b['win_rate']}%",
         lambda: f"{b['win_rate']-a['win_rate']:+.2f}%p"),
        ("최종 잔고", f"${a['final_balance']:,.0f}", f"${b['final_balance']:,.0f}",
         lambda: f"x{b['final_balance']/a['final_balance']:.1f}"),
        ("순이익", f"${a['net_profit']:+,.0f}", f"${b['net_profit']:+,.0f}",
         lambda: f"x{b['net_profit']/a['net_profit']:.1f}"),
        ("수익 배수", f"{a['multiplier']:.1f}x", f"{b['multiplier']:.1f}x",
         lambda: f"{b['multiplier']/a['multiplier']:.1f}x"),
        ("수익률", f"{a['net_pct']:+.0f}%", f"{b['net_pct']:+.0f}%", lambda: ""),
        ("연평균 수익률 (복리)", f"{a['annual_return_pct']:+.1f}%", f"{b['annual_return_pct']:+.1f}%",
         lambda: f"{b['annual_return_pct']-a['annual_return_pct']:+.1f}%p"),
        ("최대 DD", f"{a['max_dd']}%", f"{b['max_dd']}%",
         lambda: f"{b['max_dd']-a['max_dd']:+.1f}%p"),
        ("Calmar (수익/DD)", f"{a['calmar']:,}", f"{b['calmar']:,}",
         lambda: f"x{b['calmar']/a['calmar']:.1f}" if a['calmar'] > 0 else ""),
        ("평균 수익 (승)", f"${a['avg_win']}", f"${b['avg_win']}", lambda: ""),
        ("평균 손실 (패)", f"${a['avg_loss']}", f"${b['avg_loss']}", lambda: ""),
        ("최대 수익 1건", f"${a['max_win']}", f"${b['max_win']}", lambda: ""),
        ("최대 손실 1건", f"${a['max_loss']}", f"${b['max_loss']}", lambda: ""),
        ("Profit Factor", f"{a['profit_factor']}", f"{b['profit_factor']}", lambda: ""),
        ("최대 연승", f"{a['max_consec_win']}", f"{b['max_consec_win']}", lambda: ""),
        ("최대 연패", f"{a['max_consec_loss']}", f"{b['max_consec_loss']}", lambda: ""),
        ("손실 월 비율", f"{a['loss_month_ratio']}%", f"{b['loss_month_ratio']}%",
         lambda: f"{b['loss_month_ratio']-a['loss_month_ratio']:+.1f}%p"),
        ("총 수수료", f"${a['total_fee']:,}", f"${b['total_fee']:,}", lambda: ""),
    ]
    print(f"{'항목':<25} {'현재 v5':>20} {'풀 패키지':>20} {'차이':>20}")
    print("-" * 90)
    for name, va, vb, fn in metrics:
        diff = ""
        try: diff = fn()
        except: pass
        print(f"{name:<25} {va:>20} {vb:>20} {diff:>20}")


def print_yearly(a, b):
    print("\n" + "=" * 110)
    print("연도별 성과 비교 (2023 skip)")
    print("=" * 110)
    all_years = sorted(set(list(a['yearly'].keys()) + list(b['yearly'].keys())))
    print(f"{'연도':<8} {'v5 거래':>10} {'v5 승률':>8} {'v5 PnL':>14} {'풀 거래':>10} {'풀 승률':>8} {'풀 PnL':>14}")
    print("-" * 75)
    for y in all_years:
        ya = a['yearly'].get(y, {"trades":0,"wins":0,"pnl":0})
        yb = b['yearly'].get(y, {"trades":0,"wins":0,"pnl":0})
        wra = ya['wins']/ya['trades']*100 if ya['trades'] else 0
        wrb = yb['wins']/yb['trades']*100 if yb['trades'] else 0
        print(f"{y:<8} {ya['trades']:>10} {wra:>6.1f}% ${ya['pnl']:>+12,.0f} "
              f"{yb['trades']:>10} {wrb:>6.1f}% ${yb['pnl']:>+12,.0f}")


def print_by_direction(a, b):
    print("\n" + "=" * 110)
    print("방향별 성과 비교")
    print("=" * 110)
    print(f"{'방향':<10} {'v5 거래':>10} {'v5 승률':>8} {'v5 PnL':>14} {'풀 거래':>10} {'풀 승률':>8} {'풀 PnL':>14}")
    print("-" * 75)
    for d in ["LONG", "SHORT"]:
        da = a['by_direction'].get(d, {"trades":0,"wins":0,"pnl":0})
        db = b['by_direction'].get(d, {"trades":0,"wins":0,"pnl":0})
        wra = da['wins']/da['trades']*100 if da['trades'] else 0
        wrb = db['wins']/db['trades']*100 if db['trades'] else 0
        print(f"{d:<10} {da['trades']:>10} {wra:>6.1f}% ${da['pnl']:>+12,.0f} "
              f"{db['trades']:>10} {wrb:>6.1f}% ${db['pnl']:>+12,.0f}")


def print_by_coin(a, b):
    print("\n" + "=" * 110)
    print("코인별 성과 비교")
    print("=" * 110)
    print(f"{'코인':<12} {'v5 거래':>10} {'v5 승률':>8} {'v5 PnL':>14} {'풀 거래':>10} {'풀 승률':>8} {'풀 PnL':>14}")
    print("-" * 80)
    all_coins = sorted(set(list(a['by_coin'].keys()) + list(b['by_coin'].keys())))
    for c in all_coins:
        da = a['by_coin'].get(c, {"trades":0,"wins":0,"pnl":0})
        db = b['by_coin'].get(c, {"trades":0,"wins":0,"pnl":0})
        wra = da['wins']/da['trades']*100 if da['trades'] else 0
        wrb = db['wins']/db['trades']*100 if db['trades'] else 0
        print(f"{c:<12} {da['trades']:>10} {wra:>6.1f}% ${da['pnl']:>+12,.0f} "
              f"{db['trades']:>10} {wrb:>6.1f}% ${db['pnl']:>+12,.0f}")


def print_by_reason(a, b):
    print("\n" + "=" * 110)
    print("청산 사유별 성과 비교")
    print("=" * 110)
    print(f"{'사유':<12} {'v5 건수':>10} {'v5 PnL':>14} {'풀 건수':>10} {'풀 PnL':>14}")
    print("-" * 65)
    all_reasons = sorted(set(list(a['by_reason'].keys()) + list(b['by_reason'].keys())))
    for r in all_reasons:
        da = a['by_reason'].get(r, {"trades":0,"pnl":0})
        db = b['by_reason'].get(r, {"trades":0,"pnl":0})
        print(f"{r:<12} {da['trades']:>10} ${da['pnl']:>+12,.0f} "
              f"{db['trades']:>10} ${db['pnl']:>+12,.0f}")


def main():
    t0 = time.time()
    print("="*110)
    print("HERMES v6 — 현재 v5 vs 풀 패키지 최종 비교")
    print(f"시드 ${SEED:.0f}, 슬리피지 {SLIP}%, 수동 감독 (2023 skip)")
    print("="*110)

    print("\n[데이터 로드]")
    data = load_data()

    # Scenario A: 현재 v5
    print("\n[A] 현재 v5 실행 중...")
    a = run_and_analyze(data, "현재 v5",
                        max_simultaneous=2, max_leverage=5,
                        enabled_symbols=["BTCUSDT","ETHUSDT","SOLUSDT"],
                        params=BASE_PARAMS)

    # Scenario B: 풀 패키지
    print("[B] 풀 패키지 실행 중...")
    b = run_and_analyze(data, "풀 패키지",
                        max_simultaneous=3, max_leverage=7,
                        enabled_symbols=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"],
                        params=SAFE_PARAMS)

    # 출력
    print_comparison(a, b)
    print_yearly(a, b)
    print_by_direction(a, b)
    print_by_coin(a, b)
    print_by_reason(a, b)

    elapsed = time.time() - t0
    print(f"\n{'='*110}\n완료 | 소요 {elapsed:.0f}초")

    out = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "slippage": SLIP,
        "scenario_A_current_v5": a,
        "scenario_B_full_package": b,
    }
    out_path = os.path.join(RESULTS_DIR, "v6_final_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
