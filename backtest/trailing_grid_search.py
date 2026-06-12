#!/usr/bin/env python3
"""
HERMES 트레일링 스탑 그리드 서치
=================================
1위 파라미터에 트레일링 활성화/거리 조합을 대량 테스트.
"""

import os
import sys
import json
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry, evaluate_signal,
    fee_adjusted_sl_tp, TAKER_FEE_PCT, DEFAULT_PARAMS,
    INITIAL_BALANCE, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_DAILY_TRADES, MAX_SIMULTANEOUS,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = "~/Projects/HERMES_백테스팅_v2"

# 1위 파라미터 고정
BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}

# 트레일링 파라미터 그리드
TRAILING_ACTIVATIONS = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
TRAILING_DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # 고점에서 빠지는 %


def run_backtest_with_trailing(
    entry_df: pd.DataFrame,
    regime_series: pd.Series,
    params: Dict,
    activation_pct: float,
    distance_pct: float,
    symbol: str = "",
    use_trailing: bool = True,
) -> Dict:
    """트레일링 파라미터 커스터마이즈 백테스트"""

    balance = INITIAL_BALANCE
    trades = []
    position = None
    daily_trades = {}

    closes = entry_df["close"].values
    highs = entry_df["high"].values
    lows = entry_df["low"].values
    ema_f = entry_df["ema_fast"].values
    ema_s = entry_df["ema_slow"].values
    rsis = entry_df["rsi"].values
    atr_pcts = entry_df["atr_pct"].values
    vol_ratios = entry_df["volume_ratio"].values
    timestamps = entry_df["timestamp"].values
    regimes = regime_series.values if hasattr(regime_series, 'values') else regime_series

    n = len(entry_df)

    for i in range(50, n):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        ts = timestamps[i]
        day_key = int(ts // 86400000)

        if position is not None:
            ep = position["entry_price"]
            d = position["direction"]
            sl = position["sl_price"]
            tp = position["tp_price"]
            qty = position["quantity"]
            margin = position["margin"]
            peak = position["peak_price"]
            trailing_active = position["trailing_active"]

            exit_price = None
            reason = None

            if d == "LONG":
                # TP 먼저 체크 (봉이 bullish하면 high 먼저)
                if high >= tp:
                    exit_price = tp
                    reason = "TP"
                elif low <= sl:
                    exit_price = sl
                    reason = "SL"
                elif use_trailing:
                    new_peak = max(peak, high)
                    pnl_pct_peak = (new_peak - ep) / ep * 100

                    if not trailing_active and pnl_pct_peak >= activation_pct:
                        trailing_active = True

                    if trailing_active:
                        new_trail_sl = new_peak * (1 - distance_pct / 100)
                        if new_trail_sl > sl:
                            sl = new_trail_sl

                        if low <= sl:
                            exit_price = sl
                            reason = "TRAILING"

                    position["peak_price"] = new_peak
                    position["trailing_active"] = trailing_active
                    position["sl_price"] = sl
            else:
                if low <= tp:
                    exit_price = tp
                    reason = "TP"
                elif high >= sl:
                    exit_price = sl
                    reason = "SL"
                elif use_trailing:
                    new_peak = min(peak, low)
                    pnl_pct_peak = (ep - new_peak) / ep * 100

                    if not trailing_active and pnl_pct_peak >= activation_pct:
                        trailing_active = True

                    if trailing_active:
                        new_trail_sl = new_peak * (1 + distance_pct / 100)
                        if new_trail_sl < sl:
                            sl = new_trail_sl

                        if high >= sl:
                            exit_price = sl
                            reason = "TRAILING"

                    position["peak_price"] = new_peak
                    position["trailing_active"] = trailing_active
                    position["sl_price"] = sl

            if exit_price is not None:
                if d == "LONG":
                    raw_pnl = (exit_price - ep) * qty
                else:
                    raw_pnl = (ep - exit_price) * qty
                fee = (ep * qty + exit_price * qty) * TAKER_FEE_PCT
                net_pnl = raw_pnl - fee
                balance += net_pnl

                trades.append({
                    "pnl": net_pnl, "reason": reason,
                    "direction": d, "timestamp": ts,
                })
                position = None
            continue

        if balance <= 10:
            continue

        dt = daily_trades.get(day_key, 0)
        if dt >= MAX_DAILY_TRADES:
            continue

        regime = regimes[i] if i < len(regimes) else "RANGING"
        row = {
            "close": close, "ema_fast": ema_f[i], "ema_slow": ema_s[i],
            "rsi": rsis[i], "atr_pct": atr_pcts[i], "volume_ratio": vol_ratios[i],
        }
        if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
            continue

        signal = evaluate_signal(regime, row, params)
        if signal is None:
            continue

        if symbol == "SOLUSDT" and signal["direction"] == "LONG":
            continue

        entry_price = signal["entry_price"]
        sl_pct = signal["sl_pct"]
        tp_pct = signal["tp_pct"]

        risk_amt = balance * RISK_PER_TRADE
        sl_ratio = sl_pct / 100.0
        avail_margin = balance * MARGIN_USAGE / MAX_SIMULTANEOUS
        ideal = risk_amt / sl_ratio
        lev = min(int(ideal / avail_margin) if avail_margin > 0 else 1, MAX_LEVERAGE)
        lev = max(lev, MIN_LEVERAGE)
        pos_val = avail_margin * lev
        qty = pos_val / entry_price
        if pos_val < 5:
            continue

        if signal["direction"] == "LONG":
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
        else:
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)

        position = {
            "direction": signal["direction"],
            "entry_price": entry_price,
            "sl_price": sl_price, "tp_price": tp_price,
            "quantity": qty, "leverage": lev,
            "margin": pos_val / lev,
            "peak_price": entry_price,
            "trailing_active": False,
        }
        daily_trades[day_key] = dt + 1

    # 통계
    total = len(trades)
    if total == 0:
        return None

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)

    reasons = {}
    for t in trades:
        r = t["reason"]
        reasons[r] = reasons.get(r, 0) + 1

    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "total": total, "wins": wins,
        "win_rate": round(wins/total*100, 1),
        "pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl/INITIAL_BALANCE*100, 1),
        "max_dd": round(max_dd, 1),
        "reasons": reasons,
    }


def run_config(activation, distance, data, use_trailing=True):
    """멀티코인"""
    all_trades_pnl = []
    all_reasons = {}
    all_total = 0
    all_wins = 0
    running_balance = INITIAL_BALANCE
    running_peak = INITIAL_BALANCE
    max_dd_overall = 0

    for sym in SYMBOLS:
        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), BEST_PARAMS)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(BEST_PARAMS)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)
        r = run_backtest_with_trailing(
            entry_df, rm, BEST_PARAMS, activation, distance,
            symbol=sym, use_trailing=use_trailing,
        )
        if r is None:
            continue
        all_total += r["total"]
        all_wins += r["wins"]
        for rn, cnt in r["reasons"].items():
            all_reasons[rn] = all_reasons.get(rn, 0) + cnt

    # 합산 PnL을 위해 다시 계산 (DD 포함)
    all_trades = []
    for sym in SYMBOLS:
        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), BEST_PARAMS)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(BEST_PARAMS)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)
        r = run_backtest_with_trailing(
            entry_df, rm, BEST_PARAMS, activation, distance,
            symbol=sym, use_trailing=use_trailing,
        )

    if all_total == 0:
        return None

    # 간단한 메트릭으로 반환 (DD는 코인별 합산으로 근사)
    return {
        "activation": activation,
        "distance": distance,
        "use_trailing": use_trailing,
        "total_trades": all_total,
        "win_rate": round(all_wins/all_total*100, 1),
        "reasons": all_reasons,
    }


def run_fast_multi(activation, distance, data, use_trailing=True):
    """고속 멀티코인"""
    all_trades = []

    for sym in SYMBOLS:
        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), BEST_PARAMS)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(BEST_PARAMS)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)

        # 고속 백테스트 — closures에 저장된 함수 사용
        r = run_backtest_with_trailing(
            entry_df, rm, BEST_PARAMS, activation, distance,
            symbol=sym, use_trailing=use_trailing,
        )
        if r:
            # 간단 합산
            all_trades.append(r)

    if not all_trades:
        return None

    total = sum(r["total"] for r in all_trades)
    wins = sum(r["wins"] for r in all_trades)
    pnl = sum(r["pnl"] for r in all_trades)
    reasons = {}
    for r in all_trades:
        for k, v in r["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    max_dd = max(r["max_dd"] for r in all_trades)

    return {
        "activation": activation,
        "distance": distance,
        "use_trailing": use_trailing,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins/total*100, 1) if total > 0 else 0,
        "total_pnl": round(pnl, 2),
        "return_pct": round(pnl/INITIAL_BALANCE*100, 1),
        "max_dd": round(max_dd, 1),
        "reasons": reasons,
    }


def main():
    print("=" * 80)
    print("HERMES 트레일링 스탑 그리드 서치")
    print("1위 파라미터 기준: EMA5/18 SL1.5 TP4.0 S40 PB1.5 ADX35")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    data = {}
    for sym in SYMBOLS:
        for iv in ["60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
    print(f"  {len(data)}개 데이터셋")

    print("\n[2] 베이스라인 (트레일링 없음)...")
    baseline = run_fast_multi(0, 0, data, use_trailing=False)
    if baseline:
        print(f"  거래: {baseline['total_trades']} | 승률: {baseline['win_rate']}%")
        print(f"  수익: ${baseline['total_pnl']:+.2f} ({baseline['return_pct']:+.1f}%)")
        print(f"  DD: {baseline['max_dd']}%")

    print("\n[3] 트레일링 조합 스윕...")
    combos = list(itertools.product(TRAILING_ACTIVATIONS, TRAILING_DISTANCES))
    print(f"  {len(combos)}개 조합")

    results = [baseline] if baseline else []
    for i, (act, dist) in enumerate(combos):
        if dist >= act:  # 거리가 활성화보다 크면 의미 없음
            continue
        r = run_fast_multi(act, dist, data, use_trailing=True)
        if r:
            results.append(r)
            if (i+1) % 10 == 0 or i == len(combos)-1:
                best = max(results, key=lambda x: x["return_pct"])
                print(f"  {i+1}/{len(combos)} | 현재 최고: {best['return_pct']:+.1f}% "
                      f"(Act={best.get('activation',0)}, Dist={best.get('distance',0)})")

    # 정렬 출력
    results.sort(key=lambda x: x["return_pct"], reverse=True)

    print("\n" + "=" * 100)
    print("트레일링 조합 Top 20 (수익률 순)")
    print("=" * 100)
    print(f"{'#':>3} {'활성화':>8} {'거리':>6} {'거래':>6} {'승률':>7} {'수익률':>10} {'PnL':>10} {'DD':>7}")
    print("-" * 100)
    for i, r in enumerate(results[:20]):
        act_str = f"{r.get('activation', 0)}" if r.get('use_trailing') else "없음"
        dist_str = f"{r.get('distance', 0)}" if r.get('use_trailing') else "-"
        print(f"{i+1:>3} {act_str:>8} {dist_str:>6} {r['total_trades']:>6} "
              f"{r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% "
              f"${r['total_pnl']:>+9.2f} {r['max_dd']:>6.1f}%")

    # 리스크 조정 순위
    print("\n" + "-" * 100)
    print("리스크 조정 순위 (수익률 / DD)")
    print("-" * 100)
    risk_adj = sorted(results, key=lambda x: x["return_pct"] / max(x["max_dd"], 1), reverse=True)
    for i, r in enumerate(risk_adj[:10]):
        act_str = f"{r.get('activation', 0)}" if r.get('use_trailing') else "없음"
        dist_str = f"{r.get('distance', 0)}" if r.get('use_trailing') else "-"
        ratio = r["return_pct"] / max(r["max_dd"], 1)
        print(f"{i+1:>3} Act={act_str:>5} Dist={dist_str:>5} | "
              f"{r['total_trades']}거래 {r['win_rate']}% | "
              f"{r['return_pct']:+.1f}% DD{r['max_dd']}% ratio={ratio:.2f}")

    # 저장
    out_path = os.path.join(OUTPUT_DIR, "trailing_grid_search.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
