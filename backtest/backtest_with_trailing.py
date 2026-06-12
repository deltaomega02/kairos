#!/usr/bin/env python3
"""
HERMES 백테스트 (트레일링 스탑 반영)
=====================================
실제 HERMES 시스템과 동일한 트레일링 로직으로 재검증.
- 활성화: 수익 1.5% 도달 시
- 트레일링 거리: 고점에서 0.9% (activation_pct × 0.6)
"""

import os
import sys
import json
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

# 트레일링 파라미터
TRAILING_ACTIVATION_PCT = 1.5  # 1.5% 수익 시 활성화
TRAIL_DISTANCE_PCT = TRAILING_ACTIVATION_PCT * 0.6  # 0.9%


def run_backtest_trailing(
    entry_df: pd.DataFrame,
    regime_series: pd.Series,
    params: Dict,
    symbol: str = "",
    block_sol_long: bool = True,
) -> Dict:
    """트레일링 스탑 포함 백테스트"""

    balance = INITIAL_BALANCE
    trades = []
    position = None
    daily_trades = {}

    closes = entry_df["close"].values
    highs = entry_df["high"].values
    lows = entry_df["low"].values
    opens = entry_df["open"].values
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
        open_p = opens[i]
        ts = timestamps[i]
        day_key = int(ts // 86400000)

        # --- 포지션 보유 중 ---
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

            # 봉 내에서 SL/TP/트레일링 시뮬레이션
            if d == "LONG":
                # 1) 먼저 기존 SL 체크 (봉 저가)
                if low <= sl:
                    exit_price = sl
                    reason = "SL"
                # 2) TP 체크 (봉 고가)
                elif high >= tp:
                    exit_price = tp
                    reason = "TP"
                else:
                    # 3) 트레일링 로직
                    # 고가가 peak 갱신
                    new_peak = max(peak, high)
                    pnl_pct_peak = (new_peak - ep) / ep * 100

                    # 트레일링 활성화 체크
                    if not trailing_active and pnl_pct_peak >= TRAILING_ACTIVATION_PCT:
                        trailing_active = True

                    if trailing_active:
                        # 새 트레일링 SL 계산
                        new_trail_sl = new_peak * (1 - TRAIL_DISTANCE_PCT / 100)
                        # 기존 SL보다 높을 때만 갱신
                        if new_trail_sl > sl:
                            sl = new_trail_sl

                        # 봉 저가가 새 트레일링 SL에 맞았는지
                        # (보수적: 고가가 peak 갱신 후 저가에 맞았다고 가정)
                        if low <= sl:
                            exit_price = sl
                            reason = "TRAILING"

                    position["peak_price"] = new_peak
                    position["trailing_active"] = trailing_active
                    position["sl_price"] = sl

            else:  # SHORT
                if high >= sl:
                    exit_price = sl
                    reason = "SL"
                elif low <= tp:
                    exit_price = tp
                    reason = "TP"
                else:
                    new_peak = min(peak, low)
                    pnl_pct_peak = (ep - new_peak) / ep * 100

                    if not trailing_active and pnl_pct_peak >= TRAILING_ACTIVATION_PCT:
                        trailing_active = True

                    if trailing_active:
                        new_trail_sl = new_peak * (1 + TRAIL_DISTANCE_PCT / 100)
                        if new_trail_sl < sl:
                            sl = new_trail_sl

                        if high >= sl:
                            exit_price = sl
                            reason = "TRAILING"

                    position["peak_price"] = new_peak
                    position["trailing_active"] = trailing_active
                    position["sl_price"] = sl

            # 청산 처리
            if exit_price is not None:
                if d == "LONG":
                    raw_pnl = (exit_price - ep) * qty
                else:
                    raw_pnl = (ep - exit_price) * qty
                fee = (ep * qty + exit_price * qty) * TAKER_FEE_PCT
                net_pnl = raw_pnl - fee
                pnl_pct = net_pnl / margin * 100 if margin > 0 else 0
                balance += net_pnl

                trades.append({
                    "symbol": symbol,
                    "direction": d,
                    "entry_price": ep,
                    "exit_price": exit_price,
                    "quantity": qty,
                    "pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "fee": fee,
                    "timestamp": ts,
                })
                position = None

            continue  # 포지션 있으면 신규 진입 안 함

        # --- 신규 진입 평가 ---
        if balance <= 10:
            continue

        dt = daily_trades.get(day_key, 0)
        if dt >= MAX_DAILY_TRADES:
            continue

        regime = regimes[i] if i < len(regimes) else "RANGING"

        # 시그널 평가 — 기존 함수 재사용
        row = {
            "close": close, "ema_fast": ema_f[i], "ema_slow": ema_s[i],
            "rsi": rsis[i], "atr_pct": atr_pcts[i], "volume_ratio": vol_ratios[i],
        }

        if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
            continue

        signal = evaluate_signal(regime, row, params)
        if signal is None:
            continue

        if block_sol_long and symbol == "SOLUSDT" and signal["direction"] == "LONG":
            continue

        entry_price = signal["entry_price"]
        sl_pct = signal["sl_pct"]
        tp_pct = signal["tp_pct"]

        risk_amt = balance * RISK_PER_TRADE
        sl_ratio = sl_pct / 100.0
        # 마진 분배 (수정된 로직 반영)
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

        margin = pos_val / lev
        position = {
            "direction": signal["direction"],
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "quantity": qty,
            "leverage": lev,
            "margin": margin,
            "peak_price": entry_price,
            "trailing_active": False,
        }
        daily_trades[day_key] = dt + 1

    # 미청산 강제 청산
    if position is not None and len(entry_df) > 0:
        ep = position["entry_price"]
        d = position["direction"]
        qty = position["quantity"]
        exit_p = closes[-1]
        if d == "LONG":
            raw = (exit_p - ep) * qty
        else:
            raw = (ep - exit_p) * qty
        fee = (ep * qty + exit_p * qty) * TAKER_FEE_PCT
        trades.append({
            "symbol": symbol, "direction": d,
            "entry_price": ep, "exit_price": exit_p, "quantity": qty,
            "pnl": raw - fee, "pnl_pct": 0,
            "reason": "FORCE_CLOSE", "fee": fee,
            "timestamp": timestamps[-1],
        })

    return {"trades": trades}


def run_config_trailing(name, params, data):
    """멀티코인 트레일링 백테스트"""
    all_trades = []

    for sym in SYMBOLS:
        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), params)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(params)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)
        r = run_backtest_trailing(entry_df, rm, params, symbol=sym)
        all_trades.extend(r["trades"])

    all_trades.sort(key=lambda t: t["timestamp"])

    # 통계
    total = len(all_trades)
    if total == 0:
        return None

    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)

    # 이유별 분류
    reason_stats = {}
    for t in all_trades:
        r = t["reason"]
        if r not in reason_stats:
            reason_stats[r] = {"count": 0, "pnl": 0, "wins": 0}
        reason_stats[r]["count"] += 1
        reason_stats[r]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            reason_stats[r]["wins"] += 1

    # 연도별
    yearly = {}
    for t in all_trades:
        y = datetime.fromtimestamp(t["timestamp"]/1000).strftime("%Y")
        if y not in yearly:
            yearly[y] = {"trades": 0, "wins": 0, "pnl": 0}
        yearly[y]["trades"] += 1
        if t["pnl"] > 0:
            yearly[y]["wins"] += 1
        yearly[y]["pnl"] += t["pnl"]

    # DD
    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in all_trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    long_t = [t for t in all_trades if t["direction"] == "LONG"]
    short_t = [t for t in all_trades if t["direction"] == "SHORT"]

    return {
        "name": name,
        "params": params,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(INITIAL_BALANCE + total_pnl, 2),
        "return_pct": round((INITIAL_BALANCE + total_pnl) / INITIAL_BALANCE * 100 - 100, 1),
        "max_dd": round(max_dd, 1),
        "total_fees": round(total_fees, 2),
        "reason_stats": reason_stats,
        "yearly": yearly,
        "long_trades": len(long_t),
        "long_wins": sum(1 for t in long_t if t["pnl"] > 0),
        "long_pnl": round(sum(t["pnl"] for t in long_t), 2),
        "short_trades": len(short_t),
        "short_wins": sum(1 for t in short_t if t["pnl"] > 0),
        "short_pnl": round(sum(t["pnl"] for t in short_t), 2),
    }


CONFIGS = {
    "1위_EMA5_18_ADX35_PB1.5": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
    },
    "2위_EMA5_18_ADX35_PB2.0": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 2.0, "adx_enter_trending": 35,
    },
    "3위_EMA5_15_ADX25_PB1.5": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 15, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 25,
    },
    "4위_EMA7_21_ADX25_PB1.0": {
        **DEFAULT_PARAMS,
        "ema_fast": 7, "ema_slow": 21, "sl_atr_mult": 2.0,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.0, "adx_enter_trending": 25,
    },
    "5위_기존시스템": {
        **DEFAULT_PARAMS,
        "ema_fast": 9, "ema_slow": 21, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 2.5, "entry_score_threshold": 60,
        "pullback_ema_dist_pct": 1.0, "adx_enter_trending": 25,
    },
}


def main():
    print("=" * 80)
    print("HERMES 백테스트 (트레일링 스탑 반영)")
    print(f"트레일링 활성화: +{TRAILING_ACTIVATION_PCT}%")
    print(f"트레일링 거리: -{TRAIL_DISTANCE_PCT}% (고점 기준)")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    data = {}
    for sym in SYMBOLS:
        for iv in ["60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
    print(f"  {len(data)}개 데이터셋 로드")

    print("\n[2] 트레일링 반영 백테스트 실행...")
    results = []
    for name, params in CONFIGS.items():
        print(f"\n  ▶ {name}")
        r = run_config_trailing(name, params, data)
        if r:
            results.append(r)
            print(f"    거래: {r['total_trades']}회 | 승률: {r['win_rate']}%")
            print(f"    수익: ${r['total_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
            print(f"    최대DD: {r['max_dd']}%")

            print(f"    청산 사유별:")
            for reason, rs in sorted(r["reason_stats"].items(),
                                      key=lambda x: x[1]["count"], reverse=True):
                wr = round(rs["wins"]/rs["count"]*100,1) if rs["count"]>0 else 0
                print(f"      {reason}: {rs['count']}회 승률{wr}% PnL ${rs['pnl']:+.2f}")

    # 요약 테이블
    print("\n" + "=" * 90)
    print("트레일링 반영 vs 기존 (4년)")
    print("=" * 90)
    print(f"{'설정':<35} {'거래':>5} {'승률':>7} {'수익률':>10} {'PnL':>10} {'DD':>7}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["return_pct"], reverse=True):
        print(f"{r['name']:<35} {r['total_trades']:>5} {r['win_rate']:>6.1f}% "
              f"{r['return_pct']:>+9.1f}% ${r['total_pnl']:>+9.2f} {r['max_dd']:>6.1f}%")

    # 연도별
    print("\n" + "=" * 90)
    print("1위 연도별 (트레일링)")
    print("=" * 90)
    best = next((r for r in results if "1위" in r["name"]), None)
    if best:
        for y, yd in sorted(best["yearly"].items()):
            wr = round(yd["wins"]/yd["trades"]*100,1) if yd["trades"]>0 else 0
            print(f"  {y}: {yd['trades']}거래 승률{wr}% PnL ${yd['pnl']:+.2f}")

    # 저장
    out_path = os.path.join(OUTPUT_DIR, "trailing_stop_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
