#!/usr/bin/env python3
"""
HERMES 전수 조사 그리드 서치
==============================
모든 파라미터 조합을 4년 장기 데이터에서 테스트.
최적화: EMA/ADX별 지표 사전 계산 → SL/TP/Score/풀백 스윕은 재계산 없이 실행.
"""

import os
import sys
import time
import json
import itertools
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import (
    fetch_kline, calc_ema, calc_sma, calc_rsi, calc_atr, calc_adx, calc_bb, calc_macd,
    compute_regime_indicators, BacktestRegimeEngine, align_regime_to_entry,
    fee_adjusted_sl_tp, TAKER_FEE_PCT,
    DATA_DIR, RESULTS_DIR,
)

# ================================================================
# 설정
# ================================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LONG_START = "2022-01-01"
LONG_END = "2026-04-08"
INITIAL_BALANCE = 300.0
RISK_PER_TRADE = 0.015
MARGIN_USAGE = 0.80
MAX_LEVERAGE = 5
MIN_LEVERAGE = 1
MAX_DAILY_TRADES = 5

# 전수 조사 파라미터 그리드
GRID = {
    "ema_fast": [5, 7, 9, 12],
    "ema_slow": [15, 18, 21, 26, 30],
    "sl_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "tp_rr_ratio": [2.0, 2.5, 3.0, 4.0, 5.0],
    "entry_score_threshold": [30, 40, 50, 60],
    "pullback_ema_dist_pct": [0.5, 0.8, 1.0, 1.5, 2.0],
    "adx_enter_trending": [20, 25, 30, 35],
    "rsi_oversold": [30, 35, 40],
    "rsi_overbought": [60, 65, 70],
}

# 고정 파라미터
FIXED = {
    "rsi_period": 14,
    "adx_exit_trending": 20,
    "atr_high_vol_percentile": 85,
    "regime_debounce_bars": 1,
    "orderbook_imbalance_min": 0.55,
    "funding_bias_threshold": 0.0005,
}


# ================================================================
# 데이터 로드
# ================================================================

def load_long_data():
    """장기 데이터 로드 (캐시)"""
    data = {}
    for symbol in SYMBOLS:
        for interval in ["60", "240"]:
            cache = os.path.join(DATA_DIR, f"{symbol}_{interval}_long.csv")
            if os.path.exists(cache):
                df = pd.read_csv(cache)
                data[f"{symbol}_{interval}"] = df
                print(f"  ✓ {symbol} {interval}: {len(df)}개")
            else:
                print(f"  ✗ {symbol} {interval}: 없음 — 먼저 longterm_validation.py 실행")
                return None
    return data


# ================================================================
# 사전 계산: EMA 조합별 진입 지표
# ================================================================

def precompute_entry_indicators(df_raw: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    """EMA 조합별 진입 지표 사전 계산"""
    df = df_raw.copy()
    df["ema_fast"] = calc_ema(df["close"], ema_fast)
    df["ema_slow"] = calc_ema(df["close"], ema_slow)
    df["rsi"] = calc_rsi(df["close"], 14)
    atr = calc_atr(df["high"], df["low"], df["close"])
    df["atr"] = atr
    df["atr_pct"] = (atr / df["close"]) * 100
    vol_sma = calc_sma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / (vol_sma + 1e-10)
    return df


# ================================================================
# 사전 계산: ADX 조합별 레짐
# ================================================================

def precompute_regimes(regime_df_raw: pd.DataFrame, adx_enter: int) -> pd.DataFrame:
    """ADX 조합별 레짐 판독"""
    regime_df = compute_regime_indicators(regime_df_raw.copy())
    params = {**FIXED, "adx_enter_trending": adx_enter}
    re = BacktestRegimeEngine(params)
    regimes = []
    for _, row in regime_df.iterrows():
        regimes.append(re.update(row))
    regime_df["regime"] = regimes
    return regime_df


# ================================================================
# 고속 백테스트 (사전 계산 데이터 사용)
# ================================================================

def fast_backtest(
    entry_df: pd.DataFrame,
    regime_series: pd.Series,
    sl_atr_mult: float,
    tp_rr_ratio: float,
    entry_score_threshold: int,
    pullback_ema_dist_pct: float,
    rsi_oversold: float,
    rsi_overbought: float,
    symbol: str = "",
    block_sol_long: bool = True,
) -> Dict:
    """최적화된 단일 백테스트"""

    balance = INITIAL_BALANCE
    trades_pnl = []  # (pnl, fee, direction) 튜플만 저장
    position = None
    daily_trades = {}
    n = len(entry_df)

    # numpy 배열로 변환 (속도)
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

    for i in range(50, n):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        ts = timestamps[i]
        day_key = int(ts // 86400000)

        # --- 포지션 보유 중 ---
        if position is not None:
            ep = position[0]  # entry_price
            d = position[1]   # direction
            sl = position[2]  # sl_price
            tp = position[3]  # tp_price
            qty = position[4]
            margin = position[5]

            hit_sl = hit_tp = False
            if d == "LONG":
                if low <= sl: hit_sl = True
                if high >= tp: hit_tp = True
            else:
                if high >= sl: hit_sl = True
                if low <= tp: hit_tp = True

            if hit_tp and hit_sl:
                hit_tp = False

            if hit_sl or hit_tp:
                exit_p = tp if hit_tp else sl
                if d == "LONG":
                    raw = (exit_p - ep) * qty
                else:
                    raw = (ep - exit_p) * qty
                fee = (ep * qty + exit_p * qty) * TAKER_FEE_PCT
                net = raw - fee
                balance += net
                trades_pnl.append((net, fee, d))
                position = None
            continue

        # --- 신규 진입 ---
        if balance <= 10:
            continue

        dt = daily_trades.get(day_key, 0)
        if dt >= MAX_DAILY_TRADES:
            continue

        regime = regimes[i] if i < len(regimes) else "RANGING"

        if regime not in ("TRENDING_UP", "TRENDING_DOWN"):
            continue

        direction = "LONG" if regime == "TRENDING_UP" else "SHORT"

        # SOL LONG 차단
        if block_sol_long and symbol == "SOLUSDT" and direction == "LONG":
            continue

        ef = ema_f[i]
        es = ema_s[i]
        rsi = rsis[i]
        atr_pct = atr_pcts[i]
        vol = vol_ratios[i]

        if ef == 0 or es == 0 or close == 0 or pd.isna(ef) or pd.isna(es):
            continue

        # EMA 풀백
        if direction == "LONG":
            if ef <= es: continue
            dist = (ef - close) / ef * 100
        else:
            if ef >= es: continue
            dist = (close - ef) / ef * 100

        if dist < -0.1 or dist > pullback_ema_dist_pct:
            continue

        # 스코어
        score = 50
        if direction == "LONG" and rsi <= rsi_oversold:
            score += 20
        elif direction == "SHORT" and rsi >= rsi_overbought:
            score += 20
        elif (direction == "LONG" and rsi < 50) or (direction == "SHORT" and rsi > 50):
            score += 10

        if vol >= 1.3: score += 15
        elif vol >= 1.0: score += 5

        if score < entry_score_threshold:
            continue

        # SL/TP
        sl_tp = fee_adjusted_sl_tp(atr_pct, sl_atr_mult, tp_rr_ratio)
        if sl_tp[0] is None:
            continue
        sl_pct, tp_pct = sl_tp

        # 포지션 사이징
        risk_amt = balance * RISK_PER_TRADE
        sl_ratio = sl_pct / 100.0
        avail_margin = balance * MARGIN_USAGE
        ideal = risk_amt / sl_ratio
        lev = min(int(ideal / avail_margin) if avail_margin > 0 else 1, MAX_LEVERAGE)
        lev = max(lev, MIN_LEVERAGE)
        pos_val = avail_margin * lev
        qty = pos_val / close
        if pos_val < 5: continue

        if direction == "LONG":
            sl_price = close * (1 - sl_pct / 100)
            tp_price = close * (1 + tp_pct / 100)
        else:
            sl_price = close * (1 + sl_pct / 100)
            tp_price = close * (1 - tp_pct / 100)

        margin = pos_val / lev
        position = (close, direction, sl_price, tp_price, qty, margin)
        daily_trades[day_key] = dt + 1

    # 미청산 강제 청산
    if position is not None:
        ep, d, sl, tp, qty, margin = position
        exit_p = closes[-1]
        raw = (exit_p - ep) * qty if d == "LONG" else (ep - exit_p) * qty
        fee = (ep * qty + exit_p * qty) * TAKER_FEE_PCT
        trades_pnl.append((raw - fee, fee, d))

    # 통계
    total = len(trades_pnl)
    if total == 0:
        return {"total_trades": 0, "return_pct": 0, "max_dd": 0, "win_rate": 0,
                "total_pnl": 0, "fees": 0, "long_pnl": 0, "short_pnl": 0}

    wins = sum(1 for p, _, _ in trades_pnl if p > 0)
    total_pnl = sum(p for p, _, _ in trades_pnl)
    total_fees = sum(f for _, f, _ in trades_pnl)
    long_pnl = sum(p for p, _, d in trades_pnl if d == "LONG")
    short_pnl = sum(p for p, _, d in trades_pnl if d == "SHORT")

    # 최대 DD
    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for p, _, _ in trades_pnl:
        running += p
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    final = INITIAL_BALANCE + total_pnl

    return {
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((final / INITIAL_BALANCE - 1) * 100, 1),
        "max_dd": round(max_dd, 1),
        "fees": round(total_fees, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
    }


# ================================================================
# 멀티코인 고속 백테스트
# ================================================================

def fast_multi_backtest(
    precomputed_entries: Dict,  # {symbol: df}
    precomputed_regimes: Dict,  # {symbol: regime_series}
    sl, tp, score, pullback, rsi_os, rsi_ob,
) -> Dict:
    """멀티코인 합산 고속 백테스트"""
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    total_fees = 0.0
    long_pnl = 0.0
    short_pnl = 0.0
    # DD는 간소화 (코인별 합산)
    max_dd_sum = 0.0

    for symbol in SYMBOLS:
        if symbol not in precomputed_entries:
            continue
        r = fast_backtest(
            precomputed_entries[symbol],
            precomputed_regimes[symbol],
            sl, tp, score, pullback, rsi_os, rsi_ob,
            symbol=symbol, block_sol_long=True,
        )
        total_trades += r["total_trades"]
        total_wins += r["wins"]
        total_pnl += r["total_pnl"]
        total_fees += r["fees"]
        long_pnl += r["long_pnl"]
        short_pnl += r["short_pnl"]
        max_dd_sum = max(max_dd_sum, r["max_dd"])

    wr = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0
    final = INITIAL_BALANCE + total_pnl
    ret = round((final / INITIAL_BALANCE - 1) * 100, 1)

    return {
        "total_trades": total_trades,
        "win_rate": wr,
        "total_pnl": round(total_pnl, 2),
        "return_pct": ret,
        "max_dd": round(max_dd_sum, 1),
        "fees": round(total_fees, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
    }


# ================================================================
# 메인
# ================================================================

def main():
    print("=" * 80)
    print("HERMES 전수 조사 그리드 서치")
    print(f"기간: {LONG_START} ~ {LONG_END} (4년+)")
    print(f"타임프레임: 1H 진입 / 4H 레짐")
    print("=" * 80)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    raw_data = load_long_data()
    if raw_data is None:
        return

    # EMA 조합 생성 (fast < slow - 3)
    ema_combos = [(f, s) for f in GRID["ema_fast"] for s in GRID["ema_slow"] if s > f + 3]
    adx_values = GRID["adx_enter_trending"]

    # 나머지 파라미터 조합
    signal_combos = list(itertools.product(
        GRID["sl_atr_mult"],
        GRID["tp_rr_ratio"],
        GRID["entry_score_threshold"],
        GRID["pullback_ema_dist_pct"],
        GRID["rsi_oversold"],
        GRID["rsi_overbought"],
    ))

    total_combos = len(ema_combos) * len(adx_values) * len(signal_combos)
    print(f"\n  EMA 조합: {len(ema_combos)}")
    print(f"  ADX 값: {len(adx_values)}")
    print(f"  시그널 조합 (SL×TP×Score×PB×RSI): {len(signal_combos)}")
    print(f"  총 조합: {total_combos:,}")

    # 결과 저장 (Top N만 유지)
    TOP_N = 100
    top_results = []
    tested = 0
    start_time = time.time()

    # ADX별 레짐 사전 계산
    print("\n[2] ADX별 레짐 사전 계산...")
    regime_cache = {}  # (symbol, adx) -> regime_series
    for adx in adx_values:
        for symbol in SYMBOLS:
            regime_df = precompute_regimes(raw_data[f"{symbol}_240"], adx)
            entry_df_dummy = raw_data[f"{symbol}_60"]
            regime_mapped = align_regime_to_entry(regime_df, entry_df_dummy)
            regime_cache[(symbol, adx)] = regime_mapped
        print(f"  ✓ ADX={adx} 완료")

    # EMA별 지표 사전 계산 → 시그널 스윕
    print(f"\n[3] 전수 조사 시작 ({total_combos:,} 조합)...")

    for ema_idx, (ef, es) in enumerate(ema_combos):
        # EMA별 진입 지표 계산
        entry_cache = {}
        for symbol in SYMBOLS:
            entry_cache[symbol] = precompute_entry_indicators(
                raw_data[f"{symbol}_60"], ef, es
            )

        for adx in adx_values:
            # 레짐 매핑
            regime_map = {sym: regime_cache[(sym, adx)] for sym in SYMBOLS}

            # 시그널 파라미터 스윕
            for sl, tp, score, pb, rsi_os, rsi_ob in signal_combos:
                r = fast_multi_backtest(
                    entry_cache, regime_map,
                    sl, tp, score, pb, rsi_os, rsi_ob,
                )

                tested += 1
                r["params"] = {
                    "ema_fast": ef, "ema_slow": es,
                    "adx": adx, "sl": sl, "tp": tp,
                    "score": score, "pullback": pb,
                    "rsi_os": rsi_os, "rsi_ob": rsi_ob,
                }

                # Top N 유지
                if len(top_results) < TOP_N:
                    top_results.append(r)
                    top_results.sort(key=lambda x: x["return_pct"], reverse=True)
                elif r["return_pct"] > top_results[-1]["return_pct"]:
                    top_results[-1] = r
                    top_results.sort(key=lambda x: x["return_pct"], reverse=True)

            # 진행 상황
            if tested % 5000 == 0:
                elapsed = time.time() - start_time
                speed = tested / elapsed
                eta = (total_combos - tested) / speed if speed > 0 else 0
                best = top_results[0]["return_pct"] if top_results else 0
                print(f"  진행: {tested:,}/{total_combos:,} ({tested/total_combos*100:.1f}%) | "
                      f"속도: {speed:.0f}/s | ETA: {eta/60:.0f}분 | "
                      f"현재 최고: {best:+.1f}%")

        # EMA 단위 진행
        elapsed = time.time() - start_time
        print(f"  EMA {ef}/{es} 완료 ({ema_idx+1}/{len(ema_combos)}) | "
              f"경과: {elapsed/60:.1f}분 | 최고: {top_results[0]['return_pct']:+.1f}%")

    total_time = time.time() - start_time

    # ================================================================
    # 결과 출력
    # ================================================================

    print("\n" + "=" * 100)
    print(f"전수 조사 완료: {total_combos:,}개 조합 | 소요: {total_time/60:.1f}분")
    print("=" * 100)

    print(f"\n{'#':>3} {'거래':>5} {'승률':>6} {'수익률':>9} {'PnL':>10} {'DD':>6} "
          f"{'EMA':>7} {'SL':>4} {'TP':>4} {'Score':>5} {'PB':>4} {'ADX':>4} {'RSI':>8}")
    print("-" * 100)

    for i, r in enumerate(top_results[:50]):
        p = r["params"]
        print(f"{i+1:>3} {r['total_trades']:>5} {r['win_rate']:>5.1f}% {r['return_pct']:>+8.1f}% "
              f"${r['total_pnl']:>+9.2f} {r['max_dd']:>5.1f}% "
              f"{p['ema_fast']:>2}/{p['ema_slow']:<3} {p['sl']:>4} {p['tp']:>4} "
              f"{p['score']:>5} {p['pullback']:>4} {p['adx']:>4} "
              f"{p['rsi_os']}/{p['rsi_ob']}")

    # DD 대비 수익률 (리스크 조정) 상위
    print("\n" + "-" * 100)
    print("리스크 조정 순위 (수익률 / 최대DD)")
    print("-" * 100)
    risk_adj = sorted(top_results, key=lambda x: x["return_pct"] / max(x["max_dd"], 1), reverse=True)
    for i, r in enumerate(risk_adj[:20]):
        p = r["params"]
        ratio = r["return_pct"] / max(r["max_dd"], 1)
        print(f"{i+1:>3} {r['total_trades']:>5} {r['win_rate']:>5.1f}% {r['return_pct']:>+8.1f}% "
              f"DD={r['max_dd']:>5.1f}% ratio={ratio:.2f} "
              f"EMA{p['ema_fast']}/{p['ema_slow']} SL{p['sl']} TP{p['tp']} "
              f"S{p['score']} PB{p['pullback']} ADX{p['adx']} RSI{p['rsi_os']}/{p['rsi_ob']}")

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "period": f"{LONG_START} ~ {LONG_END}",
        "total_combinations": total_combos,
        "elapsed_minutes": round(total_time / 60, 1),
        "top50_by_return": [
            {**{k: v for k, v in r.items() if k != "params"}, "params": r["params"]}
            for r in top_results[:50]
        ],
        "top20_risk_adjusted": [
            {**{k: v for k, v in r.items() if k != "params"}, "params": r["params"],
             "risk_ratio": round(r["return_pct"] / max(r["max_dd"], 1), 2)}
            for r in risk_adj[:20]
        ],
    }

    path = os.path.join(RESULTS_DIR, "full_grid_search.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {path}")


if __name__ == "__main__":
    main()
