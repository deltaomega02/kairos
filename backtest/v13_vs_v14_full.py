#!/usr/bin/env python3
"""V13 vs V14 종합 백테스트.

타임프레임:
- 4H: 레짐 판정 (ADX/EMA/ATR percentile)
- 1H: 진입 시그널 (EMA/RSI/BB/ATR/Volume)
- 1D: 추세 필터 (EMA2 가격 위/아래)
- 5분: SL/TP/트레일링 정확 시뮬

V13: TRENDING_UP/DOWN만 진입 (TREND_PULLBACK)
V14: V13 + RANGING+ADX<20 BB반전 (RANGE_REVERSION)

수수료: 0.075% 왕복 (진입 maker 0.02% + 청산 taker 0.055%)
슬리피지: 0.05% (보수적)
"""

import os
import sys
import json
import gzip
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

DATA_DIR = "~/Projects/HERMES/backtest/data"

# 설정 (V13/V14 공통)
INITIAL_BALANCE = 600.0
RISK_PER_TRADE = 0.015     # 1.5%
MARGIN_USAGE = 0.80
MAX_SIMULTANEOUS = 3
MAKER_FEE = 0.0002          # 0.02% (진입 PostOnly)
TAKER_FEE = 0.00055         # 0.055% (청산 market)
SLIPPAGE = 0.0005           # 0.05%

# V13 안전망
SL_ATR_MULT = 2.0
TP_RR_RATIO = 6.0
TRAILING_ACTIVATION = 1.2    # 수익 1.2% 도달 시 활성화
TRAILING_DISTANCE = 0.1      # 고점에서 0.1% 하락 시 청산

# V14 추가 (RANGE_REVERSION)
RANGE_SL_MULT = 2.0
RANGE_RR = 2.5
RANGE_ADX_MAX = 20.0

# 진입 점수 기준
ENTRY_SCORE_THRESHOLD = 40

# 백테스트 기간 (CSV 데이터 6년치 중 최근 사용)
# 1H 데이터: 2020-03-25 ~ 2026-04-26 (약 6년)
START_DATE = "2022-01-01"   # 6년 풀 데이터 (베어/불 모두)
END_DATE = "2026-04-26"


# ============================================================
# 데이터 로드
# ============================================================

def load_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    """1H/4H/1D 데이터 로드."""
    if interval == "60":
        path = f"{DATA_DIR}/{symbol}_60_long.csv"
    elif interval == "240":
        path = f"{DATA_DIR}/{symbol}_240_long.csv"
    elif interval == "D":
        path = f"{DATA_DIR}/{symbol}_D.csv"
    elif interval == "5":
        path = f"{DATA_DIR}/{symbol}_5.csv"
    elif interval == "15":
        path = f"{DATA_DIR}/{symbol}_15.csv"
    else:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 지표 계산
# ============================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up.fillna(0)
    minus_dm = ((dn > up) & (dn > 0)) * dn.fillna(0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(n).mean() / atr_n
    minus_di = 100 * minus_dm.rolling(n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean(), plus_di, minus_di


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    diff = s.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    dn = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def compute_indicators(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    """1H 봉마다 4H/1H/1D 지표 통합."""
    df = df_1h.copy()
    df["ema_fast"] = ema(df["close"], 3)
    df["ema_slow"] = ema(df["close"], 15)
    df["atr"] = atr(df, 14)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100
    df["rsi"] = rsi(df["close"], 14)
    bb_u, bb_m, bb_l = bbands(df["close"], 20, 2.0)
    df["bb_upper"] = bb_u
    df["bb_mid"] = bb_m
    df["bb_lower"] = bb_l
    df["bb_pos"] = ((df["close"] - bb_l) / (bb_u - bb_l) * 100).clip(0, 100)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # 4H 지표 (forward fill)
    adx_4h, pdi_4h, mdi_4h = adx(df_4h, 14)
    df_4h_idx = df_4h.copy()
    df_4h_idx["adx"] = adx_4h
    df_4h_idx["pdi"] = pdi_4h
    df_4h_idx["mdi"] = mdi_4h
    df_4h_idx["ema9_4h"] = ema(df_4h["close"], 9)
    df_4h_idx["ema21_4h"] = ema(df_4h["close"], 21)
    df_4h_idx["atr_4h"] = atr(df_4h, 14)
    df_4h_idx["atr_4h_pctile"] = df_4h_idx["atr_4h"].rolling(200).rank(pct=True) * 100

    # 1H 봉의 timestamp에 가장 가까운 4H bar 찾아 매핑
    df_4h_idx = df_4h_idx[["timestamp", "adx", "pdi", "mdi", "ema9_4h", "ema21_4h",
                            "atr_4h_pctile"]].rename(columns={"timestamp": "ts_4h"})
    df = pd.merge_asof(df.sort_values("timestamp"),
                       df_4h_idx.sort_values("ts_4h"),
                       left_on="timestamp", right_on="ts_4h", direction="backward")

    # 1D 필터 (EMA2)
    df_1d_idx = df_1d.copy()
    df_1d_idx["ema_d_1d"] = ema(df_1d_idx["close"], 2)
    df_1d_idx = df_1d_idx[["timestamp", "ema_d_1d", "close"]].rename(
        columns={"timestamp": "ts_1d", "close": "close_1d"})
    df = pd.merge_asof(df.sort_values("timestamp"),
                       df_1d_idx.sort_values("ts_1d"),
                       left_on="timestamp", right_on="ts_1d", direction="backward")

    return df


# ============================================================
# 레짐 판정
# ============================================================

def regime_of(row) -> str:
    """4H ADX/EMA/ATR_pctile 기반 레짐 판정."""
    adx_v = row.get("adx", 0)
    pdi = row.get("pdi", 0)
    mdi = row.get("mdi", 0)
    ema9 = row.get("ema9_4h", 0)
    ema21 = row.get("ema21_4h", 0)
    atr_pctl = row.get("atr_4h_pctile", 50)

    if pd.isna(adx_v) or pd.isna(ema9):
        return "UNKNOWN"

    # HIGH_VOL: ATR percentile 85 이상
    if atr_pctl >= 85:
        return "HIGH_VOL"

    # TRENDING: ADX >= 30 + EMA 정렬
    if adx_v >= 30:
        if pdi > mdi and ema9 > ema21:
            return "TRENDING_UP"
        if mdi > pdi and ema9 < ema21:
            return "TRENDING_DOWN"

    # RANGING
    return "RANGING"


# ============================================================
# 시그널 평가
# ============================================================

def eval_trend_pullback(row, regime: str) -> Optional[Dict]:
    """V13/V14 공통: 추세장 풀백 진입."""
    if regime == "TRENDING_UP":
        direction = "LONG"
    elif regime == "TRENDING_DOWN":
        direction = "SHORT"
    else:
        return None

    ema_f = row.get("ema_fast", 0)
    ema_s = row.get("ema_slow", 0)
    close = row.get("close", 0)
    atr_pct = row.get("atr_pct", 0)
    rsi_v = row.get("rsi", 50)
    vol_r = row.get("volume_ratio", 1.0)
    close_1d = row.get("close_1d", 0)
    ema_1d = row.get("ema_d_1d", 0)

    if any(pd.isna(x) or x == 0 for x in [ema_f, ema_s, close, atr_pct]):
        return None

    # 1D EMA2 필터
    if ema_1d and close_1d:
        if direction == "LONG" and close_1d <= ema_1d:
            return None
        if direction == "SHORT" and close_1d >= ema_1d:
            return None

    # 풀백 조건
    if direction == "LONG":
        if ema_f <= ema_s:
            return None
        dist_pct = (ema_f - close) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > 1.5:
            return None
    else:
        if ema_f >= ema_s:
            return None
        dist_pct = (close - ema_f) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > 1.5:
            return None

    # min_atr 필터
    if atr_pct < 0.3:
        return None

    # 점수
    score = 50
    if direction == "LONG" and rsi_v <= 35:
        score += 20
    elif direction == "SHORT" and rsi_v >= 65:
        score += 20
    elif (direction == "LONG" and rsi_v < 50) or (direction == "SHORT" and rsi_v > 50):
        score += 10
    if vol_r >= 1.3:
        score += 15
    elif vol_r >= 1.0:
        score += 5

    if score < ENTRY_SCORE_THRESHOLD:
        return None

    sl_pct = max(atr_pct * SL_ATR_MULT, 0.33)  # 최소 수수료×3
    tp_pct = sl_pct * TP_RR_RATIO

    return {
        "strategy": "TREND_PULLBACK",
        "direction": direction,
        "score": score,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
    }


def eval_range_reversion(row, regime: str) -> Optional[Dict]:
    """V14 전용: 횡보장 BB 반전."""
    if regime != "RANGING":
        return None
    adx_v = row.get("adx", 30)
    if adx_v >= RANGE_ADX_MAX:
        return None  # ADX < 20 강한 횡보만

    close = row.get("close", 0)
    bb_u = row.get("bb_upper", 0)
    bb_l = row.get("bb_lower", 0)
    bb_m = row.get("bb_mid", 0)
    bb_pos = row.get("bb_pos", 50)
    rsi_v = row.get("rsi", 50)
    atr_pct = row.get("atr_pct", 0)
    vol_r = row.get("volume_ratio", 1.0)

    if any(pd.isna(x) or x == 0 for x in [close, bb_u, bb_l, atr_pct]):
        return None

    if atr_pct < 0.3:
        return None

    direction = None
    score = 0
    if close <= bb_l * 1.002 and bb_pos <= 20:
        direction = "LONG"
        score = 40
        if rsi_v <= 30:
            score += 20
        elif rsi_v <= 40:
            score += 10
        if vol_r < 0.8:
            score += 10
    elif close >= bb_u * 0.998 and bb_pos >= 80:
        direction = "SHORT"
        score = 40
        if rsi_v >= 70:
            score += 20
        elif rsi_v >= 60:
            score += 10
        if vol_r < 0.8:
            score += 10

    if direction is None:
        return None

    range_threshold = ENTRY_SCORE_THRESHOLD + 15
    if score < range_threshold:
        return None

    sl_pct = max(atr_pct * RANGE_SL_MULT, 0.33)
    tp_pct = sl_pct * RANGE_RR

    # TP를 BB 중앙으로 제한
    if direction == "LONG":
        max_tp = (bb_m - close) / close * 100
    else:
        max_tp = (close - bb_m) / close * 100
    tp_pct = min(tp_pct, max(max_tp, sl_pct * 1.5))

    return {
        "strategy": "RANGE_REVERSION",
        "direction": direction,
        "score": score,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
    }


# ============================================================
# 시뮬레이션 (5분 봉으로 SL/TP/트레일링 정확)
# ============================================================

def simulate_trade(entry_idx_1h: int, signal: Dict, df_1h: pd.DataFrame, df_5m: Optional[pd.DataFrame]) -> Dict:
    """진입 후 5분 봉으로 SL/TP/트레일링 시뮬."""
    entry_row = df_1h.iloc[entry_idx_1h]
    entry_price = entry_row["close"]
    direction = signal["direction"]
    sl_pct = signal["sl_pct"]
    tp_pct = signal["tp_pct"]
    strategy = signal["strategy"]

    # SL/TP 가격
    if direction == "LONG":
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)
    else:
        sl_price = entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 - tp_pct / 100)

    # 5분 봉 시작 인덱스
    entry_ts = entry_row["timestamp"]
    if df_5m is not None and len(df_5m) > 0:
        bars_5m = df_5m[df_5m["timestamp"] >= entry_ts].head(48)  # 최대 4시간
    else:
        # fallback: 1H high/low 사용
        bars_5m = pd.DataFrame()

    # 트레일링 상태
    trailing_active = False
    peak = entry_price
    cur_sl = sl_price
    exit_price = None
    exit_reason = None
    bars_held = 0

    if len(bars_5m) > 0:
        # 5분 봉으로 정확 시뮬
        for _, bar in bars_5m.iterrows():
            bars_held += 1
            high = bar["high"]
            low = bar["low"]

            if direction == "LONG":
                # SL 먼저 체크 (보수적)
                if low <= cur_sl:
                    exit_price = cur_sl
                    exit_reason = "SL" if not trailing_active else "TRAILING"
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
                # 트레일링
                new_peak = max(peak, high)
                gain_pct = (new_peak - entry_price) / entry_price * 100
                if not trailing_active and gain_pct >= TRAILING_ACTIVATION and strategy == "TREND_PULLBACK":
                    trailing_active = True
                if trailing_active:
                    new_trail_sl = new_peak * (1 - TRAILING_DISTANCE / 100)
                    if new_trail_sl > cur_sl:
                        cur_sl = new_trail_sl
                peak = new_peak
            else:  # SHORT
                if high >= cur_sl:
                    exit_price = cur_sl
                    exit_reason = "SL" if not trailing_active else "TRAILING"
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
                new_peak = min(peak, low)
                gain_pct = (entry_price - new_peak) / entry_price * 100
                if not trailing_active and gain_pct >= TRAILING_ACTIVATION and strategy == "TREND_PULLBACK":
                    trailing_active = True
                if trailing_active:
                    new_trail_sl = new_peak * (1 + TRAILING_DISTANCE / 100)
                    if new_trail_sl < cur_sl:
                        cur_sl = new_trail_sl
                peak = new_peak

    # 미청산 시 1H 봉 다음으로 fallback
    if exit_price is None:
        next_idx = min(entry_idx_1h + 24, len(df_1h) - 1)  # 최대 24시간
        for i in range(entry_idx_1h + 1, next_idx + 1):
            row = df_1h.iloc[i]
            high = row["high"]
            low = row["low"]
            bars_held += 1
            if direction == "LONG":
                if low <= cur_sl:
                    exit_price = cur_sl
                    exit_reason = "SL" if not trailing_active else "TRAILING"
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
            else:
                if high >= cur_sl:
                    exit_price = cur_sl
                    exit_reason = "SL" if not trailing_active else "TRAILING"
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
        if exit_price is None:
            exit_price = df_1h.iloc[next_idx]["close"]
            exit_reason = "TIMEOUT"

    # PnL 계산 (수수료 + 슬리피지 포함)
    if direction == "LONG":
        gross = (exit_price - entry_price) / entry_price
    else:
        gross = (entry_price - exit_price) / entry_price

    fee = MAKER_FEE + TAKER_FEE  # 진입 maker + 청산 taker
    slip = SLIPPAGE  # 진입+청산 슬리피지
    net_pct = gross - fee - slip
    return {
        "strategy": strategy,
        "direction": direction,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "gross_pct": gross * 100,
        "net_pct": net_pct * 100,
        "score": signal["score"],
    }


# ============================================================
# 백테스트 실행
# ============================================================

def backtest(symbol: str, mode: str = "V14") -> Dict:
    """V13: TREND only / V14: TREND + RANGE."""
    df_1h = load_ohlcv(symbol, "60")
    df_4h = load_ohlcv(symbol, "240")
    df_1d = load_ohlcv(symbol, "D")
    df_5m = load_ohlcv(symbol, "5")

    # 기간 필터
    start_ts = pd.Timestamp(START_DATE).value // 1_000_000
    end_ts = pd.Timestamp(END_DATE).value // 1_000_000
    df_1h = df_1h[(df_1h["timestamp"] >= start_ts) & (df_1h["timestamp"] <= end_ts)]

    if len(df_1h) < 100:
        return {"error": f"{symbol} 데이터 부족"}

    df = compute_indicators(df_1h.reset_index(drop=True), df_4h, df_1d)

    trades = []
    last_exit_ts = 0  # 단일 포지션 가정 (단순화)

    for i in range(len(df)):
        row = df.iloc[i]
        if row["timestamp"] <= last_exit_ts:
            continue

        regime = regime_of(row)

        # V13: TRENDING만 / V14: TRENDING + RANGING
        sig = eval_trend_pullback(row, regime)
        if sig is None and mode == "V14":
            sig = eval_range_reversion(row, regime)

        if sig is None:
            continue

        # SOL LONG 차단
        if symbol == "SOLUSDT" and sig["direction"] == "LONG":
            continue

        # 시뮬
        result = simulate_trade(i, sig, df, df_5m)
        result["symbol"] = symbol
        trades.append(result)
        last_exit_ts = row["timestamp"] + result["bars_held"] * (300_000 if df_5m is not None and len(df_5m) > 0 else 3_600_000)

    if not trades:
        return {"trades": 0, "symbol": symbol, "mode": mode}

    df_trades = pd.DataFrame(trades)
    wins = (df_trades["net_pct"] > 0).sum()
    total_pct = df_trades["net_pct"].sum()
    avg_win = df_trades[df_trades["net_pct"] > 0]["net_pct"].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades["net_pct"] <= 0]["net_pct"].mean() if (len(df_trades) - wins) > 0 else 0

    # 최대 DD 추정
    cumulative = df_trades["net_pct"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    return {
        "symbol": symbol,
        "mode": mode,
        "trades": len(trades),
        "wins": int(wins),
        "win_rate": float(wins / len(trades) * 100),
        "total_pct": float(total_pct),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "max_dd_pct": float(max_dd),
        "by_strategy": {
            s: int((df_trades["strategy"] == s).sum())
            for s in df_trades["strategy"].unique()
        },
        "by_reason": {
            r: int((df_trades["exit_reason"] == r).sum())
            for r in df_trades["exit_reason"].unique()
        },
    }


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 80)
    print(f"V13 vs V14 종합 백테스트 ({START_DATE} ~ {END_DATE})")
    print("=" * 80)
    print()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results_v13 = {}
    results_v14 = {}

    for sym in symbols:
        print(f"[{sym.replace('USDT','')}] 백테스트 중...")
        r13 = backtest(sym, mode="V13")
        r14 = backtest(sym, mode="V14")
        results_v13[sym] = r13
        results_v14[sym] = r14

    print()
    print("=" * 80)
    print(f"{'SYMBOL':>8} | {'MODE':>5} | {'거래':>5} {'승':>4} {'승률':>6} | {'총%':>8} {'avg+':>6} {'avg-':>6} | {'MaxDD%':>8}")
    print("-" * 80)

    for sym in symbols:
        for mode, r in [("V13", results_v13[sym]), ("V14", results_v14[sym])]:
            if "error" in r or r.get("trades", 0) == 0:
                print(f"{sym:>8} | {mode:>5} | (no data)")
                continue
            print(f"{sym:>8} | {mode:>5} | "
                  f"{r['trades']:>5} {r['wins']:>4} {r['win_rate']:>5.1f}% | "
                  f"{r['total_pct']:>+7.2f}% {r['avg_win_pct']:>+5.2f} {r['avg_loss_pct']:>+5.2f} | "
                  f"{r['max_dd_pct']:>+7.2f}%")
        # 비교
        if results_v13[sym].get("trades", 0) > 0 and results_v14[sym].get("trades", 0) > 0:
            diff = results_v14[sym]["total_pct"] - results_v13[sym]["total_pct"]
            extra_trades = results_v14[sym]["trades"] - results_v13[sym]["trades"]
            print(f"{sym:>8} | {'Δ':>5} | 추가거래 {extra_trades:+d}건, 총% 차이 {diff:+.2f}%")
        print()

    # 종합
    print("=" * 80)
    print("종합 비교 (3코인 합산)")
    print("=" * 80)
    for mode_name, results in [("V13", results_v13), ("V14", results_v14)]:
        total_t = sum(r.get("trades", 0) for r in results.values())
        total_w = sum(r.get("wins", 0) for r in results.values())
        total_pct = sum(r.get("total_pct", 0) for r in results.values())
        wr = total_w / total_t * 100 if total_t > 0 else 0
        print(f"{mode_name}: {total_t} 거래, {total_w} 승 ({wr:.1f}%), 총 {total_pct:+.2f}%")

    # 환경별 분석 (V14)
    print()
    print("=" * 80)
    print("V14 전략별 분포")
    print("=" * 80)
    for sym in symbols:
        if "by_strategy" in results_v14[sym]:
            print(f"  {sym}: {results_v14[sym]['by_strategy']}")

    # JSON 저장
    output = {
        "config": {
            "start": START_DATE, "end": END_DATE,
            "fees": {"maker": MAKER_FEE, "taker": TAKER_FEE, "slippage": SLIPPAGE},
            "v13_only": "TREND_PULLBACK",
            "v14_added": "RANGE_REVERSION (RANGING+ADX<20)",
        },
        "v13": results_v13,
        "v14": results_v14,
    }
    out_path = "~/Projects/HERMES/backtest/v13_vs_v14_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
