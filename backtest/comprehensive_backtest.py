#!/usr/bin/env python3
"""
HERMES Comprehensive Backtest Engine
=====================================
모든 타임프레임 × 모든 파라미터 조합 백테스트.
Bybit v5 API에서 2년치 데이터 자동 다운로드 + 캐싱.
"""

import os
import sys
import time
import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import requests

# ================================================================
# 설정
# ================================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# 진입 타임프레임 (분 단위): 5분, 15분, 1시간, 4시간
ENTRY_INTERVALS = ["5", "15", "60", "240"]
# 레짐 판독 타임프레임 (진입 TF에 따라 상위 TF 사용)
REGIME_MAP = {
    "5": "60",     # 5분 진입 → 1시간 레짐
    "15": "240",   # 15분 진입 → 4시간 레짐
    "60": "240",   # 1시간 진입 → 4시간 레짐 (현재 시스템)
    "240": "D",    # 4시간 진입 → 일봉 레짐
}

START_DATE = "2024-04-08"
END_DATE = "2026-04-08"
INITIAL_BALANCE = 300.0

TAKER_FEE_PCT = 0.00055   # 0.055% per side
MAX_LEVERAGE = 5
MIN_LEVERAGE = 1
RISK_PER_TRADE = 0.015     # 1.5%
MARGIN_USAGE = 0.80
MAX_SIMULTANEOUS = 2
MAX_DAILY_TRADES = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 파라미터 그리드
# ================================================================

# Stage 1: 타임프레임별 기본 파라미터 비교
DEFAULT_PARAMS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "pullback_ema_dist_pct": 1.0,
    "sl_atr_mult": 1.5,
    "tp_rr_ratio": 2.5,
    "entry_score_threshold": 60,
    "adx_enter_trending": 25,
    "adx_exit_trending": 20,
    "atr_high_vol_percentile": 85,
    "orderbook_imbalance_min": 0.55,
    "funding_bias_threshold": 0.0005,
}

# Stage 2: 핵심 파라미터 스윕
PARAM_GRID = {
    "sl_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "tp_rr_ratio": [1.5, 2.0, 2.5, 3.0, 4.0],
    "entry_score_threshold": [40, 50, 60, 70],
    "pullback_ema_dist_pct": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    "adx_enter_trending": [20, 25, 30, 35],
    "ema_fast": [5, 7, 9, 12],
    "ema_slow": [15, 21, 26, 30],
}


# ================================================================
# 데이터 다운로드
# ================================================================

def _bybit_interval_ms(interval: str) -> int:
    """인터벌을 밀리초로 변환"""
    if interval == "D":
        return 86400000
    return int(interval) * 60 * 1000


def fetch_kline(symbol: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Bybit v5 API에서 OHLCV 데이터 가져오기"""
    all_candles = []
    interval_ms = _bybit_interval_ms(interval)
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval if interval != "D" else "D",
            "start": current_start,
            "end": min(current_start + 999 * interval_ms, end_ts),
            "limit": 1000,
        }

        for retry in range(5):
            try:
                resp = requests.get(
                    "https://api.bybit.com/v5/market/kline",
                    params=params,
                    timeout=10,
                )
                data = resp.json()
                if data.get("retCode") == 0:
                    break
                if data.get("retCode") == 10006:  # rate limit
                    time.sleep(2 ** retry)
                    continue
                break
            except Exception:
                time.sleep(2 ** retry)
        else:
            break

        result = data.get("result", {})
        rows = result.get("list", [])
        if not rows:
            break

        for row in rows:
            # Bybit v5: [timestamp, open, high, low, close, volume, turnover]
            ts = int(row[0])
            if ts < current_start or ts >= end_ts:
                continue
            all_candles.append({
                "timestamp": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        # Bybit returns descending, so advance past oldest returned
        if rows:
            timestamps = [int(r[0]) for r in rows]
            oldest = min(timestamps)
            newest = max(timestamps)
            current_start = newest + interval_ms
        else:
            break

        time.sleep(0.15)  # rate limit

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def load_or_fetch(symbol: str, interval: str) -> pd.DataFrame:
    """캐시된 데이터 로드 또는 API에서 다운로드"""
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        if len(df) > 100:
            print(f"  ✓ 캐시 로드: {symbol} {interval} ({len(df)}개)")
            return df

    print(f"  ↓ 다운로드: {symbol} {interval}...")
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp() * 1000)

    df = fetch_kline(symbol, interval, start_ts, end_ts)
    if len(df) > 0:
        df.to_csv(cache_file, index=False)
        print(f"  ✓ 저장 완료: {symbol} {interval} ({len(df)}개)")

    return df


# ================================================================
# 지표 계산
# ================================================================

def calc_ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def calc_sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period).mean()


def calc_rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, abs(h - prev_c), abs(l - prev_c)], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, abs(h - prev_c), abs(l - prev_c)], axis=1).max(axis=1)
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=h.index)
    atr_sm = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_sm)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_sm)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def calc_bb(s: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = calc_sma(s, period)
    std = s.rolling(window=period).std()
    return sma, sma + std * std_dev, sma - std * std_dev


def calc_macd(s: pd.Series, fast=12, slow=26, sig=9):
    ema_f = calc_ema(s, fast)
    ema_s = calc_ema(s, slow)
    macd_line = ema_f - ema_s
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line, macd_line - signal_line


def compute_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """레짐 지표 계산 (4H 또는 상위 TF)"""
    adx_data = calc_adx(df["high"], df["low"], df["close"])
    df = df.copy()
    df["adx"] = adx_data["adx"]
    df["plus_di"] = adx_data["plus_di"]
    df["minus_di"] = adx_data["minus_di"]
    df["ema9_regime"] = calc_ema(df["close"], 9)
    df["ema21_regime"] = calc_ema(df["close"], 21)
    _, _, macd_hist = calc_macd(df["close"])
    df["macd_hist"] = macd_hist
    atr = calc_atr(df["high"], df["low"], df["close"])
    df["atr_regime"] = atr
    df["atr_pct_regime"] = (atr / df["close"]) * 100

    # ATR 퍼센타일 (100봉 기준)
    def _pctl(x):
        if len(x) < 10:
            return 50.0
        return (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100
    df["atr_percentile"] = df["atr_regime"].rolling(100, min_periods=10).apply(_pctl, raw=False)

    return df


def compute_entry_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """진입 지표 계산"""
    df = df.copy()
    ef = int(params.get("ema_fast", 9))
    es = int(params.get("ema_slow", 21))
    rp = int(params.get("rsi_period", 14))

    df["ema_fast"] = calc_ema(df["close"], ef)
    df["ema_slow"] = calc_ema(df["close"], es)
    df["rsi"] = calc_rsi(df["close"], rp)
    atr = calc_atr(df["high"], df["low"], df["close"])
    df["atr"] = atr
    df["atr_pct"] = (atr / df["close"]) * 100

    vol_sma = calc_sma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / (vol_sma + 1e-10)

    bb_mid, bb_up, bb_low = calc_bb(df["close"])
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_low
    df["bb_middle"] = bb_mid

    return df


# ================================================================
# 레짐 엔진 (백테스트용)
# ================================================================

class BacktestRegimeEngine:
    """히스테리시스 + 디바운스 레짐 판독"""

    def __init__(self, params: Dict):
        self.params = params
        self.current_regime = "RANGING"
        self._pending = None
        self._pending_count = 0

    def update(self, row: pd.Series) -> str:
        adx = row.get("adx", 0)
        plus_di = row.get("plus_di", 0)
        minus_di = row.get("minus_di", 0)
        ema9 = row.get("ema9_regime", 0)
        ema21 = row.get("ema21_regime", 0)
        macd_h = row.get("macd_hist", 0)
        atr_pctl = row.get("atr_percentile", 50)

        adx_enter = self.params["adx_enter_trending"]
        adx_exit = self.params.get("adx_exit_trending", 20)
        high_vol = self.params.get("atr_high_vol_percentile", 85)
        debounce = self.params.get("regime_debounce_bars", 1)

        if pd.isna(adx) or pd.isna(atr_pctl):
            return self.current_regime

        # 고변동
        if atr_pctl >= high_vol:
            raw = "HIGH_VOL"
        # 추세
        elif self._is_trending(adx, adx_enter, adx_exit):
            raw = self._direction(plus_di, minus_di, ema9, ema21, macd_h)
        else:
            raw = "RANGING"

        # 디바운스
        final = self._debounce(raw, debounce)
        self.current_regime = final
        return final

    def _is_trending(self, adx, enter, exit_th):
        currently = self.current_regime in ("TRENDING_UP", "TRENDING_DOWN")
        return adx >= (exit_th if currently else enter)

    def _direction(self, plus_di, minus_di, ema9, ema21, macd_h):
        bull = bear = 0
        if plus_di > minus_di: bull += 1
        else: bear += 1
        if ema9 > ema21: bull += 1
        else: bear += 1
        if macd_h > 0: bull += 1
        else: bear += 1
        if bull >= 2: return "TRENDING_UP"
        if bear >= 2: return "TRENDING_DOWN"
        return "RANGING"

    def _debounce(self, raw, required):
        if raw == self.current_regime:
            self._pending = None
            self._pending_count = 0
            return self.current_regime
        if raw == self._pending:
            self._pending_count += 1
        else:
            self._pending = raw
            self._pending_count = 1
        if self._pending_count >= required:
            self._pending = None
            self._pending_count = 0
            return raw
        return self.current_regime


# ================================================================
# 시그널 엔진 (백테스트용)
# ================================================================

def fee_adjusted_sl_tp(atr_pct: float, sl_mult: float, rr_ratio: float) -> Tuple[Optional[float], Optional[float]]:
    """수수료 역산 SL/TP"""
    fee_rt = TAKER_FEE_PCT * 2 * 100  # 0.11%
    raw_sl = atr_pct * sl_mult
    min_sl = fee_rt * 3
    sl_pct = max(min_sl, min(5.0, raw_sl))
    raw_tp = sl_pct * rr_ratio
    min_tp = sl_pct + 2 * fee_rt
    tp_pct = max(raw_tp, min_tp)
    real_profit = tp_pct - fee_rt
    real_loss = sl_pct + fee_rt
    real_rr = real_profit / real_loss if real_loss > 0 else 0
    if real_rr < 0.8:
        return None, None
    return sl_pct, tp_pct


def evaluate_signal(regime: str, row: pd.Series, params: Dict) -> Optional[Dict]:
    """시그널 평가 — 레짐별 풀백 전략"""

    if regime == "HIGH_VOL":
        return None
    if regime == "RANGING":
        return None  # 횡보 전략 비활성화

    if regime not in ("TRENDING_UP", "TRENDING_DOWN"):
        return None

    direction = "LONG" if regime == "TRENDING_UP" else "SHORT"

    ema_f = row.get("ema_fast", 0)
    ema_s = row.get("ema_slow", 0)
    close = row.get("close", 0)
    rsi = row.get("rsi", 50)
    vol = row.get("volume_ratio", 1.0)
    atr_pct = row.get("atr_pct", 0.5)

    if ema_f == 0 or ema_s == 0 or close == 0:
        return None

    max_dist = params["pullback_ema_dist_pct"]

    # EMA 풀백 확인
    if direction == "LONG":
        if ema_f <= ema_s:
            return None  # 역배열
        dist_pct = (ema_f - close) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None
    else:
        if ema_f >= ema_s:
            return None  # 정배열
        dist_pct = (close - ema_f) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None

    # 스코어링 (오더북 제외 — 히스토리 없음)
    score = 50

    # RSI 보너스
    if direction == "LONG" and rsi <= params.get("rsi_oversold", 35):
        score += 20
    elif direction == "SHORT" and rsi >= params.get("rsi_overbought", 65):
        score += 20
    elif (direction == "LONG" and rsi < 50) or (direction == "SHORT" and rsi > 50):
        score += 10

    # 볼륨
    if vol >= 1.3:
        score += 15
    elif vol >= 1.0:
        score += 5

    # 오더북 — 히스토리 데이터 없으므로 기본 +0
    # 펀딩 — 히스토리 없으므로 기본 NEUTRAL

    if score < params["entry_score_threshold"]:
        return None

    # SL/TP
    sl_pct, tp_pct = fee_adjusted_sl_tp(
        atr_pct, params["sl_atr_mult"], params["tp_rr_ratio"]
    )
    if sl_pct is None:
        return None

    return {
        "direction": direction,
        "score": score,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "entry_price": close,
        "atr_pct": atr_pct,
    }


# ================================================================
# 백테스트 엔진
# ================================================================

def align_regime_to_entry(regime_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.Series:
    """상위 TF 레짐을 하위 TF 타임스탬프에 매핑"""
    regime_ts = regime_df["timestamp"].values
    regimes = regime_df["regime"].values

    result = []
    regime_idx = 0
    for ts in entry_df["timestamp"]:
        while regime_idx < len(regime_ts) - 1 and regime_ts[regime_idx + 1] <= ts:
            regime_idx += 1
        if regime_idx < len(regimes):
            result.append(regimes[regime_idx])
        else:
            result.append("RANGING")

    return pd.Series(result, index=entry_df.index)


def run_single_backtest(
    entry_df: pd.DataFrame,
    regime_series: pd.Series,
    params: Dict,
    symbol: str = "",
    block_sol_long: bool = True,
) -> Dict:
    """단일 설정 백테스트 실행"""

    balance = INITIAL_BALANCE
    peak_balance = balance
    trades = []
    position = None
    daily_trades = {}

    for i in range(50, len(entry_df)):
        row = entry_df.iloc[i]
        regime = regime_series.iloc[i] if i < len(regime_series) else "RANGING"
        ts = row["timestamp"]
        day_key = str(int(ts // 86400000))

        # --- 포지션 보유 중: SL/TP 체크 ---
        if position is not None:
            entry_p = position["entry_price"]
            direction = position["direction"]
            sl_price = position["sl_price"]
            tp_price = position["tp_price"]
            qty = position["quantity"]

            hit_sl = False
            hit_tp = False

            if direction == "LONG":
                if row["low"] <= sl_price:
                    hit_sl = True
                if row["high"] >= tp_price:
                    hit_tp = True
            else:
                if row["high"] >= sl_price:
                    hit_sl = True
                if row["low"] <= tp_price:
                    hit_tp = True

            if hit_tp and hit_sl:
                # 둘 다 맞은 경우 — 보수적으로 SL 처리
                hit_tp = False

            if hit_sl or hit_tp:
                if hit_tp:
                    exit_price = tp_price
                    reason = "TP"
                else:
                    exit_price = sl_price
                    reason = "SL"

                if direction == "LONG":
                    raw_pnl = (exit_price - entry_p) * qty
                else:
                    raw_pnl = (entry_p - exit_price) * qty

                fee = (entry_p * qty + exit_price * qty) * TAKER_FEE_PCT
                net_pnl = raw_pnl - fee
                margin = position["margin"]
                pnl_pct = net_pnl / margin * 100 if margin > 0 else 0

                balance += net_pnl
                peak_balance = max(peak_balance, balance)

                trades.append({
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_p,
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

        # 일일 거래 한도
        day_count = daily_trades.get(day_key, 0)
        if day_count >= MAX_DAILY_TRADES:
            continue

        # SOL LONG 차단
        if block_sol_long and symbol == "SOLUSDT" and regime == "TRENDING_UP":
            pass  # LONG만 차단, 시그널에서 걸러짐

        signal = evaluate_signal(regime, row, params)
        if signal is None:
            continue

        # SOL LONG 차단
        if block_sol_long and symbol == "SOLUSDT" and signal["direction"] == "LONG":
            continue

        entry_price = signal["entry_price"]
        sl_pct = signal["sl_pct"]
        tp_pct = signal["tp_pct"]

        # 포지션 사이징
        risk_amount = balance * RISK_PER_TRADE
        sl_ratio = sl_pct / 100.0
        available_margin = balance * MARGIN_USAGE

        ideal_position = risk_amount / sl_ratio
        needed_lev = ideal_position / available_margin if available_margin > 0 else 1
        leverage = min(int(needed_lev), MAX_LEVERAGE)
        leverage = max(leverage, MIN_LEVERAGE)

        # ADX 기반 레버리지 제한
        # (간소화 — 항상 leverage 그대로)

        position_value = available_margin * leverage
        qty = position_value / entry_price

        if position_value < 5:
            continue

        # SL/TP 가격
        if signal["direction"] == "LONG":
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
        else:
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)

        margin = position_value / leverage
        position = {
            "direction": signal["direction"],
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "quantity": qty,
            "leverage": leverage,
            "margin": margin,
        }

        daily_trades[day_key] = day_count + 1

    # 미청산 포지션 강제 청산
    if position is not None and len(entry_df) > 0:
        last = entry_df.iloc[-1]
        exit_price = last["close"]
        entry_p = position["entry_price"]
        qty = position["quantity"]
        direction = position["direction"]

        if direction == "LONG":
            raw_pnl = (exit_price - entry_p) * qty
        else:
            raw_pnl = (entry_p - exit_price) * qty

        fee = (entry_p * qty + exit_price * qty) * TAKER_FEE_PCT
        net_pnl = raw_pnl - fee
        balance += net_pnl

        trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_p,
            "exit_price": exit_price,
            "quantity": qty,
            "pnl": net_pnl,
            "pnl_pct": net_pnl / position["margin"] * 100 if position["margin"] > 0 else 0,
            "reason": "FORCE_CLOSE",
            "fee": fee,
            "timestamp": last["timestamp"],
        })

    # 통계 계산
    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = total - wins
    total_pnl = sum(t["pnl"] for t in trades)
    total_fees = sum(t["fee"] for t in trades)
    win_rate = wins / total * 100 if total > 0 else 0
    final_balance = INITIAL_BALANCE + total_pnl

    # 최대 드로다운 계산
    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if losses > 0 else 0

    # LONG/SHORT 분리
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    long_wins = sum(1 for t in long_trades if t["pnl"] > 0)
    short_wins = sum(1 for t in short_trades if t["pnl"] > 0)
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final_balance, 2),
        "return_pct": round((final_balance / INITIAL_BALANCE - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "total_fees": round(total_fees, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "long_trades": len(long_trades),
        "long_wins": long_wins,
        "long_win_rate": round(long_wins / len(long_trades) * 100, 1) if long_trades else 0,
        "long_pnl": round(long_pnl, 2),
        "short_trades": len(short_trades),
        "short_wins": short_wins,
        "short_win_rate": round(short_wins / len(short_trades) * 100, 1) if short_trades else 0,
        "short_pnl": round(short_pnl, 2),
        "trades": trades,
    }


# ================================================================
# 멀티코인 백테스트
# ================================================================

def run_multi_coin_backtest(
    entry_interval: str,
    params: Dict,
    data_cache: Dict,
    block_sol_long: bool = True,
) -> Dict:
    """멀티코인 합산 백테스트"""
    regime_interval = REGIME_MAP.get(entry_interval, "240")

    all_trades = []
    coin_results = {}

    for symbol in SYMBOLS:
        entry_key = f"{symbol}_{entry_interval}"
        regime_key = f"{symbol}_{regime_interval}"

        if entry_key not in data_cache or regime_key not in data_cache:
            continue

        entry_df = data_cache[entry_key].copy()
        regime_df = data_cache[regime_key].copy()

        if len(entry_df) < 100 or len(regime_df) < 50:
            continue

        # 진입 지표 계산
        entry_df = compute_entry_indicators(entry_df, params)

        # 레짐 지표 계산 + 레짐 판독
        regime_df = compute_regime_indicators(regime_df)
        re = BacktestRegimeEngine(params)
        regimes = []
        for _, rr in regime_df.iterrows():
            regimes.append(re.update(rr))
        regime_df["regime"] = regimes

        # 레짐을 진입 TF에 매핑
        regime_mapped = align_regime_to_entry(regime_df, entry_df)

        # 백테스트 실행
        result = run_single_backtest(
            entry_df, regime_mapped, params,
            symbol=symbol, block_sol_long=block_sol_long
        )
        coin_results[symbol] = result
        all_trades.extend(result.get("trades", []))

    # 합산 통계
    if not all_trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "return_pct": 0, "max_drawdown_pct": 0, "final_balance": INITIAL_BALANCE,
            "coin_results": coin_results,
        }

    # 시간순 정렬 후 포트폴리오 시뮬레이션
    all_trades.sort(key=lambda t: t["timestamp"])

    total = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)
    final = INITIAL_BALANCE + total_pnl

    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in all_trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    long_trades = [t for t in all_trades if t["direction"] == "LONG"]
    short_trades = [t for t in all_trades if t["direction"] == "SHORT"]

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final, 2),
        "return_pct": round((final / INITIAL_BALANCE - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "total_fees": round(total_fees, 2),
        "long_trades": len(long_trades),
        "long_pnl": round(sum(t["pnl"] for t in long_trades), 2),
        "short_trades": len(short_trades),
        "short_pnl": round(sum(t["pnl"] for t in short_trades), 2),
        "coin_results": coin_results,
    }


# ================================================================
# Stage 1: 타임프레임 비교
# ================================================================

def stage1_timeframe_comparison(data_cache: Dict):
    """기본 파라미터로 모든 타임프레임 비교"""
    print("\n" + "=" * 70)
    print("STAGE 1: 타임프레임 비교 (기본 파라미터)")
    print("=" * 70)

    results = {}
    for interval in ENTRY_INTERVALS:
        regime_iv = REGIME_MAP.get(interval, "240")
        # 필요한 데이터 있는지 확인
        has_data = all(
            f"{sym}_{interval}" in data_cache and f"{sym}_{regime_iv}" in data_cache
            for sym in SYMBOLS
        )
        if not has_data:
            print(f"\n  [{interval}분] 데이터 부족 — 스킵")
            continue

        print(f"\n  [{interval}분 진입 / {regime_iv} 레짐] 백테스트 중...")
        r = run_multi_coin_backtest(interval, DEFAULT_PARAMS, data_cache)
        results[interval] = r

        print(f"    거래: {r['total_trades']}회 | 승률: {r['win_rate']}%")
        print(f"    수익: ${r['total_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
        print(f"    최대DD: {r['max_drawdown_pct']:.1f}% | 수수료: ${r['total_fees']:.2f}")
        print(f"    LONG: {r['long_trades']}회 ${r['long_pnl']:+.2f}")
        print(f"    SHORT: {r['short_trades']}회 ${r['short_pnl']:+.2f}")

    # 요약 테이블
    print("\n" + "-" * 70)
    print(f"{'TF':>6} {'거래':>6} {'승률':>7} {'수익률':>9} {'PnL':>10} {'최대DD':>8} {'수수료':>8}")
    print("-" * 70)
    for iv, r in sorted(results.items(), key=lambda x: x[1]["return_pct"], reverse=True):
        tf_name = {"5": "5분", "15": "15분", "60": "1시간", "240": "4시간"}.get(iv, iv)
        print(f"{tf_name:>6} {r['total_trades']:>6} {r['win_rate']:>6.1f}% {r['return_pct']:>+8.1f}% ${r['total_pnl']:>+9.2f} {r['max_drawdown_pct']:>7.1f}% ${r['total_fees']:>7.2f}")

    return results


# ================================================================
# Stage 2: 파라미터 스윕
# ================================================================

def stage2_parameter_sweep(data_cache: Dict, target_intervals: List[str]):
    """핵심 파라미터 그리드 서치"""
    print("\n" + "=" * 70)
    print("STAGE 2: 파라미터 스윕")
    print("=" * 70)

    all_results = []

    for interval in target_intervals:
        print(f"\n--- [{interval}분] 파라미터 스윕 ---")

        # 핵심 3개 파라미터 스윕 (SL, TP, Score)
        combos = list(itertools.product(
            PARAM_GRID["sl_atr_mult"],
            PARAM_GRID["tp_rr_ratio"],
            PARAM_GRID["entry_score_threshold"],
        ))
        print(f"  SL × TP × Score: {len(combos)} 조합")

        best = None
        for idx, (sl, tp, score_th) in enumerate(combos):
            params = DEFAULT_PARAMS.copy()
            params["sl_atr_mult"] = sl
            params["tp_rr_ratio"] = tp
            params["entry_score_threshold"] = score_th

            r = run_multi_coin_backtest(interval, params, data_cache)
            r["params"] = {"sl": sl, "tp": tp, "score": score_th}
            r["interval"] = interval
            all_results.append(r)

            if best is None or r["return_pct"] > best["return_pct"]:
                best = r

            if (idx + 1) % 20 == 0:
                print(f"  진행: {idx+1}/{len(combos)} | 현재 최고: {best['return_pct']:+.1f}%")

        print(f"\n  [SL×TP×Score 최고] SL={best['params']['sl']} TP={best['params']['tp']} "
              f"Score={best['params']['score']}")
        print(f"    수익: {best['return_pct']:+.1f}% | 거래: {best['total_trades']}회 | "
              f"승률: {best['win_rate']}% | DD: {best['max_drawdown_pct']:.1f}%")

        # EMA 조합 스윕
        ema_combos = [(f, s) for f in PARAM_GRID["ema_fast"]
                      for s in PARAM_GRID["ema_slow"] if s > f + 3]
        print(f"\n  EMA 조합: {len(ema_combos)} 조합")

        ema_best = None
        for ef, es in ema_combos:
            params = DEFAULT_PARAMS.copy()
            if best:
                params["sl_atr_mult"] = best["params"]["sl"]
                params["tp_rr_ratio"] = best["params"]["tp"]
                params["entry_score_threshold"] = best["params"]["score"]
            params["ema_fast"] = ef
            params["ema_slow"] = es

            r = run_multi_coin_backtest(interval, params, data_cache)
            r["params"] = {**best["params"], "ema_fast": ef, "ema_slow": es} if best else {"ema_fast": ef, "ema_slow": es}
            r["interval"] = interval
            all_results.append(r)

            if ema_best is None or r["return_pct"] > ema_best["return_pct"]:
                ema_best = r

        if ema_best:
            print(f"  [EMA 최고] EMA{ema_best['params'].get('ema_fast',9)}/{ema_best['params'].get('ema_slow',21)}")
            print(f"    수익: {ema_best['return_pct']:+.1f}% | 거래: {ema_best['total_trades']}회")

        # 풀백 거리 × ADX 스윕
        pb_adx_combos = list(itertools.product(
            PARAM_GRID["pullback_ema_dist_pct"],
            PARAM_GRID["adx_enter_trending"],
        ))
        print(f"\n  풀백 × ADX: {len(pb_adx_combos)} 조합")

        pb_best = None
        for pb, adx in pb_adx_combos:
            params = DEFAULT_PARAMS.copy()
            if ema_best:
                params.update({k: v for k, v in ema_best["params"].items()
                              if k in ("sl_atr_mult", "tp_rr_ratio", "entry_score_threshold",
                                       "ema_fast", "ema_slow")})
            params["pullback_ema_dist_pct"] = pb
            params["adx_enter_trending"] = adx

            r = run_multi_coin_backtest(interval, params, data_cache)
            r["params"] = {**params, "pullback": pb, "adx": adx}
            r["interval"] = interval
            all_results.append(r)

            if pb_best is None or r["return_pct"] > pb_best["return_pct"]:
                pb_best = r

        if pb_best:
            print(f"  [풀백×ADX 최고] 풀백={pb_best['params'].get('pullback',1.0)} "
                  f"ADX={pb_best['params'].get('adx',25)}")
            print(f"    수익: {pb_best['return_pct']:+.1f}% | 거래: {pb_best['total_trades']}회")

    # 전체 Top 20
    all_results.sort(key=lambda x: x["return_pct"], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 20 결과")
    print("=" * 70)
    print(f"{'#':>3} {'TF':>5} {'거래':>5} {'승률':>6} {'수익률':>8} {'PnL':>9} {'DD':>6} {'파라미터'}")
    print("-" * 90)
    for i, r in enumerate(all_results[:20]):
        iv = r.get("interval", "?")
        tf = {"5": "5m", "15": "15m", "60": "1H", "240": "4H"}.get(iv, iv)
        p = r.get("params", {})
        param_str = " ".join(f"{k}={v}" for k, v in p.items())
        print(f"{i+1:>3} {tf:>5} {r['total_trades']:>5} {r['win_rate']:>5.1f}% "
              f"{r['return_pct']:>+7.1f}% ${r['total_pnl']:>+8.2f} "
              f"{r['max_drawdown_pct']:>5.1f}% {param_str}")

    return all_results


# ================================================================
# Stage 3: SOL LONG 포함/제외 비교
# ================================================================

def stage3_sol_long_comparison(data_cache: Dict, best_params: Dict, interval: str):
    """SOL LONG 차단 여부 비교"""
    print("\n" + "=" * 70)
    print("STAGE 3: SOL LONG 포함 vs 제외")
    print("=" * 70)

    r_block = run_multi_coin_backtest(interval, best_params, data_cache, block_sol_long=True)
    r_allow = run_multi_coin_backtest(interval, best_params, data_cache, block_sol_long=False)

    print(f"\n  SOL LONG 차단: {r_block['return_pct']:+.1f}% | "
          f"{r_block['total_trades']}거래 | DD {r_block['max_drawdown_pct']:.1f}%")
    print(f"  SOL LONG 허용: {r_allow['return_pct']:+.1f}% | "
          f"{r_allow['total_trades']}거래 | DD {r_allow['max_drawdown_pct']:.1f}%")


# ================================================================
# 결과 저장
# ================================================================

def save_results(stage1, stage2, filename="comprehensive_results.json"):
    """결과를 JSON으로 저장"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "period": f"{START_DATE} ~ {END_DATE}",
        "initial_balance": INITIAL_BALANCE,
        "stage1_timeframe": {},
        "stage2_top20": [],
    }

    for iv, r in stage1.items():
        r2 = {k: v for k, v in r.items() if k != "coin_results" and k != "trades"}
        output["stage1_timeframe"][iv] = r2

    for r in stage2[:20]:
        r2 = {k: v for k, v in r.items() if k != "coin_results" and k != "trades"}
        output["stage2_top20"].append(r2)

    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {path}")


# ================================================================
# 메인
# ================================================================

def main():
    print("=" * 70)
    print("HERMES 종합 백테스트")
    print(f"기간: {START_DATE} ~ {END_DATE} (2년)")
    print(f"코인: {', '.join(SYMBOLS)}")
    print(f"초기 잔액: ${INITIAL_BALANCE}")
    print("=" * 70)

    # ---- 데이터 다운로드 ----
    print("\n[1/4] 데이터 다운로드...")
    data_cache = {}

    # 모든 필요한 인터벌 목록
    intervals_needed = set()
    for entry_iv in ENTRY_INTERVALS:
        intervals_needed.add(entry_iv)
        regime_iv = REGIME_MAP.get(entry_iv, "240")
        intervals_needed.add(regime_iv)

    for symbol in SYMBOLS:
        for interval in sorted(intervals_needed):
            key = f"{symbol}_{interval}"
            df = load_or_fetch(symbol, interval)
            if len(df) > 0:
                data_cache[key] = df

    print(f"\n  총 {len(data_cache)}개 데이터셋 로드 완료")

    # ---- Stage 1: 타임프레임 비교 ----
    print("\n[2/4] Stage 1: 타임프레임 비교...")
    stage1 = stage1_timeframe_comparison(data_cache)

    # 상위 2개 타임프레임 선정
    ranked = sorted(stage1.items(), key=lambda x: x[1]["return_pct"], reverse=True)
    target_intervals = [iv for iv, _ in ranked[:2] if ranked[0][1]["total_trades"] > 10]
    if not target_intervals:
        target_intervals = ["60"]  # 최소 1H는 진행

    print(f"\n  → Stage 2 대상 타임프레임: {target_intervals}")

    # ---- Stage 2: 파라미터 스윕 ----
    print("\n[3/4] Stage 2: 파라미터 스윕...")
    stage2 = stage2_parameter_sweep(data_cache, target_intervals)

    # ---- Stage 3: SOL LONG 비교 ----
    if stage2:
        best = stage2[0]
        best_iv = best.get("interval", "60")
        best_params = DEFAULT_PARAMS.copy()
        for k, v in best.get("params", {}).items():
            if k in DEFAULT_PARAMS:
                best_params[k] = v
        print("\n[4/4] Stage 3: SOL LONG 비교...")
        stage3_sol_long_comparison(data_cache, best_params, best_iv)

    # ---- 결과 저장 ----
    save_results(stage1, stage2)

    print("\n" + "=" * 70)
    print("백테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
