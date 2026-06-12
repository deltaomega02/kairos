#!/usr/bin/env python3
"""
백테스팅 v3.0 엔진
==================
확장 기능:
1. 펀딩 레이트 반영 (점수)
2. 시간대 필터 (아시아/유럽/미국)
3. 15분봉 멀티TF 확인
4. 코인별 파라미터
5. 동적 리스크 (연패 시 축소)
6. 트레일링 스탑 (v2에서 검증된 3%/0.5%)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
    fee_adjusted_sl_tp, TAKER_FEE_PCT, DEFAULT_PARAMS,
    INITIAL_BALANCE, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_DAILY_TRADES, MAX_SIMULTANEOUS,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ================================================================
# 데이터 로드
# ================================================================

def load_all_data():
    """모든 데이터 로드 (1H, 4H, 15m, 펀딩)"""
    data = {}
    for sym in SYMBOLS:
        for iv in ["15", "60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)

        # 펀딩
        f = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        if os.path.exists(f):
            data[f"{sym}_funding"] = pd.read_csv(f)
    return data


# ================================================================
# 확장 지표 계산
# ================================================================

def add_bb_width(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """볼린저 밴드 폭 (변동성 지표)"""
    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    df["bb_width"] = (upper - lower) / sma * 100
    return df


def add_atr_roc(df: pd.DataFrame, period: int = 14, roc_period: int = 10) -> pd.DataFrame:
    """ATR 변화율"""
    df = df.copy()
    if "atr" not in df.columns:
        prev_c = df["close"].shift(1)
        tr = pd.concat([df["high"]-df["low"], abs(df["high"]-prev_c), abs(df["low"]-prev_c)], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
    df["atr_roc"] = df["atr"].pct_change(roc_period) * 100
    return df


def add_stoch(df: pd.DataFrame, k_period: int = 14) -> pd.DataFrame:
    """스토캐스틱 %K"""
    df = df.copy()
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    df["stoch_k"] = (df["close"] - low_min) / (high_max - low_min + 1e-10) * 100
    return df


# ================================================================
# 펀딩 레이트 매핑
# ================================================================

def align_funding_to_entry(funding_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.Series:
    """펀딩 레이트를 진입 TF 타임스탬프에 매핑"""
    if funding_df.empty:
        return pd.Series([0.0] * len(entry_df), index=entry_df.index)

    f_ts = funding_df["timestamp"].values
    f_rates = funding_df["funding_rate"].values

    result = []
    idx = 0
    for ts in entry_df["timestamp"]:
        while idx < len(f_ts) - 1 and f_ts[idx + 1] <= ts:
            idx += 1
        if idx < len(f_rates) and f_ts[idx] <= ts:
            result.append(f_rates[idx])
        else:
            result.append(0.0)

    return pd.Series(result, index=entry_df.index)


# ================================================================
# 15분봉 멀티TF 확인
# ================================================================

def get_15m_confirm(entry_df_1h: pd.DataFrame, df_15: pd.DataFrame) -> pd.Series:
    """15분봉 RSI가 과매도/과매수인지 매핑"""
    # 15분봉에 RSI 추가
    df_15 = df_15.copy()
    delta = df_15["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_15["rsi_15"] = 100 - (100 / (1 + rs))

    # 1H 타임스탬프에 매핑 (1H 직전 15분봉 값)
    ts_15 = df_15["timestamp"].values
    rsi_15_vals = df_15["rsi_15"].values

    result = []
    idx = 0
    for ts in entry_df_1h["timestamp"]:
        while idx < len(ts_15) - 1 and ts_15[idx + 1] <= ts:
            idx += 1
        if idx < len(rsi_15_vals):
            result.append(rsi_15_vals[idx])
        else:
            result.append(50.0)

    return pd.Series(result, index=entry_df_1h.index)


# ================================================================
# 시그널 평가 (확장판)
# ================================================================

def evaluate_signal_v3(
    regime: str,
    row: Dict,
    params: Dict,
    funding_rate: float = 0,
    rsi_15: float = 50,
    use_funding: bool = True,
    use_mtf: bool = False,
    hour: int = 0,
    session_filter: Optional[List[int]] = None,
) -> Optional[Dict]:
    """v3 시그널 평가 — 펀딩/MTF/시간 필터"""

    if regime not in ("TRENDING_UP", "TRENDING_DOWN"):
        return None

    # 시간대 필터
    if session_filter is not None and hour not in session_filter:
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

    if direction == "LONG":
        if ema_f <= ema_s:
            return None
        dist_pct = (ema_f - close) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None
    else:
        if ema_f >= ema_s:
            return None
        dist_pct = (close - ema_f) / ema_f * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None

    score = 50

    # RSI
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

    # 펀딩 반영
    if use_funding:
        funding_thresh = params.get("funding_bias_threshold", 0.0005)
        if abs(funding_rate) >= funding_thresh:
            # 숏 펀딩(양수) = LONG 불리 / 숏 유리
            if (direction == "LONG" and funding_rate < 0) or (direction == "SHORT" and funding_rate > 0):
                score += 15  # 순방향
            else:
                score -= 10  # 역방향

    # 15분봉 MTF 확인
    if use_mtf:
        if direction == "LONG" and rsi_15 < 30:
            score += 10  # 15분 과매도에서 LONG → 과매도 반등
        elif direction == "SHORT" and rsi_15 > 70:
            score += 10

    if score < params["entry_score_threshold"]:
        return None

    sl_pct, tp_pct = fee_adjusted_sl_tp(atr_pct, params["sl_atr_mult"], params["tp_rr_ratio"])
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
# 백테스트 엔진 v3
# ================================================================

def run_backtest_v3(
    entry_df: pd.DataFrame,
    regime_series: pd.Series,
    funding_series: pd.Series,
    rsi_15_series: pd.Series,
    params: Dict,
    symbol: str = "",
    block_sol_long: bool = True,
    use_funding: bool = True,
    use_mtf: bool = False,
    session_filter: Optional[List[int]] = None,
    use_dynamic_risk: bool = False,
    trailing_activation: float = 3.0,
    trailing_distance: float = 0.5,
    use_trailing: bool = True,
) -> List[Dict]:
    """v3 백테스트 실행"""

    balance = INITIAL_BALANCE
    trades = []
    position = None
    daily_trades = {}
    consecutive_losses = 0

    closes = entry_df["close"].values
    highs = entry_df["high"].values
    lows = entry_df["low"].values
    ema_f = entry_df["ema_fast"].values
    ema_s = entry_df["ema_slow"].values
    rsis = entry_df["rsi"].values
    atr_pcts = entry_df["atr_pct"].values
    vol_ratios = entry_df["volume_ratio"].values
    timestamps = entry_df["timestamp"].values
    regimes = regime_series.values if hasattr(regime_series, 'values') else np.array(regime_series)
    fundings = funding_series.values if hasattr(funding_series, 'values') else np.array(funding_series)
    rsi_15s = rsi_15_series.values if hasattr(rsi_15_series, 'values') else np.array(rsi_15_series)

    n = len(entry_df)

    for i in range(50, n):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        ts = timestamps[i]
        day_key = int(ts // 86400000)
        hour = datetime.fromtimestamp(ts / 1000).hour

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

            if d == "LONG":
                if high >= tp:
                    exit_price = tp
                    reason = "TP"
                elif low <= sl:
                    exit_price = sl
                    reason = "SL"
                elif use_trailing:
                    new_peak = max(peak, high)
                    pnl_pct_peak = (new_peak - ep) / ep * 100
                    if not trailing_active and pnl_pct_peak >= trailing_activation:
                        trailing_active = True
                    if trailing_active:
                        new_trail_sl = new_peak * (1 - trailing_distance / 100)
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
                    if not trailing_active and pnl_pct_peak >= trailing_activation:
                        trailing_active = True
                    if trailing_active:
                        new_trail_sl = new_peak * (1 + trailing_distance / 100)
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

                if net_pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                trades.append({
                    "symbol": symbol, "direction": d,
                    "entry_price": ep, "exit_price": exit_price,
                    "quantity": qty, "pnl": net_pnl,
                    "fee": fee, "reason": reason,
                    "timestamp": ts,
                })
                position = None
            continue

        # --- 신규 진입 ---
        if balance <= 10:
            continue

        dt = daily_trades.get(day_key, 0)
        if dt >= MAX_DAILY_TRADES:
            continue

        regime = regimes[i] if i < len(regimes) else "RANGING"
        funding = fundings[i] if i < len(fundings) else 0
        rsi_15 = rsi_15s[i] if i < len(rsi_15s) else 50

        row = {
            "close": close, "ema_fast": ema_f[i], "ema_slow": ema_s[i],
            "rsi": rsis[i], "atr_pct": atr_pcts[i], "volume_ratio": vol_ratios[i],
        }

        if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
            continue

        signal = evaluate_signal_v3(
            regime, row, params,
            funding_rate=funding, rsi_15=rsi_15,
            use_funding=use_funding, use_mtf=use_mtf,
            hour=hour, session_filter=session_filter,
        )
        if signal is None:
            continue

        if block_sol_long and symbol == "SOLUSDT" and signal["direction"] == "LONG":
            continue

        # 동적 리스크
        risk_mult = 1.0
        if use_dynamic_risk:
            if consecutive_losses >= 3:
                risk_mult = 0.5
            elif consecutive_losses >= 2:
                risk_mult = 0.75

        entry_price = signal["entry_price"]
        sl_pct = signal["sl_pct"]
        tp_pct = signal["tp_pct"]

        risk_amt = balance * RISK_PER_TRADE * risk_mult
        sl_ratio = sl_pct / 100.0
        avail_margin = balance * MARGIN_USAGE / MAX_SIMULTANEOUS * risk_mult
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

    return trades


# ================================================================
# 멀티코인 실행
# ================================================================

def run_multi_v3(params: Dict, data: Dict, **kwargs) -> Dict:
    """멀티코인 합산 v3"""
    all_trades = []
    coin_stats = {}

    for sym in SYMBOLS:
        if f"{sym}_60" not in data:
            continue

        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), params)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(params)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)

        # 펀딩 매핑
        funding_df = data.get(f"{sym}_funding", pd.DataFrame())
        fm = align_funding_to_entry(funding_df, entry_df)

        # 15분봉 RSI
        df_15 = data.get(f"{sym}_15", pd.DataFrame())
        if not df_15.empty:
            rsi_15_series = get_15m_confirm(entry_df, df_15)
        else:
            rsi_15_series = pd.Series([50.0] * len(entry_df), index=entry_df.index)

        trades = run_backtest_v3(
            entry_df, rm, fm, rsi_15_series,
            params, symbol=sym, **kwargs,
        )
        all_trades.extend(trades)
        coin_stats[sym] = {
            "trades": len(trades),
            "pnl": sum(t["pnl"] for t in trades),
            "wins": sum(1 for t in trades if t["pnl"] > 0),
        }

    all_trades.sort(key=lambda t: t["timestamp"])
    total = len(all_trades)
    if total == 0:
        return {"total": 0, "return_pct": 0, "max_dd": 0, "win_rate": 0, "pnl": 0, "coin_stats": coin_stats}

    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)

    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in all_trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

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

    return {
        "total": total,
        "wins": wins,
        "win_rate": round(wins/total*100, 1),
        "pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl/INITIAL_BALANCE*100, 1),
        "max_dd": round(max_dd, 1),
        "coin_stats": coin_stats,
        "yearly": yearly,
    }
