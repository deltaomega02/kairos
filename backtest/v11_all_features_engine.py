#!/usr/bin/env python3
"""
v11 All-Features Engine (Opus 4.7) — 남아있는 모든 개선 아이디어 통합
======================================================================

Features (all optional via kwargs):
  [v10 기존 — 통합]
  - triple_ema_enable, d1_filter_enable, h4_long_ema_enable,
    ribbon_enable, per_direction_enable

  [v11 신규]
  1. atr_adaptive_sl: ATR < 임계값 시 sl_atr_mult 확대
     atr_low_threshold (기본 0.8), atr_low_sl_mult (기본 2.0)

  2. correlation_filter: BTC 최근 N시간 변동 > X% 시 알트 반대방향 차단
     corr_lookback_hours (기본 4), corr_threshold_pct (기본 2.0)

  3. time_filter: 특정 시간대 진입 차단
     block_hours (기본 [0, 8, 16] - 펀딩 전)
     time_filter_before_min, time_filter_after_min

  4. asymmetric_pullback: LONG/SHORT 다른 pullback_ema_dist
     long_pullback_dist (기본 None = 기존 값), short_pullback_dist

  5. partial_tp: TP 중간지점에서 50% 청산, 나머지 계속
     partial_tp_enable, partial_tp_ratio (기본 0.5 = 중간)

  6. d1_mode: "direction" | "price_above_ema" | "sma" | "adx" | "multi"
     d1_adx_enable: 1D ADX > N 필수
     d1_adx_threshold (기본 20)

  7. d1_macd_confirm: 1D MACD 히스토그램 방향도 일치해야
     d1_macd_enable

  8. volatility_regime_size: 고변동 시 포지션 축소
     vol_regime_size_pct (기본 0.5)

이전 엔진 (v4, v10) 기반 확장.
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import SYMBOLS, align_funding_to_entry
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
    fee_adjusted_sl_tp,
    TAKER_FEE_PCT, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_SIMULTANEOUS, MAX_DAILY_TRADES,
)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _resample_1h_to_1d(entry_df_1h: pd.DataFrame) -> pd.DataFrame:
    df = entry_df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    d = df[["open", "high", "low", "close"]].resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    d["timestamp"] = d.index.astype("int64") // 10**6
    d = d.reset_index(drop=True)
    return d


def _compute_adx(high, low, close, period=14):
    """간단 ADX 계산"""
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(period).mean()


def _compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    sig = _ema(macd, signal)
    return macd - sig


def prepare_symbol_v11(data: Dict, sym: str, params: Dict, **kw):
    if f"{sym}_60" not in data:
        return None
    entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), params)
    regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
    re = BacktestRegimeEngine(params)
    regimes = [re.update(row) for _, row in regime_df.iterrows()]
    regime_df["regime"] = regimes
    rm = align_regime_to_entry(regime_df, entry_df)

    funding_df = data.get(f"{sym}_funding", pd.DataFrame())
    fm = align_funding_to_entry(funding_df, entry_df)

    close_series = entry_df["close"]

    # Multi-EMA (v10 전달)
    ema_medium = None
    if kw.get("triple_ema_enable"):
        ema_medium = _ema(close_series, kw.get("ema_medium_period", 8)).values

    # Per-direction EMAs
    pd_long_fast = pd_long_slow = pd_short_fast = pd_short_slow = None
    if kw.get("per_direction_enable"):
        pd_long_fast = _ema(close_series, kw.get("ema_fast_long", 3)).values
        pd_long_slow = _ema(close_series, kw.get("ema_slow_long", 15)).values
        pd_short_fast = _ema(close_series, kw.get("ema_fast_short", 3)).values
        pd_short_slow = _ema(close_series, kw.get("ema_slow_short", 15)).values

    # 1D derivations (EMA, SMA, ADX, MACD)
    d1_data = None
    if (kw.get("d1_filter_enable") or kw.get("d1_adx_enable")
            or kw.get("d1_macd_enable")):
        d1_df = _resample_1h_to_1d(entry_df)
        period = kw.get("d1_ema_period", 10)
        if kw.get("d1_mode", "direction") == "sma":
            d1_df["indicator"] = _sma(d1_df["close"], period)
        else:
            d1_df["indicator"] = _ema(d1_df["close"], period)
        if kw.get("d1_adx_enable"):
            d1_df["adx"] = _compute_adx(d1_df["high"], d1_df["low"], d1_df["close"],
                                         kw.get("d1_adx_period", 14))
        if kw.get("d1_macd_enable"):
            d1_df["macd_hist"] = _compute_macd_hist(d1_df["close"])
        d1_data = {
            "ts": d1_df["timestamp"].values.astype(np.int64),
            "indicator": d1_df["indicator"].values,
            "close": d1_df["close"].values,
            "adx": d1_df.get("adx", pd.Series(dtype=float)).values if "adx" in d1_df.columns else None,
            "macd_hist": d1_df.get("macd_hist", pd.Series(dtype=float)).values if "macd_hist" in d1_df.columns else None,
        }

    # 4H long EMA
    h4_long_data = None
    if kw.get("h4_long_ema_enable"):
        period = kw.get("h4_long_ema_period", 50)
        h4_long_ema = _ema(regime_df["close"], period).values
        h4_long_data = {
            "ts": regime_df["timestamp"].values.astype(np.int64),
            "long_ema": h4_long_ema,
            "close": regime_df["close"].values,
        }

    # Ribbon
    ribbon_vals = {}
    if kw.get("ribbon_enable"):
        periods = kw.get("ribbon_periods") or [3, 5, 8, 13, 21, 34]
        for p in periods:
            ribbon_vals[p] = _ema(close_series, p).values

    return {
        "df": entry_df,
        "regime": rm.values if hasattr(rm, "values") else rm,
        "funding": fm.values if hasattr(fm, "values") else fm,
        "timestamp": entry_df["timestamp"].values,
        "close": entry_df["close"].values,
        "high": entry_df["high"].values,
        "low": entry_df["low"].values,
        "ema_fast": entry_df["ema_fast"].values,
        "ema_slow": entry_df["ema_slow"].values,
        "rsi": entry_df["rsi"].values,
        "atr_pct": entry_df["atr_pct"].values,
        "volume_ratio": entry_df["volume_ratio"].values,
        "ema_medium": ema_medium,
        "ribbon_vals": ribbon_vals,
        "pd_long_fast": pd_long_fast, "pd_long_slow": pd_long_slow,
        "pd_short_fast": pd_short_fast, "pd_short_slow": pd_short_slow,
        "d1_data": d1_data,
        "h4_long_data": h4_long_data,
    }


def _find_le_idx(ts_arr, t):
    return max(0, min(np.searchsorted(ts_arr, t, side="right") - 1, len(ts_arr) - 1))


def _evaluate_signal_v11(regime, row, params, funding_rate, use_funding, hour,
                         **filters):
    """v11 signal eval — 모든 필터 지원"""
    if regime not in ("TRENDING_UP", "TRENDING_DOWN"):
        return None

    direction = "LONG" if regime == "TRENDING_UP" else "SHORT"

    # Time-of-day filter
    if filters.get("time_filter_enable"):
        block_hours = filters.get("block_hours", [0, 8, 16])
        if hour in block_hours:
            return None

    # Per-direction EMA
    if filters.get("per_direction_enable"):
        if direction == "LONG":
            ema_f_use = filters["pd_fast_long"]
            ema_s_use = filters["pd_slow_long"]
        else:
            ema_f_use = filters["pd_fast_short"]
            ema_s_use = filters["pd_slow_short"]
    else:
        ema_f_use = row["ema_fast"]
        ema_s_use = row["ema_slow"]

    close = row["close"]
    rsi = row["rsi"]
    vol = row["volume_ratio"]
    atr_pct = row["atr_pct"]

    if ema_f_use == 0 or ema_s_use == 0 or close == 0:
        return None

    # Asymmetric pullback
    if filters.get("asym_pullback_enable"):
        max_dist = (filters["long_pullback_dist"] if direction == "LONG"
                    else filters["short_pullback_dist"])
    else:
        max_dist = params["pullback_ema_dist_pct"]

    # Pullback check
    if direction == "LONG":
        if ema_f_use <= ema_s_use:
            return None
        dist_pct = (ema_f_use - close) / ema_f_use * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None
    else:
        if ema_f_use >= ema_s_use:
            return None
        dist_pct = (close - ema_f_use) / ema_f_use * 100
        if dist_pct < -0.1 or dist_pct > max_dist:
            return None

    # Triple EMA
    if filters.get("triple_ema_enable"):
        medium = filters.get("ema_medium_val", 0)
        if medium <= 0:
            return None
        if direction == "LONG" and not (ema_f_use > medium > ema_s_use):
            return None
        if direction == "SHORT" and not (ema_f_use < medium < ema_s_use):
            return None

    # Ribbon
    if filters.get("ribbon_enable"):
        ribbon_cur = filters.get("ribbon_current")
        if ribbon_cur is None or len(ribbon_cur) == 0:
            return None
        sorted_periods = sorted(ribbon_cur.keys())
        vals = [ribbon_cur[p] for p in sorted_periods]
        if direction == "LONG":
            if not all(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
                return None
        else:
            if not all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
                return None

    # 1D filter
    if filters.get("d1_filter_enable"):
        d1_pass = filters.get("d1_pass", {}).get(direction, False)
        if not d1_pass:
            return None

    # 1D ADX filter
    if filters.get("d1_adx_enable"):
        if not filters.get("d1_adx_pass", False):
            return None

    # 1D MACD confirmation
    if filters.get("d1_macd_enable"):
        d1_macd_pass = filters.get("d1_macd_pass", {}).get(direction, False)
        if not d1_macd_pass:
            return None

    # 4H long EMA
    if filters.get("h4_long_ema_enable"):
        h4_pass = filters.get("h4_pass", {}).get(direction, False)
        if not h4_pass:
            return None

    # Correlation filter (alt만 적용, BTC은 스킵)
    if filters.get("corr_filter_enable") and filters.get("is_alt", False):
        btc_move_pct = filters.get("btc_recent_move_pct", 0)
        threshold = filters.get("corr_threshold_pct", 2.0)
        # BTC 하락 크면 알트 LONG 차단, BTC 상승 크면 알트 SHORT 차단
        if direction == "LONG" and btc_move_pct < -threshold:
            return None
        if direction == "SHORT" and btc_move_pct > threshold:
            return None

    # Score (기본 v3)
    score = 50
    if direction == "LONG" and rsi <= params.get("rsi_oversold", 35):
        score += 20
    elif direction == "SHORT" and rsi >= params.get("rsi_overbought", 65):
        score += 20
    elif (direction == "LONG" and rsi < 50) or (direction == "SHORT" and rsi > 50):
        score += 10

    if vol >= 1.3:
        score += 15
    elif vol >= 1.0:
        score += 5

    if use_funding:
        ft = params.get("funding_bias_threshold", 0.0005)
        if abs(funding_rate) >= ft:
            if (direction == "LONG" and funding_rate < 0) or \
               (direction == "SHORT" and funding_rate > 0):
                score += 15
            else:
                score -= 10

    if score < params["entry_score_threshold"]:
        return None

    # SL/TP with ATR adaptive
    sl_mult = params["sl_atr_mult"]
    if filters.get("atr_adaptive_sl"):
        atr_low_thr = filters.get("atr_low_threshold", 0.8)
        atr_low_mult = filters.get("atr_low_sl_mult", 2.0)
        if atr_pct < atr_low_thr:
            sl_mult = atr_low_mult

    sl_pct, tp_pct = fee_adjusted_sl_tp(atr_pct, sl_mult, params["tp_rr_ratio"])
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


def run_shared_backtest_v11(
    data, params, seed,
    *,
    use_funding=True,
    trailing_activation=1.2, trailing_distance=0.1,
    block_sol_long=True,
    skip_years=(),
    daily_cost_usd=0.0,
    ruin_threshold=15.0,
    use_cooldown=False,
    slippage_pct=0.05,
    max_simultaneous=None, risk_per_trade=None, max_leverage=None,
    max_daily_trades=None,
    enabled_symbols=None, blocked_directions=None,
    # v10 multi-EMA
    triple_ema_enable=False, ema_medium_period=8,
    d1_filter_enable=False, d1_ema_period=10,
    d1_mode="direction",
    h4_long_ema_enable=False, h4_long_ema_period=50, h4_filter_mode="direction",
    ribbon_enable=False, ribbon_periods=None,
    per_direction_enable=False,
    ema_fast_long=3, ema_slow_long=15, ema_fast_short=3, ema_slow_short=15,
    # v11 new
    atr_adaptive_sl=False, atr_low_threshold=0.8, atr_low_sl_mult=2.0,
    corr_filter_enable=False, corr_lookback_hours=4, corr_threshold_pct=2.0,
    time_filter_enable=False, block_hours=None,
    asym_pullback_enable=False, long_pullback_dist=1.5, short_pullback_dist=1.5,
    partial_tp_enable=False, partial_tp_ratio=0.5, partial_tp_pct=0.5,
    d1_adx_enable=False, d1_adx_threshold=20, d1_adx_period=14,
    d1_macd_enable=False,
    vol_regime_size_enable=False, vol_regime_size_pct=0.5,
):
    if max_simultaneous is None: max_simultaneous = MAX_SIMULTANEOUS
    if risk_per_trade is None: risk_per_trade = RISK_PER_TRADE
    if max_leverage is None: max_leverage = MAX_LEVERAGE
    if max_daily_trades is None: max_daily_trades = MAX_DAILY_TRADES
    if enabled_symbols is None: enabled_symbols = SYMBOLS
    if blocked_directions is None: blocked_directions = {}
    if block_hours is None: block_hours = [0, 8, 16]

    prep_kw = dict(
        triple_ema_enable=triple_ema_enable, ema_medium_period=ema_medium_period,
        d1_filter_enable=d1_filter_enable, d1_ema_period=d1_ema_period,
        d1_mode=d1_mode,
        d1_adx_enable=d1_adx_enable, d1_adx_period=d1_adx_period,
        d1_macd_enable=d1_macd_enable,
        h4_long_ema_enable=h4_long_ema_enable, h4_long_ema_period=h4_long_ema_period,
        ribbon_enable=ribbon_enable, ribbon_periods=ribbon_periods,
        per_direction_enable=per_direction_enable,
        ema_fast_long=ema_fast_long, ema_slow_long=ema_slow_long,
        ema_fast_short=ema_fast_short, ema_slow_short=ema_slow_short,
    )

    per_sym = {}
    for sym in enabled_symbols:
        p = prepare_symbol_v11(data, sym, params, **prep_kw)
        if p is not None:
            per_sym[sym] = p
    if not per_sym:
        raise RuntimeError("활성 심볼 없음")

    master_sym = "BTCUSDT" if "BTCUSDT" in per_sym else list(per_sym.keys())[0]
    master_ts = per_sym[master_sym]["timestamp"]
    n = len(master_ts)
    sym_ts_map = {sym: {int(t): i for i, t in enumerate(s["timestamp"])}
                  for sym, s in per_sym.items()}

    balance = seed
    peak = seed
    max_dd = 0.0
    positions: Dict[str, Dict] = {}
    daily_trades: Dict[int, int] = {}
    trades: List[Dict] = []
    ruined = False
    ruin_ts = None
    last_day: Optional[int] = None

    # BTC close array for correlation filter
    btc_close_arr = per_sym.get("BTCUSDT", {}).get("close")
    btc_ts_arr = per_sym.get("BTCUSDT", {}).get("timestamp")

    for i in range(50, n):
        ts = int(master_ts[i])
        day_key = ts // 86400000
        dt_utc = datetime.utcfromtimestamp(ts / 1000)
        year = dt_utc.year
        hour = dt_utc.hour

        if last_day is None:
            last_day = day_key
        elif day_key > last_day:
            days_passed = day_key - last_day
            balance -= days_passed * daily_cost_usd
            last_day = day_key

        if ruined:
            continue
        if balance < ruin_threshold:
            ruined = True
            ruin_ts = ts
            continue

        # Close positions
        for sym in list(positions.keys()):
            pos = positions[sym]
            s = per_sym.get(sym)
            if s is None:
                continue
            idx = sym_ts_map[sym].get(ts)
            if idx is None:
                continue

            high = s["high"][idx]
            low = s["low"][idx]
            ep = pos["entry_price"]
            d = pos["direction"]
            sl = pos["sl_price"]
            tp = pos["tp_price"]
            qty = pos["quantity"]
            peak_p = pos["peak_price"]
            trailing_active = pos["trailing_active"]
            partial_done = pos.get("partial_done", False)
            partial_tp_price = pos.get("partial_tp_price", 0)
            partial_qty = pos.get("partial_qty", 0)

            # Partial TP check (v11)
            if partial_tp_enable and not partial_done and partial_tp_price > 0:
                hit_partial = (d == "LONG" and high >= partial_tp_price) or \
                              (d == "SHORT" and low <= partial_tp_price)
                if hit_partial:
                    # 50% 청산
                    slip = slippage_pct / 100.0
                    adj_exit = partial_tp_price * ((1 - slip) if d == "LONG"
                                                    else (1 + slip))
                    if d == "LONG":
                        raw_pnl = (adj_exit - ep) * partial_qty
                    else:
                        raw_pnl = (ep - adj_exit) * partial_qty
                    fee = (ep * partial_qty + partial_tp_price * partial_qty) * TAKER_FEE_PCT
                    net_pnl = raw_pnl - fee
                    balance += net_pnl
                    trades.append({
                        "symbol": sym, "direction": d, "entry_price": ep,
                        "exit_price": partial_tp_price, "quantity": partial_qty,
                        "pnl": round(net_pnl, 4), "fee": round(fee, 4),
                        "reason": "PARTIAL_TP", "timestamp": ts, "year": year,
                    })
                    pos["quantity"] -= partial_qty
                    pos["partial_done"] = True
                    qty = pos["quantity"]

            exit_price = None
            reason = None
            if d == "LONG":
                if high >= tp:
                    exit_price = tp; reason = "TP"
                elif low <= sl:
                    exit_price = sl; reason = "SL"
                else:
                    new_peak = max(peak_p, high)
                    pnl_pct_peak = (new_peak - ep) / ep * 100
                    if not trailing_active and pnl_pct_peak >= trailing_activation:
                        trailing_active = True
                    if trailing_active:
                        new_trail_sl = new_peak * (1 - trailing_distance / 100)
                        if new_trail_sl > sl:
                            sl = new_trail_sl
                        if low <= sl:
                            exit_price = sl; reason = "TRAILING"
                    pos["peak_price"] = new_peak
                    pos["trailing_active"] = trailing_active
                    pos["sl_price"] = sl
            else:
                if low <= tp:
                    exit_price = tp; reason = "TP"
                elif high >= sl:
                    exit_price = sl; reason = "SL"
                else:
                    new_peak = min(peak_p, low)
                    pnl_pct_peak = (ep - new_peak) / ep * 100
                    if not trailing_active and pnl_pct_peak >= trailing_activation:
                        trailing_active = True
                    if trailing_active:
                        new_trail_sl = new_peak * (1 + trailing_distance / 100)
                        if new_trail_sl < sl:
                            sl = new_trail_sl
                        if high >= sl:
                            exit_price = sl; reason = "TRAILING"
                    pos["peak_price"] = new_peak
                    pos["trailing_active"] = trailing_active
                    pos["sl_price"] = sl

            if exit_price is not None:
                slip = slippage_pct / 100.0
                if d == "LONG":
                    adj_exit = exit_price * (1 - slip)
                    raw_pnl = (adj_exit - ep) * qty
                else:
                    adj_exit = exit_price * (1 + slip)
                    raw_pnl = (ep - adj_exit) * qty
                fee = (ep * qty + exit_price * qty) * TAKER_FEE_PCT
                net_pnl = raw_pnl - fee
                balance += net_pnl
                # Enriched trade record (edge hunt용)
                ent_idx = pos.get("entry_idx")
                s_ = per_sym.get(sym)
                ent_regime = s_["regime"][ent_idx] if (s_ is not None and ent_idx is not None and ent_idx < len(s_["regime"])) else None
                trades.append({
                    "symbol": sym, "direction": d, "entry_price": ep,
                    "exit_price": exit_price, "quantity": qty,
                    "pnl": round(net_pnl, 4), "fee": round(fee, 4),
                    "reason": reason, "timestamp": ts, "year": year,
                    "entry_ts": pos.get("entry_ts"),
                    "entry_rsi": pos.get("entry_rsi"),
                    "entry_atr_pct": pos.get("entry_atr_pct"),
                    "entry_vol_ratio": pos.get("entry_vol_ratio"),
                    "entry_funding": pos.get("entry_funding"),
                    "entry_regime": str(ent_regime) if ent_regime is not None else None,
                })
                del positions[sym]

        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        if balance < ruin_threshold:
            ruined = True
            ruin_ts = ts
            continue

        if year in skip_years:
            continue
        if len(positions) >= max_simultaneous:
            continue
        if daily_trades.get(day_key, 0) >= max_daily_trades:
            continue

        # Compute BTC recent move for correlation (once per timestep)
        btc_recent_move = 0
        if corr_filter_enable and btc_close_arr is not None:
            btc_idx = sym_ts_map.get("BTCUSDT", {}).get(ts)
            if btc_idx is not None and btc_idx >= corr_lookback_hours:
                curr_btc = btc_close_arr[btc_idx]
                past_btc = btc_close_arr[btc_idx - corr_lookback_hours]
                if past_btc > 0:
                    btc_recent_move = (curr_btc - past_btc) / past_btc * 100

        for sym in enabled_symbols:
            if sym in positions:
                continue
            if len(positions) >= max_simultaneous:
                break
            s = per_sym.get(sym)
            if s is None:
                continue
            idx = sym_ts_map[sym].get(ts)
            if idx is None:
                continue

            regime = s["regime"][idx] if idx < len(s["regime"]) else "RANGING"
            funding = s["funding"][idx] if idx < len(s["funding"]) else 0

            row = {
                "close": s["close"][idx],
                "ema_fast": s["ema_fast"][idx], "ema_slow": s["ema_slow"][idx],
                "rsi": s["rsi"][idx], "atr_pct": s["atr_pct"][idx],
                "volume_ratio": s["volume_ratio"][idx],
            }

            # Assemble filter context
            fctx = {
                "time_filter_enable": time_filter_enable,
                "block_hours": block_hours,
                "per_direction_enable": per_direction_enable,
                "pd_fast_long": s["pd_long_fast"][idx] if s["pd_long_fast"] is not None else 0,
                "pd_slow_long": s["pd_long_slow"][idx] if s["pd_long_slow"] is not None else 0,
                "pd_fast_short": s["pd_short_fast"][idx] if s["pd_short_fast"] is not None else 0,
                "pd_slow_short": s["pd_short_slow"][idx] if s["pd_short_slow"] is not None else 0,
                "triple_ema_enable": triple_ema_enable,
                "ema_medium_val": s["ema_medium"][idx] if s["ema_medium"] is not None else 0,
                "ribbon_enable": ribbon_enable,
                "ribbon_current": {p: v[idx] for p, v in s["ribbon_vals"].items()
                                    if idx < len(v) and not np.isnan(v[idx])} if s["ribbon_vals"] else None,
                "asym_pullback_enable": asym_pullback_enable,
                "long_pullback_dist": long_pullback_dist,
                "short_pullback_dist": short_pullback_dist,
                "atr_adaptive_sl": atr_adaptive_sl,
                "atr_low_threshold": atr_low_threshold,
                "atr_low_sl_mult": atr_low_sl_mult,
                "corr_filter_enable": corr_filter_enable,
                "is_alt": sym != "BTCUSDT",
                "btc_recent_move_pct": btc_recent_move,
                "corr_threshold_pct": corr_threshold_pct,
            }

            # 1D filters
            d1_pass_long = d1_pass_short = True
            d1_adx_pass = True
            d1_macd_pass_long = d1_macd_pass_short = True
            if (d1_filter_enable or d1_adx_enable or d1_macd_enable) and s["d1_data"]:
                d1d = s["d1_data"]
                d1_idx = _find_le_idx(d1d["ts"], ts)
                if d1_filter_enable:
                    ind_cur = d1d["indicator"][d1_idx] if d1_idx < len(d1d["indicator"]) else 0
                    ind_prev = d1d["indicator"][d1_idx - 1] if d1_idx > 0 else ind_cur
                    close_cur = d1d["close"][d1_idx]
                    if np.isnan(ind_cur) or ind_cur == 0:
                        d1_pass_long = d1_pass_short = False
                    else:
                        if d1_mode in ("direction", "sma"):
                            if not np.isnan(ind_prev):
                                d1_pass_long = ind_cur > ind_prev
                                d1_pass_short = ind_cur < ind_prev
                        elif d1_mode == "price_above_ema":
                            d1_pass_long = close_cur > ind_cur
                            d1_pass_short = close_cur < ind_cur
                if d1_adx_enable and d1d["adx"] is not None and d1_idx < len(d1d["adx"]):
                    adx_val = d1d["adx"][d1_idx]
                    d1_adx_pass = (not np.isnan(adx_val)) and adx_val >= d1_adx_threshold
                if d1_macd_enable and d1d["macd_hist"] is not None and d1_idx < len(d1d["macd_hist"]):
                    hist = d1d["macd_hist"][d1_idx]
                    if not np.isnan(hist):
                        d1_macd_pass_long = hist > 0
                        d1_macd_pass_short = hist < 0

            fctx["d1_filter_enable"] = d1_filter_enable
            fctx["d1_pass"] = {"LONG": d1_pass_long, "SHORT": d1_pass_short}
            fctx["d1_adx_enable"] = d1_adx_enable
            fctx["d1_adx_pass"] = d1_adx_pass
            fctx["d1_macd_enable"] = d1_macd_enable
            fctx["d1_macd_pass"] = {"LONG": d1_macd_pass_long, "SHORT": d1_macd_pass_short}

            # 4H long
            h4_pass_long = h4_pass_short = True
            if h4_long_ema_enable and s["h4_long_data"]:
                h4d = s["h4_long_data"]
                h4_idx = _find_le_idx(h4d["ts"], ts)
                long_ema = h4d["long_ema"][h4_idx] if h4_idx < len(h4d["long_ema"]) else 0
                h4_close = h4d["close"][h4_idx]
                if np.isnan(long_ema) or long_ema == 0:
                    h4_pass_long = h4_pass_short = False
                else:
                    if h4_filter_mode == "direction":
                        prev = h4d["long_ema"][h4_idx - 1] if h4_idx > 0 else long_ema
                        if not np.isnan(prev):
                            h4_pass_long = long_ema > prev
                            h4_pass_short = long_ema < prev
                    else:
                        h4_pass_long = h4_close > long_ema
                        h4_pass_short = h4_close < long_ema
            fctx["h4_long_ema_enable"] = h4_long_ema_enable
            fctx["h4_pass"] = {"LONG": h4_pass_long, "SHORT": h4_pass_short}

            signal = _evaluate_signal_v11(regime, row, params, funding, use_funding,
                                           hour, **fctx)
            if signal is None:
                continue
            if block_sol_long and sym == "SOLUSDT" and signal["direction"] == "LONG":
                continue
            if sym in blocked_directions and signal["direction"] in blocked_directions[sym]:
                continue

            raw_entry = signal["entry_price"]
            slip = slippage_pct / 100.0
            entry_price = raw_entry * (1 + slip) if signal["direction"] == "LONG" else raw_entry * (1 - slip)
            sl_pct = signal["sl_pct"]
            tp_pct = signal["tp_pct"]

            # Vol regime size
            size_mult = 1.0
            if vol_regime_size_enable:
                # 단순 기준: ATR 퍼센타일 70+ 시 축소 (atr_pct 기준 proxy)
                if row["atr_pct"] > 1.2:
                    size_mult = vol_regime_size_pct

            risk_amt = balance * risk_per_trade * size_mult
            sl_ratio = sl_pct / 100.0
            avail_margin = balance * MARGIN_USAGE / max_simultaneous * size_mult
            ideal = risk_amt / sl_ratio
            lev = min(int(ideal / avail_margin) if avail_margin > 0 else 1, max_leverage)
            lev = max(lev, MIN_LEVERAGE)
            pos_val = avail_margin * lev
            qty = pos_val / entry_price
            if pos_val < 5:
                continue

            if signal["direction"] == "LONG":
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_price = entry_price * (1 + tp_pct / 100)
                # Partial TP 중간지점
                partial_tp_price = (entry_price + tp_price) / 2 if partial_tp_enable else 0
            else:
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_price = entry_price * (1 - tp_pct / 100)
                partial_tp_price = (entry_price + tp_price) / 2 if partial_tp_enable else 0

            positions[sym] = {
                "direction": signal["direction"],
                "entry_price": entry_price,
                "sl_price": sl_price, "tp_price": tp_price,
                "quantity": qty, "leverage": lev,
                "peak_price": entry_price, "trailing_active": False,
                "partial_done": False,
                "partial_tp_price": partial_tp_price,
                "partial_qty": qty * partial_tp_ratio if partial_tp_enable else 0,
                # Edge-hunt용 enriched state
                "entry_idx": idx,
                "entry_ts": ts,
                "entry_rsi": row["rsi"],
                "entry_atr_pct": row["atr_pct"],
                "entry_vol_ratio": row["volume_ratio"],
                "entry_funding": funding,
            }
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

    if not ruined:
        for sym, pos in positions.items():
            s = per_sym.get(sym)
            if s is None:
                continue
            last_close = s["close"][-1]
            ep = pos["entry_price"]
            qty = pos["quantity"]
            d = pos["direction"]
            slip = slippage_pct / 100.0
            adj_close = last_close * ((1 - slip) if d == "LONG" else (1 + slip))
            raw_pnl = (adj_close - ep) * qty if d == "LONG" else (ep - adj_close) * qty
            fee = (ep * qty + last_close * qty) * TAKER_FEE_PCT
            net_pnl = raw_pnl - fee
            balance += net_pnl
            trades.append({
                "symbol": sym, "direction": d, "entry_price": ep,
                "exit_price": last_close, "quantity": qty,
                "pnl": round(net_pnl, 4), "fee": round(fee, 4),
                "reason": "END", "timestamp": int(master_ts[-1]),
                "year": datetime.utcfromtimestamp(int(master_ts[-1])/1000).year,
            })

    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "final_balance": round(balance, 2),
        "net_profit": round(balance - seed, 2),
        "max_dd": round(max_dd, 1),
        "ruined": ruined,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "_trades": trades,
    }
