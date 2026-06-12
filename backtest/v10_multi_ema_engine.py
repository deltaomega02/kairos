#!/usr/bin/env python3
"""
v10 Multi-EMA Engine (Opus 4.7)
================================
v4_shared_engine 확장 — 여러 다중 EMA 모드를 feature flag로 토글.

Supported modes (kwargs):
  - triple_ema_enable (bool): 1H에 3번째 EMA 추가. ema_medium_period 로 설정.
  - d1_filter_enable (bool): 1D(24H) EMA 방향 필터. 1H → 1D resample로 derive.
      d1_ema_period: 1D EMA N일 (기본 20)
      d1_filter_mode: "direction" (EMA 기울기) or "price_above_ema"
  - h4_long_ema_enable (bool): 4H에 장기 EMA (기본 50bars = 200H = 8.3일).
      h4_long_ema_period: 기본 50
  - ribbon_enable (bool): 1H에 여러 EMA 모두 정배열 확인.
      ribbon_periods: 기본 [3, 5, 8, 13, 21, 34]
  - per_direction_enable (bool): LONG/SHORT 다른 EMA 페어 사용.
      ema_fast_long, ema_slow_long, ema_fast_short, ema_slow_short

기본 (모든 flag False): v4와 동일 결과.
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import (
    SYMBOLS, align_funding_to_entry,
)
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
    fee_adjusted_sl_tp,
    TAKER_FEE_PCT, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_SIMULTANEOUS, MAX_DAILY_TRADES,
)


def _ema(series: pd.Series, period: int) -> pd.Series:
    """standard EMA"""
    return series.ewm(span=period, adjust=False).mean()


def _resample_1h_to_1d(entry_df_1h: pd.DataFrame) -> pd.DataFrame:
    """1H → 1D OHLC resample (UTC 기준)"""
    df = entry_df_1h.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    # 1D OHLC
    d = df[["open", "high", "low", "close"]].resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    d["timestamp"] = d.index.astype("int64") // 10**6
    d = d.reset_index(drop=True)
    return d


def prepare_symbol_v10(data: Dict, sym: str, params: Dict,
                       triple_ema_enable: bool = False,
                       ema_medium_period: int = 8,
                       d1_filter_enable: bool = False,
                       d1_ema_period: int = 20,
                       h4_long_ema_enable: bool = False,
                       h4_long_ema_period: int = 50,
                       ribbon_enable: bool = False,
                       ribbon_periods: Optional[List[int]] = None,
                       per_direction_enable: bool = False,
                       ema_fast_long: int = 3, ema_slow_long: int = 15,
                       ema_fast_short: int = 3, ema_slow_short: int = 15):
    """심볼별 지표/레짐/펀딩/멀티EMA 전처리"""
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

    # Triple EMA — medium EMA
    ema_medium = None
    if triple_ema_enable:
        ema_medium = _ema(close_series, ema_medium_period).values

    # Ribbon — multiple EMAs
    ribbon_vals = {}
    if ribbon_enable:
        if ribbon_periods is None:
            ribbon_periods = [3, 5, 8, 13, 21, 34]
        for p in ribbon_periods:
            ribbon_vals[p] = _ema(close_series, p).values

    # Per-direction EMAs
    pd_long_fast = pd_long_slow = pd_short_fast = pd_short_slow = None
    if per_direction_enable:
        pd_long_fast = _ema(close_series, ema_fast_long).values
        pd_long_slow = _ema(close_series, ema_slow_long).values
        pd_short_fast = _ema(close_series, ema_fast_short).values
        pd_short_slow = _ema(close_series, ema_slow_short).values

    # 1D filter — derived from 1H resample
    d1_align = None
    if d1_filter_enable:
        d1_df = _resample_1h_to_1d(entry_df)
        d1_df["ema"] = _ema(d1_df["close"], d1_ema_period)
        # Align 1D values back to 1H timeline
        d1_ts = d1_df["timestamp"].values.astype(np.int64)
        d1_ema = d1_df["ema"].values
        d1_close = d1_df["close"].values
        d1_align = {"ts": d1_ts, "ema": d1_ema, "close": d1_close}

    # 4H long EMA
    h4_long_align = None
    if h4_long_ema_enable:
        h4_df = regime_df.copy()
        h4_df["long_ema"] = _ema(h4_df["close"], h4_long_ema_period)
        h4_ts = h4_df["timestamp"].values.astype(np.int64)
        h4_long_ema_vals = h4_df["long_ema"].values
        h4_close = h4_df["close"].values
        h4_long_align = {"ts": h4_ts, "long_ema": h4_long_ema_vals, "close": h4_close}

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
        # New multi-EMA fields
        "ema_medium": ema_medium,
        "ribbon_vals": ribbon_vals,  # dict: period -> values
        "pd_long_fast": pd_long_fast, "pd_long_slow": pd_long_slow,
        "pd_short_fast": pd_short_fast, "pd_short_slow": pd_short_slow,
        "d1_align": d1_align,
        "h4_long_align": h4_long_align,
    }


def _find_le_index(ts_array: np.ndarray, target_ts: int) -> int:
    """target_ts 이하의 최대 index (binary search)"""
    idx = np.searchsorted(ts_array, target_ts, side="right") - 1
    return max(0, min(idx, len(ts_array) - 1))


def _evaluate_signal_multi(
    regime, row, params,
    funding_rate=0,
    use_funding=True,
    hour=0,
    # Multi-EMA state
    triple_ema_enable=False,
    ema_medium_val=0,
    ribbon_enable=False,
    ribbon_current=None,  # dict period -> current val
    per_direction_enable=False,
    pd_fast_long=0, pd_slow_long=0,
    pd_fast_short=0, pd_slow_short=0,
    # External filters
    d1_pass_long=True, d1_pass_short=True,
    h4_pass_long=True, h4_pass_short=True,
) -> Optional[Dict]:
    """v3 signal + multi-EMA filters"""
    if regime not in ("TRENDING_UP", "TRENDING_DOWN"):
        return None

    direction = "LONG" if regime == "TRENDING_UP" else "SHORT"

    # Direction-specific EMA
    if per_direction_enable:
        if direction == "LONG":
            ema_f_use = pd_fast_long
            ema_s_use = pd_slow_long
        else:
            ema_f_use = pd_fast_short
            ema_s_use = pd_slow_short
    else:
        ema_f_use = row.get("ema_fast", 0)
        ema_s_use = row.get("ema_slow", 0)

    close = row.get("close", 0)
    rsi = row.get("rsi", 50)
    vol = row.get("volume_ratio", 1.0)
    atr_pct = row.get("atr_pct", 0.5)

    if ema_f_use == 0 or ema_s_use == 0 or close == 0:
        return None

    max_dist = params["pullback_ema_dist_pct"]

    # Pullback check (using direction-specific or standard EMA)
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

    # Triple EMA: require fast > medium > slow (LONG) or fast < medium < slow (SHORT)
    if triple_ema_enable:
        if ema_medium_val <= 0:
            return None
        if direction == "LONG":
            if not (ema_f_use > ema_medium_val > ema_s_use):
                return None
        else:
            if not (ema_f_use < ema_medium_val < ema_s_use):
                return None

    # Ribbon: all EMAs aligned
    if ribbon_enable and ribbon_current is not None:
        sorted_periods = sorted(ribbon_current.keys())
        vals = [ribbon_current[p] for p in sorted_periods]
        if direction == "LONG":
            # shortest first, should be largest (descending)
            if not all(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
                return None
        else:
            if not all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
                return None

    # 1D filter
    if direction == "LONG" and not d1_pass_long:
        return None
    if direction == "SHORT" and not d1_pass_short:
        return None

    # 4H long filter
    if direction == "LONG" and not h4_pass_long:
        return None
    if direction == "SHORT" and not h4_pass_short:
        return None

    # Score (same as v3)
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
        funding_thresh = params.get("funding_bias_threshold", 0.0005)
        if abs(funding_rate) >= funding_thresh:
            if (direction == "LONG" and funding_rate < 0) or (direction == "SHORT" and funding_rate > 0):
                score += 15
            else:
                score -= 10

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


def run_shared_backtest_v10(
    data: Dict,
    params: Dict,
    seed: float,
    *,
    use_funding: bool = True,
    trailing_activation: float = 1.2,
    trailing_distance: float = 0.1,
    block_sol_long: bool = True,
    skip_years: tuple = (),
    daily_cost_usd: float = 0.0,
    ruin_threshold: float = 15.0,
    use_cooldown: bool = False,
    cooldown_after: int = 2,
    cooldown_candles: int = 1,
    daily_halt_after: int = 3,
    slippage_pct: float = 0.05,
    max_simultaneous: Optional[int] = None,
    risk_per_trade: Optional[float] = None,
    max_leverage: Optional[int] = None,
    max_daily_trades: Optional[int] = None,
    enabled_symbols: Optional[List[str]] = None,
    blocked_directions: Optional[Dict[str, List[str]]] = None,
    # Multi-EMA features
    triple_ema_enable: bool = False,
    ema_medium_period: int = 8,
    d1_filter_enable: bool = False,
    d1_ema_period: int = 20,
    d1_filter_mode: str = "direction",  # "direction" or "price_above_ema"
    h4_long_ema_enable: bool = False,
    h4_long_ema_period: int = 50,
    h4_filter_mode: str = "direction",
    ribbon_enable: bool = False,
    ribbon_periods: Optional[List[int]] = None,
    per_direction_enable: bool = False,
    ema_fast_long: int = 3, ema_slow_long: int = 15,
    ema_fast_short: int = 3, ema_slow_short: int = 15,
) -> Dict:
    if max_simultaneous is None:
        max_simultaneous = MAX_SIMULTANEOUS
    if risk_per_trade is None:
        risk_per_trade = RISK_PER_TRADE
    if max_leverage is None:
        max_leverage = MAX_LEVERAGE
    if max_daily_trades is None:
        max_daily_trades = MAX_DAILY_TRADES
    if enabled_symbols is None:
        enabled_symbols = SYMBOLS
    if blocked_directions is None:
        blocked_directions = {}

    per_sym = {}
    for sym in enabled_symbols:
        prepared = prepare_symbol_v10(
            data, sym, params,
            triple_ema_enable=triple_ema_enable,
            ema_medium_period=ema_medium_period,
            d1_filter_enable=d1_filter_enable,
            d1_ema_period=d1_ema_period,
            h4_long_ema_enable=h4_long_ema_enable,
            h4_long_ema_period=h4_long_ema_period,
            ribbon_enable=ribbon_enable,
            ribbon_periods=ribbon_periods,
            per_direction_enable=per_direction_enable,
            ema_fast_long=ema_fast_long, ema_slow_long=ema_slow_long,
            ema_fast_short=ema_fast_short, ema_slow_short=ema_slow_short,
        )
        if prepared is not None:
            per_sym[sym] = prepared

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
    consecutive_losses = 0
    trades: List[Dict] = []
    ruined = False
    ruin_ts = None
    last_day: Optional[int] = None
    current_year: Optional[int] = None
    year_end_balance: Dict[int, float] = {}
    cooldown_until_idx: int = 0
    daily_halted_day: Optional[int] = None

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

        if current_year is None:
            current_year = year
        elif year != current_year:
            year_end_balance[current_year] = round(balance, 2)
            current_year = year

        if ruined:
            continue
        if balance < ruin_threshold:
            ruined = True
            ruin_ts = ts
            continue

        # ===== Close check =====
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

                if net_pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    if use_cooldown:
                        if consecutive_losses >= daily_halt_after:
                            daily_halted_day = day_key
                        elif consecutive_losses >= cooldown_after:
                            cooldown_until_idx = i + cooldown_candles + 1

                trades.append({
                    "symbol": sym, "direction": d,
                    "entry_price": ep, "exit_price": exit_price,
                    "quantity": qty, "pnl": round(net_pnl, 4),
                    "fee": round(fee, 4), "reason": reason,
                    "timestamp": ts, "year": year,
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

        # ===== Entry =====
        if year in skip_years:
            continue
        if len(positions) >= max_simultaneous:
            continue
        if daily_trades.get(day_key, 0) >= max_daily_trades:
            continue
        if use_cooldown:
            if daily_halted_day == day_key:
                continue
            if daily_halted_day is not None and day_key > daily_halted_day:
                daily_halted_day = None
            if i < cooldown_until_idx:
                continue

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
                "rsi": s["rsi"][idx],
                "atr_pct": s["atr_pct"][idx],
                "volume_ratio": s["volume_ratio"][idx],
            }

            # Multi-EMA state assembly
            ema_medium_val = s["ema_medium"][idx] if s["ema_medium"] is not None else 0
            ribbon_cur = None
            if ribbon_enable and s["ribbon_vals"]:
                ribbon_cur = {p: vals[idx] for p, vals in s["ribbon_vals"].items()
                              if idx < len(vals) and not np.isnan(vals[idx])}
                if len(ribbon_cur) != len(s["ribbon_vals"]):
                    continue

            pd_fl = s["pd_long_fast"][idx] if s["pd_long_fast"] is not None else 0
            pd_sl = s["pd_long_slow"][idx] if s["pd_long_slow"] is not None else 0
            pd_fs = s["pd_short_fast"][idx] if s["pd_short_fast"] is not None else 0
            pd_ss = s["pd_short_slow"][idx] if s["pd_short_slow"] is not None else 0

            # 1D filter
            d1_pass_long = d1_pass_short = True
            if d1_filter_enable and s["d1_align"] is not None:
                d1a = s["d1_align"]
                d1_idx = _find_le_index(d1a["ts"], ts)
                d1_ema_v = d1a["ema"][d1_idx] if d1_idx < len(d1a["ema"]) else 0
                d1_close_v = d1a["close"][d1_idx] if d1_idx < len(d1a["close"]) else 0
                if np.isnan(d1_ema_v) or d1_ema_v == 0:
                    d1_pass_long = d1_pass_short = False
                else:
                    if d1_filter_mode == "direction":
                        # Slope: current EMA > previous EMA
                        prev_ema = d1a["ema"][d1_idx - 1] if d1_idx > 0 else d1_ema_v
                        d1_pass_long = (d1_ema_v > prev_ema) if not np.isnan(prev_ema) else True
                        d1_pass_short = (d1_ema_v < prev_ema) if not np.isnan(prev_ema) else True
                    else:  # price_above_ema
                        d1_pass_long = d1_close_v > d1_ema_v
                        d1_pass_short = d1_close_v < d1_ema_v

            # 4H long EMA filter
            h4_pass_long = h4_pass_short = True
            if h4_long_ema_enable and s["h4_long_align"] is not None:
                h4a = s["h4_long_align"]
                h4_idx = _find_le_index(h4a["ts"], ts)
                h4_long_v = h4a["long_ema"][h4_idx] if h4_idx < len(h4a["long_ema"]) else 0
                h4_close_v = h4a["close"][h4_idx] if h4_idx < len(h4a["close"]) else 0
                if np.isnan(h4_long_v) or h4_long_v == 0:
                    h4_pass_long = h4_pass_short = False
                else:
                    if h4_filter_mode == "direction":
                        prev_v = h4a["long_ema"][h4_idx - 1] if h4_idx > 0 else h4_long_v
                        h4_pass_long = (h4_long_v > prev_v) if not np.isnan(prev_v) else True
                        h4_pass_short = (h4_long_v < prev_v) if not np.isnan(prev_v) else True
                    else:
                        h4_pass_long = h4_close_v > h4_long_v
                        h4_pass_short = h4_close_v < h4_long_v

            signal = _evaluate_signal_multi(
                regime, row, params,
                funding_rate=funding, use_funding=use_funding, hour=hour,
                triple_ema_enable=triple_ema_enable,
                ema_medium_val=ema_medium_val,
                ribbon_enable=ribbon_enable, ribbon_current=ribbon_cur,
                per_direction_enable=per_direction_enable,
                pd_fast_long=pd_fl, pd_slow_long=pd_sl,
                pd_fast_short=pd_fs, pd_slow_short=pd_ss,
                d1_pass_long=d1_pass_long, d1_pass_short=d1_pass_short,
                h4_pass_long=h4_pass_long, h4_pass_short=h4_pass_short,
            )
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

            risk_amt = balance * risk_per_trade
            sl_ratio = sl_pct / 100.0
            avail_margin = balance * MARGIN_USAGE / max_simultaneous
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
            else:
                sl_price = entry_price * (1 + sl_pct / 100)
                tp_price = entry_price * (1 - tp_pct / 100)

            positions[sym] = {
                "direction": signal["direction"],
                "entry_price": entry_price,
                "sl_price": sl_price, "tp_price": tp_price,
                "quantity": qty, "leverage": lev,
                "peak_price": entry_price, "trailing_active": False,
            }
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

    # End: close open positions
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
            if d == "LONG":
                adj_close = last_close * (1 - slip)
                raw_pnl = (adj_close - ep) * qty
            else:
                adj_close = last_close * (1 + slip)
                raw_pnl = (ep - adj_close) * qty
            fee = (ep * qty + last_close * qty) * TAKER_FEE_PCT
            net_pnl = raw_pnl - fee
            balance += net_pnl
            trades.append({
                "symbol": sym, "direction": d,
                "entry_price": ep, "exit_price": last_close,
                "quantity": qty, "pnl": round(net_pnl, 4),
                "fee": round(fee, 4), "reason": "END",
                "timestamp": int(master_ts[-1]),
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
