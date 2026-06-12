#!/usr/bin/env python3
"""
v12 Realistic Engine — v11 + 현실 제약
========================================

추가된 것들:
  1. 유동성 캡 (심볼별 최대 포지션 규모)
  2. 동적 슬리피지 (포지션 크기 / 유동성 비율 × ATR 가중)
  3. API 실패 시뮬레이션 (확률적 거래 누락)
  4. 펀딩비용 실제 모델 (8시간마다 부과)
  5. Bybit 실제 포지션 한도
  6. 주문 거부 (과도한 포지션 시 skip)
"""
import os, sys, random
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import SYMBOLS as DEFAULT_SYMBOLS, align_funding_to_entry
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
    fee_adjusted_sl_tp,
    TAKER_FEE_PCT, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_SIMULTANEOUS, MAX_DAILY_TRADES,
)
from v11_all_features_engine import _ema, _resample_1h_to_1d, prepare_symbol_v11, _find_le_idx

# ============================================================
# 현실 제약 파라미터
# ============================================================

# 유동성 캡: 이 이상 포지션 잡으면 슬리피지 폭증. 시간에 따라 점진 증가 (시장 성장)
BASE_LIQUIDITY_CAPS_USD = {
    "BTCUSDT": 3_000_000,   # 2020년 기준, 연 +30%
    "ETHUSDT": 1_000_000,
    "SOLUSDT": 300_000,
    "XRPUSDT": 200_000,
}

# Bybit 실제 포지션 한도 (초과 불가)
POSITION_HARD_CAP_USD = {
    "BTCUSDT": 50_000_000,
    "ETHUSDT": 20_000_000,
    "SOLUSDT": 3_000_000,
    "XRPUSDT": 1_500_000,
}

API_FAIL_RATE = 0.05    # 5% 거래 누락 (지연/에러/rate limit)
FUNDING_INTERVAL_HOURS = 8


def liquidity_cap_at(symbol: str, year_offset: float) -> float:
    """시간에 따라 유동성 캡 성장 (연 30% 복리)."""
    base = BASE_LIQUIDITY_CAPS_USD.get(symbol, 100_000)
    return base * (1.30 ** year_offset)


def dynamic_slippage(position_value_usd: float, liq_cap: float,
                     atr_pct: float, base_slip: float = 0.05) -> float:
    """포지션/유동성 비율과 ATR에 따른 슬리피지 (%).

    - ratio ≤ 0.5: base (0.05%)
    - 0.5 < ratio ≤ 1.0: 선형 증가 (최대 2배)
    - ratio > 1.0: 지수 증가 (2% 이상 가능)
    - ATR > 1.0% 변동성 가중치 +50%
    """
    if liq_cap <= 0:
        return base_slip
    ratio = position_value_usd / liq_cap

    if ratio <= 0.5:
        slip = base_slip
    elif ratio <= 1.0:
        slip = base_slip * (1 + ratio)
    elif ratio <= 3.0:
        slip = base_slip * (2 + (ratio - 1) * 3)
    else:
        slip = base_slip * (2 + 2 * 3)  # cap at ~0.45%

    # 변동성 조정
    if atr_pct > 1.0:
        slip *= 1.5
    elif atr_pct > 0.7:
        slip *= 1.2

    return min(slip, 2.0)


def run_realistic_backtest(
    data: Dict,
    params: Dict,
    seed: float,
    start_year: int = 2020,
    *,
    use_funding: bool = True,
    trailing_activation: float = 1.2, trailing_distance: float = 0.1,
    block_sol_long: bool = True,
    skip_years: tuple = (),
    daily_cost_usd: float = 0.0,
    ruin_threshold: float = 15.0,
    slippage_pct_base: float = 0.05,
    max_simultaneous: Optional[int] = None, risk_per_trade: Optional[float] = None,
    max_leverage: Optional[int] = None, max_daily_trades: Optional[int] = None,
    enabled_symbols: Optional[List[str]] = None,
    # v11 features
    triple_ema_enable: bool = False, ema_medium_period: int = 8,
    d1_filter_enable: bool = True, d1_ema_period: int = 2,
    d1_mode: str = "price_above_ema",
    h4_long_ema_enable: bool = False, h4_long_ema_period: int = 50,
    h4_filter_mode: str = "direction",
    ribbon_enable: bool = False, ribbon_periods: Optional[List[int]] = None,
    per_direction_enable: bool = False,
    ema_fast_long: int = 3, ema_slow_long: int = 15,
    ema_fast_short: int = 3, ema_slow_short: int = 15,
    # v12 realism
    use_realism: bool = True,
    api_fail_rate: float = API_FAIL_RATE,
    funding_enabled: bool = True,
    rng_seed: int = 42,
):
    """현실 제약 반영 백테스트. v11 로직 + 유동성/슬리피지/API 실패/펀딩."""
    if max_simultaneous is None: max_simultaneous = MAX_SIMULTANEOUS
    if risk_per_trade is None: risk_per_trade = RISK_PER_TRADE
    if max_leverage is None: max_leverage = MAX_LEVERAGE
    if max_daily_trades is None: max_daily_trades = MAX_DAILY_TRADES
    if enabled_symbols is None: enabled_symbols = DEFAULT_SYMBOLS

    rng = random.Random(rng_seed)

    prep_kw = dict(
        triple_ema_enable=triple_ema_enable, ema_medium_period=ema_medium_period,
        d1_filter_enable=d1_filter_enable, d1_ema_period=d1_ema_period,
        d1_mode=d1_mode,
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

    # 펀딩 정산 (8시간 단위, UTC 00/08/16)
    funding_hours = {0, 8, 16}

    balance = seed
    peak = seed
    max_dd = 0.0
    positions: Dict[str, Dict] = {}
    daily_trades: Dict[int, int] = {}
    trades: List[Dict] = []
    ruined = False
    ruin_ts = None
    last_day: Optional[int] = None
    equity_curve = []  # [(ts, balance, peak, dd), ...]
    liquidity_skipped = 0
    api_failures = 0
    funding_paid = 0.0

    for i in range(50, n):
        ts = int(master_ts[i])
        day_key = ts // 86400000
        dt_utc = datetime.utcfromtimestamp(ts / 1000)
        year = dt_utc.year
        hour = dt_utc.hour

        year_offset = (dt_utc - datetime(start_year, 1, 1)).days / 365.25

        if last_day is None:
            last_day = day_key
        elif day_key > last_day:
            days_passed = day_key - last_day
            balance -= days_passed * daily_cost_usd
            last_day = day_key

        # 펀딩 정산
        if funding_enabled and hour in funding_hours and dt_utc.minute == 0:
            for sym, pos in list(positions.items()):
                s = per_sym.get(sym)
                if s is None:
                    continue
                idx = sym_ts_map[sym].get(ts)
                if idx is None:
                    continue
                # 실제 펀딩레이트 사용
                fr = s["funding"][idx] if idx < len(s["funding"]) else 0
                if not fr:
                    continue
                notional = pos["entry_price"] * pos["quantity"]
                # LONG은 양의 펀딩레이트에서 지불, SHORT는 수령
                if pos["direction"] == "LONG":
                    cost = notional * fr
                else:
                    cost = -notional * fr
                balance -= cost
                funding_paid += cost

        if ruined:
            continue
        if balance < ruin_threshold:
            ruined = True
            ruin_ts = ts
            continue

        # 기존 포지션 청산 체크
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
                # 동적 슬리피지 (청산 시)
                atr_pct_cur = s["atr_pct"][idx]
                pos_value = ep * qty
                liq_cap = liquidity_cap_at(sym, year_offset)
                exit_slip_pct = dynamic_slippage(pos_value, liq_cap, atr_pct_cur, slippage_pct_base)
                slip = exit_slip_pct / 100.0
                if d == "LONG":
                    adj_exit = exit_price * (1 - slip)
                    raw_pnl = (adj_exit - ep) * qty
                else:
                    adj_exit = exit_price * (1 + slip)
                    raw_pnl = (ep - adj_exit) * qty
                fee = (ep * qty + exit_price * qty) * TAKER_FEE_PCT
                net_pnl = raw_pnl - fee
                balance += net_pnl
                trades.append({
                    "symbol": sym, "direction": d, "entry_price": ep,
                    "exit_price": exit_price, "quantity": qty,
                    "pnl": round(net_pnl, 4), "fee": round(fee, 4),
                    "slip_pct": round(exit_slip_pct, 4), "reason": reason,
                    "timestamp": ts, "year": year,
                })
                del positions[sym]

        # Equity curve 월별 스냅샷
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        # 매 24시간마다 curve 저장
        if i % 24 == 0 or i == 50:
            equity_curve.append({
                "ts": ts, "balance": round(balance, 2),
                "peak": round(peak, 2), "dd": round(dd, 2)
            })

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

        # 새 진입 시도 (심볼별)
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

            # 필터 상태 조립
            fctx = {
                "time_filter_enable": False,
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
                "asym_pullback_enable": False,
                "corr_filter_enable": False,
                "atr_adaptive_sl": False,
            }

            # 1D 필터 (price_above_ema)
            d1_pass_long = d1_pass_short = True
            if d1_filter_enable and s["d1_data"]:
                d1d = s["d1_data"]
                d1_idx = _find_le_idx(d1d["ts"], ts)
                ind_cur = d1d["indicator"][d1_idx] if d1_idx < len(d1d["indicator"]) else 0
                close_cur = d1d["close"][d1_idx]
                if np.isnan(ind_cur) or ind_cur == 0:
                    d1_pass_long = d1_pass_short = False
                else:
                    if d1_mode == "price_above_ema":
                        d1_pass_long = close_cur > ind_cur
                        d1_pass_short = close_cur < ind_cur
                    else:
                        ind_prev = d1d["indicator"][d1_idx - 1] if d1_idx > 0 else ind_cur
                        if not np.isnan(ind_prev):
                            d1_pass_long = ind_cur > ind_prev
                            d1_pass_short = ind_cur < ind_prev
            fctx["d1_filter_enable"] = d1_filter_enable
            fctx["d1_pass"] = {"LONG": d1_pass_long, "SHORT": d1_pass_short}
            fctx["d1_adx_enable"] = False
            fctx["d1_macd_enable"] = False
            fctx["h4_long_ema_enable"] = h4_long_ema_enable
            fctx["h4_pass"] = {"LONG": True, "SHORT": True}

            from v11_all_features_engine import _evaluate_signal_v11
            signal = _evaluate_signal_v11(regime, row, params, funding, use_funding,
                                           hour, **fctx)
            if signal is None:
                continue
            if block_sol_long and sym == "SOLUSDT" and signal["direction"] == "LONG":
                continue

            # ====== 현실 제약 적용 ======
            # 1) API 실패 시뮬
            if use_realism and rng.random() < api_fail_rate:
                api_failures += 1
                continue

            raw_entry = signal["entry_price"]
            sl_pct = signal["sl_pct"]
            tp_pct = signal["tp_pct"]

            # 포지션 사이징 (기존 v11 로직)
            risk_amt = balance * risk_per_trade
            sl_ratio = sl_pct / 100.0
            avail_margin = balance * MARGIN_USAGE / max_simultaneous
            ideal = risk_amt / sl_ratio
            lev = min(int(ideal / avail_margin) if avail_margin > 0 else 1, max_leverage)
            lev = max(lev, MIN_LEVERAGE)
            pos_val = avail_margin * lev

            # 2) 유동성 캡 & Bybit 하드 캡
            if use_realism:
                liq_cap = liquidity_cap_at(sym, year_offset)
                hard_cap = POSITION_HARD_CAP_USD.get(sym, 1e9)
                # 하드 캡 초과 시 축소
                if pos_val > hard_cap:
                    pos_val = hard_cap
                # 유동성 × 3 초과하면 진입 스킵 (너무 큰 포지션)
                if pos_val > liq_cap * 3:
                    liquidity_skipped += 1
                    continue

            # 3) 동적 슬리피지
            atr_pct_cur = row["atr_pct"]
            if use_realism:
                liq_cap = liquidity_cap_at(sym, year_offset)
                entry_slip_pct = dynamic_slippage(pos_val, liq_cap, atr_pct_cur, slippage_pct_base)
            else:
                entry_slip_pct = slippage_pct_base
            slip = entry_slip_pct / 100.0

            entry_price = raw_entry * (1 + slip) if signal["direction"] == "LONG" else raw_entry * (1 - slip)
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
                "entry_slip_pct": entry_slip_pct,
            }
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

    # 종료 시 미청산 청산
    if not ruined:
        for sym, pos in positions.items():
            s = per_sym.get(sym)
            if s is None:
                continue
            last_close = s["close"][-1]
            ep = pos["entry_price"]
            qty = pos["quantity"]
            d = pos["direction"]
            slip = 0.001
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
        "ruin_date": datetime.utcfromtimestamp(ruin_ts/1000).strftime("%Y-%m-%d") if ruin_ts else None,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "liquidity_skipped": liquidity_skipped,
        "api_failures": api_failures,
        "funding_paid_total": round(funding_paid, 2),
        "equity_curve": equity_curve,
        "_trades": trades,
    }
