#!/usr/bin/env python3
"""
백테스팅 v4 — Shared-Balance 엔진
=================================
실제 HERMES 구조를 그대로 재현:
- 3코인 공유 잔고 (단일 wallet)
- MAX_SIMULTANEOUS=2 전역 (코인 무관)
- MAX_DAILY_TRADES 전역
- 시간순 단일 타임라인 (BTC 기준, 3코인 동일 1H 격자)
- 연패 시 동적 리스크 옵션
- 트레일링 스탑 (1.5/0.3)
- 일일 서버비 차감
- skip_years: 특정 연도 신규 진입 차단 (수동 중단 시나리오)
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import (
    SYMBOLS, align_funding_to_entry, evaluate_signal_v3,
)
from comprehensive_backtest import (
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
    TAKER_FEE_PCT, RISK_PER_TRADE, MARGIN_USAGE,
    MAX_LEVERAGE, MIN_LEVERAGE, MAX_SIMULTANEOUS, MAX_DAILY_TRADES,
)


def prepare_symbol(data: Dict, sym: str, params: Dict):
    """심볼별 지표/레짐/펀딩 전처리"""
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
    }


def run_shared_backtest(
    data: Dict,
    params: Dict,
    seed: float,
    *,
    use_funding: bool = True,
    trailing_activation: float = 1.5,
    trailing_distance: float = 0.3,
    block_sol_long: bool = True,
    skip_years: tuple = (),
    daily_cost_usd: float = 0.0,
    ruin_threshold: float = 15.0,
    use_dynamic_risk: bool = False,
    use_cooldown: bool = True,
    cooldown_after: int = 2,
    cooldown_candles: int = 1,
    daily_halt_after: int = 3,
    slippage_pct: float = 0.0,  # 편도 슬리피지 (예: 0.02 = 0.02%)
    # 시스템 파라미터 override (None = 기본값 사용)
    max_simultaneous: Optional[int] = None,
    risk_per_trade: Optional[float] = None,
    max_leverage: Optional[int] = None,
    max_daily_trades: Optional[int] = None,
    enabled_symbols: Optional[List[str]] = None,
    blocked_directions: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """실제 HERMES 구조 (shared wallet) 백테스트"""
    # override 기본값 적용
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
        prepared = prepare_symbol(data, sym, params)
        if prepared is not None:
            per_sym[sym] = prepared

    if not per_sym:
        raise RuntimeError("활성 심볼 없음")

    # 타임라인 기준: BTC 우선, 없으면 첫 심볼
    master_sym = "BTCUSDT" if "BTCUSDT" in per_sym else list(per_sym.keys())[0]
    master_ts = per_sym[master_sym]["timestamp"]
    n = len(master_ts)

    # 심볼별 타임스탬프 → 인덱스 매핑 (동일 격자 가정이지만 안전하게 매핑)
    sym_ts_map = {}
    for sym, s in per_sym.items():
        sym_ts_map[sym] = {int(t): i for i, t in enumerate(s["timestamp"])}

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
    skipped_year_entries = 0
    year_end_balance: Dict[int, float] = {}
    current_year: Optional[int] = None
    cooldown_until_idx: int = 0
    daily_halted_day: Optional[int] = None

    for i in range(50, n):
        ts = int(master_ts[i])
        day_key = ts // 86400000
        dt_utc = datetime.utcfromtimestamp(ts / 1000)
        year = dt_utc.year
        hour = dt_utc.hour

        # 일일 서버비 (날짜 바뀔 때마다 누적 차감)
        if last_day is None:
            last_day = day_key
        elif day_key > last_day:
            days_passed = day_key - last_day
            balance -= days_passed * daily_cost_usd
            last_day = day_key

        # 연도 바뀌면 직전 연말 잔고 스냅샷
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

        # === 1단계: 기존 포지션 청산 체크 ===
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
                    exit_price = tp
                    reason = "TP"
                elif low <= sl:
                    exit_price = sl
                    reason = "SL"
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
                            exit_price = sl
                            reason = "TRAILING"
                    pos["peak_price"] = new_peak
                    pos["trailing_active"] = trailing_active
                    pos["sl_price"] = sl
            else:
                if low <= tp:
                    exit_price = tp
                    reason = "TP"
                elif high >= sl:
                    exit_price = sl
                    reason = "SL"
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
                            exit_price = sl
                            reason = "TRAILING"
                    pos["peak_price"] = new_peak
                    pos["trailing_active"] = trailing_active
                    pos["sl_price"] = sl

            if exit_price is not None:
                # 청산 슬리피지 적용 — 불리한 방향
                slip = slippage_pct / 100.0
                if d == "LONG":
                    # LONG 청산 = 매도, 슬리피지만큼 불리하게 체결
                    adj_exit = exit_price * (1 - slip)
                    raw_pnl = (adj_exit - ep) * qty
                else:
                    # SHORT 청산 = 매수, 슬리피지만큼 비싸게 체결
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

        # 청산 후 잔고/DD 업데이트
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        if balance < ruin_threshold:
            ruined = True
            ruin_ts = ts
            continue

        # === 2단계: 신규 진입 (수동 중단 연도는 스킵) ===
        if year in skip_years:
            continue

        if len(positions) >= max_simultaneous:
            continue
        if daily_trades.get(day_key, 0) >= max_daily_trades:
            continue

        # 쿨다운 체크
        if use_cooldown:
            if daily_halted_day == day_key:
                continue
            if daily_halted_day is not None and day_key > daily_halted_day:
                daily_halted_day = None
            if i < cooldown_until_idx:
                continue

        # 동적 리스크
        risk_mult = 1.0
        if use_dynamic_risk:
            if consecutive_losses >= 3:
                risk_mult = 0.5
            elif consecutive_losses >= 2:
                risk_mult = 0.75

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

            ema_f = s["ema_fast"][idx]
            ema_s = s["ema_slow"][idx]
            if pd.isna(ema_f) or pd.isna(ema_s):
                continue

            regime = s["regime"][idx] if idx < len(s["regime"]) else "RANGING"
            funding = s["funding"][idx] if idx < len(s["funding"]) else 0

            row = {
                "close": s["close"][idx],
                "ema_fast": ema_f, "ema_slow": ema_s,
                "rsi": s["rsi"][idx],
                "atr_pct": s["atr_pct"][idx],
                "volume_ratio": s["volume_ratio"][idx],
            }

            signal = evaluate_signal_v3(
                regime, row, params,
                funding_rate=funding, rsi_15=50,
                use_funding=use_funding, use_mtf=False,
                hour=hour, session_filter=None,
            )
            if signal is None:
                continue
            if block_sol_long and sym == "SOLUSDT" and signal["direction"] == "LONG":
                continue
            if sym in blocked_directions and signal["direction"] in blocked_directions[sym]:
                continue

            # 진입 슬리피지: Taker 시장가 → 불리한 방향으로 체결
            raw_entry = signal["entry_price"]
            slip = slippage_pct / 100.0
            if signal["direction"] == "LONG":
                entry_price = raw_entry * (1 + slip)
            else:
                entry_price = raw_entry * (1 - slip)
            sl_pct = signal["sl_pct"]
            tp_pct = signal["tp_pct"]

            risk_amt = balance * risk_per_trade * risk_mult
            sl_ratio = sl_pct / 100.0
            avail_margin = balance * MARGIN_USAGE / max_simultaneous * risk_mult
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
                "peak_price": entry_price,
                "trailing_active": False,
            }
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

    # 종료 시 미청산 포지션 — 마지막 종가로 청산
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

    total_count = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)

    yearly = {}
    for t in trades:
        y = str(t["year"])
        if y not in yearly:
            yearly[y] = {"trades": 0, "wins": 0, "pnl": 0.0}
        yearly[y]["trades"] += 1
        if t["pnl"] > 0:
            yearly[y]["wins"] += 1
        yearly[y]["pnl"] += t["pnl"]
    for y in yearly:
        yearly[y]["pnl"] = round(yearly[y]["pnl"], 2)

    # 마지막 연도 스냅샷
    if current_year is not None:
        year_end_balance[current_year] = round(balance, 2)

    return {
        "final_balance": round(balance, 2),
        "net_profit": round(balance - seed, 2),
        "net_pct": round((balance - seed) / seed * 100, 1) if seed > 0 else 0,
        "max_dd": round(max_dd, 1),
        "ruined": ruined,
        "ruin_date": datetime.utcfromtimestamp(ruin_ts/1000).strftime("%Y-%m-%d") if ruin_ts else None,
        "total_trades": total_count,
        "wins": wins,
        "win_rate": round(wins / total_count * 100, 1) if total_count else 0,
        "skipped_years": list(skip_years),
        "yearly": yearly,
        "year_end_balance": year_end_balance,
        "_trades": trades,
    }
