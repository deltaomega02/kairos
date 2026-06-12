"""KAIROS — 모멘텀 헌팅 백테스트.

운영자 아이디어: "오를 거 같은 코인 사서 1-20% 익절"

테스트 변형:
1. Top Gainer Catching: 매 1H마다 4코인 중 24H 상승률 1위 매수, N시간 후 청산
2. Volume + Price Spike: 거래량 N배 + 가격 +X% 돌파 시 매수
3. Multi-Coin Momentum: 4코인 모두 감시, 신호 발생 시 진입

데이터: BTC/ETH/SOL/XRP 1H 캔들 (2021-12 ~ 2026-04, 모두 겹치는 기간)
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional


DATA_DIR = Path(__file__).parent / "data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_1h(symbol: str) -> List[Candle]:
    path = DATA_DIR / f"{symbol}_60_long.csv"
    candles = []
    with path.open() as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                int(row["timestamp"]),
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def align_by_timestamp(data: Dict[str, List[Candle]]) -> List[int]:
    """모든 코인이 데이터 가진 공통 timestamp만 반환."""
    all_ts = set()
    for sym, candles in data.items():
        if not all_ts:
            all_ts = set(c.timestamp for c in candles)
        else:
            all_ts &= set(c.timestamp for c in candles)
    return sorted(all_ts)


def candle_at(candles: List[Candle], ts: int) -> Optional[Candle]:
    # 이진 검색 단순화 — dict 캐싱이 효율적이지만 한 번만 사용
    for c in candles:
        if c.timestamp == ts:
            return c
    return None


def to_dict(candles: List[Candle]) -> Dict[int, Candle]:
    return {c.timestamp: c for c in candles}


def date_str(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ────────────────────────────────────────────────────────────
# 전략 1: Top Gainer Catching
# ────────────────────────────────────────────────────────────

def strategy_top_gainer(
    data: Dict[str, Dict[int, Candle]],
    timestamps: List[int],
    lookback_hours: int = 24,
    hold_hours: int = 4,
    min_gain_pct: float = 0.03,    # 24H +3% 이상이어야
    tp_pct: float = 0.05,          # +5% 익절
    sl_pct: float = 0.03,          # -3% 손절
    initial: float = 1000.0,
    fee_pct: float = 0.00055,
    slippage_pct: float = 0.0008,  # 알트 슬리피지 더 크게
    leverage: int = 2,
    position_size_pct: float = 0.30,
    label: str = "Top Gainer",
):
    balance = initial
    equity_curve = [initial]
    trades = []

    sizing = position_size_pct * leverage
    fee_rt = fee_pct * 2 + slippage_pct * 2

    in_position = False
    pos_symbol = None
    pos_entry = 0.0
    pos_entry_ts = 0
    pos_sl = 0.0
    pos_tp = 0.0

    for i, ts in enumerate(timestamps):
        if i < lookback_hours:
            equity_curve.append(balance)
            continue

        # 진입 중인 포지션 청산 체크
        if in_position:
            c = data[pos_symbol].get(ts)
            if c is None:
                equity_curve.append(balance)
                continue
            exit_reason = None
            exit_price = c.close
            if c.low <= pos_sl:
                exit_price = pos_sl
                exit_reason = "SL"
            elif c.high >= pos_tp:
                exit_price = pos_tp
                exit_reason = "TP"
            elif (ts - pos_entry_ts) / 3_600_000 >= hold_hours:
                exit_price = c.close
                exit_reason = "TIME"

            if exit_reason:
                pnl_pct = (exit_price - pos_entry) / pos_entry
                pnl_net = sizing * (pnl_pct - fee_rt)
                balance *= (1 + pnl_net)
                trades.append({
                    "symbol": pos_symbol, "entry_ts": pos_entry_ts, "exit_ts": ts,
                    "entry": pos_entry, "exit": exit_price, "reason": exit_reason,
                    "pnl_pct": pnl_pct, "pnl_net": pnl_net,
                })
                in_position = False
                pos_symbol = None

        # 진입 신호 체크
        if not in_position:
            ts_lookback = ts - lookback_hours * 3_600_000
            best_sym = None
            best_gain = -1e9
            for sym in SYMBOLS:
                cur = data[sym].get(ts)
                past = data[sym].get(ts_lookback)
                if cur is None or past is None or past.close <= 0:
                    continue
                gain = (cur.close - past.close) / past.close
                if gain > best_gain:
                    best_gain = gain
                    best_sym = sym

            if best_sym and best_gain >= min_gain_pct:
                cur = data[best_sym][ts]
                pos_entry = cur.close * (1 + slippage_pct)
                pos_symbol = best_sym
                pos_entry_ts = ts
                pos_sl = pos_entry * (1 - sl_pct)
                pos_tp = pos_entry * (1 + tp_pct)
                in_position = True

        equity_curve.append(balance)

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t["pnl_net"] > 0)
    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades if n_trades else 0.0

    peak = initial
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    days = (timestamps[-1] - timestamps[0]) / 1000 / 86400
    years = days / 365.25
    cagr = (balance / initial) ** (1/years) - 1 if years > 0 else 0
    total = (balance - initial) / initial

    return {
        "label": label,
        "balance": balance,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "total_return": total,
        "cagr": cagr,
        "trades": trades,
    }


# ────────────────────────────────────────────────────────────
# 전략 2: Volume + Price Spike
# ────────────────────────────────────────────────────────────

def strategy_volume_spike(
    data: Dict[str, Dict[int, Candle]],
    timestamps: List[int],
    vol_lookback: int = 24,
    vol_multiplier: float = 3.0,
    price_breakout_pct: float = 0.03,  # 직전 1H +3%
    hold_hours: int = 4,
    tp_pct: float = 0.05,
    sl_pct: float = 0.03,
    initial: float = 1000.0,
    fee_pct: float = 0.00055,
    slippage_pct: float = 0.0008,
    leverage: int = 2,
    position_size_pct: float = 0.30,
    label: str = "Volume Spike",
):
    balance = initial
    equity_curve = [initial]
    trades = []
    sizing = position_size_pct * leverage
    fee_rt = fee_pct * 2 + slippage_pct * 2

    in_position = False
    pos_symbol = None
    pos_entry = 0.0
    pos_entry_ts = 0
    pos_sl = 0.0
    pos_tp = 0.0

    # 코인별 거래량 history 캐시
    vol_hist: Dict[str, List[float]] = {s: [] for s in SYMBOLS}

    for i, ts in enumerate(timestamps):
        # 거래량 history 업데이트
        for sym in SYMBOLS:
            c = data[sym].get(ts)
            if c:
                vol_hist[sym].append(c.volume)
                if len(vol_hist[sym]) > vol_lookback:
                    vol_hist[sym].pop(0)

        if i < vol_lookback:
            equity_curve.append(balance)
            continue

        if in_position:
            c = data[pos_symbol].get(ts)
            if c is not None:
                exit_reason = None
                exit_price = c.close
                if c.low <= pos_sl:
                    exit_price = pos_sl; exit_reason = "SL"
                elif c.high >= pos_tp:
                    exit_price = pos_tp; exit_reason = "TP"
                elif (ts - pos_entry_ts) / 3_600_000 >= hold_hours:
                    exit_price = c.close; exit_reason = "TIME"

                if exit_reason:
                    pnl_pct = (exit_price - pos_entry) / pos_entry
                    pnl_net = sizing * (pnl_pct - fee_rt)
                    balance *= (1 + pnl_net)
                    trades.append({
                        "symbol": pos_symbol, "entry_ts": pos_entry_ts, "exit_ts": ts,
                        "entry": pos_entry, "exit": exit_price, "reason": exit_reason,
                        "pnl_pct": pnl_pct, "pnl_net": pnl_net,
                    })
                    in_position = False; pos_symbol = None

        if not in_position:
            for sym in SYMBOLS:
                cur = data[sym].get(ts)
                if cur is None or len(vol_hist[sym]) < vol_lookback:
                    continue
                avg_vol = sum(vol_hist[sym][:-1]) / max(len(vol_hist[sym])-1, 1)
                if avg_vol <= 0:
                    continue
                vol_ratio = cur.volume / avg_vol
                # 직전 1H 가격 변화
                price_change = (cur.close - cur.open) / cur.open if cur.open > 0 else 0

                if vol_ratio >= vol_multiplier and price_change >= price_breakout_pct:
                    pos_entry = cur.close * (1 + slippage_pct)
                    pos_symbol = sym
                    pos_entry_ts = ts
                    pos_sl = pos_entry * (1 - sl_pct)
                    pos_tp = pos_entry * (1 + tp_pct)
                    in_position = True
                    break

        equity_curve.append(balance)

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t["pnl_net"] > 0)
    win_rate = n_wins / n_trades if n_trades else 0.0
    peak = initial; max_dd = 0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0: max_dd = max(max_dd, (peak - v) / peak)
    days = (timestamps[-1] - timestamps[0]) / 1000 / 86400
    years = days / 365.25
    cagr = (balance / initial) ** (1/years) - 1 if years > 0 else 0

    return {
        "label": label, "balance": balance,
        "n_trades": n_trades, "n_wins": n_wins, "win_rate": win_rate,
        "max_dd": max_dd, "total_return": (balance - initial) / initial,
        "cagr": cagr, "trades": trades,
    }


# ────────────────────────────────────────────────────────────
# 출력
# ────────────────────────────────────────────────────────────

def print_result(r):
    print(f"\n{'='*80}")
    print(f"📊 {r['label']}")
    print(f"{'─'*80}")
    print(f"거래 수      {r['n_trades']}회 ({r['n_wins']}승 / 승률 {r['win_rate']*100:.1f}%)")
    print(f"누적 수익률  {r['total_return']*100:+.1f}%")
    print(f"연환산 CAGR  {r['cagr']*100:+.1f}%")
    print(f"최대 DD      {r['max_dd']*100:.1f}%")
    print(f"최종 잔고    ${r['balance']:,.2f}")

    # 청산 사유 분석
    if r['trades']:
        reasons = {}
        for t in r['trades']:
            reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
        symbol_dist = {}
        for t in r['trades']:
            symbol_dist[t['symbol']] = symbol_dist.get(t['symbol'], 0) + 1
        print(f"청산 사유    {reasons}")
        print(f"코인 분포    {symbol_dist}")


def main():
    print("데이터 로드 중...")
    raw = {sym: load_1h(sym) for sym in SYMBOLS}
    data = {sym: to_dict(raw[sym]) for sym in SYMBOLS}
    timestamps = align_by_timestamp(raw)
    print(f"공통 1H 캔들: {len(timestamps)}개")
    print(f"기간: {date_str(timestamps[0])} ~ {date_str(timestamps[-1])}")

    # === Top Gainer 변형 ===
    configs_top = [
        dict(lookback_hours=24, hold_hours=4, min_gain_pct=0.03, tp_pct=0.05, sl_pct=0.03,
             label="Top Gainer 24H>3% / 4h hold / TP5%-SL3%"),
        dict(lookback_hours=24, hold_hours=8, min_gain_pct=0.03, tp_pct=0.10, sl_pct=0.05,
             label="Top Gainer 24H>3% / 8h hold / TP10%-SL5%"),
        dict(lookback_hours=4, hold_hours=2, min_gain_pct=0.02, tp_pct=0.03, sl_pct=0.02,
             label="Top Gainer 4H>2% / 2h hold / TP3%-SL2%"),
        dict(lookback_hours=24, hold_hours=24, min_gain_pct=0.05, tp_pct=0.15, sl_pct=0.07,
             label="Top Gainer 24H>5% / 24h hold / TP15%-SL7%"),
    ]
    for cfg in configs_top:
        r = strategy_top_gainer(data, timestamps, **cfg)
        print_result(r)

    # === Volume Spike 변형 ===
    configs_vol = [
        dict(vol_multiplier=3.0, price_breakout_pct=0.03, hold_hours=4, tp_pct=0.05, sl_pct=0.03,
             label="Volume×3 + 1H>3% / 4h hold / TP5%-SL3%"),
        dict(vol_multiplier=2.0, price_breakout_pct=0.02, hold_hours=4, tp_pct=0.04, sl_pct=0.02,
             label="Volume×2 + 1H>2% / 4h hold / TP4%-SL2%"),
        dict(vol_multiplier=5.0, price_breakout_pct=0.05, hold_hours=8, tp_pct=0.10, sl_pct=0.05,
             label="Volume×5 + 1H>5% / 8h hold / TP10%-SL5%"),
    ]
    for cfg in configs_vol:
        r = strategy_volume_spike(data, timestamps, **cfg)
        print_result(r)

    # === 벤치마크 ===
    print(f"\n{'='*80}")
    print(f"💡 같은 기간 BTC Buy & Hold 벤치마크")
    print(f"{'─'*80}")
    btc = raw["BTCUSDT"]
    btc_aligned = [c for c in btc if c.timestamp in set(timestamps)]
    bh_return = (btc_aligned[-1].close / btc_aligned[0].close - 1)
    days = (timestamps[-1] - timestamps[0]) / 1000 / 86400
    bh_cagr = (1 + bh_return) ** (365.25 / days) - 1 if days > 0 else 0
    print(f"누적: {bh_return*100:+.1f}%   CAGR: {bh_cagr*100:+.1f}%")


if __name__ == "__main__":
    main()
