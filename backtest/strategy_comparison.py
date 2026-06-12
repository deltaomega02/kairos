"""KAIROS — 다양한 전략 종합 비교 백테스트.

6년치 BTC 데이터(4H → 일봉 집계)로 다음을 비교:
1. Buy & Hold (벤치마크)
2. DCA (매주 월요일 정액 매수)
3. 변동성 돌파 (K=0.6, 다양한 SL/추세 필터 조합)
4. 이평 모멘텀 (5/20 골든크로스)
5. RSI 평균회귀 (30 매수 / 70 매도)

목적: 어떤 전략이 통계적으로 의미 있는지 객관 비교.

사용:
    python3 -m backtest.strategy_comparison
    python3 -m backtest.strategy_comparison --source 4h_long  # 6년 데이터
    python3 -m backtest.strategy_comparison --source d        # 2년 일봉
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict


DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StrategyResult:
    name: str
    final_balance: float
    initial_balance: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    max_drawdown_pct: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float = 0.0          # 일별 수익률 표준편차 기반
    notes: str = ""


# ────────────────────────────────────────────────────────────
# 데이터 로드 + 집계
# ────────────────────────────────────────────────────────────

def load_candles(path: Path) -> List[Candle]:
    candles = []
    with path.open() as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    candles.sort(key=lambda c: c.timestamp)
    return candles


def aggregate_to_daily(intra_candles: List[Candle]) -> List[Candle]:
    """4H/1H 캔들을 UTC 일봉으로 집계."""
    by_day: Dict[str, List[Candle]] = {}
    for c in intra_candles:
        day_key = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day.setdefault(day_key, []).append(c)

    daily = []
    for day_key in sorted(by_day.keys()):
        bucket = sorted(by_day[day_key], key=lambda c: c.timestamp)
        if not bucket:
            continue
        # UTC 00:00 timestamp
        ts = int(datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        daily.append(Candle(
            timestamp=ts,
            open=bucket[0].open,
            high=max(c.high for c in bucket),
            low=min(c.low for c in bucket),
            close=bucket[-1].close,
            volume=sum(c.volume for c in bucket),
        ))
    return daily


def date_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ────────────────────────────────────────────────────────────
# 공통 지표 계산
# ────────────────────────────────────────────────────────────

def compute_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def compute_cagr(initial: float, final: float, days: float) -> float:
    if days <= 0 or initial <= 0:
        return 0.0
    years = days / 365.25
    return (final / initial) ** (1 / years) - 1


def compute_sharpe(daily_returns: List[float]) -> float:
    """단순 Sharpe (연환산, 무위험수익률 0 가정)."""
    if not daily_returns or len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std = var ** 0.5
    if std == 0:
        return 0.0
    return (mean / std) * (365.25 ** 0.5)


# ────────────────────────────────────────────────────────────
# 전략 1: Buy & Hold (벤치마크)
# ────────────────────────────────────────────────────────────

def strategy_buy_and_hold(candles: List[Candle], initial: float = 1000.0,
                          fee_pct: float = 0.00055) -> StrategyResult:
    entry = candles[0].close
    final_price = candles[-1].close
    qty = initial * (1 - fee_pct) / entry  # 매수 수수료
    final_balance = qty * final_price * (1 - fee_pct)  # 매도 수수료

    equity = [qty * c.close for c in candles]
    daily_ret = [(equity[i] / equity[i-1] - 1) for i in range(1, len(equity))]

    days = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400

    return StrategyResult(
        name="Buy & Hold",
        initial_balance=initial,
        final_balance=final_balance,
        n_trades=1,
        n_wins=1 if final_balance > initial else 0,
        n_losses=0 if final_balance > initial else 1,
        win_rate=1.0 if final_balance > initial else 0.0,
        max_drawdown_pct=compute_drawdown(equity),
        total_return_pct=(final_balance - initial) / initial,
        cagr_pct=compute_cagr(initial, final_balance, days),
        sharpe=compute_sharpe(daily_ret),
        notes="단순 매수 후 보유 (수수료 0.055% × 2회)",
    )


# ────────────────────────────────────────────────────────────
# 전략 2: DCA (매주 월요일 정액 매수)
# ────────────────────────────────────────────────────────────

def strategy_dca(candles: List[Candle], initial: float = 1000.0,
                 weekly_amount: float = 0.0, weeks: int = 0,
                 fee_pct: float = 0.00055) -> StrategyResult:
    """DCA: 시드 1000을 N주에 걸쳐 분할 매수.

    weekly_amount=0이면 자동 계산 (initial / weeks).
    weeks=0이면 전체 기간 매주.
    """
    days_total = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400
    if weeks == 0:
        weeks = max(1, int(days_total / 7))
    if weekly_amount == 0:
        weekly_amount = initial / weeks

    cash = initial
    btc_qty = 0.0
    n_buys = 0
    equity_curve = []
    last_buy_week = -1

    for c in candles:
        dt = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc)
        # 월요일 (weekday 0) + 주가 바뀌었으면 매수
        week_num = (c.timestamp / 1000) // (7 * 86400)
        if dt.weekday() == 0 and week_num != last_buy_week and cash >= weekly_amount and n_buys < weeks:
            buy_amt = min(weekly_amount, cash)
            qty = buy_amt * (1 - fee_pct) / c.close
            btc_qty += qty
            cash -= buy_amt
            n_buys += 1
            last_buy_week = week_num
        equity_curve.append(cash + btc_qty * c.close)

    final_price = candles[-1].close
    final_balance = cash + btc_qty * final_price * (1 - fee_pct)  # 가상 청산 수수료

    daily_ret = [(equity_curve[i] / equity_curve[i-1] - 1) if equity_curve[i-1] > 0 else 0
                 for i in range(1, len(equity_curve))]

    return StrategyResult(
        name=f"DCA (주간 ${weekly_amount:.0f}, {n_buys}회 매수)",
        initial_balance=initial,
        final_balance=final_balance,
        n_trades=n_buys,
        n_wins=n_buys if final_balance > initial else 0,
        n_losses=0 if final_balance > initial else n_buys,
        win_rate=1.0 if final_balance > initial else 0.0,
        max_drawdown_pct=compute_drawdown(equity_curve),
        total_return_pct=(final_balance - initial) / initial,
        cagr_pct=compute_cagr(initial, final_balance, days_total),
        sharpe=compute_sharpe(daily_ret),
        notes=f"매주 월요일 ${weekly_amount:.0f} 매수, 분할 {weeks}회",
    )


# ────────────────────────────────────────────────────────────
# 전략 3: 변동성 돌파 (Larry Williams)
# ────────────────────────────────────────────────────────────

def strategy_breakout(candles: List[Candle], k: float = 0.6,
                      sl_pct: float = 0.05, leverage: int = 2,
                      position_size_pct: float = 0.20,
                      trend_ma: int = 20, initial: float = 1000.0,
                      fee_pct: float = 0.00055,
                      slippage_pct: float = 0.0005) -> StrategyResult:
    balance = initial
    equity_curve = [initial]
    n_trades = 0
    n_wins = 0
    n_losses = 0

    sizing_factor = position_size_pct * leverage
    fee_round_trip = fee_pct * 2 + slippage_pct * 2

    closes_history: List[float] = []

    for i in range(1, len(candles) - 1):
        yesterday = candles[i - 1]
        today = candles[i]
        tomorrow = candles[i + 1]

        closes_history.append(yesterday.close)

        prev_range = yesterday.high - yesterday.low
        if today.open <= 0 or prev_range / today.open < 0.01:
            equity_curve.append(balance)
            continue

        if trend_ma > 0:
            if len(closes_history) < trend_ma:
                equity_curve.append(balance)
                continue
            ma = sum(closes_history[-trend_ma:]) / trend_ma
            if yesterday.close < ma:
                equity_curve.append(balance)
                continue

        target = today.open + prev_range * k
        if today.high < target:
            equity_curve.append(balance)
            continue

        entry = target
        sl = entry * (1 - sl_pct)

        if today.low <= sl:
            exit_price = sl
        else:
            exit_price = tomorrow.open

        pnl_pct = (exit_price - entry) / entry
        pnl_pct_net = sizing_factor * (pnl_pct - fee_round_trip)
        balance *= (1 + pnl_pct_net)
        n_trades += 1
        if pnl_pct_net > 0:
            n_wins += 1
        else:
            n_losses += 1
        equity_curve.append(balance)

    days = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400
    daily_ret = [(equity_curve[i] / equity_curve[i-1] - 1) if equity_curve[i-1] > 0 else 0
                 for i in range(1, len(equity_curve))]

    return StrategyResult(
        name=f"변동성 돌파 K={k} SL={sl_pct*100:.0f}% MA{trend_ma} {leverage}x",
        initial_balance=initial,
        final_balance=balance,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / n_trades if n_trades else 0.0,
        max_drawdown_pct=compute_drawdown(equity_curve),
        total_return_pct=(balance - initial) / initial,
        cagr_pct=compute_cagr(initial, balance, days),
        sharpe=compute_sharpe(daily_ret),
    )


# ────────────────────────────────────────────────────────────
# 전략 4: 이평 모멘텀 (5/20 골든크로스)
# ────────────────────────────────────────────────────────────

def strategy_ma_crossover(candles: List[Candle], fast: int = 5, slow: int = 20,
                          initial: float = 1000.0, fee_pct: float = 0.00055,
                          slippage_pct: float = 0.0005) -> StrategyResult:
    balance = initial
    in_position = False
    entry_price = 0.0
    equity_curve = [initial]
    btc_qty = 0.0
    n_trades = 0
    n_wins = 0
    n_losses = 0

    closes: List[float] = []

    for c in candles:
        closes.append(c.close)

        if len(closes) < slow + 1:
            equity_curve.append(balance + btc_qty * c.close if in_position else balance)
            continue

        ma_fast = sum(closes[-fast:]) / fast
        ma_slow = sum(closes[-slow:]) / slow
        ma_fast_prev = sum(closes[-fast-1:-1]) / fast
        ma_slow_prev = sum(closes[-slow-1:-1]) / slow

        # 골든크로스 → 진입
        if not in_position and ma_fast_prev <= ma_slow_prev and ma_fast > ma_slow:
            entry_price = c.close * (1 + slippage_pct)
            btc_qty = balance * (1 - fee_pct) / entry_price
            balance = 0
            in_position = True

        # 데드크로스 → 청산
        elif in_position and ma_fast_prev >= ma_slow_prev and ma_fast < ma_slow:
            exit_price = c.close * (1 - slippage_pct)
            balance = btc_qty * exit_price * (1 - fee_pct)
            n_trades += 1
            if exit_price > entry_price:
                n_wins += 1
            else:
                n_losses += 1
            btc_qty = 0
            in_position = False

        equity = balance + btc_qty * c.close if in_position else balance
        equity_curve.append(equity)

    if in_position:
        balance = btc_qty * candles[-1].close * (1 - fee_pct)

    days = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400
    daily_ret = [(equity_curve[i] / equity_curve[i-1] - 1) if equity_curve[i-1] > 0 else 0
                 for i in range(1, len(equity_curve))]

    return StrategyResult(
        name=f"MA 모멘텀 ({fast}/{slow})",
        initial_balance=initial,
        final_balance=balance,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / n_trades if n_trades else 0.0,
        max_drawdown_pct=compute_drawdown(equity_curve),
        total_return_pct=(balance - initial) / initial,
        cagr_pct=compute_cagr(initial, balance, days),
        sharpe=compute_sharpe(daily_ret),
        notes="골든크로스 진입 / 데드크로스 청산 (현물 1x)",
    )


# ────────────────────────────────────────────────────────────
# 전략 5: RSI 평균회귀
# ────────────────────────────────────────────────────────────

def compute_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def strategy_rsi_meanrevert(candles: List[Candle], rsi_period: int = 14,
                             buy_thr: float = 30, sell_thr: float = 70,
                             initial: float = 1000.0, fee_pct: float = 0.00055,
                             slippage_pct: float = 0.0005) -> StrategyResult:
    balance = initial
    in_position = False
    entry_price = 0.0
    btc_qty = 0.0
    equity_curve = [initial]
    n_trades = 0
    n_wins = 0
    n_losses = 0
    closes: List[float] = []

    for c in candles:
        closes.append(c.close)
        if len(closes) < rsi_period + 1:
            equity_curve.append(balance)
            continue

        rsi = compute_rsi(closes, rsi_period)

        if not in_position and rsi <= buy_thr:
            entry_price = c.close * (1 + slippage_pct)
            btc_qty = balance * (1 - fee_pct) / entry_price
            balance = 0
            in_position = True
        elif in_position and rsi >= sell_thr:
            exit_price = c.close * (1 - slippage_pct)
            balance = btc_qty * exit_price * (1 - fee_pct)
            n_trades += 1
            if exit_price > entry_price:
                n_wins += 1
            else:
                n_losses += 1
            btc_qty = 0
            in_position = False

        equity = balance + btc_qty * c.close if in_position else balance
        equity_curve.append(equity)

    if in_position:
        balance = btc_qty * candles[-1].close * (1 - fee_pct)

    days = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400
    daily_ret = [(equity_curve[i] / equity_curve[i-1] - 1) if equity_curve[i-1] > 0 else 0
                 for i in range(1, len(equity_curve))]

    return StrategyResult(
        name=f"RSI 평균회귀 ({buy_thr}/{sell_thr})",
        initial_balance=initial,
        final_balance=balance,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / n_trades if n_trades else 0.0,
        max_drawdown_pct=compute_drawdown(equity_curve),
        total_return_pct=(balance - initial) / initial,
        cagr_pct=compute_cagr(initial, balance, days),
        sharpe=compute_sharpe(daily_ret),
        notes=f"RSI<={buy_thr} 매수 / RSI>={sell_thr} 매도 (현물 1x)",
    )


# ────────────────────────────────────────────────────────────
# 출력
# ────────────────────────────────────────────────────────────

def print_table(results: List[StrategyResult], data_label: str):
    print(f"\n{'='*100}")
    print(f"📊 전략 비교 — {data_label}")
    print(f"{'='*100}")
    print(f"{'전략':50} {'거래':>5} {'승률':>6} {'CAGR':>8} {'누적':>9} {'최대DD':>8} {'Sharpe':>7}")
    print(f"{'─'*100}")
    for r in results:
        name = r.name[:48]
        print(f"{name:50} {r.n_trades:>5} {r.win_rate*100:>5.1f}% "
              f"{r.cagr_pct*100:>+7.1f}% {r.total_return_pct*100:>+8.1f}% "
              f"{r.max_drawdown_pct*100:>7.1f}% {r.sharpe:>7.2f}")

    bh = next((r for r in results if r.name == "Buy & Hold"), None)
    if bh:
        print(f"\n{'─'*100}")
        print(f"💡 Buy & Hold 대비 초과 수익률 (Alpha)")
        print(f"{'─'*100}")
        for r in results:
            if r.name == "Buy & Hold":
                continue
            alpha = r.total_return_pct - bh.total_return_pct
            outperform = "🟢" if alpha > 0 else "🔴"
            print(f"{outperform} {r.name[:60]:60} {alpha*100:>+8.1f}%p")


def main():
    parser = argparse.ArgumentParser(description="KAIROS 전략 종합 비교")
    parser.add_argument("--source", default="4h_long",
                        choices=["d", "4h_long", "1h_long"],
                        help="데이터 소스: d(2년 일봉) / 4h_long(6년→일봉) / 1h_long(6년→일봉)")
    args = parser.parse_args()

    if args.source == "d":
        candles = load_candles(DATA_DIR / "BTCUSDT_D.csv")
        label = "BTC 일봉 (2년 직접)"
    elif args.source == "4h_long":
        intra = load_candles(DATA_DIR / "BTCUSDT_240_long.csv")
        candles = aggregate_to_daily(intra)
        label = "BTC 일봉 (6년, 4H→일봉 집계)"
    else:
        intra = load_candles(DATA_DIR / "BTCUSDT_60_long.csv")
        candles = aggregate_to_daily(intra)
        label = "BTC 일봉 (6년, 1H→일봉 집계)"

    print(f"\n데이터: {label} · {len(candles)}일")
    print(f"기간: {date_str(candles[0].timestamp)} ~ {date_str(candles[-1].timestamp)}")

    results = []
    results.append(strategy_buy_and_hold(candles))
    results.append(strategy_dca(candles))
    results.append(strategy_breakout(candles, k=0.6, sl_pct=0.05, leverage=2,
                                     position_size_pct=0.20, trend_ma=20))
    results.append(strategy_breakout(candles, k=0.6, sl_pct=0.05, leverage=3,
                                     position_size_pct=0.30, trend_ma=20))
    results.append(strategy_breakout(candles, k=0.5, sl_pct=0.03, leverage=2,
                                     position_size_pct=0.20, trend_ma=20))
    results.append(strategy_breakout(candles, k=0.7, sl_pct=0.05, leverage=2,
                                     position_size_pct=0.20, trend_ma=50))
    results.append(strategy_ma_crossover(candles, fast=5, slow=20))
    results.append(strategy_ma_crossover(candles, fast=10, slow=50))
    results.append(strategy_ma_crossover(candles, fast=20, slow=100))
    results.append(strategy_rsi_meanrevert(candles, rsi_period=14, buy_thr=30, sell_thr=70))
    results.append(strategy_rsi_meanrevert(candles, rsi_period=14, buy_thr=25, sell_thr=75))

    print_table(results, label)


if __name__ == "__main__":
    main()
