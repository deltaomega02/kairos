"""KAIROS v2 — Larry Williams 변동성 돌파 백테스트.

일봉 OHLCV 데이터로 다음을 시뮬레이션:
1. 매일 시가 기준 target = open + range_yesterday × K
2. 그날 high ≥ target → target 가격에 LONG 진입
3. 그날 low ≤ entry × (1 - sl_pct) → SL 청산
4. 아니면 다음날 시가에 청산

출력:
- K값 sweep 결과 표
- 최적 K의 자본 곡선 (텍스트)
- 핵심 지표: 누적 수익률, 거래 수, 승률, 평균 R, 최대 DD

사용:
    python3 -m backtest.breakout_backtest
    python3 -m backtest.breakout_backtest --k 0.6
    python3 -m backtest.breakout_backtest --sweep 0.3 0.9 0.05
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


DATA_PATH = Path(__file__).parent / "data" / "BTCUSDT_D.csv"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    date_ms: int
    entry: float
    exit: float
    reason: str  # "SL" or "NEXT_OPEN"
    pnl_pct: float          # 가격 변화율 (수수료 X)
    pnl_pct_net: float      # 수수료 차감 후 (시드 대비)


@dataclass
class BacktestResult:
    k: float
    initial_balance: float
    final_balance: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    rr: float                  # 평균 승 / 평균 패
    max_drawdown_pct: float
    total_return_pct: float
    cagr_pct: float            # 연환산 (단순)
    trades: List[Trade] = field(default_factory=list)


def load_daily_candles(path: Path) -> List[Candle]:
    candles: List[Candle] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def _ma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def run_backtest(
    candles: List[Candle],
    k: float = 0.6,
    sl_pct: float = 0.02,
    min_range_pct: float = 0.01,
    fee_pct: float = 0.00055,        # taker 0.055% × 2회 (진입+청산)
    slippage_pct: float = 0.0005,    # 슬리피지 0.05%
    leverage: int = 3,
    position_size_pct: float = 0.30, # 시드의 30%를 마진으로
    initial_balance: float = 1000.0,
    trend_ma: int = 0,               # 0이면 비활성. >0이면 어제 close가 trend_ma일 이평 위에서만 진입
) -> BacktestResult:
    """변동성 돌파 백테스트.

    포지션 사이징:
        margin = balance × position_size_pct
        notional = margin × leverage
        가격이 1% 변하면 시드 대비: position_size_pct × leverage × 1%

    예: position_size=0.3, lev=3 → 가격 1% = 시드 0.9%
    """
    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0

    trades: List[Trade] = []

    sizing_factor = position_size_pct * leverage  # 시드 대비 가격변화 배수
    fee_round_trip = fee_pct * 2 + slippage_pct * 2  # 왕복 수수료+슬리피지 (가격 대비)

    closes_history: List[float] = []

    for i in range(1, len(candles) - 1):
        yesterday = candles[i - 1]
        today = candles[i]
        tomorrow = candles[i + 1]

        closes_history.append(yesterday.close)

        prev_range = yesterday.high - yesterday.low
        if today.open <= 0:
            continue

        # 변동성 필터
        if prev_range / today.open < min_range_pct:
            continue

        # 추세 필터 (어제 close가 N일 이평 위인지)
        if trend_ma > 0:
            ma = _ma(closes_history, trend_ma)
            if ma is None or yesterday.close < ma:
                continue

        target = today.open + prev_range * k

        # 그날 high가 target 이상이면 진입 (가정: target에 정확 체결)
        if today.high < target:
            continue

        entry = target
        sl = entry * (1 - sl_pct)

        # 청산 결정:
        # 1) 그날 low가 SL 이하 → SL 청산
        # 2) 아니면 다음날 시가에 청산
        if today.low <= sl:
            exit_price = sl
            reason = "SL"
        else:
            exit_price = tomorrow.open
            reason = "NEXT_OPEN"

        pnl_pct = (exit_price - entry) / entry           # 가격 변화율
        # 시드 대비: sizing_factor × pnl_pct, 수수료 차감
        pnl_pct_net = sizing_factor * (pnl_pct - fee_round_trip)

        balance *= (1 + pnl_pct_net)

        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance
        max_dd = max(max_dd, dd)

        trades.append(Trade(
            date_ms=today.timestamp,
            entry=entry,
            exit=exit_price,
            reason=reason,
            pnl_pct=pnl_pct,
            pnl_pct_net=pnl_pct_net,
        ))

    n_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct_net > 0]
    losses = [t for t in trades if t.pnl_pct_net <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades else 0.0
    avg_win = sum(t.pnl_pct_net for t in wins) / n_wins if wins else 0.0
    avg_loss = sum(t.pnl_pct_net for t in losses) / n_losses if losses else 0.0
    rr = (avg_win / -avg_loss) if avg_loss < 0 else 0.0

    total_return = (balance - initial_balance) / initial_balance

    # CAGR (단순)
    days = (candles[-1].timestamp - candles[0].timestamp) / 1000 / 86400
    years = days / 365.25
    cagr = ((balance / initial_balance) ** (1 / years) - 1) if years > 0 else 0.0

    return BacktestResult(
        k=k,
        initial_balance=initial_balance,
        final_balance=balance,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        rr=rr,
        max_drawdown_pct=max_dd,
        total_return_pct=total_return,
        cagr_pct=cagr,
        trades=trades,
    )


def print_result(r: BacktestResult, verbose: bool = False):
    print(f"\n{'='*60}")
    print(f"K = {r.k:.2f}")
    print(f"{'─'*60}")
    print(f"기간 거래        {r.n_trades}회 ({r.n_wins}승 {r.n_losses}패)")
    print(f"승률             {r.win_rate*100:.1f}%")
    print(f"평균 승          {r.avg_win_pct*100:+.2f}%/거래 (시드 대비)")
    print(f"평균 패          {r.avg_loss_pct*100:+.2f}%/거래 (시드 대비)")
    print(f"R:R              {r.rr:.2f}")
    print(f"최대 DD          {r.max_drawdown_pct*100:.1f}%")
    print(f"누적 수익률      {r.total_return_pct*100:+.1f}%")
    print(f"연환산 (CAGR)    {r.cagr_pct*100:+.1f}%")
    print(f"최종 잔고        ${r.final_balance:,.2f} (시작 ${r.initial_balance:,.2f})")

    if verbose and r.trades:
        print(f"\n최근 10거래:")
        for t in r.trades[-10:]:
            from datetime import datetime, timezone
            d = datetime.fromtimestamp(t.date_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {d} {t.reason:9} entry=${t.entry:,.0f} exit=${t.exit:,.0f} "
                  f"({t.pnl_pct*100:+.2f}% / 시드 {t.pnl_pct_net*100:+.2f}%)")


def sweep(candles: List[Candle], k_min: float, k_max: float, k_step: float,
          **kwargs) -> List[BacktestResult]:
    """K값 sweep."""
    results = []
    k = k_min
    while k <= k_max + 1e-9:
        r = run_backtest(candles, k=k, **kwargs)
        results.append(r)
        k += k_step
    return results


def print_sweep(results: List[BacktestResult]):
    print(f"\n{'='*78}")
    print(f"K값 SWEEP 결과")
    print(f"{'='*78}")
    print(f"{'K':>5}  {'거래':>5}  {'승률':>6}  {'R:R':>5}  {'누적%':>9}  "
          f"{'CAGR%':>8}  {'최대DD%':>8}")
    print(f"{'─'*78}")
    for r in results:
        print(f"{r.k:>5.2f}  {r.n_trades:>5}  {r.win_rate*100:>5.1f}%  "
              f"{r.rr:>5.2f}  {r.total_return_pct*100:>+8.1f}%  "
              f"{r.cagr_pct*100:>+7.1f}%  {r.max_drawdown_pct*100:>7.1f}%")

    # 최적 K
    best = max(results, key=lambda r: r.total_return_pct)
    print(f"\n→ 최적 K = {best.k:.2f} (누적 {best.total_return_pct*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="KAIROS 변동성 돌파 백테스트")
    parser.add_argument("--k", type=float, default=None, help="단일 K값 백테스트")
    parser.add_argument("--sweep", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                        default=None, help="K값 sweep")
    parser.add_argument("--sl", type=float, default=0.02, help="SL %% (기본 0.02)")
    parser.add_argument("--lev", type=int, default=3, help="레버리지 (기본 3)")
    parser.add_argument("--size", type=float, default=0.30,
                        help="포지션 사이즈 (시드 대비, 기본 0.30)")
    parser.add_argument("--seed", type=float, default=1000.0, help="시작 잔고")
    parser.add_argument("--ma", type=int, default=0,
                        help="추세 필터: N일 이평 위에서만 진입 (0=비활성)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="최근 10거래 상세 출력")
    args = parser.parse_args()

    candles = load_daily_candles(DATA_PATH)
    print(f"데이터: {DATA_PATH.name} · {len(candles)}일 일봉")
    from datetime import datetime, timezone
    s = datetime.fromtimestamp(candles[0].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    e = datetime.fromtimestamp(candles[-1].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"기간: {s} ~ {e}")

    common = dict(sl_pct=args.sl, leverage=args.lev,
                  position_size_pct=args.size, initial_balance=args.seed,
                  trend_ma=args.ma)

    if args.sweep:
        results = sweep(candles, args.sweep[0], args.sweep[1], args.sweep[2], **common)
        print_sweep(results)
    elif args.k is not None:
        r = run_backtest(candles, k=args.k, **common)
        print_result(r, verbose=args.verbose)
    else:
        # 기본: K=0.6 + 자동 sweep 0.3~0.9
        r = run_backtest(candles, k=0.6, **common)
        print_result(r, verbose=args.verbose)
        results = sweep(candles, 0.30, 0.90, 0.05, **common)
        print_sweep(results)


if __name__ == "__main__":
    main()
