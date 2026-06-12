"""리스크 관리 레이어.

- 포지션 크기 계산 (risk per trade 역산)
- 일일 손실 한도 / 드로다운 셧다운
- 펀딩 정산 전후 진입 차단
- 연패 카운터 (현재 쿨다운 로직은 제거됨, 통계용)
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from config import get_logger, TRADING

logger = get_logger("risk_manager")


@dataclass
class DailyStats:
    """하루 단위 거래 통계. 자정(로컬) 기준 자동 리셋."""
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    halted: bool = False

    def reset(self):
        """로컬 날짜가 바뀌었으면 카운터를 0으로 되돌린다. 호출은 메서드 진입 시마다 가능."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.date != today:
            self.date = today
            self.trades = 0
            self.wins = 0
            self.losses = 0
            self.consecutive_losses = 0
            self.daily_pnl = 0.0
            self.halted = False


class RiskManager:
    """사이징, 일일 한도, 최대 DD, 펀딩 회피를 한 곳에서 관리."""

    def __init__(self):
        self._lock = threading.Lock()
        self._daily = DailyStats()
        self._initial_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._shutdown_triggered: bool = False

    def initialize(self, balance: float):
        """부팅 또는 입금 시 피크 잔고를 재설정. 기존 피크가 더 크면 유지한다."""
        with self._lock:
            self._initial_balance = balance
            self._peak_balance = max(self._peak_balance, balance)
            self._daily.reset()
            logger.info(f"RiskManager 초기화: 잔액={balance:.2f} 피크={self._peak_balance:.2f}")

    def can_trade(self, current_balance: float, active_positions: int = 0) -> tuple:
        """전 조건을 통과해야 True. 하나라도 막히면 이유 문자열을 돌려준다."""
        with self._lock:
            self._daily.reset()

            # 셧다운 상태
            if self._shutdown_triggered:
                return False, "시스템 셧다운 (최대 드로다운 초과)"

            # 일일 거래 중단 (일일 손실 한도 기준만 유지)
            if self._daily.halted:
                return False, "일일 손실 한도 초과로 거래 중단"

            # 일일 거래 횟수
            if self._daily.trades >= TRADING.MAX_DAILY_TRADES:
                return False, f"일일 거래 한도 초과 ({TRADING.MAX_DAILY_TRADES}회)"

            # 일일 손실 한도 (현재 잔고 기준 — 입금/출금 자동 반영)
            if current_balance > 0 and self._daily.daily_pnl < 0:
                daily_loss_pct = abs(self._daily.daily_pnl) / current_balance
                if daily_loss_pct >= TRADING.MAX_DAILY_LOSS_PCT:
                    self._daily.halted = True
                    return False, f"일일 손실 한도 초과 ({daily_loss_pct*100:.1f}%)"

            # 최대 드로다운
            self._peak_balance = max(self._peak_balance, current_balance)
            if self._peak_balance > 0:
                drawdown = (self._peak_balance - current_balance) / self._peak_balance
                if drawdown >= TRADING.MAX_DRAWDOWN_PCT:
                    self._shutdown_triggered = True
                    return False, f"최대 드로다운 초과 ({drawdown*100:.1f}%)"

            # 펀딩비 회피
            if self._is_near_funding():
                return False, "펀딩 정산 시간 근접"

            # 동시 포지션 한도
            if active_positions >= TRADING.MAX_SIMULTANEOUS_POSITIONS:
                return False, f"동시 포지션 한도 ({TRADING.MAX_SIMULTANEOUS_POSITIONS}개)"

            return True, "OK"

    def calculate_position_size(
        self,
        current_balance: float,
        entry_price: float,
        sl_pct: float,
        max_leverage: int,
        symbol: str = "BTCUSDT"
    ) -> Dict[str, Any]:
        """목표 손실 금액 (= 잔고 × risk%) 에서 역산해 수량과 레버리지를 결정.

        드로다운 경고 구간이면 자동으로 50% 축소된다.
        """
        # 운영 정책 요청: 손실 시 시스템적 사이즈 축소 X
        # DD 단계화 제거 — 항상 풀 사이즈 진입
        size_multiplier = 1.0

        # 마진 계산 (동시 포지션 수로 분배)
        available_margin = current_balance * TRADING.MARGIN_USAGE_PCT * size_multiplier / TRADING.MAX_SIMULTANEOUS_POSITIONS

        risk_amount = current_balance * TRADING.RISK_PER_TRADE_PCT * size_multiplier
        sl_ratio = sl_pct / 100.0

        if sl_ratio <= 0:
            logger.error("SL 거리가 0 이하")
            return None

        ideal_position = risk_amount / sl_ratio
        needed_leverage = ideal_position / available_margin
        leverage = min(int(needed_leverage), max_leverage)
        leverage = max(leverage, TRADING.MIN_LEVERAGE)

        position_value = available_margin * leverage
        quantity = position_value / entry_price

        min_qty = TRADING.MIN_ORDER_QTYS.get(symbol, 0.001)
        qty_precision = TRADING.QTY_PRECISIONS.get(symbol, 3)

        quantity = round(quantity, qty_precision)

        if quantity < min_qty:
            logger.warning(f"주문 수량 부족: {quantity} < {min_qty}")
            return None

        actual_risk = position_value * sl_ratio
        if actual_risk > risk_amount * 1.1:
            logger.warning(f"리스크 초과: {actual_risk:.2f} > {risk_amount:.2f}")
            # 레버리지 하향 조정
            leverage = max(1, leverage - 1)
            position_value = available_margin * leverage
            quantity = round(position_value / entry_price, qty_precision)
            if quantity < min_qty:
                return None

        # 청산가
        mmr = TRADING.MAINTENANCE_MARGIN_RATE
        liq_long = entry_price * (1 - 1/leverage + mmr) if leverage > 1 else 0
        liq_short = entry_price * (1 + 1/leverage - mmr) if leverage > 1 else entry_price * 100

        return {
            "margin": round(available_margin, 2),
            "leverage": leverage,
            "quantity": quantity,
            "position_value": round(position_value, 2),
            "risk_amount": round(actual_risk, 2),
            "liquidation_price_long": round(liq_long, 2),
            "liquidation_price_short": round(liq_short, 2),
        }

    def record_trade_result(self, pnl: float, is_win: bool):
        """일일 카운터와 연패 스트릭을 갱신. 포지션이 닫힐 때마다 호출."""
        with self._lock:
            self._daily.reset()
            self._daily.trades += 1
            self._daily.daily_pnl += pnl

            if is_win:
                self._daily.wins += 1
                self._daily.consecutive_losses = 0
            else:
                self._daily.losses += 1
                self._daily.consecutive_losses += 1

                if self._daily.consecutive_losses >= 5:
                    logger.warning(f"연속 {self._daily.consecutive_losses}패 주의")

            logger.info(
                f"거래 기록: {'승' if is_win else '패'} | PnL={pnl:+.2f} | "
                f"일일: {self._daily.wins}승 {self._daily.losses}패 | "
                f"연속패: {self._daily.consecutive_losses}"
            )

    def _is_near_funding(self) -> bool:
        """UTC 00/08/16 시 전 N분 이내면 진입 차단 (펀딩 비용 회피)."""
        now = datetime.utcnow()
        avoidance = TRADING.FUNDING_AVOIDANCE_MINUTES

        # 다음 펀딩 시간 (00:00, 08:00, 16:00 UTC)
        current_hour = now.hour
        if current_hour < 8:
            next_funding_hour = 8
        elif current_hour < 16:
            next_funding_hour = 16
        else:
            next_funding_hour = 24  # 다음 날 00:00

        next_funding = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_funding += timedelta(hours=next_funding_hour)

        minutes_until = (next_funding - now).total_seconds() / 60
        if 0 <= minutes_until <= avoidance:
            logger.info(f"펀딩 정산 {minutes_until:.0f}분 전 → 진입 보류")
            return True

        return False

    def get_daily_stats(self) -> Dict[str, Any]:
        """리포트/텔레그램용 일일 통계 스냅샷."""
        with self._lock:
            self._daily.reset()
            return {
                "date": self._daily.date,
                "trades": self._daily.trades,
                "wins": self._daily.wins,
                "losses": self._daily.losses,
                "consecutive_losses": self._daily.consecutive_losses,
                "daily_pnl": round(self._daily.daily_pnl, 2),
                "halted": self._daily.halted,
                "shutdown": self._shutdown_triggered
            }

    def reset_shutdown(self):
        """DD 셧다운을 운영자가 수동으로 해제할 때 사용."""
        with self._lock:
            self._shutdown_triggered = False
            logger.info("셧다운 해제됨")
