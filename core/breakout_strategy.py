"""KAIROS v2 — Larry Williams 변동성 돌파 전략.

매일 시가 기준으로 어제 변동폭(range = high - low)의 K배만큼 돌파하면
LONG 진입. 다음날 시가에 청산. SL은 진입가 -2%.

핵심 단순함:
- 지표 1개 (어제의 고가/저가/오늘 시가)
- 시간봉 1개 (1D, UTC)
- 방향 1개 (LONG only)
- 코인 1개 (BTCUSDT)
- 진입 1회/일

이 단순함이 KAIROS의 정체성. 추가 지표/필터를 넣고 싶을 때
안티패턴 룰(공개 저장소에는 미포함된 내부 문서)을 다시 읽을 것.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from config import BREAKOUT, get_logger

logger = get_logger("breakout")


@dataclass
class BreakoutSignal:
    """돌파 진입 신호."""
    direction: str            # "LONG" (KAIROS는 LONG only)
    entry_price: float        # 진입가 (현재가)
    target_price: float       # 돌파 임계값 (시가 + range × K)
    stop_loss: float          # SL (진입가 × (1 - SL_PCT))
    prev_range: float         # 어제 range (high - low)
    today_open: float         # 오늘 시가
    next_open_time_ms: int    # 다음 일봉 시작 (청산 트리거)


class BreakoutStrategy:
    """Larry Williams 변동성 돌파."""

    def __init__(self):
        self.k = BREAKOUT.K
        self.sl_pct = BREAKOUT.SL_PCT
        self.min_range_pct = BREAKOUT.MIN_RANGE_PCT
        logger.info(
            f"BreakoutStrategy 초기화 K={self.k} SL={self.sl_pct*100:.1f}% "
            f"MIN_RANGE={self.min_range_pct*100:.1f}%"
        )

    @staticmethod
    def _next_utc_midnight_ms(reference_ms: int) -> int:
        """주어진 시각 다음 UTC 00:00 (다음 일봉 시작) 의 ms 타임스탬프."""
        ref = datetime.fromtimestamp(reference_ms / 1000, tz=timezone.utc)
        next_day = (ref + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(next_day.timestamp() * 1000)

    def calculate_target(self, prev_high: float, prev_low: float,
                         today_open: float) -> float:
        """돌파 임계 가격 계산."""
        return today_open + (prev_high - prev_low) * self.k

    def evaluate(self, daily_candles: List[Dict[str, Any]],
                 current_price: float, now_ms: int) -> Optional[BreakoutSignal]:
        """진입 신호 평가.

        Args:
            daily_candles: 최근 일봉 리스트 (오래된→최신). 최소 2개 필요.
            current_price: 현재가 (지난 분봉의 종가 또는 ticker)
            now_ms: 현재 시각 ms

        Returns:
            BreakoutSignal 또는 None
        """
        if len(daily_candles) < 2:
            logger.warning(f"일봉 부족 ({len(daily_candles)}개) — 최소 2개 필요")
            return None

        yesterday = daily_candles[-2]
        today = daily_candles[-1]

        prev_range = yesterday["high"] - yesterday["low"]
        today_open = today["open"]

        # 변동성 필터
        if today_open <= 0 or prev_range / today_open < self.min_range_pct:
            return None

        target = today_open + prev_range * self.k

        if current_price < target:
            return None

        # 다음 일봉 시작 시각 = 오늘 일봉의 다음 UTC 00:00
        next_open_ms = self._next_utc_midnight_ms(today["timestamp"])

        signal = BreakoutSignal(
            direction="LONG",
            entry_price=current_price,
            target_price=target,
            stop_loss=current_price * (1 - self.sl_pct),
            prev_range=prev_range,
            today_open=today_open,
            next_open_time_ms=next_open_ms,
        )
        logger.info(
            f"진입 신호: target=${target:,.2f} 현재=${current_price:,.2f} "
            f"SL=${signal.stop_loss:,.2f} range=${prev_range:,.2f}"
        )
        return signal

    def check_exit(self, position: Dict[str, Any], current_price: float,
                   now_ms: int) -> Optional[str]:
        """청산 사유 체크.

        Args:
            position: dict {"stop_loss": ..., "next_open_time_ms": ..., "direction": "LONG"}
            current_price: 현재가
            now_ms: 현재 시각 ms

        Returns:
            "STOP_LOSS" / "NEXT_OPEN" / None
        """
        # 1. SL 우선
        if position["direction"] == "LONG" and current_price <= position["stop_loss"]:
            return "STOP_LOSS"

        # 2. 다음 일봉 시작 = 청산
        if now_ms >= position["next_open_time_ms"]:
            return "NEXT_OPEN"

        return None


breakout_strategy = BreakoutStrategy()
