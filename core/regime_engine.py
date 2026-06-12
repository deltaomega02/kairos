"""4H 레짐 판독 엔진.

심볼별로 ADX / EMA / MACD / ATR 퍼센타일을 조합해 상태를 분류한다.
히스테리시스 (진입 임계값 > 이탈 임계값) 와 디바운스 (연속 봉 수 요구) 로
잦은 전환을 억제하고, 1H 급변 감지 시 HIGH_VOL 긴급 오버라이드를 건다.
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import get_logger, PARAM_REGISTRY

logger = get_logger("regime_engine")


class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class RegimeResult:
    """레짐 판독 한 번의 결과 스냅샷."""
    regime: MarketRegime
    confidence: int          # 0~10, 높을수록 판정 신뢰도
    adx: float
    atr_percentile: float
    detail: str


class RegimeEngine:
    """심볼별 4H 레짐 상태 머신."""

    EMERGENCY_COOLDOWN_SEC = 3600  # 1H 급변으로 HIGH_VOL 진입 후 복귀까지 대기

    def __init__(self):
        self._regimes: Dict[str, MarketRegime] = {}
        self._pending: Dict[str, Optional[MarketRegime]] = {}
        self._pending_counts: Dict[str, int] = {}
        self._regime_bars: Dict[str, int] = {}
        self._last_timestamps: Dict[str, int] = {}
        self._emergency_until: Dict[str, float] = {}

    def _get_regime(self, symbol: str) -> MarketRegime:
        return self._regimes.get(symbol, MarketRegime.RANGING)

    def get_regime(self, symbol: str) -> MarketRegime:
        """심볼의 현재 확정 레짐."""
        return self._get_regime(symbol)

    def update(self, symbol: str, indicators_4h: Dict[str, Any]) -> RegimeResult:
        """새 4H 지표로 레짐을 재평가하고 필요하면 전환을 확정한다."""
        current_regime = self._get_regime(symbol)
        timestamp = indicators_4h.get("timestamp", 0)

        if timestamp == self._last_timestamps.get(symbol, 0):
            return RegimeResult(
                current_regime, 5,
                indicators_4h.get("adx", 0),
                indicators_4h.get("atr_percentile", 50),
                "캐시된 레짐"
            )

        self._last_timestamps[symbol] = timestamp

        # 긴급 쿨다운 중
        emergency_until = self._emergency_until.get(symbol, 0)
        if emergency_until > 0 and time.time() < emergency_until:
            remaining = int(emergency_until - time.time())
            return RegimeResult(
                MarketRegime.HIGH_VOL, 9,
                indicators_4h.get("adx", 0),
                indicators_4h.get("atr_percentile", 50),
                f"긴급 쿨다운 {remaining}초"
            )

        # 쿨다운 만료
        if emergency_until > 0:
            self._emergency_until[symbol] = 0

        adx_enter = PARAM_REGISTRY.get("adx_enter_trending")
        adx_exit = PARAM_REGISTRY.get("adx_exit_trending")
        high_vol_pctl = PARAM_REGISTRY.get("atr_high_vol_percentile")
        debounce_bars = PARAM_REGISTRY.get_int("regime_debounce_bars")

        adx = indicators_4h["adx"]
        atr_percentile = indicators_4h["atr_percentile"]
        plus_di = indicators_4h["plus_di"]
        minus_di = indicators_4h["minus_di"]
        ema_9 = indicators_4h["ema_9"]
        ema_21 = indicators_4h["ema_21"]
        macd_hist = indicators_4h["macd_histogram"]

        # 고변동
        if atr_percentile >= high_vol_pctl:
            raw_regime = MarketRegime.HIGH_VOL
            confidence = 8
            detail = f"ATR퍼센타일 {atr_percentile:.0f}% ≥ {high_vol_pctl:.0f}%"

        # 추세
        elif self._is_trending(symbol, adx, adx_enter, adx_exit):
            raw_regime, confidence, detail = self._determine_direction(
                plus_di, minus_di, ema_9, ema_21, macd_hist, adx
            )

        else:
            raw_regime = MarketRegime.RANGING
            confidence = 5
            detail = f"ADX {adx:.1f} (추세부재)"

        # 디바운스 적용
        final_regime = self._apply_debounce(symbol, raw_regime, debounce_bars)

        result = RegimeResult(final_regime, confidence, adx, atr_percentile, detail)

        if final_regime != current_regime:
            old = current_regime
            self._regimes[symbol] = final_regime
            self._regime_bars[symbol] = 0
            logger.info(f"[{symbol}] 레짐: {old.value} → {final_regime.value} | {detail}")
        else:
            self._regime_bars[symbol] = self._regime_bars.get(symbol, 0) + 1

        return result

    def emergency_override(self, symbol: str, indicators_1h: Dict[str, Any]) -> bool:
        """1H 3% 이상 변동이나 ATR 2% 이상이면 HIGH_VOL 로 강제 전환하고 쿨다운 세팅."""
        now = time.time()
        current_regime = self._get_regime(symbol)
        emergency_until = self._emergency_until.get(symbol, 0)

        if emergency_until > 0:
            if now < emergency_until:
                return current_regime == MarketRegime.HIGH_VOL
            else:
                self._emergency_until[symbol] = 0
                self._regimes[symbol] = MarketRegime.RANGING
                logger.info(f"[{symbol}] 긴급 쿨다운 만료")
                return False

        if current_regime == MarketRegime.HIGH_VOL:
            return False

        atr_pct = indicators_1h.get("atr_pct", 0)
        close = indicators_1h.get("close", 0)
        open_price = indicators_1h.get("open", 0)

        triggered = False
        reason = ""

        # 1H 캔들 3% 이상 변동
        if close > 0 and open_price > 0:
            move = abs(close - open_price) / open_price * 100
            if move >= 3.0:
                triggered = True
                reason = f"1H봉 {move:.1f}% 변동"

        # ATR 극단
        if not triggered and atr_pct >= 2.0:
            triggered = True
            reason = f"1H ATR {atr_pct:.2f}%"

        if triggered:
            self._regimes[symbol] = MarketRegime.HIGH_VOL
            self._emergency_until[symbol] = now + self.EMERGENCY_COOLDOWN_SEC
            self._pending[symbol] = None
            self._pending_counts[symbol] = 0
            logger.warning(f"[{symbol}] 긴급 오버라이드: {reason} → HIGH_VOL (1시간 쿨다운)")
            return True

        return False

    def _is_trending(self, symbol: str, adx, enter_threshold, exit_threshold) -> bool:
        """이미 추세면 이탈 임계값만 넘기면 유지, 아니면 진입 임계값을 요구 (히스테리시스)."""
        current_regime = self._get_regime(symbol)
        currently_trending = current_regime in (
            MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN
        )
        if currently_trending:
            return adx >= exit_threshold
        return adx >= enter_threshold

    def _determine_direction(self, plus_di, minus_di, ema_9, ema_21, macd_hist, adx):
        """+DI/-DI, EMA 배열, MACD 부호 3표 중 2표 이상으로 방향 결정."""
        bullish = bearish = 0
        reasons = []

        if plus_di > minus_di:
            bullish += 1
            reasons.append(f"+DI>-DI")
        else:
            bearish += 1
            reasons.append(f"-DI>+DI")

        if ema_9 > ema_21:
            bullish += 1
            reasons.append("EMA9>21")
        else:
            bearish += 1
            reasons.append("EMA9<21")

        if macd_hist > 0:
            bullish += 1
            reasons.append("MACD>0")
        else:
            bearish += 1
            reasons.append("MACD<0")

        detail = f"ADX={adx:.1f} | {', '.join(reasons)}"

        if bullish >= 2:
            conf = 6 + min(bullish, 3) + (1 if adx >= 35 else 0)
            return MarketRegime.TRENDING_UP, min(conf, 10), detail
        elif bearish >= 2:
            conf = 6 + min(bearish, 3) + (1 if adx >= 35 else 0)
            return MarketRegime.TRENDING_DOWN, min(conf, 10), detail
        return MarketRegime.RANGING, 4, detail

    def _apply_debounce(self, symbol: str, raw_regime, required_bars):
        """같은 새 레짐이 연속 N 봉 이어져야 전환을 확정한다 (잡음 방지)."""
        current_regime = self._get_regime(symbol)
        pending = self._pending.get(symbol)
        pending_count = self._pending_counts.get(symbol, 0)

        if raw_regime == current_regime:
            self._pending[symbol] = None
            self._pending_counts[symbol] = 0
            return current_regime

        if raw_regime == pending:
            pending_count += 1
            self._pending_counts[symbol] = pending_count
        else:
            self._pending[symbol] = raw_regime
            self._pending_counts[symbol] = 1
            pending_count = 1

        if pending_count >= required_bars:
            self._pending[symbol] = None
            self._pending_counts[symbol] = 0
            return raw_regime

        return current_regime

    def force_regime(self, symbol: str, regime: MarketRegime):
        """운영자가 외부에서 레짐을 수동으로 덮어쓸 때 사용 (테스트/긴급 조치)."""
        current_regime = self._get_regime(symbol)
        logger.warning(f"[{symbol}] 강제: {current_regime.value} → {regime.value}")
        self._regimes[symbol] = regime
        self._pending[symbol] = None
        self._pending_counts[symbol] = 0
