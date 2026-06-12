"""진입 신호 엔진.

4H 레짐 결과와 1H 지표·오더북·펀딩·일봉 추세를 종합해 LONG/SHORT 여부를 결정한다.
Range reversion 전략은 장기 백테스트 -93% 가 확인되어 비활성화 상태.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from core.regime_engine import MarketRegime
from config import get_logger, PARAM_REGISTRY, TRADING

logger = get_logger("signal_engine")


@dataclass
class SignalResult:
    """단일 진입 신호의 스냅샷."""
    direction: str             # "LONG" | "SHORT"
    strategy: str              # "TREND_PULLBACK" | "RANGE_REVERSION"
    score: int                 # 0-100, 복합 신뢰 점수
    reason: str
    entry_price: float         # None 이면 현재가 기준 시장가
    sl_pct: float              # SL 까지 거리 (%)
    tp_pct: float              # TP 까지 거리 (%)
    max_leverage: int
    funding_bias: str          # "ALIGNED" | "NEUTRAL" | "AGAINST"


def _fee_adjusted_sl_tp(
    atr_pct: float,
    sl_mult: float,
    rr_ratio: float,
    maker_both_sides: bool = False
) -> tuple:
    """수수료를 반영해 SL/TP 거리를 계산.

    ATR 과 multiplier 로 기본 SL 을 잡은 뒤, 최소값으로 `수수료 왕복 × 3` 을 강제한다.
    TP 는 SL × R:R 이지만 `SL + 2 × 수수료` 아래로 내려가지 않도록 보정.
    실질 R:R 이 0.8 미만이면 None 을 돌려 거래 자체를 거부한다.
    """
    if maker_both_sides:
        fee_roundtrip = TRADING.MAKER_FEE_PCT * 2 * 100  # 0.04% (지정가 양방향)
    else:
        fee_roundtrip = TRADING.TAKER_FEE_PCT * 2 * 100  # 0.11% (시장가 양방향)

    raw_sl = atr_pct * sl_mult
    min_sl = fee_roundtrip * 3
    sl_pct = max(min_sl, min(5.0, raw_sl))

    raw_tp = sl_pct * rr_ratio
    min_tp = sl_pct + 2 * fee_roundtrip
    tp_pct = max(raw_tp, min_tp)

    real_profit = tp_pct - fee_roundtrip
    real_loss = sl_pct + fee_roundtrip
    real_rr = real_profit / real_loss if real_loss > 0 else 0

    if real_rr < 0.8:
        logger.warning(f"실질 R:R {real_rr:.2f} < 0.8 → 거래 불가")
        return None, None

    return sl_pct, tp_pct


class SignalEngine:
    """레짐 분기 → 전략별 평가. 현재 활성은 추세 풀백만."""

    def evaluate(
        self,
        regime: MarketRegime,
        indicators_1h: Dict[str, Any],
        indicators_4h: Dict[str, Any],
        orderbook: Optional[Dict[str, Any]] = None,
        funding_rate: float = 0.0,
        daily_trend: Optional[Dict[str, Any]] = None,
        current_hour_kst: Optional[int] = None,
        consecutive_losses: int = 0,
    ) -> Optional[SignalResult]:
        """현재 레짐과 지표를 바탕으로 진입 신호를 생성 (없으면 None).

        Args:
            daily_trend: 1D EMA 정보 {direction, slope_pct, ema, close}.
                         `d1_filter_mode` 설정에 따라 slope 방향 또는 price-above-EMA 로 해석.
            current_hour_kst: 현재 KST 시간 (0-23). 특정 시간 차단 필터용.
            consecutive_losses: 연속 패배 수. 3회 이상이면 진입 차단 (edge hunt 검증).
        """
        # V14: HIGH_VOL 진입 차단 유지 (학술적으로 ADX 30+ 매수는 통계 손실, fakeout 41%)
        # _evaluate_breakout 메서드는 코드 보존 (5/10 데이터 검토 후 활성 여부 재결정)
        if regime == MarketRegime.HIGH_VOL:
            return None

        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return self._evaluate_trend_pullback(regime, indicators_1h, indicators_4h,
                                                  orderbook, funding_rate,
                                                  daily_trend=daily_trend,
                                                  current_hour_kst=current_hour_kst,
                                                  consecutive_losses=consecutive_losses)

        # V14 백테스트 결과 6개 환경 모두에서 V13보다 -12~-83% 열위 확인
        # (2022 베어, 2023 횡보, 2024 강세, 2025 H1/H2, 2026 현재)
        # RANGE_REVERSION이 본업 환경(횡보)에서도 -76% 손실 → 본질적 결함
        # 코드 보존하되 라우팅 비활성: RANGING 진입 X (V13 운영 복귀)
        if regime == MarketRegime.RANGING:
            return None

        return None

    def _evaluate_trend_pullback(
        self, regime, ind, ind_4h, orderbook, funding_rate,
        daily_trend: Optional[Dict[str, Any]] = None,
        current_hour_kst: Optional[int] = None,
        consecutive_losses: int = 0,
    ) -> Optional[SignalResult]:
        """추세 레짐에서 EMA 풀백을 기다렸다 진입하는 기본 전략."""
        direction = "LONG" if regime == MarketRegime.TRENDING_UP else "SHORT"

        # [edge hunt 필터 1] 연속 패배 차단 (3회+ 시 진입 거부)
        max_losses = int(PARAM_REGISTRY.get("max_consecutive_losses"))
        if max_losses > 0 and consecutive_losses >= max_losses:
            return None

        # [edge hunt 필터 2] 특정 시간대 차단 (기본 19시 KST)
        blocked_hour = int(PARAM_REGISTRY.get("blocked_hour_kst"))
        if blocked_hour >= 0 and current_hour_kst is not None and current_hour_kst == blocked_hour:
            return None

        # [edge hunt 필터 3] 저변동 차단 (ATR 0.3% 미만)
        min_atr = PARAM_REGISTRY.get("min_atr_pct")
        atr_pct = ind.get("atr_pct", 0)
        if min_atr > 0 and atr_pct < min_atr:
            return None

        # 1D 추세 필터: 큰 그림이 반대 방향이면 진입을 포기한다.
        # mode 0 = EMA 기울기, mode 1 = 종가가 EMA 위/아래에 있는지.
        if PARAM_REGISTRY.get("d1_filter_enable") >= 1 and daily_trend:
            mode = int(PARAM_REGISTRY.get("d1_filter_mode"))
            if mode == 0:
                d1_dir = daily_trend.get("direction", "FLAT")
                if direction == "LONG" and d1_dir != "UP":
                    return None
                if direction == "SHORT" and d1_dir != "DOWN":
                    return None
            else:
                d1_close = daily_trend.get("close", 0)
                d1_ema = daily_trend.get("ema", 0)
                if d1_ema == 0:
                    return None
                if direction == "LONG" and d1_close <= d1_ema:
                    return None
                if direction == "SHORT" and d1_close >= d1_ema:
                    return None

        # EMA 풀백 확인
        pullback = self._check_pullback(direction, ind)
        if not pullback:
            return None

        # [edge hunt 필터 4] 오더북 비대칭 적용 — LONG에만, SHORT는 구조적 노이즈로 스킵
        ob_long_enabled = PARAM_REGISTRY.get("ob_filter_long") >= 1
        ob_short_enabled = PARAM_REGISTRY.get("ob_filter_short") >= 1
        if direction == "LONG" and ob_long_enabled:
            if not self._check_orderbook(direction, orderbook):
                return None
        elif direction == "SHORT" and ob_short_enabled:
            if not self._check_orderbook(direction, orderbook):
                return None

        score = 50
        ob_reason = "오더북확인" if (direction == "LONG" and ob_long_enabled) or (direction == "SHORT" and ob_short_enabled) else "오더북생략"
        reasons = [pullback["reason"], ob_reason]

        # RSI 가 과매수/과매도 영역이면 가중치 추가
        rsi = ind.get("rsi", 50)
        if direction == "LONG" and rsi <= PARAM_REGISTRY.get("rsi_oversold"):
            score += 20
            reasons.append(f"RSI{rsi:.0f}")
        elif direction == "SHORT" and rsi >= PARAM_REGISTRY.get("rsi_overbought"):
            score += 20
            reasons.append(f"RSI{rsi:.0f}")
        elif (direction == "LONG" and rsi < 50) or (direction == "SHORT" and rsi > 50):
            score += 10

        # 거래량이 평균 대비 높을수록 신호 신뢰도 상승
        vol = ind.get("volume_ratio", 1.0)
        if vol >= 1.3:
            score += 15
            reasons.append(f"Vol{vol:.1f}x")
        elif vol >= 1.0:
            score += 5

        # 펀딩 (V13+: SHORT는 crowded long 역풍에 구조적 취약 → AGAINST 페널티 강화)
        funding_bias = self._check_funding(direction, funding_rate)
        if funding_bias == "ALIGNED":
            score += 15
            reasons.append("펀딩순방향")
        elif funding_bias == "AGAINST":
            if direction == "SHORT":
                score -= 20  # SHORT: 펀딩 양수 = crowded long 역풍, 실전 0-3 BTC SHORT 방어
                reasons.append("펀딩역풍-SHORT페널티")
            else:
                score -= 10

        if score < PARAM_REGISTRY.get("entry_score_threshold"):
            return None

        # SL/TP 는 ATR 기반이지만 수수료 회수가 보장되게 하한을 둔다.
        sl_pct, tp_pct = _fee_adjusted_sl_tp(
            ind.get("atr_pct", 0.5),
            PARAM_REGISTRY.get("sl_atr_mult"),
            PARAM_REGISTRY.get("tp_rr_ratio")
        )
        if sl_pct is None:
            return None

        adx = ind_4h.get("adx", 25)
        max_lev = TRADING.MAX_LEVERAGE if adx >= 30 else min(TRADING.MAX_LEVERAGE, 3)

        # V13+: 강신호(score 70+) 또는 스프레드 과다 시 market, 그 외는 Bid/Ask PostOnly limit
        # - LONG: best_bid 가격으로 Buy → ask보다 낮아 maker 확정
        # - SHORT: best_ask 가격으로 Sell → bid보다 높아 maker 확정
        # - 슬립 = 스프레드 한 칸 (메이저 코인 ~0.005%), 수수료 절감 0.035%/편으로 순이득
        # - 5분 타임아웃 후 market 폴백 (position_manager에서 처리)
        close = ind.get("close", 0)
        best_bid = (orderbook or {}).get("best_bid", 0)
        best_ask = (orderbook or {}).get("best_ask", 0)
        spread_pct = (orderbook or {}).get("spread_pct", 0)

        # 결정 로직:
        # 1) score 70+ → 강신호는 즉시성 우선 (market)
        # 2) 스프레드 > 0.1% → 비정상 시장 (변동 심함, market 안전)
        # 3) bid/ask 유효값 없음 → market 폴백
        # 4) 그 외 → PostOnly limit (bid for LONG, ask for SHORT)
        entry_price = None
        if score < 70 and spread_pct < 0.1 and best_bid > 0 and best_ask > 0:
            if direction == "LONG":
                entry_price = round(best_bid, 2)
            else:
                entry_price = round(best_ask, 2)

        return SignalResult(
            direction=direction,
            strategy="TREND_PULLBACK",
            score=score,
            reason=f"풀백 {direction}: {', '.join(reasons)} (스코어{score})",
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            max_leverage=max_lev,
            funding_bias=funding_bias
        )

    def _evaluate_range_reversion(
        self, ind, ind_4h, orderbook, funding_rate
    ) -> Optional[SignalResult]:
        """V14: 횡보장 전용 BB 반전 전략. V13 안전망(SL ATR×2.0) 정렬."""
        close = ind.get("close", 0)
        bb_upper = ind.get("bb_upper", 0)
        bb_lower = ind.get("bb_lower", 0)
        bb_middle = ind.get("bb_middle", 0)
        price_pos = ind.get("price_position", 50)
        rsi = ind.get("rsi", 50)

        if close == 0 or bb_upper == 0:
            return None

        # V14 추가 안전 조건: 4H ADX < 20 (강한 횡보 확신)
        # ADX 20~30 구간은 약한 추세일 수 있어 BB 반전이 트랩 될 위험
        adx_4h = ind_4h.get("adx", 25)
        if adx_4h >= 20:
            return None

        # V14: 저변동 차단 (1H ATR < 0.3% 시 진입 X)
        atr_pct = ind.get("atr_pct", 0.5)
        min_atr = PARAM_REGISTRY.get("min_atr_pct")
        if min_atr > 0 and atr_pct < min_atr:
            return None

        direction = None
        score = 0
        reasons = []

        # 롱: BB 하단 + 바닥
        if close <= bb_lower * 1.002 and price_pos <= 20:
            direction = "LONG"
            score = 40
            reasons.append(f"BB하단터치")
            reasons.append(f"레인지바닥{price_pos:.0f}%")

            # RSI 보너스
            if rsi <= 30:
                score += 20
                reasons.append(f"RSI과매도{rsi:.0f}")
            elif rsi <= 40:
                score += 10

            # 오더북 매수벽
            if orderbook and orderbook.get("bid_ratio", 0.5) >= 0.55:
                score += 15
                reasons.append("매수벽확인")

            # 볼륨 감소
            vol = ind.get("volume_ratio", 1.0)
            if vol < 0.8:
                score += 10
                reasons.append("매도소진")

        # 숏: BB 상단 + 천장
        elif close >= bb_upper * 0.998 and price_pos >= 80:
            direction = "SHORT"
            score = 40
            reasons.append(f"BB상단터치")
            reasons.append(f"레인지천장{price_pos:.0f}%")

            if rsi >= 70:
                score += 20
                reasons.append(f"RSI과매수{rsi:.0f}")
            elif rsi >= 60:
                score += 10

            if orderbook and (1 - orderbook.get("bid_ratio", 0.5)) >= 0.55:
                score += 15
                reasons.append("매도벽확인")

            vol = ind.get("volume_ratio", 1.0)
            if vol < 0.8:
                score += 10
                reasons.append("매수소진")

        if direction is None:
            return None

        # 펀딩
        funding_bias = self._check_funding(direction, funding_rate)
        if funding_bias == "ALIGNED":
            score += 15
            reasons.append("펀딩순방향")
        elif funding_bias == "AGAINST":
            score -= 10

        # 횡보 임계값 (추세보다 +15 높게)
        range_threshold = PARAM_REGISTRY.get("entry_score_threshold") + 15
        if score < range_threshold:
            return None

        # V14: V13 안전망 정렬 (sl_mult 1.0 → 2.0, R:R 1.5 → 2.5)
        sl_pct, tp_pct = _fee_adjusted_sl_tp(
            atr_pct,
            sl_mult=2.0,   # V14: V13과 정렬, whipsaw 회피
            rr_ratio=2.5,  # V14: BB 반전은 빠른 회전이라 1:2.5 적정 (1:6은 비현실적)
            maker_both_sides=False
        )
        if sl_pct is None:
            return None

        # TP를 BB 중앙으로 제한 (BB 반전은 중앙선 회귀가 자연스러움)
        if direction == "LONG":
            max_tp_pct = (bb_middle - close) / close * 100
            tp_pct = min(tp_pct, max(max_tp_pct, sl_pct * 1.5))
        else:
            max_tp_pct = (close - bb_middle) / close * 100
            tp_pct = min(tp_pct, max(max_tp_pct, sl_pct * 1.5))

        max_lev = min(TRADING.MAX_LEVERAGE, 3)  # 평균회귀는 3x로 제한

        # V14: Bid/Ask PostOnly 진입 (V13 hybrid 동일 로직)
        best_bid = (orderbook or {}).get("best_bid", 0)
        best_ask = (orderbook or {}).get("best_ask", 0)
        spread_pct = (orderbook or {}).get("spread_pct", 0)
        entry_price = None
        if score < 70 and spread_pct < 0.1 and best_bid > 0 and best_ask > 0:
            if direction == "LONG":
                entry_price = round(best_bid, 2)
            else:
                entry_price = round(best_ask, 2)

        return SignalResult(
            direction=direction,
            strategy="RANGE_REVERSION",
            score=score,
            reason=f"BB반전 {direction}: {', '.join(reasons)} (스코어{score})",
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            max_leverage=max_lev,
            funding_bias=funding_bias
        )

    def _check_pullback(self, direction: str, ind: Dict[str, Any]) -> Optional[Dict]:
        """가격이 빠른 EMA 근처까지 되돌아왔는지 + EMA 배열이 추세 방향인지 검사."""
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        close = ind.get("close", 0)

        if ema_fast == 0 or ema_slow == 0 or close == 0:
            return None

        max_dist = PARAM_REGISTRY.get("pullback_ema_dist_pct")

        if direction == "LONG":
            if ema_fast <= ema_slow:
                return None
            dist_pct = (ema_fast - close) / ema_fast * 100
            if dist_pct < -0.1 or dist_pct > max_dist:
                return None
            return {"reason": f"EMA풀백({dist_pct:.2f}%)"}
        else:
            if ema_fast >= ema_slow:
                return None
            dist_pct = (close - ema_fast) / ema_fast * 100
            if dist_pct < -0.1 or dist_pct > max_dist:
                return None
            return {"reason": f"EMA풀백({dist_pct:.2f}%)"}

    def _evaluate_breakout(
        self, ind, ind_4h, orderbook, funding_rate,
        daily_trend: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """V14.5: HIGH_VOL 레짐에서 방향 명확한 변동성 폭발 진입.

        조건이 매우 까다로움 (보수적):
        - 4H ADX ≥ 35 (매우 강한 추세)
        - +DI/-DI 차이 큼 (방향 명확)
        - 1D 추세 일치
        - 1H 모멘텀 + 볼륨 폭증
        """
        plus_di = ind_4h.get("plus_di", 0)
        minus_di = ind_4h.get("minus_di", 0)
        adx_4h = ind_4h.get("adx", 0)

        # 1. ADX 매우 강함 + DI 차이 큼 (방향 명확)
        if adx_4h < 35:
            return None
        di_diff = abs(plus_di - minus_di)
        if di_diff < 15:
            return None  # 방향 불명확 = 패닉, 진입 X

        # 방향 결정
        direction = "LONG" if plus_di > minus_di else "SHORT"

        # 2. 1D 추세 일치 필수 (V13 d1_filter와 동일)
        if PARAM_REGISTRY.get("d1_filter_enable") >= 1 and daily_trend:
            mode = int(PARAM_REGISTRY.get("d1_filter_mode"))
            if mode == 1:
                d1_close = daily_trend.get("close", 0)
                d1_ema = daily_trend.get("ema", 0)
                if d1_ema == 0:
                    return None
                if direction == "LONG" and d1_close <= d1_ema:
                    return None
                if direction == "SHORT" and d1_close >= d1_ema:
                    return None

        # 3. 1H 모멘텀 확인
        ema_fast = ind.get("ema_fast", 0)
        ema_slow = ind.get("ema_slow", 0)
        close = ind.get("close", 0)
        if ema_fast == 0 or ema_slow == 0:
            return None

        if direction == "LONG":
            if not (ema_fast > ema_slow and close > ema_fast):
                return None
        else:
            if not (ema_fast < ema_slow and close < ema_fast):
                return None

        # 4. 볼륨 폭증 필수 (1.5x 이상)
        vol = ind.get("volume_ratio", 1.0)
        if vol < 1.5:
            return None

        # 5. RSI 모멘텀 확인 (LONG: > 55, SHORT: < 45)
        rsi = ind.get("rsi", 50)
        if direction == "LONG" and rsi < 55:
            return None
        if direction == "SHORT" and rsi > 45:
            return None

        # 점수 계산 (BREAKOUT은 매우 보수적, 임계 +30)
        score = 50
        reasons = [f"ADX{adx_4h:.0f}", f"DI차이{di_diff:.0f}", f"Vol{vol:.1f}x", f"RSI{rsi:.0f}"]

        if vol >= 2.0:
            score += 20
            reasons.append("볼륨급증")
        else:
            score += 10

        if di_diff >= 20:
            score += 15
            reasons.append("방향확정")

        # 펀딩 (V13 SHORT 페널티 정렬)
        funding_bias = self._check_funding(direction, funding_rate)
        if funding_bias == "ALIGNED":
            score += 15
            reasons.append("펀딩순방향")
        elif funding_bias == "AGAINST":
            if direction == "SHORT":
                score -= 20
            else:
                score -= 10

        # BREAKOUT 임계: TREND threshold + 30 (매우 보수적)
        breakout_threshold = PARAM_REGISTRY.get("entry_score_threshold") + 30
        if score < breakout_threshold:
            return None

        # SL/TP: BREAKOUT은 빠른 결판
        atr_pct = ind.get("atr_pct", 0.5)
        sl_pct, tp_pct = _fee_adjusted_sl_tp(
            atr_pct,
            sl_mult=1.5,   # 가짜 브레이크아웃 시 빠른 손절
            rr_ratio=3.0,  # 성공 시 큰 수익
            maker_both_sides=False
        )
        if sl_pct is None:
            return None

        # BREAKOUT은 강신호니까 즉시 market (V13 hybrid 로직)
        # score 70+면 자동 market, BREAKOUT은 임계 70+이라 거의 항상 market
        max_lev = min(TRADING.MAX_LEVERAGE, 5)  # 추세 7x보다 보수, 평균회귀 3x보다 적극

        return SignalResult(
            direction=direction,
            strategy="BREAKOUT",
            score=score,
            reason=f"브레이크아웃 {direction}: {', '.join(reasons)} (스코어{score})",
            entry_price=None,  # 강신호 → market
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            max_leverage=max_lev,
            funding_bias=funding_bias
        )

    def _check_orderbook(self, direction: str, orderbook: Optional[Dict]) -> bool:
        """매수/매도 불균형이 진입 방향에 유리해야 True."""
        if not orderbook:
            return True
        bid_ratio = orderbook.get("bid_ratio", 0.5)
        threshold = PARAM_REGISTRY.get("orderbook_imbalance_min")
        if direction == "LONG":
            return bid_ratio >= threshold
        return (1 - bid_ratio) >= threshold

    def _check_funding(self, direction: str, funding_rate: float) -> str:
        """펀딩레이트가 진입 방향에 대해 유리/중립/불리 중 무엇인지 분류."""
        threshold = PARAM_REGISTRY.get("funding_bias_threshold")
        if abs(funding_rate) < threshold:
            return "NEUTRAL"
        if funding_rate > threshold:
            return "ALIGNED" if direction == "SHORT" else "AGAINST"
        return "ALIGNED" if direction == "LONG" else "AGAINST"
