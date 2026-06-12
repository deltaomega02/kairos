"""KAIROS — Καιρός: Decisive Moment / Right-Time Trading Engine

Bybit USDT 무기한 선물 기반 v2 자동매매 시스템.
HERMES v1의 학습을 적용한 단순화 버전 — 결정적 타이밍에 집중.

핵심 차별점 (HERMES 대비):
- 단순화 우선 (지표 최소화, 워뇨띠 원칙)
- v# 잦은 변경 금지 (안티패턴 차단)
- 별도 자금 / 별도 텔레그램 / 같은 거래소

"""

import sys
import signal
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any

from config import (
    setup_logging, get_logger,
    TRADING, SCHEDULER, PARAM_REGISTRY
)
from exchange.bybit_client import bybit_client
from core.technical_analysis import (
    get_regime_indicators_4h, get_entry_indicators_1h, get_daily_trend,
    calculate_orderbook_imbalance
)
from core.regime_engine import RegimeEngine, MarketRegime
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.websocket_watcher import ScalpingWatcher
from database.db_manager import db_manager
from utils.telegram_bot import telegram_notifier
from backtest.optimizer import WalkForwardOptimizer

setup_logging()
logger = get_logger("main")

def _coin_name(symbol: str) -> str:
    """BTCUSDT → BTC 형태로 심볼 축약."""
    return symbol.replace("USDT", "")


class Kairos:
    """Top-level orchestrator. Owns the main loop and per-symbol state."""

    def __init__(self):
        self.running = False
        self.regime_engine = RegimeEngine()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        self.optimizer = WalkForwardOptimizer()

        # 심볼별 상태
        self._pos_lock = threading.Lock()
        self.positions: Dict[str, Optional[Dict[str, Any]]] = {}
        self.watchers: Dict[str, Optional[ScalpingWatcher]] = {}
        self._last_4h_ts: Dict[str, int] = {}
        self._last_1h_ts: Dict[str, int] = {}
        self._last_1d_ts: Dict[str, int] = {}
        self._cached_4h: Dict[str, Dict[str, Any]] = {}
        self._cached_1d: Dict[str, Dict[str, Any]] = {}

        self._halt_notified: bool = False  # 거래 중단 알림 중복 방지
        self._last_summary_ts: float = 0.0  # V13+: 30분 주기 누적 PnL 알림

        for symbol in TRADING.SYMBOLS:
            self.positions[symbol] = None
            self.watchers[symbol] = None
            self._last_4h_ts[symbol] = 0
            self._last_1h_ts[symbol] = 0
            self._last_1d_ts[symbol] = 0
            self._cached_4h[symbol] = {}
            self._cached_1d[symbol] = {}

    # ================================================================
    # 시작
    # ================================================================

    def start(self):
        """부팅: 로깅 세팅 확인, 잔액 조회, DB 내 활성 포지션 복구, 메인 루프 진입."""
        logger.info("=" * 60)
        logger.info("KAIROS Starting (multi-coin)")
        logger.info(f"Symbols: {', '.join(TRADING.SYMBOLS)}")
        logger.info("=" * 60)

        self.running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        wallet = bybit_client.get_wallet_balance()
        balance = wallet.get("available_balance", 0)
        if balance <= 0:
            logger.error("잔액 없음")
            return

        self.risk_manager.initialize(balance)

        # DB에서 모든 활성 포지션 복구
        active_list = db_manager.get_active_positions()
        for active in active_list:
            sym = active.get("symbol", "BTCUSDT")
            if sym in TRADING.SYMBOLS:
                logger.info(
                    f"활성 포지션 복구: {_coin_name(sym)} "
                    f"{active['direction']} {active['leverage']}x"
                )
                self._resume_monitoring(sym, active)

        active_count = sum(1 for p in self.positions.values() if p is not None)
        telegram_notifier.send_system_start(
            balance=balance,
            position_info=(
                {"direction": f"{active_count}개 활성", "leverage": ""}
                if active_count > 0 else None
            )
        )
        self._main_loop()

    # ================================================================
    # 메인 루프
    # ================================================================

    def _main_loop(self):
        """메인 루프: 심볼별 4H 레짐 갱신 + 1D 추세 갱신 + 1H 시그널 체크 + 진입."""
        logger.info("메인 루프 시작 (멀티코인 4H 레짐 / 1H 시그널)")

        while self.running:
            try:
                # 지갑 잔고는 사이클당 1회만 조회해 API 콜을 절감한다.
                cached_balance = None
                try:
                    wallet = bybit_client.get_wallet_balance()
                    cached_balance = wallet.get("available_balance", 0)
                except Exception as e:
                    logger.warning(f"지갑 조회 실패 (사이클 스킵 가능): {e}")

                for symbol in TRADING.SYMBOLS:
                    self._update_regime(symbol)
                    self._update_daily_trend(symbol)

                    with self._pos_lock:
                        has_position = self.positions[symbol] is not None
                    if not has_position:
                        self._check_signals(symbol, cached_balance=cached_balance)

                if self.optimizer.should_run():
                    self.optimizer.run()

                # V13.2: 30분 주기 요약 제거 (운영 정책)

                time.sleep(SCHEDULER.SIGNAL_CHECK_INTERVAL_SEC)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"루프 오류: {e}", exc_info=True)
                telegram_notifier.send_system_error("MainLoop", str(e), "main_loop")
                time.sleep(60)

    # ================================================================
    # 30분 주기 누적 PnL 요약 (V13+)
    # ================================================================

    SUMMARY_INTERVAL_SEC = 1800  # 30분

    def _maybe_send_summary(self, current_balance: Optional[float]):
        """30분 간격으로 누적 상태 텔레그램 발송. 잔고 미조회 시 스킵."""
        if current_balance is None or current_balance <= 0:
            return
        now = time.time()
        if now - self._last_summary_ts < self.SUMMARY_INTERVAL_SEC:
            return

        try:
            stats = self.risk_manager.get_daily_stats()
            peak = self.risk_manager._peak_balance
            active_count = sum(1 for p in self.positions.values() if p is not None)
            telegram_notifier.send_periodic_summary(
                balance=current_balance,
                daily_stats=stats,
                peak_balance=peak,
                active_positions=active_count,
            )
            self._last_summary_ts = now
        except Exception as e:
            logger.warning(f"주기 요약 전송 실패: {e}")

    # ================================================================
    # 4H 레짐 판독
    # ================================================================

    def _update_regime(self, symbol: str):
        """4H 캔들을 읽어 ADX / EMA / MACD 기반 레짐을 갱신한다."""
        try:
            candles_4h = bybit_client.get_kline(
                symbol=symbol, interval="240", limit=100
            )
            if not candles_4h:
                return

            latest_ts = candles_4h[-1]["timestamp"]
            if latest_ts == self._last_4h_ts[symbol]:
                return

            self._last_4h_ts[symbol] = latest_ts

            indicators_4h = get_regime_indicators_4h(candles_4h)
            if not indicators_4h:
                return

            old_regime = self.regime_engine.get_regime(symbol)
            result = self.regime_engine.update(symbol, indicators_4h)

            changed = result.regime != old_regime
            telegram_notifier.send_regime_update(
                symbol, result.regime.value, indicators_4h,
                changed, old_regime.value
            )

            self._cached_4h[symbol] = indicators_4h

        except Exception as e:
            logger.error(f"[{_coin_name(symbol)}] 레짐 오류: {e}")

    # ================================================================
    # 1D 추세 판독 — 진입 필터용
    # ================================================================

    def _update_daily_trend(self, symbol: str):
        """일봉 EMA 상태를 갱신해 진입 필터(direction / price-above-EMA)에서 사용한다."""
        try:
            d1_period = int(PARAM_REGISTRY.get("d1_ema_period"))
            candles_1d = bybit_client.get_kline(
                symbol=symbol, interval="D", limit=d1_period + 20
            )
            if not candles_1d:
                return

            latest_ts = candles_1d[-1]["timestamp"]
            if latest_ts == self._last_1d_ts[symbol]:
                return  # 같은 1D 캔들 → 스킵

            self._last_1d_ts[symbol] = latest_ts
            trend = get_daily_trend(candles_1d, ema_period=d1_period)
            if trend:
                self._cached_1d[symbol] = trend
                logger.info(
                    f"[{_coin_name(symbol)}] 1D EMA{d1_period}: "
                    f"{trend['direction']} (slope {trend['slope_pct']:+.3f}%)"
                )

        except Exception as e:
            logger.warning(f"[{_coin_name(symbol)}] 1D 추세 오류: {e}")

    # ================================================================
    # 1H 시그널 평가 및 진입
    # ================================================================

    def _check_signals(self, symbol: str, cached_balance: Optional[float] = None):
        """1H 지표를 계산하고 리스크/레짐/신호 엔진을 통과하면 포지션을 연다."""
        try:
            candles_1h = bybit_client.get_kline(
                symbol=symbol, interval="60", limit=100
            )
            if not candles_1h:
                return

            latest_ts = candles_1h[-1]["timestamp"]
            if latest_ts == self._last_1h_ts[symbol]:
                return
            self._last_1h_ts[symbol] = latest_ts

            # 잔고는 메인 루프에서 이미 조회했으면 재사용하고, 없으면 직접 호출한다.
            if cached_balance is not None and cached_balance > 0:
                balance = cached_balance
            else:
                wallet = bybit_client.get_wallet_balance()
                balance = wallet.get("available_balance", 0)
            active_count = sum(1 for p in self.positions.values() if p is not None)
            can_trade, reason = self.risk_manager.can_trade(
                balance, active_positions=active_count
            )
            if not can_trade:
                logger.info(f"[{_coin_name(symbol)}] 거래 불가: {reason}")
                if not self._halt_notified:
                    self._halt_notified = True
                    telegram_notifier.send_risk_alert(
                        "TRADE_HALTED",
                        f"거래 중단: {reason}\n잔고: ${balance:,.2f}"
                    )
                return
            # 거래 가능 → 중단 알림 플래그 리셋
            self._halt_notified = False

            params = PARAM_REGISTRY.get_all()
            indicators_1h = get_entry_indicators_1h(candles_1h, params)
            if not indicators_1h:
                return

            if self.regime_engine.emergency_override(symbol, indicators_1h):
                telegram_notifier.send_risk_alert(
                    "EMERGENCY_HIGH_VOL",
                    f"[{_coin_name(symbol)}] 1H 급변"
                )
                return

            orderbook = self._fetch_orderbook(symbol)
            funding_rate = self._fetch_funding_rate(symbol)

            regime = self.regime_engine.get_regime(symbol)
            daily_trend = self._cached_1d.get(symbol, {}) or None

            # [edge hunt] 시간/연패 기반 필터용 추가 컨텍스트
            # KST = UTC + 9. datetime.now() 이 로컬이므로 서버가 UTC면 KST로 환산.
            # main.py 실행 환경이 UTC 기준 GCP 서버이므로 utcnow()+9 를 계산.
            from datetime import datetime as _dt, timezone, timedelta
            kst_now = _dt.now(timezone.utc) + timedelta(hours=9)
            current_hour_kst = kst_now.hour
            risk_stats = self.risk_manager.get_daily_stats()
            consecutive_losses = risk_stats.get("consecutive_losses", 0)

            signal_result = self.signal_engine.evaluate(
                regime=regime,
                indicators_1h=indicators_1h,
                indicators_4h=self._cached_4h[symbol],
                orderbook=orderbook,
                funding_rate=funding_rate,
                daily_trend=daily_trend,
                current_hour_kst=current_hour_kst,
                consecutive_losses=consecutive_losses,
            )

            # 진입 실패 시 텔레그램으로 사유를 돌려주기 위한 분류
            reject = ""
            if not signal_result:
                # edge hunt 신규 필터 우선 체크
                max_losses = int(PARAM_REGISTRY.get("max_consecutive_losses"))
                blocked_hour = int(PARAM_REGISTRY.get("blocked_hour_kst"))
                min_atr = PARAM_REGISTRY.get("min_atr_pct")
                atr_pct = indicators_1h.get("atr_pct", 0)

                if regime == MarketRegime.HIGH_VOL:
                    reject = "고변동 레짐 → 거래 중단"
                elif regime == MarketRegime.RANGING:
                    reject = "횡보 → 추세 부재"
                elif max_losses > 0 and consecutive_losses >= max_losses:
                    reject = f"{consecutive_losses}연패로 진입 차단 (edge 필터)"
                elif blocked_hour >= 0 and current_hour_kst == blocked_hour:
                    reject = f"{blocked_hour}시 KST 진입 차단 (edge 필터)"
                elif min_atr > 0 and atr_pct < min_atr:
                    reject = f"저변동 차단 (ATR {atr_pct:.2f}% < {min_atr}%)"
                else:
                    # 1D 필터 확인 (direction / price-above-EMA 모드 분기)
                    d1_enable = PARAM_REGISTRY.get("d1_filter_enable") >= 1
                    d1_mode = int(PARAM_REGISTRY.get("d1_filter_mode")) if d1_enable else 0
                    d1_blocked_long = d1_blocked_short = False
                    if d1_enable and daily_trend:
                        if d1_mode == 0:
                            d1_dir = daily_trend.get("direction", "FLAT")
                            d1_blocked_long = (d1_dir != "UP")
                            d1_blocked_short = (d1_dir != "DOWN")
                        else:
                            d1_close = daily_trend.get("close", 0)
                            d1_ema = daily_trend.get("ema", 0)
                            d1_blocked_long = (d1_close <= d1_ema) if d1_ema else False
                            d1_blocked_short = (d1_close >= d1_ema) if d1_ema else False
                    if d1_enable and regime == MarketRegime.TRENDING_UP and d1_blocked_long:
                        if d1_mode == 1:
                            reject = "1D 필터 차단 (일봉 종가 ≤ EMA2, LONG 불허)"
                        else:
                            d1_dir = (daily_trend or {}).get("direction", "FLAT")
                            reject = f"1D 필터 차단 (일봉 {d1_dir} 중 LONG 시도)"
                    elif d1_enable and regime == MarketRegime.TRENDING_DOWN and d1_blocked_short:
                        if d1_mode == 1:
                            reject = "1D 필터 차단 (일봉 종가 ≥ EMA2, SHORT 불허)"
                        else:
                            d1_dir = (daily_trend or {}).get("direction", "FLAT")
                            reject = f"1D 필터 차단 (일봉 {d1_dir} 중 SHORT 시도)"
                    else:
                        ema_f = indicators_1h.get("ema_fast", 0)
                        ema_s = indicators_1h.get("ema_slow", 0)
                        close = indicators_1h.get("close", 0)
                        dist = abs(close - ema_f) / ema_f * 100 if ema_f > 0 else 0
                        if regime == MarketRegime.TRENDING_UP and ema_f <= ema_s:
                            reject = "1H EMA 역배열 (상승 약화)"
                        elif regime == MarketRegime.TRENDING_UP and close > ema_f * 1.001:
                            reject = f"풀백 조건 미달 (EMA fast 위 {dist:.2f}%)"
                        elif regime == MarketRegime.TRENDING_DOWN and ema_f >= ema_s:
                            reject = "1H EMA 정배열 (하락 약화)"
                        elif regime == MarketRegime.TRENDING_DOWN and close < ema_f * 0.999:
                            reject = f"풀백 조건 미달 (EMA fast 아래 {dist:.2f}%)"
                        else:
                            ob_ratio = orderbook.get("bid_ratio", 0.5) if orderbook else 0.5
                            reject = f"오더북({ob_ratio:.0%}) 또는 점수 미달"

            telegram_notifier.send_signal_check(
                symbol, regime.value, indicators_1h,
                signal_result, reject, orderbook, funding_rate,
                daily_trend=daily_trend,
            )

            if not signal_result:
                return

            # V13.2: SOL LONG 차단 제거 (운영 정책: 모든 시그널 진입 허용)

            coin = _coin_name(symbol)
            logger.info(
                f"[{coin}] 시그널: {signal_result.direction} "
                f"스코어={signal_result.score} "
                f"펀딩={signal_result.funding_bias}"
            )

            entry_price = signal_result.entry_price or indicators_1h["close"]
            sizing = self.risk_manager.calculate_position_size(
                current_balance=balance,
                entry_price=entry_price,
                sl_pct=signal_result.sl_pct,
                max_leverage=signal_result.max_leverage,
                symbol=symbol
            )
            if not sizing:
                return

            liq_price = (sizing["liquidation_price_long"]
                        if signal_result.direction == "LONG"
                        else sizing["liquidation_price_short"])

            position = self.position_manager.open_position(
                signal=signal_result,
                leverage=sizing["leverage"],
                quantity=sizing["quantity"],
                liquidation_price=liq_price,
                margin_used=sizing["margin"],
                symbol=symbol
            )

            if not position:
                logger.info(f"[{coin}] 진입 실패 (지정가 미체결 가능)")
                return

            with self._pos_lock:
                self.positions[symbol] = position

            telegram_notifier.send_position_opened(
                direction=f"{coin} {position['direction']}",
                leverage=position["leverage"],
                entry_price=position["entry_price"],
                qty=position["quantity"],
                stop_loss=position["sl_price"],
                take_profit=position["tp_price"],
                margin_used=position["margin_used"],
                strategy=position.get("strategy", "TREND_PULLBACK"),
                entry_fee=position.get("entry_fee", 0),
                score=signal_result.score
            )

            self._start_monitoring(symbol, position)

        except Exception as e:
            logger.error(
                f"[{_coin_name(symbol)}] 시그널 오류: {e}", exc_info=True
            )

    # ================================================================
    # 오더북 / 펀딩레이트
    # ================================================================

    def _fetch_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """매수/매도 호가 불균형 계산 → 시그널 엔진의 확인 필터로 사용."""
        try:
            params = {
                "category": TRADING.CATEGORY,
                "symbol": symbol,
                "limit": TRADING.ORDERBOOK_DEPTH
            }
            result = bybit_client._request("GET", "/v5/market/orderbook", params)

            bids = result.get("b", [])
            asks = result.get("a", [])

            return calculate_orderbook_imbalance({"bids": bids, "asks": asks})

        except Exception as e:
            logger.warning(f"[{_coin_name(symbol)}] 오더북 조회 실패: {e}")
            return None

    def _fetch_funding_rate(self, symbol: str) -> float:
        """티커에서 현재 펀딩레이트만 추출."""
        try:
            ticker = bybit_client.get_ticker(symbol=symbol)
            return ticker.get("funding_rate", 0)
        except Exception:
            return 0

    # ================================================================
    # 포지션 모니터링
    # ================================================================

    def _start_monitoring(self, symbol: str, position: Dict[str, Any]):
        """포지션이 열리면 전용 WebSocket 워처를 돌려 SL/TP/트레일링을 실시간 추적."""
        self.watchers[symbol] = ScalpingWatcher(
            position_info=position,
            on_close_callback=lambda reason: self._on_position_closed(
                symbol, reason
            ),
            symbol=symbol
        )
        self.watchers[symbol].start()

    def _resume_monitoring(self, symbol: str, db_position: Dict[str, Any]):
        """재시작 시 DB에 남아 있는 활성 포지션을 메모리로 복구하고 워처를 재가동."""
        position = {
            "position_uuid": db_position["position_uuid"],
            "direction": db_position["direction"],
            "entry_price": db_position["entry_price"],
            "quantity": db_position["entry_quantity"],
            "leverage": db_position["leverage"],
            "sl_price": db_position["stop_loss_price"],
            "tp_price": db_position["take_profit_price"],
            "liquidation_price": db_position.get("liquidation_price", 0),
            "strategy": db_position.get("strategy", "TREND_PULLBACK"),
            "margin_used": 0,
            "entry_fee": db_position.get("entry_fee", 0)
        }
        self.positions[symbol] = position
        self._start_monitoring(symbol, position)

    # ================================================================
    # 포지션 청산 콜백
    # ================================================================

    def _on_position_closed(self, symbol: str, reason: str):
        """워처에서 청산 이벤트가 들어올 때 호출: 주문 실행, DB 업데이트, 알림 전송."""
        position = self.positions.get(symbol)
        if not position:
            return

        coin = _coin_name(symbol)

        # 서버 체결(SL/TP) vs 수동 청산 분기
        result = self.position_manager.close_position(
            position_uuid=position["position_uuid"],
            direction=position["direction"],
            entry_price=position["entry_price"],
            quantity=position["quantity"],
            leverage=position["leverage"],
            reason=reason,
            symbol=symbol
        )

        # 보유 시간 계산
        db_pos = db_manager.get_position_by_uuid(position["position_uuid"])
        hold_time = "?"
        if db_pos and db_pos.get("entry_timestamp"):
            try:
                entry_dt = datetime.fromisoformat(db_pos["entry_timestamp"])
                hold_sec = (datetime.now() - entry_dt).total_seconds()
                if hold_sec < 3600:
                    hold_time = f"{int(hold_sec / 60)}분"
                else:
                    hold_time = f"{hold_sec / 3600:.1f}시간"
            except (ValueError, TypeError):
                pass

        if result:
            # 우리가 청산 주문을 실행한 경우
            is_win = result["realized_pnl"] > 0
            self.risk_manager.record_trade_result(result["realized_pnl"], is_win)

            telegram_notifier.send_position_closed(
                direction=f"{coin} {position['direction']}",
                reason=reason,
                entry_price=position["entry_price"],
                exit_price=result["exit_price"],
                pnl=result["realized_pnl"],
                pnl_pct=result["realized_pnl_pct"],
                hold_time=hold_time,
                total_fee=result.get("exit_fee", 0),
                strategy=position.get("strategy", "?")
            )
        else:
            # 서버가 이미 청산한 경우 (SL/TP 서버 체결)
            logger.info(
                f"[{coin}] 서버 체결 감지 — Bybit에서 청산 정보 조회"
            )
            try:
                ticker = bybit_client.get_ticker(symbol=symbol)
                exit_price = ticker.get("last_price", 0)

                # PnL 계산
                entry_price = position["entry_price"]
                qty = position["quantity"]
                leverage = position["leverage"]

                if position["direction"] == "LONG":
                    raw_pnl = (exit_price - entry_price) * qty
                else:
                    raw_pnl = (entry_price - exit_price) * qty

                fee_estimate = exit_price * qty * TRADING.TAKER_FEE_PCT * 2
                realized_pnl = raw_pnl - fee_estimate
                margin = entry_price * qty / leverage
                realized_pnl_pct = (
                    (realized_pnl / margin * 100) if margin > 0 else 0
                )

                # DB 업데이트
                db_manager.close_position(
                    position_uuid=position["position_uuid"],
                    exit_price=exit_price,
                    exit_reason=reason,
                    realized_pnl=realized_pnl,
                    realized_pnl_percentage=realized_pnl_pct,
                    exit_fee=fee_estimate / 2
                )

                is_win = realized_pnl > 0
                self.risk_manager.record_trade_result(realized_pnl, is_win)

                telegram_notifier.send_position_closed(
                    direction=f"{coin} {position['direction']}",
                    reason=reason,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    pnl_pct=realized_pnl_pct,
                    hold_time=hold_time,
                    total_fee=fee_estimate,
                    strategy=position.get("strategy", "?")
                )
            except Exception as e:
                logger.error(f"[{coin}] 서버 체결 정보 조회 실패: {e}")
                telegram_notifier.send_risk_alert(
                    "SERVER_CLOSE",
                    f"{coin} {position['direction']} 포지션 서버 청산 감지. "
                    f"로그 확인 필요."
                )

        # 워쳐 정리 + 포지션 해제
        watcher = self.watchers.get(symbol)
        if watcher:
            watcher.stop()
            self.watchers[symbol] = None
        with self._pos_lock:
            self.positions[symbol] = None
        logger.info(f"[{coin}] 청산 완료: {reason}")

    # ================================================================
    # 종료
    # ================================================================

    def _shutdown(self, signum=None, frame=None):
        """SIGINT/SIGTERM 처리: 워처 정리 후 프로세스 종료."""
        logger.info("종료 요청...")
        self.running = False
        for symbol, watcher in self.watchers.items():
            if watcher:
                watcher.stop()
        telegram_notifier.info("[KAIROS] 시스템 종료")
        sys.exit(0)


if __name__ == "__main__":
    Kairos().start()
