"""실시간 포지션 감시자.

WebSocket mark price 를 구독해 SL/TP 를 로컬에서도 감시한다.
서버 트리거 SL/TP 는 Bybit 측에서도 걸려 있으므로 이중 안전망이 된다.
Dead Man's Switch 로 WebSocket 이 끊기면 REST 로 폴백.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable
from config import get_logger, TRADING, SCHEDULER, PARAM_REGISTRY
from exchange.bybit_client import bybit_client
from exchange.bybit_websocket import BybitWebSocket

logger = get_logger("watcher")


class ScalpingWatcher:
    """포지션 하나당 하나의 워처. 가격 tick 을 받아 SL/TP/트레일링을 처리."""

    def __init__(
        self,
        position_info: Dict[str, Any],
        on_close_callback: Callable[[str], None],
        symbol: str = "BTCUSDT"
    ):
        self.position = position_info
        self.on_close = on_close_callback
        self._symbol = symbol

        self._running = False
        self._lock = threading.Lock()

        # 트레일링 스탑
        self._peak_price: float = position_info["entry_price"]
        self._trailing_active: bool = False
        self._trailing_sl: Optional[float] = None

        # WebSocket
        self._ws: Optional[BybitWebSocket] = None
        self._last_price: float = 0.0
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self):
        """WebSocket 구독 + Dead Man's Switch 스레드 가동."""
        if self._running:
            return

        self._running = True

        self._ws = BybitWebSocket(
            on_price_callback=self._on_price_update,
            on_position_closed_callback=self._on_ws_position_closed,
            symbol=self._symbol
        )
        self._ws.set_position_tracking(True)
        self._ws.set_liquidation_tracking(
            self.position["liquidation_price"],
            self.position["direction"]
        )
        self._ws.start()

        # Dead Man's Switch
        self._monitor_thread = threading.Thread(
            target=self._dead_mans_switch_loop,
            daemon=True
        )
        self._monitor_thread.start()

        logger.info(
            f"[{self._symbol}] 감시 시작: {self.position['direction']} "
            f"SL={self.position['sl_price']:,.2f} "
            f"TP={self.position['tp_price']:,.2f}"
        )

    def stop(self):
        """외부에서 강제 종료. 청산 콜백 이후에도 호출해 자원을 회수한다."""
        self._running = False
        if self._ws:
            self._ws.stop()

    def update_targets(self, sl: Optional[float] = None, tp: Optional[float] = None):
        """로컬 상태는 즉시 갱신하고, Bybit 서버에 set_trading_stop 을 시도.

        서버 호출 실패는 치명적이지 않다 (로컬 워처가 계속 감시하므로). 경고만 남긴다.
        """
        with self._lock:
            if sl is not None:
                self.position["sl_price"] = sl
            if tp is not None:
                self.position["tp_price"] = tp

        # 서버 반영 — 실패해도 로컬 워처가 계속 감시하므로 치명적 아님
        try:
            bybit_client.set_trading_stop(
                symbol=self._symbol, stop_loss=sl, take_profit=tp
            )
        except Exception as e:
            logger.warning(
                f"[{self._symbol}] 서버 SL/TP 업데이트 실패: {e} "
                f"(로컬 워처는 계속 감시 중)"
            )

    def _on_price_update(self, price: float):
        """WebSocket 에서 mark price 가 들어올 때마다 호출. 여기가 실시간 판정 지점."""
        with self._lock:
            self._last_price = price

        direction = self.position["direction"]
        sl = self.position["sl_price"]
        tp = self.position["tp_price"]

        # 로컬 SL/TP 백업. 트레일링이 활성화된 뒤 SL 에 맞으면 사유를 TRAILING_STOP 으로 분류.
        sl_reason = "TRAILING_STOP" if self._trailing_active else "STOP_LOSS"
        if direction == "LONG":
            if price <= sl:
                self._trigger_close(sl_reason)
                return
            if price >= tp:
                self._trigger_close("TAKE_PROFIT")
                return
        else:
            if price >= sl:
                self._trigger_close(sl_reason)
                return
            if price <= tp:
                self._trigger_close("TAKE_PROFIT")
                return

        # 청산가 근접 알림 (유지마진 부족 위험)
        liq = self.position["liquidation_price"]
        if liq > 0:
            dist = abs(price - liq) / price
            if dist < TRADING.LIQUIDATION_WARN_PCT:
                logger.warning(f"[{self._symbol}] 청산가 근접! 현재가={price:,.2f} 청산가={liq:,.2f} 거리={dist*100:.1f}%")

        # 추세 풀백 전략에만 트레일링 적용 (레인지 반전 전략은 현재 비활성)
        if self.position.get("strategy") == "TREND_PULLBACK":
            self._update_trailing_stop(price)

    def _update_trailing_stop(self, price: float):
        """수익이 activation % 에 도달하면 트레일링을 켜고, 이후 고점에서 distance 만큼 아래로 SL 을 따라간다."""
        direction = self.position["direction"]
        entry = self.position["entry_price"]
        activation_pct = PARAM_REGISTRY.get("trailing_activation_pct")
        trail_distance_pct = PARAM_REGISTRY.get("trailing_distance_pct")

        # 수익률
        if direction == "LONG":
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl_pct = (entry - price) / entry * 100

        # 활성화
        if pnl_pct >= activation_pct and not self._trailing_active:
            self._trailing_active = True
            logger.info(f"[{self._symbol}] 트레일링 스탑 활성화: PnL={pnl_pct:.2f}%")

        if not self._trailing_active:
            return

        # 피크 갱신
        with self._lock:
            if direction == "LONG":
                if price > self._peak_price:
                    self._peak_price = price
            else:
                if price < self._peak_price or abs(self._peak_price - entry) < 0.01:
                    self._peak_price = price

        if direction == "LONG":
            new_trailing_sl = round(self._peak_price * (1 - trail_distance_pct / 100), 2)
            current_sl = self.position["sl_price"]

            if new_trailing_sl > current_sl:
                self.update_targets(sl=new_trailing_sl)
                logger.info(f"[{self._symbol}] 트레일링 SL 상향: {current_sl:,.2f} → {new_trailing_sl:,.2f}")
        else:
            new_trailing_sl = round(self._peak_price * (1 + trail_distance_pct / 100), 2)
            current_sl = self.position["sl_price"]

            if new_trailing_sl < current_sl:
                self.update_targets(sl=new_trailing_sl)
                logger.info(f"[{self._symbol}] 트레일링 SL 하향: {current_sl:,.2f} → {new_trailing_sl:,.2f}")

    def _on_ws_position_closed(self, reason: str):
        """Bybit 가 먼저 SL/TP 를 체결한 케이스. 워처가 아직 감지 못 했을 때 콜백으로 들어온다."""
        if self._running:
            logger.info(f"[{self._symbol}] WS 포지션 종료 감지: {reason}")
            # 트레일링 활성화 상태에서 서버 체결 → 트레일링 익절로 라벨
            if reason == "SERVER_TRIGGERED" and self._trailing_active:
                reason = "TRAILING_STOP"
            self._trigger_close(reason)

    def _trigger_close(self, reason: str):
        """메인 스레드에 청산 사유 전달. 중복 호출 방지 플래그도 세팅."""
        if not self._running:
            return

        self._running = False

        if self._ws:
            self._ws.set_external_close_in_progress(True)

        logger.info(f"[{self._symbol}] 청산 트리거: {reason}")

        # 콜백
        threading.Thread(
            target=self.on_close,
            args=(reason,),
            daemon=True
        ).start()

    def _dead_mans_switch_loop(self):
        """WebSocket 이 N 초 이상 조용하면 살아있는지 의심하고 REST 폴백을 띄운다."""
        timeout = SCHEDULER.DEAD_MANS_SWITCH_TIMEOUT_SEC

        while self._running:
            time.sleep(10)

            if not self._ws:
                continue

            last_data = self._ws.get_last_data_time()
            if last_data > 0:
                elapsed = time.time() - last_data
                if elapsed > timeout:
                    logger.warning(f"[{self._symbol}] Dead Man's Switch: {elapsed:.0f}초 데이터 없음 → REST 폴백")
                    self._rest_fallback()

    def _rest_fallback(self):
        """REST 티커를 한 번 가져와 가격 업데이트 파이프라인을 강제로 한 번 돌린다."""
        try:
            ticker = bybit_client.get_ticker(symbol=self._symbol)
            if ticker:
                price = ticker.get("mark_price", 0) or ticker.get("last_price", 0)
                if price > 0:
                    self._on_price_update(price)
        except Exception as e:
            logger.error(f"[{self._symbol}] REST 폴백 실패: {e}")
