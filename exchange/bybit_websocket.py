# exchange/bybit_websocket.py — Bybit WebSocket 스트림

import json
import time
import hmac
import hashlib
import threading
from typing import Callable, Optional, Dict, Any

import websocket

from config import BYBIT, TRADING, get_logger

logger = get_logger("bybit_websocket")


class BybitWebSocket:
    """Bybit WebSocket — 자동 재연결 + ping/pong heartbeat"""

    PING_INTERVAL = 20
    RECONNECT_DELAY = 5

    def __init__(
        self,
        on_price_callback: Optional[Callable[[float], None]] = None,
        on_position_closed_callback: Optional[Callable[[str], None]] = None,
        symbol: Optional[str] = None
    ):
        """WebSocket 클라이언트 초기화"""
        self.on_price_callback = on_price_callback
        self.on_position_closed_callback = on_position_closed_callback
        self._symbol = symbol or TRADING.SYMBOL

        if BYBIT.USE_TESTNET:
            self.ws_public_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.ws_private_url = "wss://stream-testnet.bybit.com/v5/private"
            self.api_key = BYBIT.TESTNET_API_KEY
            self.secret = BYBIT.TESTNET_SECRET
        else:
            self.ws_public_url = "wss://stream.bybit.com/v5/public/linear"
            self.ws_private_url = "wss://stream.bybit.com/v5/private"
            self.api_key = BYBIT.API_KEY
            self.secret = BYBIT.SECRET

        self.ws_public: Optional[websocket.WebSocketApp] = None
        self.ws_private: Optional[websocket.WebSocketApp] = None

        self._running = False
        self._lock = threading.Lock()
        self._last_price: float = 0.0
        self._last_data_time: float = 0.0

        # 포지션 추적
        self._had_position = False
        self._position_closed_handled = False

        # 외부 청산 플래그
        self._external_close_in_progress = False

        # 청산 판별
        self._tracked_liquidation_price: Optional[float] = None
        self._tracked_direction: Optional[str] = None

        # 스레드 핸들
        self._public_thread: Optional[threading.Thread] = None
        self._private_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None

    def _generate_auth_signature(self) -> tuple:
        """Private 스트림 인증 서명 생성"""
        expires = int((time.time() + 10) * 1000)
        sign_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self.secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return expires, signature

    # ========== Public Stream ==========

    def _on_public_open(self, ws):
        """Public WebSocket 연결 시 구독 처리"""
        logger.info(f"[{self._symbol}] Public WebSocket 연결됨")

        # ticker 구독
        subscribe_msg = {
            "op": "subscribe",
            "args": [f"tickers.{self._symbol}"]
        }
        ws.send(json.dumps(subscribe_msg))

    def _on_public_message(self, ws, message):
            """Public 메시지 수신 처리"""
            try:
                data = json.loads(message)

                # 핑퐁
                if data.get("op") == "pong":
                    return

                # 구독 응답
                if data.get("op") == "subscribe":
                    if data.get("success"):
                        logger.info(f"Public 구독 성공: {data.get('conn_id')}")
                    return

                # Ticker 데이터
                topic = data.get("topic", "")
                if topic == f"tickers.{self._symbol}":
                    ticker_data = data.get("data", {})

                    # 가격 추출
                    new_price = float(ticker_data.get("markPrice", 0))
                    if new_price == 0:
                        new_price = float(ticker_data.get("lastPrice", 0))

                    with self._lock:
                        if new_price > 0:
                            self._last_price = new_price

                        self._last_data_time = time.time()
                        current_price = self._last_price

                    if self.on_price_callback and current_price > 0:
                        self.on_price_callback(current_price)

            except json.JSONDecodeError:
                logger.warning(f"JSON 파싱 실패: {message[:100]}")
            except Exception as e:
                logger.error(f"Public 메시지 처리 오류: {e}")

    def _on_public_error(self, ws, error):
        """Public WebSocket 에러 처리"""
        logger.error(f"Public WebSocket 에러: {error}")

    def _on_public_close(self, ws, close_status_code, close_msg):
        """Public WebSocket 종료 시 재연결"""
        logger.warning(f"Public WebSocket 종료: {close_status_code} - {close_msg}")

        if self._running:
            time.sleep(self.RECONNECT_DELAY)
            self._connect_public()

    def _connect_public(self):
        """Public WebSocket 연결"""
        self.ws_public = websocket.WebSocketApp(
            self.ws_public_url,
            on_open=self._on_public_open,
            on_message=self._on_public_message,
            on_error=self._on_public_error,
            on_close=self._on_public_close
        )

        self._public_thread = threading.Thread(
            target=self.ws_public.run_forever,
            daemon=True
        )
        self._public_thread.start()

    # ========== Private Stream ==========

    def _on_private_open(self, ws):
        """Private WebSocket 연결 시 인증 처리"""
        logger.info("Private WebSocket 연결됨")

        # 인증
        expires, signature = self._generate_auth_signature()
        auth_msg = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        ws.send(json.dumps(auth_msg))

    def _on_private_message(self, ws, message):
        """Private 메시지 수신 처리"""
        try:
            data = json.loads(message)

            # pong
            if data.get("op") == "pong":
                return

            # 인증
            if data.get("op") == "auth":
                if data.get("success"):
                    logger.info("Private 인증 성공")
                    # 구독
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": ["position"]
                    }
                    ws.send(json.dumps(subscribe_msg))
                else:
                    logger.error(f"Private 인증 실패: {data}")
                return

            if data.get("op") == "subscribe":
                if data.get("success"):
                    logger.info("Position 구독 성공")
                return

            # position
            topic = data.get("topic", "")
            if topic == "position":
                self._handle_position_update(data.get("data", []))

        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {message[:100]}")
        except Exception as e:
            logger.error(f"Private 메시지 처리 오류: {e}")

    def _handle_position_update(self, positions: list):
        """포지션 업데이트 처리"""
        for pos in positions:
            if pos.get("symbol") != self._symbol:
                continue

            size = float(pos.get("size", 0))
            side = pos.get("side", "")
            unrealized_pnl = float(pos.get("unrealisedPnl", 0))

            logger.info(f"[{self._symbol}] Position 업데이트: {side} size={size} pnl={unrealized_pnl}")

            with self._lock:
                if self._external_close_in_progress:
                    logger.info("외부 청산 처리 중 - Private WS 콜백 무시")
                    if size == 0:
                        self._had_position = False
                        self._position_closed_handled = True
                    continue

                # 포지션 소멸 감지
                if self._had_position and size == 0:
                    if not self._position_closed_handled:
                        self._position_closed_handled = True

                        # 강제청산 vs SL/TP 판별
                        close_reason = self._determine_close_reason()
                        logger.warning(f"Private WS: 포지션 청산 감지 - 판정: {close_reason}")

                        if self.on_position_closed_callback:
                            threading.Thread(
                                target=self.on_position_closed_callback,
                                args=(close_reason,),
                                daemon=True
                            ).start()

                # 상태 갱신
                self._had_position = size > 0
                if size > 0:
                    self._position_closed_handled = False

    def _determine_close_reason(self) -> str:
        """강제청산 vs SL/TP 서버체결 판별"""
        if not hasattr(self, '_tracked_liquidation_price') or self._tracked_liquidation_price is None:
            logger.warning("청산가 정보 없음 - SERVER_TRIGGERED로 판정")
            return "SERVER_TRIGGERED"

        current_price = self._last_price
        if current_price <= 0:
            logger.warning("현재 시장가 없음 - SERVER_TRIGGERED로 판정")
            return "SERVER_TRIGGERED"

        liq_price = self._tracked_liquidation_price
        direction = self._tracked_direction

        # 괴리율
        diff_pct = abs(current_price - liq_price) / liq_price * 100

        logger.info(f"청산 판별: 현재가={current_price:.2f} 청산가={liq_price:.2f} 괴리율={diff_pct:.2f}%")

        if diff_pct < 1.0:
            if direction == "LONG" and current_price <= liq_price * 1.01:
                logger.warning(f"강제청산 확정: LONG 포지션, 현재가({current_price:.2f}) ≤ 청산가({liq_price:.2f})")
                return "LIQUIDATION"
            elif direction == "SHORT" and current_price >= liq_price * 0.99:
                logger.warning(f"강제청산 확정: SHORT 포지션, 현재가({current_price:.2f}) ≥ 청산가({liq_price:.2f})")
                return "LIQUIDATION"

        # SL/TP 서버 체결
        logger.info(f"SL/TP 서버 체결로 판정: 괴리율 {diff_pct:.2f}% > 1%")
        return "SERVER_TRIGGERED"

    def set_liquidation_tracking(self, liquidation_price: float, direction: str):
        """청산 판별용 청산가/방향 설정"""
        with self._lock:
            self._tracked_liquidation_price = liquidation_price
            self._tracked_direction = direction
        logger.info(f"청산 추적 설정: 청산가={liquidation_price:.2f} 방향={direction}")

    def _on_private_error(self, ws, error):
        """Private WebSocket 에러 처리"""
        logger.error(f"Private WebSocket 에러: {error}")

    def _on_private_close(self, ws, close_status_code, close_msg):
        """Private WebSocket 종료 시 재연결"""
        logger.warning(f"Private WebSocket 종료: {close_status_code} - {close_msg}")

        if self._running:
            time.sleep(self.RECONNECT_DELAY)
            self._connect_private()

    def _connect_private(self):
        """Private WebSocket 연결"""
        self.ws_private = websocket.WebSocketApp(
            self.ws_private_url,
            on_open=self._on_private_open,
            on_message=self._on_private_message,
            on_error=self._on_private_error,
            on_close=self._on_private_close
        )

        self._private_thread = threading.Thread(
            target=self.ws_private.run_forever,
            daemon=True
        )
        self._private_thread.start()

    # ========== Ping/Pong ==========

    def _ping_loop(self):
        """주기적 ping 전송"""
        while self._running:
            time.sleep(self.PING_INTERVAL)

            ping_msg = json.dumps({"op": "ping"})

            try:
                if self.ws_public and self.ws_public.sock:
                    self.ws_public.send(ping_msg)

                if self.ws_private and self.ws_private.sock:
                    self.ws_private.send(ping_msg)
            except Exception as e:
                logger.warning(f"Ping 전송 실패: {e}")

    # ========== Public API ==========

    def start(self):
        """WebSocket 연결 시작"""
        if self._running:
            return

        self._running = True

        self._connect_public()
        self._connect_private()

        # Ping
        self._ping_thread = threading.Thread(
            target=self._ping_loop,
            daemon=True
        )
        self._ping_thread.start()

        logger.info(f"[{self._symbol}] WebSocket 시작됨")

    def stop(self):
        """WebSocket 연결 종료"""
        self._running = False

        if self.ws_public:
            self.ws_public.close()
        if self.ws_private:
            self.ws_private.close()

        logger.info(f"[{self._symbol}] WebSocket 종료됨")

    def get_last_price(self) -> float:
        """마지막 수신 가격 반환"""
        with self._lock:
            return self._last_price

    def get_last_data_time(self) -> float:
        """마지막 데이터 수신 시각 반환"""
        with self._lock:
            return self._last_data_time

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return (
            self._running and
            self.ws_public and
            self.ws_public.sock and
            self.ws_public.sock.connected
        )

    def set_position_tracking(self, has_position: bool):
        """포지션 추적 상태 설정 (외부에서 초기화 시 호출)"""
        with self._lock:
            self._had_position = has_position
            self._position_closed_handled = False
            self._external_close_in_progress = False

    def set_external_close_in_progress(self, in_progress: bool):
        """외부 청산 처리 상태 (중복 콜백 방지)"""
        with self._lock:
            self._external_close_in_progress = in_progress
            if in_progress:
                logger.info("외부 청산 처리 시작 - Private WS 콜백 비활성화")
            else:
                logger.info("외부 청산 처리 완료 - Private WS 콜백 활성화")
