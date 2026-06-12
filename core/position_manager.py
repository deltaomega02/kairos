"""포지션 실행 레이어.

- 시장가/지정가 진입과 체결 확인
- SL/TP 서버 트리거 등록
- DB 기록
- 수동 청산 (시장가)
"""

import time
from typing import Optional, Dict, Any
from config import get_logger, TRADING
from exchange.bybit_client import bybit_client, BybitClientError
from database.db_manager import db_manager
from core.signal_engine import SignalResult

logger = get_logger("position_manager")


class PositionManager:
    """진입/청산 오더 실행과 DB 기록을 담당한다."""

    LIMIT_ORDER_TIMEOUT_SEC = TRADING.LIMIT_ORDER_TIMEOUT_SEC  # 지정가 미체결 시 대기 한도
    FILL_CHECK_INTERVAL_SEC = 10

    def open_position(
        self,
        signal: SignalResult,
        leverage: int,
        quantity: float,
        liquidation_price: float,
        margin_used: float,
        symbol: str = "BTCUSDT"
    ) -> Optional[Dict[str, Any]]:
        """레버리지 설정 → 진입 주문 → 체결 확인 → 서버 SL/TP 등록 → DB insert."""
        side = "Buy" if signal.direction == "LONG" else "Sell"

        try:
            # 레버리지
            bybit_client.set_leverage(symbol=symbol, leverage=leverage)

            # 주문 실행 (V13+: limit 먼저 → 타임아웃 시 market 폴백)
            order = None
            if signal.entry_price is not None:
                order = self._execute_limit_order(side, quantity, signal.entry_price, symbol=symbol)
                if not order:
                    logger.info(f"[{symbol}] 지정가 미체결 → 시장가 폴백")
                    order = self._execute_market_order(side, quantity, symbol=symbol)
            else:
                order = self._execute_market_order(side, quantity, symbol=symbol)

            if not order:
                return None

            order_id = order["order_id"]

            # 체결 조회
            time.sleep(1)
            detail = bybit_client.get_execution_detail(order_id)

            if not detail or detail["total_qty"] <= 0:
                logger.error(f"체결 확인 실패: {order_id}")
                return None

            entry_price = detail["avg_price"]
            filled_qty = detail["total_qty"]
            entry_fee = abs(detail["exec_fee"])

            # SL/TP 계산
            if signal.direction == "LONG":
                sl_price = round(entry_price * (1 - signal.sl_pct / 100), 2)
                tp_price = round(entry_price * (1 + signal.tp_pct / 100), 2)
            else:
                sl_price = round(entry_price * (1 + signal.sl_pct / 100), 2)
                tp_price = round(entry_price * (1 - signal.tp_pct / 100), 2)

            # 서버 SL/TP
            bybit_client.set_trading_stop(
                symbol=symbol,
                stop_loss=sl_price,
                take_profit=tp_price
            )

            # DB
            position_uuid = db_manager.create_position(
                symbol=symbol,
                direction=signal.direction,
                strategy=signal.strategy,
                leverage=leverage,
                entry_price=entry_price,
                entry_quantity=filled_qty,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                liquidation_price=liquidation_price,
                signal_score=signal.score,
                signal_reason=signal.reason,
                entry_fee=entry_fee
            )

            logger.info(
                f"[{symbol}] 포지션 오픈: {signal.direction} {leverage}x @ {entry_price:,.2f} "
                f"qty={filled_qty} SL={sl_price:,.2f} TP={tp_price:,.2f} "
                f"전략={signal.strategy}"
            )

            return {
                "position_uuid": position_uuid,
                "entry_price": entry_price,
                "quantity": filled_qty,
                "direction": signal.direction,
                "leverage": leverage,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "liquidation_price": liquidation_price,
                "entry_fee": entry_fee,
                "strategy": signal.strategy,
                "margin_used": margin_used
            }

        except BybitClientError as e:
            logger.error(f"[{symbol}] 포지션 오픈 실패: {e}")
            return None

    def _execute_market_order(self, side: str, quantity: float, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """시장가 주문 실행"""
        return bybit_client.place_market_order(symbol=symbol, side=side, qty=quantity)

    def _execute_limit_order(self, side: str, quantity: float, price: float, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """지정가 PostOnly 주문 + 타임아웃 취소"""
        order = bybit_client.place_limit_order(
            symbol=symbol,
            side=side,
            qty=quantity,
            price=price,
            time_in_force="PostOnly"
        )

        if not order or not order.get("order_id"):
            return None

        order_id = order["order_id"]

        # 체결 폴링
        waited = 0
        while waited < self.LIMIT_ORDER_TIMEOUT_SEC:
            time.sleep(self.FILL_CHECK_INTERVAL_SEC)
            waited += self.FILL_CHECK_INTERVAL_SEC

            detail = bybit_client.get_execution_detail(order_id)
            if detail and detail["total_qty"] > 0:
                logger.info(f"지정가 체결 완료: {waited}초 후")
                return order

        # 타임아웃
        bybit_client.cancel_order(order_id=order_id)
        logger.info(f"지정가 미체결 → 취소 ({self.LIMIT_ORDER_TIMEOUT_SEC}초 초과)")
        return None

    def close_position(
        self,
        position_uuid: str,
        direction: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        reason: str,
        symbol: str = "BTCUSDT"
    ) -> Optional[Dict[str, Any]]:
        """시장가 청산 + PnL 계산 + DB 업데이트"""
        try:
            order = bybit_client.close_position(symbol=symbol, direction=direction)

            if not order or not order.get("order_id"):
                logger.error(f"[{symbol}] 청산 주문 실패")
                return None

            time.sleep(1)
            detail = bybit_client.get_execution_detail(order["order_id"])

            if not detail:
                logger.error(f"[{symbol}] 청산 체결 확인 실패")
                return None

            exit_price = detail["avg_price"]
            exit_fee = abs(detail["exec_fee"])

            # PnL
            if direction == "LONG":
                raw_pnl = (exit_price - entry_price) * quantity
            else:
                raw_pnl = (entry_price - exit_price) * quantity

            # 수수료 차감
            pos_record = db_manager.get_position_by_uuid(position_uuid)
            entry_fee_val = pos_record.get("entry_fee", 0) if pos_record else 0
            realized_pnl = raw_pnl - exit_fee - entry_fee_val
            realized_pnl_pct = (realized_pnl / (entry_price * quantity / leverage)) * 100

            # DB
            db_manager.close_position(
                position_uuid=position_uuid,
                exit_price=exit_price,
                exit_reason=reason,
                realized_pnl=realized_pnl,
                realized_pnl_percentage=realized_pnl_pct,
                exit_fee=exit_fee
            )

            logger.info(
                f"[{symbol}] 포지션 청산: {reason} | {direction} @ {exit_price:,.2f} | "
                f"PnL={realized_pnl:+.2f} ({realized_pnl_pct:+.2f}%)"
            )

            return {
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "exit_fee": exit_fee
            }

        except BybitClientError as e:
            logger.error(f"[{symbol}] 포지션 청산 실패: {e}")
            return None
