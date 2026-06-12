# exchange/bybit_client.py — Bybit REST API 클라이언트

import time
import hmac
import hashlib
import json
import uuid
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode

import requests

from config import BYBIT, TRADING, get_logger

logger = get_logger("bybit_client")


def safe_float(value, default: float = 0.0) -> float:
    """빈 문자열 또는 None을 안전하게 float로 변환"""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class BybitClientError(Exception):
    """Bybit API 에러"""
    pass


class InsufficientBalanceError(BybitClientError):
    """잔고 부족"""
    pass


class InvalidLeverageError(BybitClientError):
    """레버리지 범위 초과"""
    pass


class BybitClient:
    """Bybit v5 REST API 클라이언트 (USDT Perpetual)"""

    RECV_WINDOW = 5000

    def __init__(self):
        """Bybit 클라이언트 초기화"""
        if BYBIT.USE_TESTNET:
            self.api_key = BYBIT.TESTNET_API_KEY
            self.secret = BYBIT.TESTNET_SECRET
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.api_key = BYBIT.API_KEY
            self.secret = BYBIT.SECRET
            self.base_url = "https://api.bybit.com"

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })

    def _generate_signature(self, timestamp: str, params: str) -> str:
        """HMAC SHA256 서명 생성"""
        param_str = f"{timestamp}{self.api_key}{self.RECV_WINDOW}{params}"
        return hmac.new(
            self.secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """API 요청 실행 (exponential backoff retry)"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        last_exception = None

        for attempt in range(max_retries):
            headers = {}
            if signed:
                timestamp = str(int(time.time() * 1000))

                if method == "GET":
                    param_str = urlencode(params) if params else ""
                else:
                    param_str = json.dumps(params) if params else ""

                signature = self._generate_signature(timestamp, param_str)

                headers = {
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": str(self.RECV_WINDOW)
                }

            try:
                if method == "GET":
                    response = self.session.get(url, params=params, headers=headers, timeout=10)
                else:
                    response = self.session.post(url, json=params, headers=headers, timeout=10)

                response.raise_for_status()
                data = response.json()

                ret_code = data.get("retCode", 0)
                if ret_code != 0:
                    error_msg = data.get("retMsg", "Unknown error")

                    # v6: orderLinkId 중복 = 첫 요청이 이미 성공한 것 (재시도 phantom)
                    # retCode 110072 = duplicate orderLinkId (Bybit v5)
                    if ret_code == 110072 or "duplicate" in error_msg.lower():
                        logger.warning(
                            f"중복 orderLinkId 감지 (첫 요청이 이미 성공): {error_msg} "
                            f"— 재시도 스킵하고 성공 처리"
                        )
                        # 첫 요청이 이미 성공했을 것이므로 빈 결과 반환 (호출자가 조회로 확인)
                        return {"duplicate_detected": True}

                    # 비즈니스 에러: 즉시 raise
                    if "insufficient" in error_msg.lower():
                        raise InsufficientBalanceError(error_msg)

                    # 서버 에러: retry
                    if ret_code >= 10000:
                        last_exception = BybitClientError(error_msg)
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt
                            logger.warning(
                                f"Bybit 서버 에러 ({attempt+1}/{max_retries}): "
                                f"retCode={ret_code} {error_msg} → {wait}s 후 재시도"
                            )
                            time.sleep(wait)
                            continue
                        logger.error(f"Bybit API Error (재시도 소진): {error_msg}")
                        raise last_exception

                    # 기타 에러
                    logger.error(f"Bybit API Error: {error_msg}")
                    raise BybitClientError(error_msg)

                if attempt > 0:
                    logger.info(f"API 요청 성공 (재시도 {attempt}회 후): {endpoint}")
                return data.get("result", {})

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = BybitClientError(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"네트워크 에러 ({attempt+1}/{max_retries}): {e} → {wait}s 후 재시도"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Request failed (재시도 소진): {e}")
                raise last_exception

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                raise BybitClientError(f"Request failed: {e}")

        raise last_exception or BybitClientError("Max retries exceeded")

    # ========== Market Data ==========

    def get_kline(
        self,
        symbol: str = TRADING.SYMBOL,
        interval: str = "240",
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """OHLCV 캔들 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        result = self._request("GET", "/v5/market/kline", params)

        # 오래된 순 정렬
        candles = []
        for item in reversed(result.get("list", [])):
            candles.append({
                "timestamp": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "turnover": float(item[6])
            })

        return candles

    def get_ticker(self, symbol: str = TRADING.SYMBOL) -> Dict[str, Any]:
        """현재 시세 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol
        }

        result = self._request("GET", "/v5/market/tickers", params)

        if result.get("list"):
            ticker = result["list"][0]
            return {
                "symbol": ticker.get("symbol"),
                "last_price": safe_float(ticker.get("lastPrice")),
                "index_price": safe_float(ticker.get("indexPrice")),
                "mark_price": safe_float(ticker.get("markPrice")),
                "funding_rate": safe_float(ticker.get("fundingRate")),
                "next_funding_time": int(ticker.get("nextFundingTime", 0)),
                "open_interest": safe_float(ticker.get("openInterest")),
                "bid_price": safe_float(ticker.get("bid1Price")),
                "ask_price": safe_float(ticker.get("ask1Price")),
                "volume_24h": safe_float(ticker.get("volume24h")),
                "price_change_24h_pct": safe_float(ticker.get("price24hPcnt"))
            }

        return {}

    # ========== Account ==========

    def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, Any]:
            """지갑 잔고 조회"""
            params = {
                "accountType": "UNIFIED",
                "coin": coin
            }

            try:
                result = self._request("GET", "/v5/account/wallet-balance", params, signed=True)

                for account in result.get("list", []):
                    for c in account.get("coin", []):
                        if c.get("coin") == coin:
                            wallet_balance = safe_float(c.get("walletBalance"))
                            equity = safe_float(c.get("equity"))
                            raw_available = safe_float(c.get("availableToWithdraw"))

                            if raw_available <= 0 and wallet_balance > 0:
                                final_available = wallet_balance * 0.90
                            else:
                                safe_cap = wallet_balance * 0.90
                                final_available = raw_available if raw_available < safe_cap else safe_cap

                            logger.info(f"자산 전달 (USDT): 지갑잔고={wallet_balance}, 사용가능금액(보정됨)={final_available}")

                            return {
                                "coin": coin,
                                "wallet_balance": wallet_balance,
                                "available_balance": final_available,
                                "total_equity": equity,
                                "unrealized_pnl": safe_float(c.get("unrealisedPnl"))
                            }

                return {"coin": coin, "wallet_balance": 0, "available_balance": 0}

            except Exception as e:
                logger.error(f"잔고 조회 중 오류 발생: {e}")
                return {"coin": coin, "wallet_balance": 0, "available_balance": 0}

    # ========== Position ==========

    def get_position(self, symbol: str = TRADING.SYMBOL) -> Optional[Dict[str, Any]]:
        """현재 포지션 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol
        }

        result = self._request("GET", "/v5/position/list", params, signed=True)

        for pos in result.get("list", []):
            size = safe_float(pos.get("size"))
            if size > 0:
                return {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "size": size,
                    "entry_price": safe_float(pos.get("avgPrice")),
                    "mark_price": safe_float(pos.get("markPrice")),
                    "liquidation_price": safe_float(pos.get("liqPrice")),
                    "leverage": int(pos.get("leverage", 1)),
                    "unrealized_pnl": safe_float(pos.get("unrealisedPnl")),
                    "position_value": safe_float(pos.get("positionValue")),
                    "position_margin": safe_float(pos.get("positionIM"))
                }

        return None

    def set_leverage(
        self,
        symbol: str = TRADING.SYMBOL,
        leverage: int = 1
    ) -> bool:
        """레버리지 설정"""
        if not TRADING.MIN_LEVERAGE <= leverage <= TRADING.MAX_LEVERAGE:
            raise InvalidLeverageError(
                f"레버리지는 {TRADING.MIN_LEVERAGE}~{TRADING.MAX_LEVERAGE} 범위여야 합니다"
            )

        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }

        try:
            self._request("POST", "/v5/position/set-leverage", params, signed=True)
            logger.info(f"레버리지 설정 완료: {leverage}x")
            return True
        except BybitClientError as e:
            # 동일 레버리지 설정은 무시
            if "leverage not modified" in str(e).lower():
                return True
            raise

    # ========== Order ==========

    def place_market_order(
        self,
        symbol: str = TRADING.SYMBOL,
        side: str = "Buy",
        qty: float = 0,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """시장가 주문 (v6: orderLinkId로 중복 방지)"""
        # 중복 주문 방지용 고유 ID — 재시도 시 Bybit이 중복 거절
        link_id = f"hermes-{uuid.uuid4().hex[:16]}"
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty) if qty > 0 else "0",
            "timeInForce": "GTC",
            "reduceOnly": reduce_only,
            "orderLinkId": link_id,
        }

        result = self._request("POST", "/v5/order/create", params, signed=True)

        order_id = result.get("orderId", "")
        logger.info(f"시장가 주문 체결: {side} {qty} {symbol}, ID: {order_id}")

        return {
            "order_id": order_id,
            "order_link_id": result.get("orderLinkId", ""),
            "symbol": symbol,
            "side": side,
            "qty": qty
        }

    def place_limit_order(
        self,
        symbol: str = TRADING.SYMBOL,
        side: str = "Buy",
        qty: float = 0,
        price: float = 0,
        reduce_only: bool = False,
        time_in_force: str = "PostOnly"
    ) -> Dict[str, Any]:
        """지정가 주문 (PostOnly = Maker 보장, v6: orderLinkId 추가)"""
        link_id = f"hermes-{uuid.uuid4().hex[:16]}"
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(round(price, 2)),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "orderLinkId": link_id,
        }

        result = self._request("POST", "/v5/order/create", params, signed=True)

        order_id = result.get("orderId", "")
        logger.info(f"지정가 주문: {side} {qty} {symbol} @ {price}, ID: {order_id}")

        return {
            "order_id": order_id,
            "order_link_id": result.get("orderLinkId", ""),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price
        }

    def cancel_order(
        self,
        symbol: str = TRADING.SYMBOL,
        order_id: str = ""
    ) -> bool:
        """주문 취소"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "orderId": order_id
        }

        try:
            self._request("POST", "/v5/order/cancel", params, signed=True)
            logger.info(f"주문 취소: {order_id}")
            return True
        except BybitClientError as e:
            if "order not exists" in str(e).lower():
                logger.info(f"이미 체결/취소된 주문: {order_id}")
                return True
            raise

    def get_open_orders(
        self,
        symbol: str = TRADING.SYMBOL
    ) -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol
        }

        result = self._request("GET", "/v5/order/realtime", params, signed=True)

        orders = []
        for order in result.get("list", []):
            orders.append({
                "order_id": order.get("orderId"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "order_type": order.get("orderType"),
                "price": safe_float(order.get("price")),
                "qty": safe_float(order.get("qty")),
                "status": order.get("orderStatus"),
                "created_time": int(order.get("createdTime", 0))
            })

        return orders

    def close_position(
        self,
        symbol: str = TRADING.SYMBOL,
        direction: str = "LONG"
    ) -> Dict[str, Any]:
        """포지션 전량 청산"""
        close_side = "Sell" if direction == "LONG" else "Buy"
        position = self.get_position(symbol)
        if not position:
            logger.warning("청산할 포지션이 없습니다")
            return {}

        return self.place_market_order(
            symbol=symbol,
            side=close_side,
            qty=position["size"],
            reduce_only=True
        )

    def get_order_history(
        self,
        symbol: str = TRADING.SYMBOL,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """주문 내역 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "limit": limit
        }

        result = self._request("GET", "/v5/order/history", params, signed=True)

        orders = []
        for order in result.get("list", []):
            orders.append({
                "order_id": order.get("orderId"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "order_type": order.get("orderType"),
                "price": safe_float(order.get("price")),
                "qty": safe_float(order.get("qty")),
                "avg_price": safe_float(order.get("avgPrice")),
                "status": order.get("orderStatus"),
                "created_time": int(order.get("createdTime", 0)),
                "updated_time": int(order.get("updatedTime", 0))
            })

        return orders

    # ========== Execution ==========

    def get_execution_list(
        self,
        symbol: str = TRADING.SYMBOL,
        order_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """체결 내역 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "limit": limit
        }

        if order_id:
            params["orderId"] = order_id

        result = self._request("GET", "/v5/execution/list", params, signed=True)

        executions = []
        for exec_item in result.get("list", []):
            executions.append({
                "exec_id": exec_item.get("execId"),
                "order_id": exec_item.get("orderId"),
                "symbol": exec_item.get("symbol"),
                "side": exec_item.get("side"),
                "exec_price": safe_float(exec_item.get("execPrice")),
                "exec_qty": safe_float(exec_item.get("execQty")),
                "exec_fee": safe_float(exec_item.get("execFee")),
                "fee_rate": safe_float(exec_item.get("feeRate")),
                "exec_type": exec_item.get("execType"),
                "exec_time": int(exec_item.get("execTime", 0))
            })

        return executions

    def get_execution_detail(self, order_id: str) -> Optional[Dict[str, Any]]:
        """주문 체결 상세 (부분체결 합산)"""
        executions = self.get_execution_list(order_id=order_id, limit=10)

        if not executions:
            return None

        # 합산
        total_qty = 0.0
        total_value = 0.0
        total_fee = 0.0

        for ex in executions:
            qty = ex["exec_qty"]
            price = ex["exec_price"]
            fee = ex["exec_fee"]

            total_qty += qty
            total_value += qty * price
            total_fee += fee

        avg_price = total_value / total_qty if total_qty > 0 else 0

        return {
            "order_id": order_id,
            "avg_price": round(avg_price, 2),
            "total_qty": round(total_qty, 6),
            "exec_fee": round(total_fee, 6),
            "exec_count": len(executions)
        }

    def get_execution_fee(self, order_id: str) -> float:
        """주문 수수료 조회"""
        detail = self.get_execution_detail(order_id)
        if detail:
            return detail["exec_fee"]
        return 0.0

    def get_funding_history(
        self,
        symbol: str = TRADING.SYMBOL,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """펀딩비 내역 조회 (transaction-log 기반, 양수=수취, 음수=지불)"""
        params = {
            "category": TRADING.CATEGORY,
            "type": "SETTLEMENT",  # 펀딩비는 SETTLEMENT 타입으로 기록됨
            "limit": limit
        }

        if symbol:
            params["symbol"] = symbol

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        result = self._request("GET", "/v5/account/transaction-log", params, signed=True)

        funding_records = []
        for item in result.get("list", []):
            item_type = item.get("type", "")
            item_symbol = item.get("symbol", "")

            # SETTLEMENT 타입만 수집
            if item_type == "SETTLEMENT" and (not symbol or item_symbol == symbol):
                funding_value = abs(safe_float(item.get("funding")))
                fee_rate = safe_float(item.get("feeRate"))
                side = item.get("side", "")

                if funding_value != 0:
                    # 부호: feeRate > 0 = 롱 지불/숏 수취, < 0 = 반대
                    if side == "Sell":
                        actual_funding = funding_value if fee_rate > 0 else -funding_value
                    else:
                        actual_funding = funding_value if fee_rate < 0 else -funding_value

                    funding_records.append({
                        "symbol": item_symbol,
                        "funding_rate": fee_rate,
                        "funding_fee": actual_funding,
                        "timestamp": int(item.get("transactionTime", 0)),
                        "size": safe_float(item.get("size"))
                    })

        return funding_records

    def get_closed_pnl_funding(
        self,
        symbol: str = TRADING.SYMBOL,
        start_time: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """청산 PnL 내역 조회"""
        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "limit": limit
        }

        if start_time:
            params["startTime"] = start_time

        result = self._request("GET", "/v5/position/closed-pnl", params, signed=True)

        records = []
        for item in result.get("list", []):
            records.append({
                "symbol": item.get("symbol"),
                "order_id": item.get("orderId"),
                "side": item.get("side"),
                "qty": safe_float(item.get("qty")),
                "entry_price": safe_float(item.get("avgEntryPrice")),
                "exit_price": safe_float(item.get("avgExitPrice")),
                "closed_pnl": safe_float(item.get("closedPnl")),
                "cum_entry_value": safe_float(item.get("cumEntryValue")),
                "cum_exit_value": safe_float(item.get("cumExitValue")),
                "created_time": int(item.get("createdTime", 0)),
                "updated_time": int(item.get("updatedTime", 0))
            })

        return records

    def get_total_funding_fee_for_position(
        self,
        symbol: str = TRADING.SYMBOL,
        entry_time_ms: int = 0
    ) -> float:
        """진입 시점 이후 누적 펀딩비 조회"""
        try:
            now_ms = int(time.time() * 1000)
            records = self.get_funding_history(
                symbol=symbol,
                start_time=entry_time_ms,
                end_time=now_ms,
                limit=100
            )

            if records:
                total_fee = sum(r["funding_fee"] for r in records)
                logger.info(f"펀딩비 조회: {len(records)}건, 총 {total_fee:.4f} USDT")
                return total_fee

            logger.info(f"펀딩비 조회: 0건, 총 0.0000 USDT")
            return 0.0

        except Exception as e:
            logger.warning(f"펀딩비 조회 실패: {e}")
            return 0.0

    def set_trading_stop(
        self,
        symbol: str = TRADING.SYMBOL,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """포지션 SL/TP 설정"""
        if stop_loss is None and take_profit is None:
            logger.warning("set_trading_stop: SL/TP 둘 다 None")
            return False

        params = {
            "category": TRADING.CATEGORY,
            "symbol": symbol,
            "positionIdx": 0  # One-Way Mode
        }

        if stop_loss is not None:
            params["stopLoss"] = str(round(stop_loss, 2))

        if take_profit is not None:
            params["takeProfit"] = str(round(take_profit, 2))

        try:
            self._request("POST", "/v5/position/trading-stop", params, signed=True)
            logger.info(f"Trading Stop 설정: SL={stop_loss}, TP={take_profit}")
            return True
        except BybitClientError as e:
            # 동일 값이면 성공 처리
            if "not modified" in str(e).lower():
                logger.debug(f"Trading Stop 변경 없음 (동일 값): SL={stop_loss}, TP={take_profit}")
                return True
            logger.error(f"Trading Stop 설정 실패: {e}")
            return False


# 싱글톤 인스턴스
bybit_client = BybitClient()
