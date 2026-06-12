"""텔레그램 알림 포맷 + 전송.

KAIROS 메시지 정책 (HERMES 대비 노이즈 대폭 축소):
- 보냄: 진입 / 청산 / 시스템 시작 · 종료 / 시스템 에러 / 리스크 셧다운
- 안 보냄: 30분 요약 / 레짐 업데이트 / 시그널 체크 / 레짐 변경 / 쿨다운
  → 메서드는 호환성을 위해 유지하되 내부에서 즉시 return (no-op)
  → 정보는 logs/ 에서 확인

스타일 원칙:
- 해설/요약 문구 금지 (숫자와 상태만)
- 구분선은 정보 블록 구분용 1회
- 이모지는 상태(🟢🟡🔴) / 결과(✅❌) / 경고(🚨)만 최소
"""

import requests
from enum import Enum
from typing import Optional, Dict, Any

from config import TELEGRAM, get_logger

logger = get_logger("telegram")

USDKRW = 1470
SEP = "─" * 28


def _krw(usd: float) -> str:
    krw = usd * USDKRW
    if abs(krw) >= 1000:
        return f"₩{krw:+,.0f}"
    return f"₩{krw:+.0f}"


def _regime_ko(regime: str) -> str:
    return {
        "TRENDING_UP": "상승추세",
        "TRENDING_DOWN": "하락추세",
        "RANGING": "횡보",
        "HIGH_VOL": "고변동",
    }.get(regime.upper(), regime)


def _direction_ko(direction: str) -> str:
    return "LONG" if "LONG" in direction.upper() else "SHORT"


def _fmt_hold(raw: str) -> str:
    return raw if raw and raw != "?" else "—"


class AlertPriority(Enum):
    P0_EMERGENCY = "emergency"
    P1_TRADE = "trade"
    P2_INFO = "info"


class TelegramNotifier:
    """Bybit → KAIROS → 사용자로 이어지는 마지막 레이어. 모든 메시지를 동기 전송."""

    def __init__(self):
        self.bot_token = TELEGRAM.BOT_TOKEN
        self.chat_id = TELEGRAM.CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        if not self.bot_token or not self.chat_id:
            logger.warning("텔레그램 설정 누락")

    def _send_request(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        try:
            payload = {"chat_id": self.chat_id, "text": message}
            response = requests.post(self.base_url, json=payload, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")
            return False

    def send(self, message: str, priority: AlertPriority = AlertPriority.P2_INFO) -> bool:
        return self._send_request(message)

    def emergency(self, message: str) -> bool:
        return self.send(message, AlertPriority.P0_EMERGENCY)

    def trade(self, message: str) -> bool:
        return self.send(message, AlertPriority.P1_TRADE)

    def info(self, message: str) -> bool:
        return self.send(message, AlertPriority.P2_INFO)

    def status(self, message: str) -> bool:
        return self.send(message, AlertPriority.P2_INFO)

    # ================================================================
    # 시스템
    # ================================================================

    def send_system_start(self, balance: float, position_info=None, is_restart=False):
        pos = "없음"
        if position_info:
            d = position_info.get("direction", "?")
            l = position_info.get("leverage", 0)
            pos = f"{d} {l}x" if l else f"{d}"

        return self.info(
            f"🟢 KAIROS 가동\n"
            f"{SEP}\n"
            f"잔고  ${balance:,.2f} ({_krw(balance).replace('+', '')})\n"
            f"포지션  {pos}"
        )

    def send_periodic_summary(self, *args, **kwargs):
        # KAIROS: 노이즈 차단 — 30분 누적 요약은 보내지 않음
        return True

    def send_system_error(self, error_type, error_msg, location):
        return self.emergency(
            f"🚨 시스템 오류\n"
            f"{SEP}\n"
            f"위치  {location}\n"
            f"유형  {error_type}\n"
            f"{error_msg}"
        )

    # ================================================================
    # 4H 레짐 판독
    # ================================================================

    def send_regime_update(self, *args, **kwargs):
        # KAIROS: 노이즈 차단 — 4H 레짐 업데이트는 보내지 않음 (logs 확인)
        return True

    # ================================================================
    # 1H 시그널 체크
    # ================================================================

    def send_signal_check(self, *args, **kwargs):
        # KAIROS: 노이즈 차단 — 1H 시그널 체크는 보내지 않음 (체결만 알림)
        return True

    # ================================================================
    # 포지션 진입
    # ================================================================

    def send_position_opened(self, direction, leverage, entry_price, qty,
                             stop_loss, take_profit, margin_used,
                             strategy="", entry_fee=0, score=0):
        coin_sym = direction.split()[0] if " " in direction else direction
        dir_type = direction.split()[-1] if " " in direction else direction
        dir_ko = _direction_ko(dir_type)

        sl_pct = abs((stop_loss - entry_price) / entry_price * 100)
        tp_pct = abs((take_profit - entry_price) / entry_price * 100)
        rr = tp_pct / sl_pct if sl_pct > 0 else 0

        if "LONG" in dir_type:
            expected_profit = (take_profit - entry_price) * qty
            expected_loss = (entry_price - stop_loss) * qty
        else:
            expected_profit = (entry_price - take_profit) * qty
            expected_loss = (stop_loss - entry_price) * qty

        fee_estimate = entry_price * qty * 0.00055 * 2
        net_profit = expected_profit - fee_estimate
        net_loss = expected_loss + fee_estimate

        profit_pct_m = (net_profit / margin_used * 100) if margin_used > 0 else 0
        loss_pct_m = (net_loss / margin_used * 100) if margin_used > 0 else 0

        return self.trade(
            f"✅ [{coin_sym}] 진입 · {dir_ko} · {leverage}x\n"
            f"{SEP}\n"
            f"전략  {strategy} · 점수 {score:.0f}\n"
            f"진입  ${entry_price:,.2f} · 수량 {qty}\n"
            f"마진  ${margin_used:.2f} · 수수료 ~${fee_estimate:.2f}\n"
            f"{SEP}\n"
            f"TP  ${take_profit:,.2f}  +{tp_pct:.2f}%  →  +${net_profit:.2f} ({_krw(net_profit)}) / 마진 {profit_pct_m:+.1f}%\n"
            f"SL  ${stop_loss:,.2f}  -{sl_pct:.2f}%  →  -${net_loss:.2f} ({_krw(-net_loss)}) / 마진 {loss_pct_m:-.1f}%\n"
            f"RR  1:{rr:.1f}"
        )

    # ================================================================
    # 포지션 청산
    # ================================================================

    def send_position_closed(self, direction, reason, entry_price, exit_price,
                             pnl, pnl_pct, hold_time="", total_fee=0, strategy=""):
        coin_sym = direction.split()[0] if " " in direction else direction
        dir_type = direction.split()[-1] if " " in direction else direction
        dir_ko = _direction_ko(dir_type)

        is_profit = pnl > 0
        result_emoji = "🟢" if is_profit else "🔴"

        label_map = {
            "LIQUIDATION": "강제청산",
            "DEAD_MANS_SWITCH": "긴급청산",
            "TAKE_PROFIT": "익절",
            "TRAILING_STOP": "트레일링 익절" if is_profit else "트레일링 손절",
            "STOP_LOSS": "트레일링 익절" if is_profit else "손절",
            "SERVER_TRIGGERED": "서버 체결",
        }
        label = label_map.get(reason, reason)

        price_change = (exit_price - entry_price) / entry_price * 100
        hold_str = _fmt_hold(hold_time)

        return self.trade(
            f"{result_emoji} [{coin_sym}] 청산 · {label}\n"
            f"{SEP}\n"
            f"방향  {dir_ko} · {strategy} · 보유 {hold_str}\n"
            f"진입 ${entry_price:,.2f}  →  청산 ${exit_price:,.2f} ({price_change:+.2f}%)\n"
            f"수수료  ~${total_fee:.2f}\n"
            f"{SEP}\n"
            f"PnL   {pnl:+.2f} USDT ({_krw(pnl)})\n"
            f"마진  {pnl_pct:+.2f}%"
        )

    # ================================================================
    # 레짐 변경 (하위호환)
    # ================================================================

    def send_regime_change(self, *args, **kwargs):
        # KAIROS: 노이즈 차단 — 레짐 변경은 보내지 않음
        return True

    # ================================================================
    # 리스크 알림
    # ================================================================

    def send_risk_alert(self, alert_type, detail=""):
        type_map = {
            "TRADE_HALTED": "거래 중단",
            "EMERGENCY_HIGH_VOL": "긴급 고변동",
            "SERVER_CLOSE": "서버 청산 감지",
            "DD_WARNING": "드로다운 경고",
            "SHUTDOWN": "시스템 셧다운",
        }
        title = type_map.get(alert_type, alert_type)

        body = f"{title}"
        if detail:
            body += f"\n{detail}"

        return self.emergency(
            f"🚨 리스크\n"
            f"{SEP}\n"
            f"{body}"
        )

    def send_cooldown_activated(self, *args, **kwargs):
        # KAIROS: 노이즈 차단 — 쿨다운 알림은 보내지 않음
        return True


telegram_notifier = TelegramNotifier()
