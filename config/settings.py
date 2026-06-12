# config/settings.py — 거래소, 거래, 스케줄러 설정

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Tuple

load_dotenv()

_USE_TESTNET = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"


@dataclass(frozen=True)
class BybitConfig:
    """Bybit REST / WebSocket endpoints + API 자격증명."""
    API_KEY: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    SECRET: str = field(default_factory=lambda: os.getenv("BYBIT_SECRET", ""))
    TESTNET_API_KEY: str = field(default_factory=lambda: os.getenv("BYBIT_TESTNET_API_KEY", ""))
    TESTNET_SECRET: str = field(default_factory=lambda: os.getenv("BYBIT_TESTNET_SECRET", ""))
    USE_TESTNET: bool = _USE_TESTNET
    BASE_URL: str = "https://api-testnet.bybit.com" if _USE_TESTNET else "https://api.bybit.com"
    WS_PUBLIC: str = "wss://stream-testnet.bybit.com/v5/public/linear" if _USE_TESTNET else "wss://stream.bybit.com/v5/public/linear"
    WS_PRIVATE: str = "wss://stream-testnet.bybit.com/v5/private" if _USE_TESTNET else "wss://stream.bybit.com/v5/private"


@dataclass(frozen=True)
class TradingConfig:
    """거래·리스크 관련 정적 설정.

    KAIROS v2: BTC 단일 코인 / 일봉 변동성 돌파 / 3x 보수.
    동적 튜닝 대상은 `tunable_params.json`. 여기는 상한/상수/구조적 값만.
    """
    SYMBOL: str = "BTCUSDT"
    CATEGORY: str = "linear"

    # 레버리지 상한 (KAIROS: 3x 고정 권장, HERMES 7x 대비 보수)
    MIN_LEVERAGE: int = 1
    MAX_LEVERAGE: int = 3

    # 리스크 파라미터
    RISK_PER_TRADE_PCT: float = 0.015     # 거래당 잔액의 1.5%
    MARGIN_USAGE_PCT: float = 0.80
    STOP_LOSS_MARGIN_PCT: float = 0.02
    LIQUIDATION_WARN_PCT: float = 0.03

    # 수수료
    TAKER_FEE_PCT: float = 0.00055   # 0.055%
    MAKER_FEE_PCT: float = 0.0002    # 0.02%

    # 유지마진율
    MAINTENANCE_MARGIN_RATE: float = 0.004

    # BTC 주문 단위
    MIN_ORDER_QTY: float = 0.001
    QTY_PRECISION: int = 3

    # KAIROS v2: BTC 단일 코인. 동시 1포지션.
    SYMBOLS: tuple = ("BTCUSDT",)
    MAX_SIMULTANEOUS_POSITIONS: int = 1

    # 코인별 최소 주문 수량
    MIN_ORDER_QTYS: dict = None  # __post_init__에서 설정
    QTY_PRECISIONS: dict = None

    def __post_init__(self):
        """거래소가 심볼별로 요구하는 최소 주문 수량과 소수점 자릿수."""
        object.__setattr__(self, 'MIN_ORDER_QTYS', {
            "BTCUSDT": 0.001,
        })
        object.__setattr__(self, 'QTY_PRECISIONS', {
            "BTCUSDT": 3,
        })

    # 일일/누적 리스크 한도 (운영 정책: 손실 기반 시스템적 차단/축소 X)
    MAX_DAILY_LOSS_PCT: float = 1.00     # 사실상 비활성 (100% = 잔고 0 되어야 작동)
    MAX_DAILY_TRADES: int = 9999
    MAX_DRAWDOWN_PCT: float = 1.00       # 사실상 비활성 (100%)
    DRAWDOWN_WARNING_PCT: float = 1.00   # 사이즈 축소 비활성 (100%)

    # 펀딩비 (V13.2 운영 정책: 시간 차단 비활성)
    FUNDING_AVOIDANCE_MINUTES: int = 0
    FUNDING_EXTREME_POSITIVE: float = 0.0005
    FUNDING_EXTREME_NEGATIVE: float = -0.0005

    # 오더북
    ORDERBOOK_DEPTH: int = 25
    ORDERBOOK_IMBALANCE_THRESHOLD: float = 0.6

    # 지정가 진입
    LIMIT_ORDER_TIMEOUT_SEC: int = 300
    LIMIT_ORDER_OFFSET_PCT: float = 0.01


@dataclass(frozen=True)
class BreakoutConfig:
    """KAIROS v2 변동성 돌파 전략 (Larry Williams) 설정.

    매일 시가 기준으로 어제 변동폭의 K배만큼 돌파하면 매수,
    다음날 시가에 청산.
    """
    # K값 (어제 range × K = 돌파 임계값). 백테스트에서 결정.
    K: float = 0.6

    # 진입 시 시드 사용 비율 (마진 기준)
    POSITION_SIZE_PCT: float = 0.30

    # 손절 폭 (진입가 대비 %, 일봉 평균 변동성 고려)
    SL_PCT: float = 0.02

    # 다음 일봉 시작 시 청산 (Bybit 일봉 = UTC 00:00)
    EXIT_AT_NEXT_OPEN: bool = True

    # 진입 후보 체크 주기 (초). 일봉이라 자주 안 봐도 됨.
    CHECK_INTERVAL_SEC: int = 300

    # 최소 변동성 필터 (어제 range가 시가의 X% 미만이면 진입 X)
    MIN_RANGE_PCT: float = 0.01  # 1%


@dataclass(frozen=True)
class TelegramConfig:
    BOT_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))


@dataclass(frozen=True)
class SchedulerConfig:
    """메인 루프의 폴링/감시 주기."""
    REGIME_CHECK_INTERVAL_SEC: int = 60
    SIGNAL_CHECK_INTERVAL_SEC: int = 60
    ORDERBOOK_UPDATE_INTERVAL_SEC: int = 10

    # Dead Man's Switch
    DEAD_MANS_SWITCH_TIMEOUT_SEC: int = 120

    # 일일 리포트
    DAILY_REPORT_HOUR: int = 9
    DAILY_REPORT_MINUTE: int = 0


# 모듈 로드 시 한 번만 인스턴스화 → 애플리케이션 전역에서 공유
BYBIT = BybitConfig()
TRADING = TradingConfig()
BREAKOUT = BreakoutConfig()
TELEGRAM = TelegramConfig()
SCHEDULER = SchedulerConfig()
