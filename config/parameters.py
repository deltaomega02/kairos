"""Tunable parameters with bounds and max-delta constraints.

`tunable_params.json` 을 로드해 값을 덮어쓴다. 런타임에 Optuna 또는 수동으로
변경하고 싶을 때 `update()` 를 쓰면 min/max/max_delta 검증을 거친다.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from config.logging_config import get_logger

logger = get_logger("parameters")

PARAMS_FILE = Path(__file__).parent.parent / "config" / "tunable_params.json"


@dataclass
class ParamSpec:
    """단일 튜닝 파라미터. `max_delta` 는 한 번의 update() 최대 변경 폭."""
    name: str
    value: float
    min_val: float
    max_val: float
    max_delta: float
    description: str

    def validate_change(self, new_value: float) -> bool:
        """범위·변경폭 제약을 동시에 만족하는지 확인."""
        if not (self.min_val <= new_value <= self.max_val):
            return False
        if abs(new_value - self.value) > self.max_delta:
            return False
        return True


class TunableParameters:
    """Thread-safe 파라미터 레지스트리. 전역 싱글톤으로 사용."""

    def __init__(self):
        self._lock = threading.Lock()
        self._params: Dict[str, ParamSpec] = {}
        self._init_defaults()
        self._load_from_file()

    def _init_defaults(self):
        """기본값 + 검증 범위 등록. 이후 JSON 파일이 있으면 값을 덮어쓴다."""
        defaults = [
            # 1H 진입 지표
            ParamSpec("ema_fast", 3, 2, 15, 3, "1H EMA fast period"),
            ParamSpec("ema_slow", 15, 8, 30, 4, "1H EMA slow period"),
            ParamSpec("rsi_period", 14, 10, 20, 2, "1H RSI period"),
            ParamSpec("rsi_oversold", 35, 25, 40, 3, "풀백 LONG RSI 임계값"),
            ParamSpec("rsi_overbought", 65, 60, 75, 3, "풀백 SHORT RSI 임계값"),
            ParamSpec("pullback_ema_dist_pct", 1.5, 0.3, 2.0, 0.2, "EMA 풀백 허용 거리 (%)"),

            # SL / TP
            ParamSpec("sl_atr_mult", 2.0, 1.0, 3.0, 0.5, "SL = ATR × multiplier"),
            ParamSpec("tp_rr_ratio", 6.0, 1.5, 8.0, 0.3, "TP R:R 비율"),

            # 진입 점수
            ParamSpec("entry_score_threshold", 60, 30, 80, 10, "최소 진입 점수 (V13.1: 60 — 약신호 차단으로 승률 +10%p)"),

            # 4H 레짐
            ParamSpec("adx_enter_trending", 30, 20, 45, 3, "TRENDING 진입 ADX"),
            ParamSpec("adx_exit_trending", 20, 15, 25, 3, "TRENDING 이탈 ADX"),
            ParamSpec("atr_high_vol_percentile", 85, 70, 95, 5, "HIGH_VOL 전환 ATR 퍼센타일"),
            ParamSpec("regime_debounce_bars", 1, 1, 3, 1, "레짐 전환 디바운스 (4H bars)"),

            # 오더북 / 펀딩
            ParamSpec("orderbook_imbalance_min", 0.45, 0.40, 0.70, 0.10, "오더북 불균형 최소 (LONG 전용)"),
            ParamSpec("funding_bias_threshold", 0.0005, 0.0002, 0.001, 0.0001, "펀딩 편향 임계값"),

            # 오더북 비대칭 필터 (edge hunt 8개월 검증 결과)
            # 0 = 해당 방향에 오더북 필터 적용 안함
            # 1 = 적용 (threshold = orderbook_imbalance_min)
            ParamSpec("ob_filter_long", 1, 0, 1, 1, "LONG에 오더북 필터 적용"),
            ParamSpec("ob_filter_short", 0, 0, 1, 1, "SHORT에 오더북 필터 적용 (0=구조적 노이즈로 스킵)"),

            # 진입 환경 필터 (V13.2 운영 정책: 모든 차단 비활성)
            ParamSpec("min_atr_pct", 0.0, 0.0, 1.0, 0.1, "진입 허용 최소 ATR% (0=비활성)"),
            ParamSpec("blocked_hour_kst", -1, -1, 23, 24, "진입 차단 시간 KST (-1=비활성)"),
            ParamSpec("max_consecutive_losses", 0, 0, 10, 3, "진입 차단 연속 패배 (0=비활성)"),

            # 트레일링 스탑
            ParamSpec("trailing_activation_pct", 1.2, 0.3, 10.0, 0.1, "트레일링 활성화 수익 (%)"),
            ParamSpec("trailing_distance_pct", 0.1, 0.05, 3.0, 0.05, "트레일링 추적 거리 (%)"),

            # 1D (일봉) 추세 필터
            ParamSpec("d1_filter_enable", 1, 0, 1, 1, "1D 필터 ON/OFF"),
            ParamSpec("d1_ema_period", 2, 2, 30, 3, "1D EMA 기간 (일)"),
            ParamSpec("d1_filter_mode", 1, 0, 1, 1, "1D 필터 모드 (0=slope direction, 1=price_above_ema)"),
        ]

        for p in defaults:
            self._params[p.name] = p

    def _load_from_file(self):
        """`tunable_params.json` 이 있으면 값을 덮어쓴다. 파일이 없으면 기본값 유지."""
        if PARAMS_FILE.exists():
            try:
                with open(PARAMS_FILE) as f:
                    saved = json.load(f)
                for name, value in saved.items():
                    if name in self._params:
                        self._params[name].value = float(value)
                logger.info(f"파라미터 로드: {PARAMS_FILE}")
            except Exception as e:
                logger.warning(f"파라미터 로드 실패: {e}")

    def save_to_file(self):
        """현재 값을 JSON 파일에 기록. Optuna 가 채택한 값을 영속화할 때 사용."""
        with self._lock:
            data = {name: p.value for name, p in self._params.items()}
            with open(PARAMS_FILE, "w") as f:
                json.dump(data, f, indent=2)

    def get(self, name: str) -> float:
        """값 조회. 모르는 이름이면 KeyError. 호출측 실수를 빨리 드러내는 것이 목적."""
        with self._lock:
            if name not in self._params:
                raise KeyError(f"Unknown parameter: {name}")
            return self._params[name].value

    def get_int(self, name: str) -> int:
        """정수형이 필요한 곳에서 사용 (예: regime_debounce_bars)."""
        return int(self.get(name))

    def update(self, name: str, new_value: float) -> bool:
        """검증을 통과한 경우에만 값을 변경. 성공 여부를 bool 로 반환."""
        with self._lock:
            if name not in self._params:
                logger.error(f"Unknown parameter: {name}")
                return False
            param = self._params[name]
            if not param.validate_change(new_value):
                logger.warning(
                    f"변경 거부: {name} = {param.value} → {new_value} "
                    f"(범위: {param.min_val}~{param.max_val}, ±{param.max_delta})"
                )
                return False
            old = param.value
            param.value = new_value
            logger.info(f"파라미터: {name} = {old} → {new_value}")
            return True

    def get_all(self) -> Dict[str, float]:
        """모든 파라미터를 {name: value} 로 스냅샷 (신호 엔진이 사이클당 1회 사용)."""
        with self._lock:
            return {name: p.value for name, p in self._params.items()}

    def get_specs(self) -> Dict[str, Dict[str, Any]]:
        """Optuna 가 탐색 범위를 알아야 할 때 사용."""
        with self._lock:
            return {
                name: {
                    "value": p.value, "min": p.min_val, "max": p.max_val,
                    "max_delta": p.max_delta, "description": p.description
                }
                for name, p in self._params.items()
            }


PARAM_REGISTRY = TunableParameters()
