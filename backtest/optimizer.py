# backtest/optimizer.py — Walk-Forward Optuna 최적화

import json
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
from config import get_logger, PARAM_REGISTRY
from database.db_manager import db_manager

logger = get_logger("optimizer")


class WalkForwardOptimizer:
    """Walk-Forward 최적화 — 주 1회, 50거래 이상, Train/Val 분리

    v8 (2026-04-19) 부로 비활성화. MIN_TRADES=99999로 should_run()이 영구 False.
    v8 (EMA 3/15) 실전 검증 중. 검증 완료 후 MIN_TRADES를 50으로 되돌려 재활성화.
    """

    MIN_TRADES = 99999   # v8 검증 기간 동안 Optuna 비활성 (원래 50)
    TRAIN_SIZE = 200
    VALIDATION_SIZE = 100
    OPTUNA_TRIALS = 80
    COOLDOWN_DAYS = 7
    MIN_IMPROVEMENT_PCT = 2.0

    TUNABLE_KEYS = [
        "sl_atr_mult",
        "tp_rr_ratio",
        "entry_score_threshold",
        "pullback_ema_dist_pct",
        "adx_enter_trending",
        "rsi_oversold",
    ]

    def __init__(self):
        """옵티마이저 초기화"""
        self._last_run: Optional[datetime] = None
        self._optuna_available = False
        self._check_optuna()

    def _check_optuna(self):
        """Optuna 설치 여부 확인"""
        try:
            import optuna  # noqa: F401
            self._optuna_available = True
        except ImportError:
            logger.warning("Optuna 미설치 — 최적화 비활성. pip install optuna")
            self._optuna_available = False

    def should_run(self) -> bool:
        """실행 조건 확인"""
        if not self._optuna_available:
            return False

        stats = db_manager.get_trade_stats(days=30)
        if stats["total_trades"] < self.MIN_TRADES:
            return False

        if self._last_run:
            days_since = (datetime.now() - self._last_run).days
            if days_since < self.COOLDOWN_DAYS:
                return False

        return True

    def run(self) -> Optional[Dict[str, Any]]:
        """최적화 실행 → 변경 dict 또는 None"""
        if not self.should_run():
            return None

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("=" * 50)
        logger.info("Walk-Forward 최적화 시작")
        self._last_run = datetime.now()

        try:
            return self._run_optimization()
        except Exception as e:
            logger.error(f"최적화 오류: {e}", exc_info=True)
            return None

    def _run_optimization(self) -> Optional[Dict[str, Any]]:
        """최적화 핵심 로직"""
        import optuna

        # 거래 데이터 로드
        all_trades = db_manager.get_recent_trades(
            limit=self.TRAIN_SIZE + self.VALIDATION_SIZE
        )
        if len(all_trades) < self.MIN_TRADES:
            logger.info(f"거래 부족: {len(all_trades)}건 < {self.MIN_TRADES}건")
            return None

        # 시간순 정렬
        all_trades.reverse()

        # Train/Val 분리
        split = min(self.TRAIN_SIZE, len(all_trades) - 20)
        train_trades = all_trades[:split]
        val_trades = all_trades[split:]

        if len(val_trades) < 20:
            logger.info(f"Validation 데이터 부족: {len(val_trades)}건")
            return None

        # 베이스라인
        baseline_train = self._score_trades(train_trades)
        baseline_val = self._score_trades(val_trades)

        logger.info(
            f"베이스라인 — Train: {baseline_train:.2f} ({len(train_trades)}건) | "
            f"Val: {baseline_val:.2f} ({len(val_trades)}건)"
        )

        # Optuna 탐색
        all_specs = PARAM_REGISTRY.get_specs()
        specs = {k: all_specs[k] for k in self.TUNABLE_KEYS if k in all_specs}

        # 비탐색 파라미터 고정
        fixed_params = {k: v["value"] for k, v in all_specs.items() if k not in specs}

        def objective(trial):
            suggested = dict(fixed_params)  # 고정값 복사
            for name, spec in specs.items():
                current = spec["value"]
                lo = max(spec["min"], current - spec["max_delta"])
                hi = min(spec["max"], current + spec["max_delta"])

                if isinstance(current, float) and not float(current).is_integer():
                    suggested[name] = trial.suggest_float(name, lo, hi)
                else:
                    suggested[name] = trial.suggest_float(name, lo, hi, step=1.0)

            return self._simulate_with_params(train_trades, suggested)

        logger.info(f"Optuna 탐색: {list(specs.keys())} ({len(specs)}차원, {self.OPTUNA_TRIALS} trials)")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.OPTUNA_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_train = study.best_value

        # Train 개선 확인
        if baseline_train != 0:
            train_improve = (best_train - baseline_train) / abs(baseline_train) * 100
        else:
            train_improve = 100 if best_train > 0 else 0

        logger.info(f"Optuna 최적: Train {best_train:.2f} (개선 {train_improve:+.1f}%)")

        if train_improve < self.MIN_IMPROVEMENT_PCT:
            logger.info(f"Train 개선 부족 → 폐기")
            self._log_run(best_params, False, f"Train개선부족 {train_improve:.1f}%")
            return None

        # Validation 검증
        val_score = self._simulate_with_params(val_trades, best_params)
        if baseline_val != 0:
            val_improve = (val_score - baseline_val) / abs(baseline_val) * 100
        else:
            val_improve = 100 if val_score > 0 else 0

        logger.info(f"Validation: {val_score:.2f} (개선 {val_improve:+.1f}%)")

        if val_improve < 0:
            logger.warning(f"Validation 악화 → 과적합 → 폐기")
            self._log_run(best_params, False, f"Validation악화 {val_improve:.1f}%")
            return None

        # 적용
        changes = {}
        stats_before = db_manager.get_trade_stats(days=30)

        for name, new_val in best_params.items():
            old_val = PARAM_REGISTRY.get(name)
            if abs(new_val - old_val) < 0.001:
                continue  # 변화 없음

            if PARAM_REGISTRY.update(name, new_val):
                changes[name] = {"old": round(old_val, 4), "new": round(new_val, 4)}
                db_manager.log_param_change(
                    param_name=name,
                    old_value=old_val,
                    new_value=new_val,
                    trigger_type="OPTUNA",
                    trades_since=stats_before["total_trades"],
                    win_rate_before=stats_before["win_rate"],
                    notes=f"T+{train_improve:.1f}% V+{val_improve:.1f}%"
                )

        if changes:
            PARAM_REGISTRY.save_to_file()
            self._log_run(best_params, True,
                          f"T+{train_improve:.1f}% V+{val_improve:.1f}% 변경{len(changes)}개")
            logger.info(f"파라미터 {len(changes)}개 적용: {changes}")
        else:
            logger.info("최적값이 현재값과 동일 → 변경 없음")

        logger.info("=" * 50)
        return changes if changes else None

    def _score_trades(self, trades: List[Dict[str, Any]]) -> float:
        """성과 스코어 = (승률 - 0.5) * 평균수익률 * sqrt(거래수)"""
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if (t.get("realized_pnl") or 0) > 0)
        total = len(trades)
        win_rate = wins / total

        pnl_pcts = [t.get("realized_pnl_percentage") or 0 for t in trades]
        avg_pnl = sum(pnl_pcts) / len(pnl_pcts)

        return (win_rate - 0.5) * avg_pnl * math.sqrt(total)

    def _simulate_with_params(
        self,
        trades: List[Dict[str, Any]],
        params: Dict[str, float]
    ) -> float:
        """파라미터로 거래 필터링 + 수익률 근사 조정"""
        if not trades:
            return 0.0

        threshold = params.get("entry_score_threshold", 60)
        sl_mult = params.get("sl_atr_mult", 1.5)
        tp_rr = params.get("tp_rr_ratio", 2.5)

        filtered = []
        for t in trades:
            score = t.get("signal_score") or 0
            pnl_pct = t.get("realized_pnl_percentage") or 0

            # 스코어 필터
            if score < threshold:
                continue

            rr_factor = tp_rr / 2.5

            adjusted_pnl = pnl_pct
            if pnl_pct > 0:
                adjusted_pnl *= rr_factor * 0.9
            else:
                adjusted_pnl *= sl_mult / 1.5

            filtered.append({
                "realized_pnl": t.get("realized_pnl", 0),
                "realized_pnl_percentage": adjusted_pnl,
            })

        return self._score_trades(filtered)

    def _log_run(self, params: Dict[str, float], applied: bool, notes: str = ""):
        """최적화 실행 결과 DB 기록"""
        try:
            conn = db_manager._get_connection()
            conn.execute("""
                INSERT INTO optimizer_runs (timestamp, trades_evaluated, best_params, applied, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                db_manager.get_trade_stats(days=30)["total_trades"],
                json.dumps({k: round(v, 4) for k, v in params.items()}),
                applied,
                notes
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"최적화 로깅 실패: {e}")
