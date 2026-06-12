# database/db_manager.py — SQLite CRUD

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from config import get_logger

logger = get_logger("db_manager")

DB_PATH = Path(__file__).parent / "kairos.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DBManager:
    def __init__(self, db_path: Path = DB_PATH):
        """DB 매니저 초기화"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """스키마 초기화"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            with open(SCHEMA_PATH, "r") as f:
                conn.executescript(f.read())
            conn.commit()
            logger.info(f"DB 초기화: {self.db_path}")
        finally:
            if conn:
                conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결 반환"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ========== Position CRUD ==========

    def create_position(
        self,
        direction: str,
        strategy: str,
        leverage: int,
        entry_price: float,
        entry_quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
        liquidation_price: float,
        signal_score: int = 0,
        signal_reason: str = "",
        regime: str = "",
        entry_fee: float = 0,
        symbol: str = "BTCUSDT",
        # 하위호환 파라미터
        confidence_score: int = 0,
        ai_reason: str = "",
        strategy_json: str = ""
    ) -> str:
        """포지션 생성 및 DB 기록"""
        position_uuid = str(uuid.uuid4())
        entry_timestamp = datetime.now().isoformat()

        # 레거시 파라미터 매핑
        if not signal_reason and ai_reason:
            signal_reason = ai_reason
        if not signal_score and confidence_score:
            signal_score = confidence_score
        if not strategy and strategy_json:
            strategy = "BREAKOUT"

        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO positions (
                    position_uuid, symbol, direction, strategy, leverage,
                    entry_price, entry_quantity, entry_timestamp,
                    stop_loss_price, take_profit_price, liquidation_price,
                    signal_score, signal_reason, regime, status, entry_fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
            """, (
                position_uuid, symbol, direction, strategy, leverage,
                entry_price, entry_quantity, entry_timestamp,
                stop_loss_price, take_profit_price, liquidation_price,
                signal_score, signal_reason, regime, entry_fee
            ))
            conn.commit()
            logger.info(f"포지션 생성: {position_uuid[:8]}... {symbol} {direction} {strategy} {leverage}x")
            return position_uuid
        finally:
            if conn:
                conn.close()

    def get_active_position(self) -> Optional[Dict[str, Any]]:
        """첫 번째 활성 포지션 조회"""
        conn = None
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM positions WHERE status = 'ACTIVE' LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
        finally:
            if conn:
                conn.close()

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """모든 활성 포지션 조회 (멀티코인)"""
        conn = None
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'ACTIVE'"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if conn:
                conn.close()

    def get_active_position_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """특정 코인의 활성 포지션 조회"""
        conn = None
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM positions WHERE status = 'ACTIVE' AND symbol = ? LIMIT 1",
                (symbol,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            if conn:
                conn.close()

    def get_position_by_uuid(self, position_uuid: str) -> Optional[Dict[str, Any]]:
        """UUID로 포지션 조회 (상태 무관)"""
        conn = None
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM positions WHERE position_uuid = ? LIMIT 1",
                (position_uuid,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            if conn:
                conn.close()

    def close_position(
        self,
        position_uuid: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl: float,
        realized_pnl_percentage: float,
        exit_fee: float = 0
    ) -> bool:
        """포지션 청산 및 DB 업데이트"""
        exit_timestamp = datetime.now().isoformat()
        conn = None
        try:
            conn = self._get_connection()

            row = conn.execute(
                "SELECT entry_fee FROM positions WHERE position_uuid = ?",
                (position_uuid,)
            ).fetchone()
            entry_fee = row["entry_fee"] if row else 0
            total_fee = entry_fee + exit_fee

            conn.execute("""
                UPDATE positions SET
                    status = 'CLOSED',
                    exit_price = ?,
                    exit_timestamp = ?,
                    exit_reason = ?,
                    realized_pnl = ?,
                    realized_pnl_percentage = ?,
                    exit_fee = ?,
                    total_fee = ?
                WHERE position_uuid = ?
            """, (
                exit_price, exit_timestamp, exit_reason,
                realized_pnl, realized_pnl_percentage,
                exit_fee, total_fee, position_uuid
            ))
            conn.commit()
            logger.info(f"포지션 청산: {position_uuid[:8]}... {exit_reason} PnL={realized_pnl:.2f}")
            return True
        finally:
            if conn:
                conn.close()

    def update_position_targets(
        self,
        position_uuid: str,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> bool:
        """포지션 SL/TP 업데이트"""
        updates = []
        params = []
        if stop_loss_price is not None:
            updates.append("stop_loss_price = ?")
            params.append(stop_loss_price)
        if take_profit_price is not None:
            updates.append("take_profit_price = ?")
            params.append(take_profit_price)
        if not updates:
            return False

        params.append(position_uuid)
        conn = None
        try:
            conn = self._get_connection()
            conn.execute(
                f"UPDATE positions SET {', '.join(updates)} WHERE position_uuid = ?",
                params
            )
            conn.commit()
            return True
        finally:
            if conn:
                conn.close()

    # ========== Parameter History ==========

    def log_param_change(
        self,
        param_name: str,
        old_value: float,
        new_value: float,
        trigger_type: str = "OPTUNA",
        trades_since: int = 0,
        win_rate_before: float = 0,
        notes: str = ""
    ):
        """파라미터 변경 이력 기록"""
        conn = None
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO parameter_history (
                    timestamp, param_name, old_value, new_value,
                    trigger_type, trades_since_last_change, win_rate_before, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), param_name, old_value,
                new_value, trigger_type, trades_since, win_rate_before, notes
            ))
            conn.commit()
        finally:
            if conn:
                conn.close()

    # ========== Statistics ==========

    def get_trade_stats(self, days: int = 7) -> Dict[str, Any]:
        """기간별 거래 통계 조회"""
        conn = None
        try:
            conn = self._get_connection()
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(realized_pnl), 0) as total_pnl,
                    COALESCE(AVG(realized_pnl_percentage), 0) as avg_pnl_pct,
                    COALESCE(SUM(total_fee), 0) as total_fees
                FROM positions
                WHERE status = 'CLOSED'
                AND created_at >= datetime('now', ?)
            """, (f'-{days} days',)).fetchone()

            total = row["total_trades"] or 0
            wins = row["wins"] or 0

            return {
                "total_trades": total,
                "wins": wins,
                "losses": row["losses"] or 0,
                "win_rate": (wins / total * 100) if total else 0,
                "total_pnl": row["total_pnl"] or 0,
                "avg_pnl_pct": row["avg_pnl_pct"] or 0,
                "total_fees": row["total_fees"] or 0
            }
        finally:
            if conn:
                conn.close()

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 거래 내역 조회"""
        conn = None
        try:
            conn = self._get_connection()
            rows = conn.execute("""
                SELECT * FROM positions
                WHERE status = 'CLOSED'
                ORDER BY exit_timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            if conn:
                conn.close()

    def get_strategy_stats(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """전략별 통계"""
        conn = None
        try:
            conn = self._get_connection()
            result = {}
            for strategy in ["BREAKOUT", "REVERSION"]:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as trades,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        COALESCE(SUM(realized_pnl), 0) as pnl
                    FROM positions
                    WHERE status = 'CLOSED' AND strategy = ?
                    AND created_at >= datetime('now', ?)
                """, (strategy, f'-{days} days')).fetchone()

                trades = row["trades"] or 0
                wins = row["wins"] or 0
                result[strategy] = {
                    "trades": trades,
                    "wins": wins,
                    "win_rate": (wins / trades * 100) if trades else 0,
                    "pnl": row["pnl"] or 0
                }
            return result
        finally:
            if conn:
                conn.close()


# 싱글톤
db_manager = DBManager()
