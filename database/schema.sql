-- HERMES Database Schema
-- 스캘핑 포지션 + 파라미터 이력 + 일일 통계

CREATE TABLE IF NOT EXISTS positions (
    position_uuid TEXT PRIMARY KEY,
    symbol TEXT NOT NULL DEFAULT 'BTCUSDT',  -- 멀티코인 지원
    direction TEXT NOT NULL,           -- LONG | SHORT
    strategy TEXT NOT NULL,            -- BREAKOUT | REVERSION
    leverage INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_quantity REAL NOT NULL,
    entry_timestamp TEXT NOT NULL,
    stop_loss_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    liquidation_price REAL,
    signal_score INTEGER,
    signal_reason TEXT,
    regime TEXT,                        -- 진입 시 레짐
    status TEXT DEFAULT 'ACTIVE',      -- ACTIVE | CLOSED | LIQUIDATED
    exit_price REAL,
    exit_timestamp TEXT,
    exit_reason TEXT,
    realized_pnl REAL,
    realized_pnl_percentage REAL,
    entry_fee REAL DEFAULT 0,
    exit_fee REAL DEFAULT 0,
    total_fee REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parameter_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    param_name TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    trigger_type TEXT,                 -- OPTUNA | MANUAL | ROLLBACK
    trades_since_last_change INTEGER,
    win_rate_before REAL,
    win_rate_after REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS optimizer_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    trades_evaluated INTEGER,
    best_params TEXT,                  -- JSON
    validation_result TEXT,            -- JSON {win_rate, pnl, sharpe}
    applied BOOLEAN DEFAULT FALSE,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    total_fees REAL DEFAULT 0,
    max_drawdown_pct REAL DEFAULT 0,
    breakout_trades INTEGER DEFAULT 0,
    reversion_trades INTEGER DEFAULT 0,
    breakout_pnl REAL DEFAULT 0,
    reversion_pnl REAL DEFAULT 0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_created ON positions(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_param_history_name ON parameter_history(param_name);
CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);
