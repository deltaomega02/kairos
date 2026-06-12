"""기술적 지표 계산 모듈.

EMA / SMA / RSI / MACD / BB / ATR / ADX 기본 지표와,
타임프레임별 지표 묶음을 돌려주는 high-level 함수들을 제공한다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from config import get_logger

logger = get_logger("technical_analysis")


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing 근사)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD 라인 + 시그널 라인 + 히스토그램."""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands: 중앙(SMA) / 상단 / 하단 / 밴드 폭 (%)."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {
        "middle": sma,
        "upper": upper,
        "lower": lower,
        "width": (upper - lower) / sma * 100
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (단순 rolling mean 방식)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """ADX / +DI / -DI 세트. 추세 세기와 방향을 동시에 판단하는 데 쓴다."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)

    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def calculate_atr_percentile(atr_series: pd.Series, lookback: int = 100) -> float:
    """최근 N 봉 대비 현재 ATR 이 어느 분위에 있는지 (HIGH_VOL 판정용)."""
    if len(atr_series) < lookback:
        lookback = len(atr_series)

    recent = atr_series.iloc[-lookback:]
    current = atr_series.iloc[-1]

    percentile = (recent < current).sum() / len(recent) * 100
    return float(percentile)


def get_regime_indicators_4h(candles_4h: list) -> Optional[Dict[str, Any]]:
    """4H 봉에서 레짐 판독용 지표를 한 번에 계산해 dict 로 반환."""
    if len(candles_4h) < 50:
        logger.warning(f"4H 캔들 부족: {len(candles_4h)}개")
        return None

    df = pd.DataFrame(candles_4h)

    adx_data = calculate_adx(df["high"], df["low"], df["close"])
    ema_9 = calculate_ema(df["close"], 9)
    ema_21 = calculate_ema(df["close"], 21)
    macd_data = calculate_macd(df["close"], 12, 26, 9)
    atr = calculate_atr(df["high"], df["low"], df["close"])
    atr_pct = (atr / df["close"]) * 100
    atr_percentile = calculate_atr_percentile(atr)

    current = df.iloc[-1]

    return {
        "adx": round(float(adx_data["adx"].iloc[-1]), 2),
        "plus_di": round(float(adx_data["plus_di"].iloc[-1]), 2),
        "minus_di": round(float(adx_data["minus_di"].iloc[-1]), 2),
        "ema_9": round(float(ema_9.iloc[-1]), 2),
        "ema_21": round(float(ema_21.iloc[-1]), 2),
        "macd_histogram": round(float(macd_data["histogram"].iloc[-1]), 6),
        "atr": round(float(atr.iloc[-1]), 2),
        "atr_pct": round(float(atr_pct.iloc[-1]), 4),
        "atr_percentile": round(atr_percentile, 1),
        "close": float(current["close"]),
        "timestamp": int(current["timestamp"])
    }


def get_entry_indicators_1h(candles_1h: list, params: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """1H 봉에서 진입 신호용 지표 세트 계산. EMA 기간은 파라미터로 주입."""
    if len(candles_1h) < 50:
        logger.warning(f"1H 캔들 부족: {len(candles_1h)}개")
        return None

    df = pd.DataFrame(candles_1h)

    ema_fast_period = int(params.get("ema_fast", 9))
    ema_slow_period = int(params.get("ema_slow", 21))
    rsi_period = int(params.get("rsi_period", 14))

    ema_fast = calculate_ema(df["close"], ema_fast_period)
    ema_slow = calculate_ema(df["close"], ema_slow_period)
    rsi = calculate_rsi(df["close"], rsi_period)

    atr = calculate_atr(df["high"], df["low"], df["close"])
    atr_pct = (atr / df["close"]) * 100

    vol_sma = calculate_sma(df["volume"], 20)
    volume_ratio = df["volume"] / vol_sma

    current = df.iloc[-1]

    # Bollinger Bands
    bb = calculate_bollinger_bands(df["close"], 20, 2.0)

    # 최근 고/저
    recent_high = float(df["high"].iloc[-20:].max())
    recent_low = float(df["low"].iloc[-20:].min())
    price_range = recent_high - recent_low
    price_position = (float(current["close"]) - recent_low) / price_range * 100 if price_range > 0 else 50

    return {
        "ema_fast": round(float(ema_fast.iloc[-1]), 2),
        "ema_slow": round(float(ema_slow.iloc[-1]), 2),
        "rsi": round(float(rsi.iloc[-1]), 2),
        "atr": round(float(atr.iloc[-1]), 2),
        "atr_pct": round(float(atr_pct.iloc[-1]), 4),
        "volume_ratio": round(float(volume_ratio.iloc[-1]), 2),
        "close": float(current["close"]),
        "open": float(current["open"]),
        "high": float(current["high"]),
        "low": float(current["low"]),
        "timestamp": int(current["timestamp"]),
        # BB / 레인지
        "bb_upper": round(float(bb["upper"].iloc[-1]), 2),
        "bb_lower": round(float(bb["lower"].iloc[-1]), 2),
        "bb_middle": round(float(bb["middle"].iloc[-1]), 2),
        "bb_width": round(float(bb["width"].iloc[-1]), 4),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "price_position": round(price_position, 1),
    }


def get_daily_trend(candles_1d: list, ema_period: int = 10) -> Optional[Dict[str, Any]]:
    """일봉 EMA 상태 요약. 진입 필터(direction 또는 price-above-EMA)에서 사용.

    Returns:
        dict with keys:
            ema         — 현재 1D EMA 값
            prev_ema    — 직전 1D EMA 값
            close       — 현재 1D 종가
            direction   — "UP" / "DOWN" / "FLAT" (기울기 ±0.05% 기준)
            slope_pct   — EMA 기울기, 직전 대비 % 변화
    """
    if len(candles_1d) < ema_period + 2:
        logger.warning(f"1D 캔들 부족: {len(candles_1d)}개 (최소 {ema_period + 2})")
        return None

    df = pd.DataFrame(candles_1d)
    ema = calculate_ema(df["close"], ema_period)

    cur_ema = float(ema.iloc[-1])
    prev_ema = float(ema.iloc[-2])
    cur_close = float(df["close"].iloc[-1])

    if prev_ema == 0:
        return None

    slope_pct = (cur_ema - prev_ema) / prev_ema * 100

    # 기울기 ±0.05% 이상일 때만 방향으로 확정. 그 안쪽은 FLAT 으로 간주해 진입 보류.
    if slope_pct > 0.05:
        direction = "UP"
    elif slope_pct < -0.05:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return {
        "ema": round(cur_ema, 2),
        "prev_ema": round(prev_ema, 2),
        "close": round(cur_close, 2),
        "direction": direction,
        "slope_pct": round(slope_pct, 4),
        "timestamp": int(df["timestamp"].iloc[-1]),
    }


def calculate_orderbook_imbalance(orderbook_data: Dict[str, Any]) -> Dict[str, float]:
    """호가창 bid/ask 총량 비율 + 스프레드(%) 계산. 시그널 엔진이 방향 확인에 사용."""
    bids = orderbook_data.get("bids", [])
    asks = orderbook_data.get("asks", [])

    if not bids or not asks:
        return {"bid_ratio": 0.5, "total_bid_qty": 0, "total_ask_qty": 0,
                "spread_pct": 0, "best_bid": 0, "best_ask": 0}

    total_bid = sum(float(b[1]) for b in bids)
    total_ask = sum(float(a[1]) for a in asks)

    total = total_bid + total_ask
    bid_ratio = total_bid / total if total > 0 else 0.5

    # 스프레드 계산
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

    return {
        "bid_ratio": round(bid_ratio, 4),
        "total_bid_qty": round(total_bid, 4),
        "total_ask_qty": round(total_ask, 4),
        "spread_pct": round(spread_pct, 6),
        "best_bid": best_bid,
        "best_ask": best_ask,
    }
