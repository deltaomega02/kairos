#!/usr/bin/env python3
"""
v6 스트레스 시나리오 테스트
=============================
현실에서 발생 가능한 악몽 시나리오로 시스템을 시험.

시나리오:
1. 저변동 횡보 (2023 스타일) — 2년간 ATR 0.3%
2. 플래시 크래시 — 정상 → 1H 내 -15% → 48h 회복
3. 장기 하락장 — 6개월간 -40%
4. 휩쏘 (whipsaw) — 매 4-8시간 반전
5. 변동성 폭발 — 3x ATR 급등
6. 조합: 크래시 + 저변동 회복 (2020-03 스타일)

각 시나리오는 4코인 모두에 동시 적용. 풀 패키지 설정.
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

FULL_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30,
}
SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# 기준 가격 (현재가 근사)
BASE_PRICES = {
    "BTCUSDT": 74000.0,
    "ETHUSDT": 2350.0,
    "SOLUSDT": 86.0,
    "XRPUSDT": 1.37,
}


def generate_1h_timestamps(start_date: str, num_hours: int) -> np.ndarray:
    """1H 타임스탬프 시퀀스 생성"""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    return np.array([start_ts + i * 3600000 for i in range(num_hours)])


def make_ohlcv(closes: np.ndarray, timestamps: np.ndarray,
               intrabar_range: float = 0.003,
               volume: float = 1000.0,
               rng: np.random.Generator = None) -> pd.DataFrame:
    """종가 시퀀스로부터 OHLC 생성 (intrabar noise 추가)"""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(closes)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)

    opens[0] = closes[0]
    for i in range(n):
        # open = prev close
        if i > 0:
            opens[i] = closes[i - 1]

    for i in range(n):
        # intrabar high/low: close ± random small
        spread_h = abs(rng.normal(0, intrabar_range)) * closes[i]
        spread_l = abs(rng.normal(0, intrabar_range)) * closes[i]
        highs[i] = max(opens[i], closes[i]) + spread_h
        lows[i] = min(opens[i], closes[i]) - spread_l
        lows[i] = max(lows[i], closes[i] * 0.5)  # 하한
    vols = np.full(n, volume) + rng.normal(0, volume * 0.2, n)
    vols = np.abs(vols)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })


def reconstruct_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    ts_col = df_1h["timestamp"].values
    four_h_ms = 14400000
    result = []
    i = 0
    n = len(df_1h)
    while i < n:
        cur_4h = (int(ts_col[i]) // four_h_ms) * four_h_ms
        idxs = []
        while i < n and (int(ts_col[i]) // four_h_ms) * four_h_ms == cur_4h:
            idxs.append(i)
            i += 1
        if not idxs:
            break
        g = df_1h.iloc[idxs]
        result.append({
            "timestamp": cur_4h,
            "open": float(g["open"].iloc[0]),
            "high": float(g["high"].max()),
            "low": float(g["low"].min()),
            "close": float(g["close"].iloc[-1]),
            "volume": float(g["volume"].sum()),
        })
    return pd.DataFrame(result)


def make_fake_funding(timestamps: np.ndarray) -> pd.DataFrame:
    """8시간마다 펀딩 레이트 (전부 0)"""
    funding_interval_ms = 8 * 3600 * 1000
    first_ts = int(timestamps[0])
    last_ts = int(timestamps[-1])
    fund_ts = []
    t = (first_ts // funding_interval_ms) * funding_interval_ms
    while t <= last_ts:
        fund_ts.append(t)
        t += funding_interval_ms
    return pd.DataFrame({
        "timestamp": fund_ts,
        "funding_rate": [0.0] * len(fund_ts),
    })


# ================================================================
# 시나리오 생성기
# ================================================================

def scenario_low_vol_sideways(base_price: float, num_hours: int,
                               daily_range_pct: float = 0.3,
                               rng: np.random.Generator = None) -> np.ndarray:
    """저변동 횡보 — 가격이 좁은 범위에서 노이즈"""
    if rng is None:
        rng = np.random.default_rng(42)
    hourly_std = daily_range_pct / 100 / np.sqrt(24)
    rets = rng.normal(0, hourly_std, num_hours)
    # 약한 평균 회귀
    levels = np.cumsum(rets)
    for i in range(1, num_hours):
        levels[i] -= levels[i] * 0.001  # mean reversion
    prices = base_price * np.exp(levels)
    return prices


def scenario_flash_crash(base_price: float, num_hours: int,
                          crash_hour: int, crash_pct: float = -15,
                          recovery_hours: int = 48,
                          recovery_pct: float = 12,
                          rng: np.random.Generator = None) -> np.ndarray:
    """플래시 크래시 — 1시간 급락 후 회복"""
    if rng is None:
        rng = np.random.default_rng(42)
    prices = np.ones(num_hours) * base_price
    normal_vol = 0.5 / 100 / np.sqrt(24)
    noise = rng.normal(0, normal_vol, num_hours)

    for i in range(1, num_hours):
        if i == crash_hour:
            prices[i] = prices[i-1] * (1 + crash_pct / 100)
        elif crash_hour < i <= crash_hour + recovery_hours:
            # 점진적 회복
            progress = (i - crash_hour) / recovery_hours
            target = base_price * (1 + crash_pct / 100) * (1 + (recovery_pct / 100) * progress)
            prices[i] = target * (1 + noise[i])
        else:
            prices[i] = prices[i-1] * (1 + noise[i])
    return prices


def scenario_sustained_downtrend(base_price: float, num_hours: int,
                                  total_decline_pct: float = -40,
                                  rng: np.random.Generator = None) -> np.ndarray:
    """지속적 하락장"""
    if rng is None:
        rng = np.random.default_rng(42)
    hourly_drift = (total_decline_pct / 100) / num_hours
    hourly_vol = 0.8 / 100 / np.sqrt(24)
    rets = rng.normal(hourly_drift, hourly_vol, num_hours)
    prices = base_price * np.exp(np.cumsum(rets))
    return prices


def scenario_whipsaw(base_price: float, num_hours: int,
                     flip_min: int = 4, flip_max: int = 8,
                     swing_pct: float = 2.5,
                     rng: np.random.Generator = None) -> np.ndarray:
    """휩쏘 — 짧은 주기 방향 반전"""
    if rng is None:
        rng = np.random.default_rng(42)
    prices = [base_price]
    i = 0
    direction = 1
    while i < num_hours - 1:
        flip_len = rng.integers(flip_min, flip_max + 1)
        swing = rng.uniform(swing_pct * 0.5, swing_pct * 1.5) / 100
        for j in range(min(flip_len, num_hours - 1 - i)):
            step = direction * swing / flip_len
            noise = rng.normal(0, 0.002)
            new_price = prices[-1] * (1 + step + noise)
            prices.append(new_price)
        direction *= -1
        i += flip_len
    return np.array(prices[:num_hours])


def scenario_vol_explosion(base_price: float, num_hours: int,
                            vol_multiplier: float = 3.0,
                            rng: np.random.Generator = None) -> np.ndarray:
    """변동성 폭발 — 정상의 3배 ATR"""
    if rng is None:
        rng = np.random.default_rng(42)
    normal_std = 0.8 / 100 / np.sqrt(24)
    explosive_std = normal_std * vol_multiplier
    rets = rng.normal(0, explosive_std, num_hours)
    prices = base_price * np.exp(np.cumsum(rets))
    return prices


def scenario_crash_then_lowvol(base_price: float, num_hours: int,
                                rng: np.random.Generator = None) -> np.ndarray:
    """크래시 + 저변동 회복 (2020-03 스타일)"""
    if rng is None:
        rng = np.random.default_rng(42)
    # 1주일 정상
    phase1_hours = 24 * 7
    # 2일 크래시 -30%
    phase2_hours = 24 * 2
    # 나머지 저변동 횡보
    phase3_hours = num_hours - phase1_hours - phase2_hours

    prices = []
    # phase 1: 정상
    for _ in range(phase1_hours):
        rets = rng.normal(0, 0.003)
        prev = prices[-1] if prices else base_price
        prices.append(prev * (1 + rets))
    # phase 2: 크래시
    crash_per_hour = (-30 / 100) / phase2_hours
    for _ in range(phase2_hours):
        rets = rng.normal(crash_per_hour, 0.015)
        prices.append(prices[-1] * (1 + rets))
    # phase 3: 저변동
    for _ in range(phase3_hours):
        rets = rng.normal(0, 0.002)
        prices.append(prices[-1] * (1 + rets))
    return np.array(prices[:num_hours])


# ================================================================
# 시나리오 실행
# ================================================================

def build_data_from_scenario(price_generator, num_hours: int, seed: int,
                              label: str) -> dict:
    """시나리오로부터 4코인 가짜 데이터 생성"""
    rng = np.random.default_rng(seed)
    timestamps = generate_1h_timestamps("2022-01-01", num_hours)

    data = {}
    for sym in SYMBOLS:
        # 각 코인마다 조금 다른 seed로
        sym_rng = np.random.default_rng(seed + hash(sym) % 10000)
        prices = price_generator(BASE_PRICES[sym], num_hours, rng=sym_rng)
        df_1h = make_ohlcv(prices, timestamps, rng=sym_rng)
        data[f"{sym}_60"] = df_1h
        data[f"{sym}_240"] = reconstruct_4h(df_1h)
        data[f"{sym}_funding"] = make_fake_funding(timestamps)

    return data


def run_backtest(data: dict) -> dict:
    try:
        r = run_shared_backtest(
            data, FULL_PARAMS, SEED,
            use_funding=True,
            trailing_activation=1.2, trailing_distance=0.1,
            block_sol_long=True,
            skip_years=(),  # 스트레스는 모든 기간 거래
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=SLIP,
            max_simultaneous=3,
            max_leverage=7,
            enabled_symbols=SYMBOLS,
        )
        return {
            "net_profit": r["net_profit"],
            "max_dd": r["max_dd"],
            "final_balance": r["final_balance"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "ruined": r["ruined"],
        }
    except Exception as e:
        return {"error": str(e), "ruined": True,
                "net_profit": 0, "max_dd": 0, "final_balance": 0,
                "total_trades": 0, "win_rate": 0}


def print_result(label: str, result: dict):
    if result.get("ruined") or "error" in result:
        err = result.get("error", "RUINED")
        print(f"  {label:<45} RUINED / {err[:40]}")
    else:
        print(f"  {label:<45} 거래 {result['total_trades']:>5} | 승률 {result['win_rate']:>5.1f}% "
              f"| 순이익 ${result['net_profit']:>+12,.0f} | DD {result['max_dd']:>5.1f}%")


def main():
    t0 = time.time()
    print("="*120)
    print("HERMES v6 — 스트레스 시나리오 테스트")
    print("각 시나리오는 4코인 모두에 동시 적용. 풀 패키지 설정. 2년 기간 (~17,520 1H 캔들)")
    print("="*120)

    num_hours = 24 * 365 * 2  # 2년
    print(f"\n[시나리오 기간: {num_hours}시간 (2년)]")

    results = {}

    # Scenario 1: 저변동 횡보
    print("\n[1] 저변동 횡보 (2023 스타일) — ATR 0.3%/day 유지 2년")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_low_vol_sideways(bp, nh, 0.3, rng),
        num_hours, seed=1, label="low_vol"
    ))
    results["low_vol_sideways"] = r
    print_result("저변동 횡보", r)

    # Scenario 2: 플래시 크래시
    print("\n[2] 플래시 크래시 — 1H -15% + 48h 회복 +12%")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_flash_crash(bp, nh, nh // 2, -15, 48, 12, rng),
        num_hours, seed=2, label="flash_crash"
    ))
    results["flash_crash"] = r
    print_result("플래시 크래시", r)

    # Scenario 3: 장기 하락장
    print("\n[3] 장기 하락장 — 2년간 -60% smooth decline")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_sustained_downtrend(bp, nh, -60, rng),
        num_hours, seed=3, label="downtrend"
    ))
    results["sustained_downtrend"] = r
    print_result("장기 하락장 -60%", r)

    # Scenario 3b: 상승장
    print("\n[3b] 장기 상승장 — 2년간 +100%")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_sustained_downtrend(bp, nh, 100, rng),
        num_hours, seed=31, label="uptrend"
    ))
    results["sustained_uptrend"] = r
    print_result("장기 상승장 +100%", r)

    # Scenario 4: 휩쏘
    print("\n[4] 휩쏘 — 4~8시간 주기로 방향 반전, ±2.5% 스윙")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_whipsaw(bp, nh, 4, 8, 2.5, rng),
        num_hours, seed=4, label="whipsaw"
    ))
    results["whipsaw"] = r
    print_result("휩쏘", r)

    # Scenario 4b: 강한 휩쏘
    print("\n[4b] 강한 휩쏘 — 2~4시간 주기 ±3.5% 스윙")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_whipsaw(bp, nh, 2, 4, 3.5, rng),
        num_hours, seed=41, label="whipsaw_strong"
    ))
    results["whipsaw_strong"] = r
    print_result("강한 휩쏘", r)

    # Scenario 5: 변동성 폭발
    print("\n[5] 변동성 폭발 — 3x ATR 상시")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_vol_explosion(bp, nh, 3.0, rng),
        num_hours, seed=5, label="vol_explosion"
    ))
    results["vol_explosion"] = r
    print_result("변동성 폭발 3x", r)

    # Scenario 5b: 극한 변동성
    print("\n[5b] 극한 변동성 — 5x ATR")
    r = run_backtest(build_data_from_scenario(
        lambda bp, nh, rng: scenario_vol_explosion(bp, nh, 5.0, rng),
        num_hours, seed=51, label="vol_extreme"
    ))
    results["vol_extreme"] = r
    print_result("극한 변동성 5x", r)

    # Scenario 6: 크래시 + 저변동
    print("\n[6] 크래시 + 저변동 회복 (2020-03 스타일)")
    r = run_backtest(build_data_from_scenario(
        scenario_crash_then_lowvol,
        num_hours, seed=6, label="crash_lowvol"
    ))
    results["crash_then_lowvol"] = r
    print_result("크래시 + 저변동 회복", r)

    # ==== 종합 ====
    print(f"\n{'='*120}")
    print("종합 결과")
    print(f"{'='*120}")
    print(f"{'시나리오':<35} {'결과':>12} {'순이익':>14} {'DD':>7} {'승률':>7}")
    print("-" * 90)

    scenarios = [
        ("저변동 횡보 (2023 스타일)", "low_vol_sideways"),
        ("플래시 크래시 -15%", "flash_crash"),
        ("장기 하락장 -60% (2년)", "sustained_downtrend"),
        ("장기 상승장 +100% (2년)", "sustained_uptrend"),
        ("휩쏘 ±2.5%/6h", "whipsaw"),
        ("강한 휩쏘 ±3.5%/3h", "whipsaw_strong"),
        ("변동성 폭발 3x ATR", "vol_explosion"),
        ("극한 변동성 5x ATR", "vol_extreme"),
        ("크래시 + 저변동 회복", "crash_then_lowvol"),
    ]
    for label, key in scenarios:
        r = results.get(key, {})
        if r.get("ruined") or "error" in r:
            status = "❌ 파산/오류"
            pnl_str = "-"
            dd_str = "-"
            wr_str = "-"
        elif r.get("net_profit", 0) > 0:
            status = "✅ 생존 (수익)"
            pnl_str = f"${r['net_profit']:+,.0f}"
            dd_str = f"{r['max_dd']:.1f}%"
            wr_str = f"{r['win_rate']:.1f}%"
        else:
            status = "⚠ 생존 (손실)"
            pnl_str = f"${r['net_profit']:+,.0f}"
            dd_str = f"{r['max_dd']:.1f}%"
            wr_str = f"{r['win_rate']:.1f}%"
        print(f"{label:<35} {status:>12} {pnl_str:>14} {dd_str:>7} {wr_str:>7}")

    # 통계
    valid_results = [r for r in results.values() if not r.get("ruined")]
    profitable = sum(1 for r in valid_results if r.get("net_profit", 0) > 0)
    ruined_count = sum(1 for r in results.values() if r.get("ruined"))

    print(f"\n[요약]")
    print(f"  전체 시나리오: {len(results)}")
    print(f"  생존: {len(valid_results)}/{len(results)} ({len(valid_results)/len(results)*100:.0f}%)")
    print(f"  수익: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
    print(f"  파산: {ruined_count}/{len(results)}")

    if valid_results:
        pnls = [r["net_profit"] for r in valid_results]
        dds = [r["max_dd"] for r in valid_results]
        print(f"  평균 순이익: ${sum(pnls)/len(pnls):+,.0f}")
        print(f"  최악 손실:   ${min(pnls):+,.0f}")
        print(f"  최고 수익:   ${max(pnls):+,.0f}")
        print(f"  평균 DD:     {sum(dds)/len(dds):.1f}%")
        print(f"  최악 DD:     {max(dds):.1f}%")

    elapsed = time.time() - t0
    print(f"\n총 소요: {elapsed:.0f}s")

    out_path = os.path.join(RESULTS_DIR, "v6_stress_scenarios.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_hours": num_hours,
            "seed": SEED,
            "slippage": SLIP,
            "config": "full package (4코인, 3pos, lev7, TP6, ADX30, trailing 1.2/0.1)",
            "scenarios": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
