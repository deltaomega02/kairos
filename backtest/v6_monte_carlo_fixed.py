#!/usr/bin/env python3
"""
v6 Monte Carlo (log-return bootstrap — 수정판)
==============================================
이전 버전의 문제: OHLC 블록을 직접 이어붙여서 비현실적 가격 점프 발생 → 수익 왜곡
수정: log return과 OHLC 비율을 블록 단위로 샘플링 → 가격 연속성 유지

방법:
1. 각 캔들을 4개 비율로 표현:
   - ret_close = close[i] / close[i-1]  (종가 대비 종가)
   - ratio_open = open[i] / close[i-1]  (갭)
   - ratio_high = high[i] / close[i]    (상단 꼬리)
   - ratio_low  = low[i] / close[i]     (하단 꼬리)
2. 24시간 블록 단위로 ratio 벡터 샘플링
3. 재조합 시 실제 초기 가격에서 누적곱으로 복원 → 가격 연속성 보장
4. 4H는 1H에서 재구성
"""
import os
import sys
import json
import time
from datetime import datetime
from statistics import mean, median, stdev

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v6"
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
N_SIMULATIONS = 300
BLOCK_SIZE_HOURS = 24
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_real_data():
    data = {}
    for sym in SYMBOLS:
        for iv in ["60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
        f = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        if os.path.exists(f):
            data[f"{sym}_funding"] = pd.read_csv(f)
    return data


def compute_ratios(df_1h: pd.DataFrame) -> dict:
    """각 캔들을 OHLC 비율 벡터로 변환"""
    closes = df_1h["close"].values
    opens = df_1h["open"].values
    highs = df_1h["high"].values
    lows = df_1h["low"].values
    vols = df_1h["volume"].values

    # i=0은 prev_close 없으므로 i=1부터
    prev_close = closes[:-1]
    cur_close = closes[1:]
    cur_open = opens[1:]
    cur_high = highs[1:]
    cur_low = lows[1:]
    cur_vol = vols[1:]

    # 0 방지
    prev_close = np.where(prev_close <= 0, 1e-10, prev_close)
    cur_close_safe = np.where(cur_close <= 0, 1e-10, cur_close)

    ret_close = cur_close / prev_close              # 종가 수익률
    ratio_open = cur_open / prev_close              # 갭 비율
    ratio_high = cur_high / cur_close_safe          # 상단 꼬리
    ratio_low = cur_low / cur_close_safe            # 하단 꼬리

    return {
        "ret_close": ret_close,
        "ratio_open": ratio_open,
        "ratio_high": ratio_high,
        "ratio_low": ratio_low,
        "volume": cur_vol,
        "initial_close": float(closes[0]),
        "initial_open": float(opens[0]),
        "initial_high": float(highs[0]),
        "initial_low": float(lows[0]),
        "initial_volume": float(vols[0]),
        "timestamps": df_1h["timestamp"].values,
    }


def block_bootstrap_ratios(ratios: dict, block_size: int, rng: np.random.Generator) -> dict:
    """ratio 벡터를 블록 단위로 샘플링"""
    n = len(ratios["ret_close"])
    if n < block_size:
        return ratios

    num_blocks = (n // block_size) + 1
    max_start = n - block_size
    starts = rng.integers(0, max_start, size=num_blocks)

    # 각 비율을 블록으로 concat
    def sample(arr):
        pieces = [arr[s:s+block_size] for s in starts]
        return np.concatenate(pieces)[:n]

    return {
        "ret_close": sample(ratios["ret_close"]),
        "ratio_open": sample(ratios["ratio_open"]),
        "ratio_high": sample(ratios["ratio_high"]),
        "ratio_low": sample(ratios["ratio_low"]),
        "volume": sample(ratios["volume"]),
        "initial_close": ratios["initial_close"],
        "initial_open": ratios["initial_open"],
        "initial_high": ratios["initial_high"],
        "initial_low": ratios["initial_low"],
        "initial_volume": ratios["initial_volume"],
        "timestamps": ratios["timestamps"],
    }


def reconstruct_1h_from_ratios(sampled: dict) -> pd.DataFrame:
    """샘플된 비율에서 OHLC 재구성 (가격 연속성 유지)"""
    n = len(sampled["ret_close"])
    closes = np.zeros(n + 1)
    opens = np.zeros(n + 1)
    highs = np.zeros(n + 1)
    lows = np.zeros(n + 1)
    vols = np.zeros(n + 1)

    # 초기값
    closes[0] = sampled["initial_close"]
    opens[0] = sampled["initial_open"]
    highs[0] = sampled["initial_high"]
    lows[0] = sampled["initial_low"]
    vols[0] = sampled["initial_volume"]

    ret_close = sampled["ret_close"]
    ratio_open = sampled["ratio_open"]
    ratio_high = sampled["ratio_high"]
    ratio_low = sampled["ratio_low"]
    vols_sampled = sampled["volume"]

    # 누적 재구성
    for i in range(n):
        prev_c = closes[i]
        new_close = prev_c * ret_close[i]
        new_open = prev_c * ratio_open[i]
        new_high = new_close * ratio_high[i]
        new_low = new_close * ratio_low[i]

        # Sanity: open/close 사이에 high/low 포함되도록
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)

        closes[i + 1] = max(new_close, 1e-4)
        opens[i + 1] = max(new_open, 1e-4)
        highs[i + 1] = max(new_high, 1e-4)
        lows[i + 1] = max(new_low, 1e-4)
        vols[i + 1] = max(vols_sampled[i], 0)

    ts = sampled["timestamps"][:n + 1]

    return pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })


def reconstruct_4h_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1H → 4H 재구성"""
    ts_col = df_1h["timestamp"].values
    four_h_ms = 14400000

    result = []
    i = 0
    n = len(df_1h)

    while i < n:
        current_4h_ts = (int(ts_col[i]) // four_h_ms) * four_h_ms
        group_indices = []
        while i < n and (int(ts_col[i]) // four_h_ms) * four_h_ms == current_4h_ts:
            group_indices.append(i)
            i += 1
        if not group_indices:
            break
        group = df_1h.iloc[group_indices]
        result.append({
            "timestamp": current_4h_ts,
            "open": float(group["open"].iloc[0]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group["close"].iloc[-1]),
            "volume": float(group["volume"].sum()),
        })
    return pd.DataFrame(result)


def create_synthetic(real_data: dict, sim_seed: int) -> dict:
    rng = np.random.default_rng(sim_seed)
    synthetic = {}
    for sym in SYMBOLS:
        key_1h = f"{sym}_60"
        if key_1h not in real_data:
            continue
        ratios = compute_ratios(real_data[key_1h])
        sampled = block_bootstrap_ratios(ratios, BLOCK_SIZE_HOURS, rng)
        new_1h = reconstruct_1h_from_ratios(sampled)
        synthetic[key_1h] = new_1h
        synthetic[f"{sym}_240"] = reconstruct_4h_from_1h(new_1h)
        fund_key = f"{sym}_funding"
        if fund_key in real_data:
            synthetic[fund_key] = real_data[fund_key]
    return synthetic


def run_backtest(data: dict) -> dict:
    try:
        r = run_shared_backtest(
            data, FULL_PARAMS, SEED,
            use_funding=True,
            trailing_activation=1.2, trailing_distance=0.1,
            block_sol_long=True,
            skip_years=(2023,),
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


def pct(vals, p):
    if not vals:
        return 0
    idx = int(len(vals) * p / 100)
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def main():
    t0 = time.time()
    print("="*110)
    print(f"HERMES v6 — Monte Carlo (log-return bootstrap) N={N_SIMULATIONS}")
    print(f"가격 연속성 유지 + 실제 분포 기반")
    print("="*110)

    print("\n[실제 데이터 로드]")
    real = load_real_data()
    print(f"  로드 완료: {len(real)} 데이터셋")

    print("\n[기준 — 실제 데이터]")
    baseline = run_backtest(real)
    if baseline.get("ruined"):
        print(f"  ⚠ baseline 파산")
    else:
        print(f"  순이익: ${baseline['net_profit']:+,.0f}")
        print(f"  DD: {baseline['max_dd']}%")
        print(f"  승률: {baseline['win_rate']}%")

    print(f"\n[Monte Carlo {N_SIMULATIONS}회 실행]")
    results = []
    t_sim = time.time()

    for i in range(N_SIMULATIONS):
        syn = create_synthetic(real, sim_seed=i + 1)
        r = run_backtest(syn)
        results.append(r)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_sim
            rate = (i + 1) / elapsed
            eta = (N_SIMULATIONS - i - 1) / rate
            valid = [rr for rr in results if not rr.get("ruined")]
            if valid:
                profits = sorted([rr["net_profit"] for rr in valid])
                med = pct(profits, 50)
                print(f"  {i+1:>4}/{N_SIMULATIONS} | {elapsed:.0f}s | "
                      f"ETA {eta:.0f}s | 중앙값 ${med:+,.0f}")

    elapsed = time.time() - t_sim
    print(f"  완료 | {elapsed:.0f}s")

    valid = [r for r in results if not r.get("ruined")]
    ruined = N_SIMULATIONS - len(valid)
    ruin_rate = ruined / N_SIMULATIONS * 100

    if not valid:
        print("모두 파산")
        return

    profits = sorted([r["net_profit"] for r in valid])
    dds = sorted([r["max_dd"] for r in valid])
    wrs = sorted([r["win_rate"] for r in valid])
    balances = sorted([r["final_balance"] for r in valid])

    print(f"\n{'='*110}")
    print(f"결과 분석 (valid N={len(valid)})")
    print(f"{'='*110}")

    def stats(name, vals):
        print(f"\n  [{name}]")
        print(f"    평균:    {mean(vals):>+14,.0f}")
        print(f"    중앙값:  {pct(vals, 50):>+14,.0f}")
        print(f"    표준편차:{stdev(vals) if len(vals)>1 else 0:>14,.0f}")
        print(f"    5%:      {pct(vals, 5):>+14,.0f}  ← 최악 5%")
        print(f"    25%:     {pct(vals, 25):>+14,.0f}")
        print(f"    75%:     {pct(vals, 75):>+14,.0f}")
        print(f"    95%:     {pct(vals, 95):>+14,.0f}  ← 최상 5%")
        print(f"    최솟값:  {min(vals):>+14,.0f}")
        print(f"    최댓값:  {max(vals):>+14,.0f}")

    stats("순이익 ($)", profits)
    stats("최대 DD (%)", dds)
    stats("승률 (%)", wrs)

    print(f"\n  [파산률]")
    print(f"    {ruined}/{N_SIMULATIONS} = {ruin_rate:.1f}%")

    # 기준 대비
    if not baseline.get("ruined"):
        base = baseline["net_profit"]
        better = sum(1 for p in profits if p > base)
        worse = sum(1 for p in profits if p < base)
        print(f"\n  [기준 대비 — 실제 데이터 ${base:+,.0f}]")
        print(f"    이보다 나은: {better}/{len(profits)} = {better/len(profits)*100:.1f}%")
        print(f"    이보다 나쁜: {worse}/{len(profits)} = {worse/len(profits)*100:.1f}%")

    # 수익성
    profitable = sum(1 for p in profits if p > 0)
    print(f"\n  [수익성]")
    print(f"    플러스: {profitable}/{len(profits)} = {profitable/len(profits)*100:.1f}%")

    # 분포 버킷
    print(f"\n  [수익 분포 버킷]")
    buckets = [
        ("파산", lambda p: False),
        ("손실 < -$100k", lambda p: p < -100000),
        ("손실 -$100k ~ -$10k", lambda p: -100000 <= p < -10000),
        ("손실 -$10k ~ 0", lambda p: -10000 <= p < 0),
        ("수익 $0 ~ $10k", lambda p: 0 <= p < 10000),
        ("수익 $10k ~ $100k", lambda p: 10000 <= p < 100000),
        ("수익 $100k ~ $500k", lambda p: 100000 <= p < 500000),
        ("수익 $500k ~ $1M", lambda p: 500000 <= p < 1000000),
        ("수익 > $1M", lambda p: p >= 1000000),
    ]
    print(f"    {'구간':<22} {'개수':>6} {'비율':>7}")
    print(f"    " + "-" * 38)
    print(f"    {'파산':<22} {ruined:>6} {ruin_rate:>6.1f}%")
    for label, cond in buckets[1:]:
        c = sum(1 for p in profits if cond(p))
        print(f"    {label:<22} {c:>6} {c/N_SIMULATIONS*100:>6.1f}%")

    # 손실률 안전성
    losing = sum(1 for p in profits if p < 0)
    print(f"\n  [리스크 지표]")
    print(f"    손실 확률: {losing/len(profits)*100:.1f}%")
    print(f"    원금 보존률(>= $580): {sum(1 for b in balances if b >= SEED)/len(balances)*100:.1f}%")
    print(f"    DD 평균: {mean(dds):.1f}%")
    print(f"    DD > 40% 비율: {sum(1 for d in dds if d > 40)/len(dds)*100:.1f}%")
    print(f"    DD > 50% 비율: {sum(1 for d in dds if d > 50)/len(dds)*100:.1f}%")

    total_elapsed = time.time() - t0
    print(f"\n총 소요: {total_elapsed:.0f}s ({total_elapsed/60:.1f}분)")

    out = {
        "timestamp": datetime.now().isoformat(),
        "n_simulations": N_SIMULATIONS,
        "method": "log-return block bootstrap (continuity preserved)",
        "block_size_hours": BLOCK_SIZE_HOURS,
        "baseline": baseline,
        "ruin_rate_pct": ruin_rate,
        "stats": {
            "profit": {
                "mean": mean(profits), "median": pct(profits, 50),
                "p5": pct(profits, 5), "p25": pct(profits, 25),
                "p75": pct(profits, 75), "p95": pct(profits, 95),
                "min": min(profits), "max": max(profits),
            },
            "dd": {
                "mean": mean(dds), "median": pct(dds, 50),
                "p5": pct(dds, 5), "p95": pct(dds, 95),
                "min": min(dds), "max": max(dds),
            },
            "win_rate": {
                "mean": mean(wrs), "median": pct(wrs, 50),
                "p5": pct(wrs, 5), "p95": pct(wrs, 95),
            },
        },
        "profitable_rate_pct": profitable / len(profits) * 100,
        "losing_rate_pct": losing / len(profits) * 100,
    }

    out_path = os.path.join(RESULTS_DIR, "v6_monte_carlo_fixed.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
