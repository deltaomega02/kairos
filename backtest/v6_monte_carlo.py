#!/usr/bin/env python3
"""
v6 Monte Carlo Bootstrap 검증
==============================
실제 BTC/ETH/SOL/XRP의 1H 데이터를 블록 단위로 재샘플링해서
수백 개의 "평행우주" 가격 경로 생성. 각 경로에서 풀 패키지 돌려
결과 분포 확인.

목적: "우리가 운이 좋았던 건 아닌가?"의 통계적 답변.

방법:
1. 1H 캔들을 24개(1일) 블록 단위로 랜덤 샘플링 (block bootstrap)
2. 원본과 같은 총 길이로 재조합 (~37000 1H 캔들)
3. 4H는 1H에서 재구성 (consistency 유지)
4. 펀딩 레이트는 실제값 유지 (노이즈 최소화)
5. 풀 패키지 백테스트 실행
6. N = 500회 반복
7. 결과 분포 (평균, 중앙값, 95% CI, 최악 5%)
"""
import os
import sys
import json
import time
import random
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

# 풀 패키지 설정
FULL_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30,
}
SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470
N_SIMULATIONS = 500  # 평행우주 개수
BLOCK_SIZE_HOURS = 24  # 블록 크기 (1일)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_real_data():
    """실제 BTC/ETH/SOL/XRP 데이터 로드"""
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


def block_bootstrap_1h(df_1h: pd.DataFrame, block_size: int, rng: np.random.Generator) -> pd.DataFrame:
    """1H 블록 부트스트랩"""
    n = len(df_1h)
    target_len = n
    max_start = n - block_size

    if max_start < 1:
        return df_1h.copy()

    # 필요한 블록 수
    num_blocks = (target_len // block_size) + 1

    # 랜덤 시작점들
    starts = rng.integers(0, max_start, size=num_blocks)

    # 블록들 concat
    pieces = []
    for s in starts:
        pieces.append(df_1h.iloc[s:s+block_size])
    combined = pd.concat(pieces, ignore_index=True)

    # 원본 길이로 자르기
    combined = combined.iloc[:target_len].reset_index(drop=True)

    # 타임스탬프는 원본 그대로 유지 (시간 축 복원)
    combined["timestamp"] = df_1h["timestamp"].values[:len(combined)]

    return combined


def reconstruct_4h_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1H에서 4H 재구성 (consistency 유지)"""
    # 1H의 timestamp를 기준으로 4H 경계에 정렬
    # 첫 4H 경계 찾기
    ts_col = df_1h["timestamp"].values
    first_ts = int(ts_col[0])
    four_h_ms = 14400000

    # 첫 4H 정렬 시작점
    aligned_start = (first_ts // four_h_ms) * four_h_ms

    result = []
    i = 0
    n = len(df_1h)

    while i < n:
        # 현재 4H 블록의 타임스탬프
        current_4h_ts = (int(ts_col[i]) // four_h_ms) * four_h_ms

        # 이 4H에 속하는 1H 캔들들
        group_indices = []
        while i < n and (int(ts_col[i]) // four_h_ms) * four_h_ms == current_4h_ts:
            group_indices.append(i)
            i += 1

        if len(group_indices) == 0:
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


def create_synthetic_data(real_data: dict, seed: int) -> dict:
    """실제 데이터 기반 가상 데이터 생성"""
    rng = np.random.default_rng(seed)
    synthetic = {}

    for sym in SYMBOLS:
        key_1h = f"{sym}_60"
        if key_1h not in real_data:
            continue

        # 1H 부트스트랩
        new_1h = block_bootstrap_1h(real_data[key_1h], BLOCK_SIZE_HOURS, rng)
        synthetic[key_1h] = new_1h

        # 4H 재구성
        synthetic[f"{sym}_240"] = reconstruct_4h_from_1h(new_1h)

        # 펀딩 레이트는 실제값 사용 (노이즈 최소화)
        fund_key = f"{sym}_funding"
        if fund_key in real_data:
            synthetic[fund_key] = real_data[fund_key]

    return synthetic


def run_full_package(data: dict) -> dict:
    """풀 패키지로 백테스트"""
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


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * pct / 100)
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def main():
    t0 = time.time()
    print("="*110)
    print(f"HERMES v6 — Monte Carlo Bootstrap 검증 (N={N_SIMULATIONS})")
    print(f"풀 패키지: 4코인, 3pos, lev7, TP6, ADX30, trailing 1.2/0.1")
    print(f"방법: 1H 블록 부트스트랩 (block size {BLOCK_SIZE_HOURS}h)")
    print("="*110)

    print("\n[실제 데이터 로드]")
    real = load_real_data()
    print(f"  로드 완료: {len(real)} 데이터셋")
    for sym in SYMBOLS:
        key = f"{sym}_60"
        if key in real:
            print(f"  {sym}: {len(real[key])}개 1H 캔들")

    # 실제 데이터로 기준 결과
    print("\n[기준 — 실제 데이터 풀 패키지]")
    baseline = run_full_package(real)
    if baseline.get("ruined"):
        print(f"  ⚠ baseline 파산: {baseline.get('error', 'RUINED')}")
    else:
        print(f"  순이익: ${baseline['net_profit']:+,.0f}")
        print(f"  최종잔고: ${baseline['final_balance']:,.0f}")
        print(f"  DD: {baseline['max_dd']}%")
        print(f"  승률: {baseline['win_rate']}%")
        print(f"  거래: {baseline['total_trades']}")

    # Monte Carlo 시뮬레이션
    print(f"\n[Monte Carlo {N_SIMULATIONS}회 실행]")
    results = []
    t_sim_start = time.time()

    for i in range(N_SIMULATIONS):
        synthetic = create_synthetic_data(real, seed=i + 1)
        r = run_full_package(synthetic)
        results.append(r)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_sim_start
            rate = (i + 1) / elapsed
            eta = (N_SIMULATIONS - i - 1) / rate if rate > 0 else 0
            # 중간 통계
            valid = [rr for rr in results if not rr.get("ruined")]
            if valid:
                profits = sorted([rr["net_profit"] for rr in valid])
                med = percentile(profits, 50)
                print(f"  {i+1:>4}/{N_SIMULATIONS} | {elapsed:.0f}s 경과 | "
                      f"ETA {eta:.0f}s | 현재 중앙값 ${med:+,.0f}")

    elapsed = time.time() - t_sim_start
    print(f"  완료 | {elapsed:.0f}s")

    # 분석
    print(f"\n{'='*110}")
    print(f"결과 분석 (N={N_SIMULATIONS})")
    print(f"{'='*110}")

    valid_results = [r for r in results if not r.get("ruined")]
    ruined_count = N_SIMULATIONS - len(valid_results)
    ruin_rate = ruined_count / N_SIMULATIONS * 100

    if not valid_results:
        print("  모든 시뮬레이션 파산. 데이터 문제일 수 있음.")
        return

    profits = sorted([r["net_profit"] for r in valid_results])
    dds = sorted([r["max_dd"] for r in valid_results])
    balances = sorted([r["final_balance"] for r in valid_results])
    win_rates = sorted([r["win_rate"] for r in valid_results])
    trades = sorted([r["total_trades"] for r in valid_results])

    def stats_block(name, sorted_vals, is_profit=False):
        print(f"\n  [{name}]")
        print(f"    평균:       {mean(sorted_vals):>+12,.1f}")
        print(f"    중앙값:     {percentile(sorted_vals, 50):>+12,.1f}")
        print(f"    표준편차:   {stdev(sorted_vals) if len(sorted_vals)>1 else 0:>12,.1f}")
        print(f"    최솟값:     {min(sorted_vals):>+12,.1f}")
        print(f"    5% 분위:    {percentile(sorted_vals, 5):>+12,.1f}  ← 최악 5%")
        print(f"    25% 분위:   {percentile(sorted_vals, 25):>+12,.1f}")
        print(f"    75% 분위:   {percentile(sorted_vals, 75):>+12,.1f}")
        print(f"    95% 분위:   {percentile(sorted_vals, 95):>+12,.1f}  ← 최상 5%")
        print(f"    최댓값:     {max(sorted_vals):>+12,.1f}")

    stats_block("순이익 ($)", profits, is_profit=True)
    stats_block("최대 DD (%)", dds)
    stats_block("최종 잔고 ($)", balances)
    stats_block("승률 (%)", win_rates)
    stats_block("거래 수", trades)

    print(f"\n  [파산률]")
    print(f"    파산 시뮬레이션: {ruined_count}/{N_SIMULATIONS} = {ruin_rate:.1f}%")

    # 기준 대비 위치
    if not baseline.get("ruined"):
        base_profit = baseline["net_profit"]
        better = sum(1 for p in profits if p > base_profit)
        worse = sum(1 for p in profits if p < base_profit)
        pct_better = better / len(profits) * 100
        pct_worse = worse / len(profits) * 100
        print(f"\n  [기준 대비 — 실제 데이터 결과 ${base_profit:+,.0f}]")
        print(f"    이보다 나은 시뮬레이션: {better}/{len(profits)} = {pct_better:.1f}%")
        print(f"    이보다 나쁜 시뮬레이션: {worse}/{len(profits)} = {pct_worse:.1f}%")
        if pct_worse > 50:
            print(f"    → 우리가 '평균보다 잘 봤음' (운이 좋은 편)")
        elif pct_better > 50:
            print(f"    → 우리가 '평균보다 덜 봤음' (운이 평균 이하)")
        else:
            print(f"    → '평균 근처' (대표성 있는 경로)")

    # 수익성 요약
    profitable = sum(1 for p in profits if p > 0)
    print(f"\n  [수익성]")
    print(f"    플러스 시뮬레이션: {profitable}/{len(profits)} = {profitable/len(profits)*100:.1f}%")
    print(f"    마이너스:        {len(profits)-profitable}/{len(profits)}")

    # 시나리오별 분포 버킷
    print(f"\n  [수익 분포 버킷]")
    buckets = [
        ("파산", lambda p: False),
        ("손실 > -$10k", lambda p: p <= -10000),
        ("손실 -$10k ~ 0", lambda p: -10000 < p <= 0),
        ("수익 $0 ~ $10k", lambda p: 0 < p <= 10000),
        ("수익 $10k ~ $50k", lambda p: 10000 < p <= 50000),
        ("수익 $50k ~ $100k", lambda p: 50000 < p <= 100000),
        ("수익 $100k ~ $500k", lambda p: 100000 < p <= 500000),
        ("수익 > $500k", lambda p: p > 500000),
    ]
    print(f"    {'구간':<22} {'개수':>8} {'비율':>8}")
    print(f"    " + "-" * 40)
    print(f"    {'파산':<22} {ruined_count:>8} {ruin_rate:>7.1f}%")
    for label, cond in buckets[1:]:
        count = sum(1 for p in profits if cond(p))
        pct = count / N_SIMULATIONS * 100
        print(f"    {label:<22} {count:>8} {pct:>7.1f}%")

    # 저장
    total_elapsed = time.time() - t0
    print(f"\n총 소요: {total_elapsed:.0f}s ({total_elapsed/60:.1f}분)")

    out = {
        "timestamp": datetime.now().isoformat(),
        "n_simulations": N_SIMULATIONS,
        "block_size_hours": BLOCK_SIZE_HOURS,
        "seed": SEED,
        "slippage": SLIP,
        "config": "full package (4코인, 3pos, lev7, TP6, ADX30, trailing 1.2/0.1)",
        "baseline_real_data": baseline,
        "n_ruined": ruined_count,
        "ruin_rate_pct": ruin_rate,
        "n_profitable": profitable,
        "profitable_rate_pct": profitable / len(profits) * 100 if profits else 0,
        "stats": {
            "net_profit": {
                "mean": mean(profits),
                "median": percentile(profits, 50),
                "std": stdev(profits) if len(profits) > 1 else 0,
                "p5": percentile(profits, 5),
                "p25": percentile(profits, 25),
                "p75": percentile(profits, 75),
                "p95": percentile(profits, 95),
                "min": min(profits),
                "max": max(profits),
            },
            "max_dd": {
                "mean": mean(dds),
                "median": percentile(dds, 50),
                "p5": percentile(dds, 5),
                "p95": percentile(dds, 95),
                "min": min(dds),
                "max": max(dds),
            },
            "win_rate": {
                "mean": mean(win_rates),
                "median": percentile(win_rates, 50),
                "p5": percentile(win_rates, 5),
                "p95": percentile(win_rates, 95),
            },
        },
        "all_results": results,
    }

    out_path = os.path.join(RESULTS_DIR, "v6_monte_carlo.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
