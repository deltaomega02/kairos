#!/usr/bin/env python3
"""
백테스팅 v4 — 최소 시드 분석 (Shared-Balance 버전)
====================================================
실제 HERMES 구조 (shared wallet) 기반 시드 스윕.
두 시나리오 비교:
  A) 전체 기간 (2022-2026, 4년)
  B) 2023년 수동 중단 시나리오 (일일 수동 감독으로 저변동 구간 스킵)

일일 서버비 ₩1,150 차감.
"""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v4"
os.makedirs(RESULTS_DIR, exist_ok=True)

# v3 확정 파라미터
BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
TRAILING_ACT = 1.5
TRAILING_DIST = 0.3

# 서버비 (GCP e2-small 서울)
DAILY_COST_KRW = 1150
USDKRW = 1470
DAILY_COST_USD = DAILY_COST_KRW / USDKRW

SEEDS = [100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000]
RUIN_THRESHOLD = 15.0


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_row(seed, r):
    outcome = f"파산 {r['ruin_date']}" if r["ruined"] else "✓ 수익" if r["net_profit"] > 0 else "생존(손실)"
    print(f"${seed:>6} {r['total_trades']:>6} {r['win_rate']:>6.1f}% "
          f"${r['final_balance']:>10,.2f} ${r['net_profit']:>+10,.2f} "
          f"{r['net_pct']:>+9.1f}% {r['max_dd']:>7.1f}% {outcome:>14}")


def run_scenario(data, skip_years, label):
    print_header(f"[{label}] skip_years={skip_years or '없음'}")
    print(f"{'시드':>7} {'거래':>6} {'승률':>7} {'최종잔고':>12} {'순이익':>11} "
          f"{'순수익률':>10} {'최대DD':>8} {'결과':>14}")
    print("-" * 100)

    results = {}
    for seed in SEEDS:
        r = run_shared_backtest(
            data, BEST_PARAMS, float(seed),
            use_funding=True,
            trailing_activation=TRAILING_ACT,
            trailing_distance=TRAILING_DIST,
            block_sol_long=True,
            skip_years=skip_years,
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=RUIN_THRESHOLD,
        )
        results[seed] = r
        print_row(seed, r)
    return results


def summarize(results, label):
    print(f"\n[{label}] 요약")
    survivors = {s: r for s, r in results.items()
                 if not r["ruined"] and r["net_profit"] > 0}
    if survivors:
        min_profit = min(survivors.keys())
        mp = survivors[min_profit]
        print(f"  최소 수익 시드: ${min_profit} (순이익 ${mp['net_profit']:,.2f}, DD {mp['max_dd']}%)")
        safe = {s: r for s, r in survivors.items() if r["max_dd"] <= 50}
        if safe:
            ms = min(safe.keys())
            sr = safe[ms]
            print(f"  안전권 (DD ≤ 50%): ${ms} (순이익 ${sr['net_profit']:,.2f}, DD {sr['max_dd']}%)")
        safer = {s: r for s, r in survivors.items() if r["max_dd"] <= 35}
        if safer:
            mss = min(safer.keys())
            srr = safer[mss]
            print(f"  안정권 (DD ≤ 35%): ${mss} (순이익 ${srr['net_profit']:,.2f}, DD {srr['max_dd']}%)")
    else:
        print("  ⚠ 모든 시드에서 수익 실패 또는 파산")


def main():
    print("=" * 100)
    print("HERMES v4 — Shared-Balance 시드 분석")
    print(f"실제 HERMES 구조 그대로: 공유 잔고, MAX_SIMULTANEOUS=2, daily trade cap")
    print(f"파라미터: EMA5/18 SL1.5 TP4.0 ADX35 PB1.5 score40 + trailing {TRAILING_ACT}/{TRAILING_DIST} + funding")
    print(f"서버비: ₩{DAILY_COST_KRW:,}/일 (${DAILY_COST_USD:.3f}/일) @ ₩{USDKRW}/$")
    print(f"파산 기준: 잔고 ${RUIN_THRESHOLD} 미만")
    print("=" * 100)

    print("\n[데이터 로드]")
    data = load_all_data()
    print(f"  로드 완료: {len(data)} 데이터셋")

    # 시나리오 A: 전체 기간 (자동화된 HERMES만)
    scenario_a = run_scenario(data, skip_years=(), label="A. 자동 운영 (전 기간)")

    # 시나리오 B: 2023년 수동 중단 (일일 수동 감독)
    scenario_b = run_scenario(data, skip_years=(2023,), label="B. 2023년 Claude 중단 권고 시나리오")

    summarize(scenario_a, "시나리오 A")
    summarize(scenario_b, "시나리오 B")

    # 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "engine": "shared-balance (실제 HERMES 구조)",
        "config": {
            "best_params": {k: v for k, v in BEST_PARAMS.items()
                            if isinstance(v, (int, float, str, bool))},
            "trailing_activation": TRAILING_ACT,
            "trailing_distance": TRAILING_DIST,
            "daily_cost_krw": DAILY_COST_KRW,
            "usdkrw": USDKRW,
            "daily_cost_usd": round(DAILY_COST_USD, 4),
            "ruin_threshold": RUIN_THRESHOLD,
        },
        "scenario_a_full": {str(s): r for s, r in scenario_a.items()},
        "scenario_b_skip_2023": {str(s): r for s, r in scenario_b.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "v4_seed_sweep.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
