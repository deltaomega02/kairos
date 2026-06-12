#!/usr/bin/env python3
"""
v4 복리 효과 시각화
====================
현재 시드($600) 기준 + 증액 시나리오 비교.
연말 잔고, 연도별 성장률, 최종 복리 효과를 표로 출력.
"""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v4"

BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
DAILY_COST_KRW = 1150
USDKRW = 1470
DAILY_COST_USD = DAILY_COST_KRW / USDKRW

# 현재 $600 baseline + 증액 시나리오들
SCENARIOS = [
    ("현재 $600", 600),
    ("+$400 → $1,000", 1000),
    ("+$900 → $1,500", 1500),
    ("+$1,400 → $2,000", 2000),
    ("+$2,400 → $3,000", 3000),
    ("+$4,400 → $5,000", 5000),
]


def main():
    print("=" * 100)
    print("HERMES v4 — 복리 효과 시각화")
    print("시나리오: 일일 수동 감독 (2023년 수동 중단)")
    print(f"파라미터: EMA5/18 + trailing 1.5/0.3 + funding | 서버비 ₩{DAILY_COST_KRW}/일")
    print("=" * 100)

    print("\n[데이터 로드]")
    data = load_all_data()

    all_results = {}
    for label, seed in SCENARIOS:
        print(f"  ▶ {label} 백테스트 중...")
        r = run_shared_backtest(
            data, BEST_PARAMS, float(seed),
            use_funding=True,
            trailing_activation=1.5, trailing_distance=0.3,
            block_sol_long=True,
            skip_years=(2023,),
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
        )
        all_results[seed] = {"label": label, "result": r}

    # 연말 잔고 테이블
    years = [2022, 2023, 2024, 2025, 2026]

    print("\n" + "=" * 100)
    print("📊 연말 잔고 추이 (수동 감독 시나리오)")
    print("=" * 100)
    print(f"{'시나리오':<22} {'시작':>10} {'2022말':>12} {'2023말':>12} "
          f"{'2024말':>12} {'2025말':>12} {'2026년현재':>13}")
    print("-" * 100)

    for seed, rec in all_results.items():
        label = rec["label"]
        r = rec["result"]
        yeb = r["year_end_balance"]
        row = f"{label:<22} ${seed:>9,}"
        for y in years:
            if y in yeb:
                row += f" ${yeb[y]:>10,.0f}"
            else:
                row += f" {'-':>11}"
        print(row)

    # 최종 성과 비교
    print("\n" + "=" * 100)
    print("💰 최종 성과 비교")
    print("=" * 100)
    print(f"{'시나리오':<22} {'시드':>10} {'최종잔고':>14} {'순이익':>14} "
          f"{'수익배수':>10} {'DD':>7}")
    print("-" * 100)

    base_profit = None
    for seed, rec in all_results.items():
        r = rec["result"]
        label = rec["label"]
        mult = r["final_balance"] / seed
        row = (f"{label:<22} ${seed:>9,} ${r['final_balance']:>12,.0f} "
               f"${r['net_profit']:>+12,.0f} {mult:>9.1f}x "
               f"{r['max_dd']:>6.1f}%")
        print(row)
        if seed == 600:
            base_profit = r["net_profit"]

    # 증액 대비 추가 이익 (레버리지 효과)
    if base_profit:
        print("\n" + "=" * 100)
        print("🚀 시드 증액 효율 — $600 대비 (4년 백테스트 기준)")
        print("=" * 100)
        print(f"{'증액':<22} {'추가 투입':>12} {'추가 순이익':>16} {'ROI (추가분)':>16} {'배수효과':>12}")
        print("-" * 100)

        for seed, rec in all_results.items():
            if seed == 600:
                continue
            r = rec["result"]
            extra_seed = seed - 600
            extra_profit = r["net_profit"] - base_profit
            roi = (extra_profit / extra_seed * 100) if extra_seed > 0 else 0
            mult = extra_profit / extra_seed if extra_seed > 0 else 0
            print(f"{rec['label']:<22} ${extra_seed:>+10,} ${extra_profit:>+14,.0f} "
                  f"{roi:>14,.0f}% {mult:>10,.1f}x")

    # 저장
    out = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {
            str(seed): {
                "label": rec["label"],
                "final_balance": rec["result"]["final_balance"],
                "net_profit": rec["result"]["net_profit"],
                "net_pct": rec["result"]["net_pct"],
                "max_dd": rec["result"]["max_dd"],
                "year_end_balance": rec["result"]["year_end_balance"],
                "total_trades": rec["result"]["total_trades"],
                "win_rate": rec["result"]["win_rate"],
            }
            for seed, rec in all_results.items()
        }
    }

    out_path = os.path.join(RESULTS_DIR, "v4_compound_viz.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
