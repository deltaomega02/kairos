#!/usr/bin/env python3
"""
v5 슬리피지 민감도 테스트
===========================
실전 슬리피지를 모델링하여 최적 트레일링 조합이 실전에서 얼마나 견디는지 확인.

슬리피지 수준:
- 0.00%: 순수 백테스트 (이상적)
- 0.02%: Bybit BTC 현실 최소 (스프레드 + 체결 지연)
- 0.05%: 평균적 실전 (ETH/SOL 포함)
- 0.10%: 보수적 가정 (불리한 시장 상황)
- 0.15%: 극보수 (유동성 낮은 구간)

각 슬리피지 수준에서 후보 조합들의 성과를 비교한다.
"""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v5"
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
SEED = 600.0
DAILY_COST_USD = 1150 / 1470

# 후보 트레일링 조합 (v5 final optimization TOP 후보들)
CANDIDATES = [
    (0.5, 0.1),
    (0.6, 0.1),
    (0.7, 0.1),
    (0.8, 0.1),   # v5 복합 1위
    (0.9, 0.1),   # v5 복합 2위
    (1.0, 0.1),   # v5 복합 6위
    (1.1, 0.1),
    (1.2, 0.1),
    (0.8, 0.12),
    (1.0, 0.12),  # 제가 권한 "안전" 옵션
    (1.2, 0.12),
    (0.8, 0.15),
    (1.0, 0.15),
    (1.2, 0.15),
    (1.5, 0.3),   # 현재 운영 (비교용)
]

# 슬리피지 수준 (편도, %)
SLIPPAGE_LEVELS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]


def run_one(data, act, dist, slippage, skip_years=(2023,)):
    r = run_shared_backtest(
        data, BEST_PARAMS, SEED,
        use_funding=True,
        trailing_activation=act,
        trailing_distance=dist,
        block_sol_long=True,
        skip_years=skip_years,
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        slippage_pct=slippage,
    )
    return {
        "net_profit": r["net_profit"],
        "max_dd": r["max_dd"],
        "final_balance": r["final_balance"],
        "total_trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "ruined": r["ruined"],
    }


def main():
    print("=" * 110)
    print("HERMES v5 — 슬리피지 민감도 테스트")
    print(f"시나리오: 수동 감독 (2023 skip) | 시드 $600 | 4년 데이터")
    print(f"후보 {len(CANDIDATES)}개 × 슬리피지 {len(SLIPPAGE_LEVELS)}수준 = {len(CANDIDATES)*len(SLIPPAGE_LEVELS)} 실행")
    print("=" * 110)

    print("\n[데이터 로드]")
    data = load_all_data()

    # 결과: {(act,dist): {slippage: result}}
    results = {}
    total = len(CANDIDATES) * len(SLIPPAGE_LEVELS)
    done = 0

    for (act, dist) in CANDIDATES:
        results[(act, dist)] = {}
        for slip in SLIPPAGE_LEVELS:
            r = run_one(data, act, dist, slip)
            results[(act, dist)][slip] = r
            done += 1
        print(f"  {act}/{dist} 완료 ({done}/{total})")

    # === 출력 ===
    print("\n" + "=" * 110)
    print("슬리피지 레벨별 순이익 (수동 감독, 4년)")
    print("=" * 110)
    header = f"{'Act':>5} {'Dist':>5} " + " ".join(f"{'slip '+str(s)+'%':>12}" for s in SLIPPAGE_LEVELS)
    print(header)
    print("-" * 110)
    for key in CANDIDATES:
        act, dist = key
        row = f"{act:>5} {dist:>5} "
        for slip in SLIPPAGE_LEVELS:
            r = results[key][slip]
            if r["ruined"]:
                val = "RUINED"
            else:
                val = f"${r['net_profit']:>+10,.0f}"
            row += f" {val:>12}"
        marker = " ← 현재" if key == (1.5, 0.3) else ""
        print(row + marker)

    # DD 비교
    print("\n" + "=" * 110)
    print("슬리피지 레벨별 최대 DD (%)")
    print("=" * 110)
    print(header)
    print("-" * 110)
    for key in CANDIDATES:
        act, dist = key
        row = f"{act:>5} {dist:>5} "
        for slip in SLIPPAGE_LEVELS:
            r = results[key][slip]
            if r["ruined"]:
                val = "RUINED"
            else:
                val = f"{r['max_dd']:>8.1f}%"
            row += f" {val:>12}"
        marker = " ← 현재" if key == (1.5, 0.3) else ""
        print(row + marker)

    # 열화율 분석 (slip 0% 대비 0.05%, 0.10% 감소폭)
    print("\n" + "=" * 110)
    print("슬리피지 내성 분석 — slip 0% 대비 순이익 유지율")
    print("=" * 110)
    print(f"{'Act':>5} {'Dist':>5} "
          f"{'slip 0%':>14} {'slip 0.05%':>14} {'유지율':>10} "
          f"{'slip 0.10%':>14} {'유지율':>10}")
    print("-" * 80)
    for key in CANDIDATES:
        act, dist = key
        base = results[key][0.00]["net_profit"]
        mid = results[key][0.05]["net_profit"]
        high = results[key][0.10]["net_profit"]

        if base <= 0 or results[key][0.00]["ruined"]:
            print(f"{act:>5} {dist:>5} baseline 실패")
            continue

        mid_ratio = mid / base * 100 if base > 0 else 0
        high_ratio = high / base * 100 if base > 0 else 0

        marker = " ← 현재" if key == (1.5, 0.3) else ""
        print(f"{act:>5} {dist:>5} "
              f"${base:>+12,.0f} ${mid:>+12,.0f} {mid_ratio:>8.1f}% "
              f"${high:>+12,.0f} {high_ratio:>8.1f}%{marker}")

    # 슬리피지 0.05%와 0.10% 시점의 순이익 기준 재랭킹
    for slip_level in [0.02, 0.05, 0.10]:
        print(f"\n{'=' * 110}")
        print(f"슬리피지 {slip_level}% 환경에서의 랭킹")
        print(f"{'=' * 110}")

        valid = []
        for key in CANDIDATES:
            r = results[key][slip_level]
            if not r["ruined"]:
                valid.append((key, r))

        valid.sort(key=lambda x: x[1]["net_profit"], reverse=True)

        print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'순이익':>14} {'DD':>7} {'승률':>7}")
        print("-" * 60)
        for i, (key, r) in enumerate(valid):
            marker = " ← 현재" if key == (1.5, 0.3) else ""
            print(f"{i+1:>4} {key[0]:>5} {key[1]:>5} "
                  f"${r['net_profit']:>+12,.0f} {r['max_dd']:>6.1f}% "
                  f"{r['win_rate']:>6.1f}%{marker}")

    # 최종 결정 기준
    print("\n" + "=" * 110)
    print("슬리피지 0.05% 실전 기준 최종 추천")
    print("=" * 110)
    mid_results = [(k, results[k][0.05]) for k in CANDIDATES if not results[k][0.05]["ruined"]]
    mid_sorted = sorted(mid_results, key=lambda x: x[1]["net_profit"], reverse=True)
    winner = mid_sorted[0]
    current = next((x for x in mid_sorted if x[0] == (1.5, 0.3)), None)

    print(f"\n  슬리피지 0.05% 가정 1위: {winner[0][0]}/{winner[0][1]}")
    print(f"    순이익: ${winner[1]['net_profit']:+,.0f}")
    print(f"    DD: {winner[1]['max_dd']}%")
    print(f"    승률: {winner[1]['win_rate']}%")

    if current:
        print(f"\n  현재 1.5/0.3 (슬리피지 0.05%):")
        print(f"    순이익: ${current[1]['net_profit']:+,.0f}")
        print(f"    개선배수: {winner[1]['net_profit']/current[1]['net_profit']:.1f}x")

    # 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "candidates": CANDIDATES,
        "slippage_levels": SLIPPAGE_LEVELS,
        "results": {
            f"{k[0]}_{k[1]}": {
                str(s): r for s, r in sl_data.items()
            }
            for k, sl_data in results.items()
        },
    }

    out_path = os.path.join(RESULTS_DIR, "v5_slippage_test.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
