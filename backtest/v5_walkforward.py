#!/usr/bin/env python3
"""
v5 Walk-Forward 검증 — 과적합 확인
====================================
목적: v5 전수 그리드의 결과가 과적합인지 확인
방법:
1. 학습 기간(2022-2024)에서 최적 찾기
2. 검증 기간(2025-2026)에서 해당 설정 성과 확인
3. 또한 반대로: 2025-2026 최적 → 2022-2024 확인
4. 전체 기간 최적과 각 구간 최적의 일관성 체크
"""
import os
import sys
import json
from datetime import datetime
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest, prepare_symbol
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v5"

BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
SEED = 600.0
DAILY_COST_USD = 1150 / 1470

# 그리드 (좀더 coarse — 빠른 검증용)
ACTIVATIONS = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
DISTANCES = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8]


def filter_data_by_date(data, start_date, end_date):
    """데이터를 특정 기간으로 필터링"""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    filtered = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and "timestamp" in v.columns:
            mask = (v["timestamp"] >= start_ts) & (v["timestamp"] < end_ts)
            filtered[k] = v[mask].reset_index(drop=True)
        else:
            filtered[k] = v
    return filtered


def run_grid(data, label, skip_years=()):
    """그리드 실행"""
    results = []
    valid = [(a, d) for a, d in product(ACTIVATIONS, DISTANCES) if d < a]
    print(f"\n  [{label}] {len(valid)} 조합 실행 중...")

    for i, (act, dist) in enumerate(valid):
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
        )
        if r.get("total_trades", 0) > 0:
            results.append({
                "activation": act,
                "distance": dist,
                "net_profit": r["net_profit"],
                "max_dd": r["max_dd"],
                "total_trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "final_balance": r["final_balance"],
                "ruined": r["ruined"],
            })
        if (i + 1) % 20 == 0:
            print(f"    진행: {i+1}/{len(valid)}")

    return results


def rank_map(results, metric="net_profit", reverse=True):
    """결과를 순위 맵으로 변환 (act, dist) → rank"""
    valid = [r for r in results if not r["ruined"]]
    sorted_r = sorted(valid, key=lambda x: x[metric], reverse=reverse)
    return {(r["activation"], r["distance"]): i + 1 for i, r in enumerate(sorted_r)}


def print_comparison(label_a, results_a, label_b, results_b):
    """두 기간 결과 비교"""
    rank_a = rank_map(results_a)
    rank_b = rank_map(results_b)

    # A에서 상위 10
    valid_a = sorted([r for r in results_a if not r["ruined"]],
                     key=lambda x: x["net_profit"], reverse=True)

    print(f"\n{'=' * 110}")
    print(f"[{label_a}] 상위 10의 {label_b} 순위 비교")
    print(f"{'=' * 110}")
    print(f"{'Act':>5} {'Dist':>5} "
          f"{'[' + label_a + '] 순이익':>20} {'순위':>6} "
          f"{'[' + label_b + '] 순이익':>20} {'순위':>6} "
          f"{'안정':>8}")
    print("-" * 110)

    for r in valid_a[:10]:
        key = (r["activation"], r["distance"])
        r_b = next((x for x in results_b if x["activation"] == key[0]
                    and x["distance"] == key[1]), None)
        ra = rank_a.get(key, "-")
        rb = rank_b.get(key, "-")
        pnl_b = r_b["net_profit"] if r_b else 0
        dd_b = r_b["max_dd"] if r_b else 0

        # 안정성: 두 기간 순위 차이가 적으면 안정
        if isinstance(ra, int) and isinstance(rb, int):
            total = len(valid_a)
            rank_diff = abs(ra - rb) / total
            if rank_diff < 0.1:
                stab = "✓ 강건"
            elif rank_diff < 0.25:
                stab = "△ 보통"
            else:
                stab = "✗ 변동"
        else:
            stab = "?"

        print(f"{r['activation']:>5} {r['distance']:>5} "
              f"${r['net_profit']:>+18,.0f} {ra:>6} "
              f"${pnl_b:>+18,.0f} {rb:>6} {stab:>8}")


def main():
    print("=" * 110)
    print("HERMES v5 Walk-Forward 검증 — 과적합 확인")
    print("=" * 110)

    print("\n[전체 데이터 로드]")
    full_data = load_all_data()

    # 기간 분할
    train_data = filter_data_by_date(full_data, "2022-01-01", "2024-07-01")
    test_data = filter_data_by_date(full_data, "2024-07-01", "2026-04-12")

    print(f"\n  학습 기간: 2022-01 ~ 2024-06 (2.5년)")
    print(f"  검증 기간: 2024-07 ~ 2026-04 (1.75년)")

    # 학습 기간 실행
    train_results = run_grid(train_data, "TRAIN 2022-2024.06")
    print(f"  TRAIN 완료: {len(train_results)} 유효 조합")

    # 검증 기간 실행
    test_results = run_grid(test_data, "TEST 2024.07-2026")
    print(f"  TEST 완료: {len(test_results)} 유효 조합")

    # 전체 기간 (참고용)
    print(f"\n[전체 기간 참고]")
    full_results = run_grid(full_data, "FULL 2022-2026")

    # 비교
    print_comparison("TRAIN", train_results, "TEST", test_results)
    print_comparison("TEST", test_results, "TRAIN", train_results)

    # 현재 설정과 과거 추천 설정 위치
    print(f"\n{'=' * 110}")
    print(f"후보 설정 — 각 기간에서의 순위")
    print(f"{'=' * 110}")
    print(f"{'Act':>5} {'Dist':>5} "
          f"{'TRAIN pnl':>15} {'T순위':>7} "
          f"{'TEST pnl':>15} {'T순위':>7} "
          f"{'FULL pnl':>15} {'F순위':>7}")
    print("-" * 100)

    candidates = [
        (1.5, 0.3),  # 현재
        (0.8, 0.1),  # v5 1위
        (1.0, 0.1),  # v5 3위
        (1.2, 0.1),  # v5 2위
        (1.2, 0.15),
        (0.5, 0.1),
        (1.5, 0.1),
        (2.0, 0.3),
        (3.0, 0.3),
    ]

    rank_train = rank_map(train_results)
    rank_test = rank_map(test_results)
    rank_full = rank_map(full_results)
    total_train = len([r for r in train_results if not r["ruined"]])
    total_test = len([r for r in test_results if not r["ruined"]])
    total_full = len([r for r in full_results if not r["ruined"]])

    for act, dist in candidates:
        t = next((r for r in train_results if r["activation"] == act
                  and r["distance"] == dist), None)
        e = next((r for r in test_results if r["activation"] == act
                  and r["distance"] == dist), None)
        f = next((r for r in full_results if r["activation"] == act
                  and r["distance"] == dist), None)

        t_rank = f"{rank_train.get((act,dist),'-')}/{total_train}" if t else "-"
        e_rank = f"{rank_test.get((act,dist),'-')}/{total_test}" if e else "-"
        f_rank = f"{rank_full.get((act,dist),'-')}/{total_full}" if f else "-"

        t_pnl = f"${t['net_profit']:+,.0f}" if t else "-"
        e_pnl = f"${e['net_profit']:+,.0f}" if e else "-"
        f_pnl = f"${f['net_profit']:+,.0f}" if f else "-"

        marker = " ← 현재" if (act, dist) == (1.5, 0.3) else ""
        print(f"{act:>5} {dist:>5} {t_pnl:>15} {t_rank:>7} "
              f"{e_pnl:>15} {e_rank:>7} {f_pnl:>15} {f_rank:>7}{marker}")

    # 일관성 분석
    print(f"\n{'=' * 110}")
    print(f"일관성 분석 — TRAIN/TEST 모두 상위 20에 들어간 설정")
    print(f"{'=' * 110}")

    valid_train_top20 = set((r["activation"], r["distance"]) for r in
                            sorted([x for x in train_results if not x["ruined"]],
                                   key=lambda x: x["net_profit"], reverse=True)[:20])
    valid_test_top20 = set((r["activation"], r["distance"]) for r in
                           sorted([x for x in test_results if not x["ruined"]],
                                  key=lambda x: x["net_profit"], reverse=True)[:20])

    intersection = valid_train_top20 & valid_test_top20
    print(f"\n  TRAIN 상위 20 ∩ TEST 상위 20: {len(intersection)}개")
    print(f"  → 두 기간 모두 상위에 드는 로버스트한 설정:")
    for act, dist in sorted(intersection):
        t = next(r for r in train_results if r["activation"] == act and r["distance"] == dist)
        e = next(r for r in test_results if r["activation"] == act and r["distance"] == dist)
        print(f"    {act}/{dist}: TRAIN ${t['net_profit']:+,.0f} (DD {t['max_dd']}%) | "
              f"TEST ${e['net_profit']:+,.0f} (DD {e['max_dd']}%)")

    # 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "train_period": "2022-01 ~ 2024-06",
        "test_period": "2024-07 ~ 2026-04",
        "train_results": sorted(train_results, key=lambda x: x["net_profit"], reverse=True),
        "test_results": sorted(test_results, key=lambda x: x["net_profit"], reverse=True),
        "full_results": sorted(full_results, key=lambda x: x["net_profit"], reverse=True),
        "robust_intersection": list(intersection),
    }

    out_path = os.path.join(RESULTS_DIR, "v5_walkforward.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
