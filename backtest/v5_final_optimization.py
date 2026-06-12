#!/usr/bin/env python3
"""
v5 최종 트레일링 스탑 최적화 — 완벽 검증
===========================================
다차원 평가:
1. 전체 기간 성과 (시나리오 A/B)
2. 연도별 성과 (2022~2026 각각)
3. Walk-forward (다중 분할)
4. 리스크-조정 수익률 (Calmar)
5. 최악 연도 / 최악 분기
6. 민감도 (이웃 조합 일관성)
7. 승률 안정성 (연도별 std)
8. 복합 점수로 최종 순위

최적의 단일 조합을 찾는다. 과적합 배제.
"""
import os
import sys
import json
from datetime import datetime
from itertools import product
from statistics import mean, stdev
from collections import defaultdict

import pandas as pd

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

# 로버스트 존 집중 그리드
ACTIVATIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
DISTANCES = [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3]


def filter_data_by_date(data, start_date, end_date):
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


def run_single(data, act, dist, skip_years=()):
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
    return {
        "net_profit": r["net_profit"],
        "max_dd": r["max_dd"] if r["max_dd"] > 0 else 0.01,
        "final_balance": r["final_balance"],
        "total_trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "ruined": r["ruined"],
    }


def main():
    print("=" * 110)
    print("HERMES v5 — 최종 트레일링 스탑 최적화 (완벽 검증)")
    print(f"그리드: Act {len(ACTIVATIONS)}종 × Dist {len(DISTANCES)}종 = {len(ACTIVATIONS)*len(DISTANCES)} 조합")
    print("=" * 110)

    print("\n[데이터 로드]")
    full_data = load_all_data()

    # 유효 조합
    combos = [(a, d) for a, d in product(ACTIVATIONS, DISTANCES) if d < a]
    print(f"  유효 조합: {len(combos)}")

    # 기간 정의
    periods = {
        "full_A": (full_data, "2022~2026 전체", ()),
        "full_B": (full_data, "수동 감독 (2023 skip)", (2023,)),
        "2022": (filter_data_by_date(full_data, "2022-01-01", "2023-01-01"), "2022", ()),
        "2023": (filter_data_by_date(full_data, "2023-01-01", "2024-01-01"), "2023 저변동", ()),
        "2024": (filter_data_by_date(full_data, "2024-01-01", "2025-01-01"), "2024", ()),
        "2025": (filter_data_by_date(full_data, "2025-01-01", "2026-01-01"), "2025", ()),
        "2026": (filter_data_by_date(full_data, "2026-01-01", "2026-04-12"), "2026 YTD", ()),
        "wf1_train": (filter_data_by_date(full_data, "2022-01-01", "2024-01-01"), "WF1 train 22-23", ()),
        "wf1_test":  (filter_data_by_date(full_data, "2024-01-01", "2026-04-12"), "WF1 test 24-26", ()),
        "wf2_train": (filter_data_by_date(full_data, "2022-01-01", "2025-01-01"), "WF2 train 22-24", ()),
        "wf2_test":  (filter_data_by_date(full_data, "2025-01-01", "2026-04-12"), "WF2 test 25-26", ()),
    }

    print(f"  기간 수: {len(periods)}")
    print(f"  총 실행: {len(combos) * len(periods)}")

    # 실행
    results = {}  # (act, dist) → {period_key: stats}
    for (act, dist) in combos:
        results[(act, dist)] = {}

    total_runs = len(combos) * len(periods)
    run_count = 0

    for pkey, (pdata, plabel, skip) in periods.items():
        print(f"\n[{pkey}] {plabel} 실행 중...")
        for (act, dist) in combos:
            r = run_single(pdata, act, dist, skip_years=skip)
            results[(act, dist)][pkey] = r
            run_count += 1
        print(f"  완료: {run_count}/{total_runs}")

    # === 메트릭 계산 ===
    print("\n[메트릭 계산]")

    metrics = {}
    annual_keys = ["2022", "2023", "2024", "2025", "2026"]

    for (act, dist), periods_data in results.items():
        full_a = periods_data["full_A"]
        full_b = periods_data["full_B"]

        if full_a["ruined"] or full_b["ruined"]:
            continue

        # 연도별 순이익
        annual_profits = [periods_data[y]["net_profit"] for y in annual_keys]
        annual_drawdowns = [periods_data[y]["max_dd"] for y in annual_keys]
        annual_win_rates = [periods_data[y]["win_rate"] for y in annual_keys
                            if periods_data[y]["total_trades"] > 0]

        worst_year_pnl = min(annual_profits)
        best_year_pnl = max(annual_profits)
        years_positive = sum(1 for p in annual_profits if p > 0)
        worst_year_dd = max(annual_drawdowns)

        wr_mean = mean(annual_win_rates) if annual_win_rates else 0
        wr_std = stdev(annual_win_rates) if len(annual_win_rates) > 1 else 0

        # WF 성과
        wf1_train_pnl = periods_data["wf1_train"]["net_profit"]
        wf1_test_pnl = periods_data["wf1_test"]["net_profit"]
        wf2_train_pnl = periods_data["wf2_train"]["net_profit"]
        wf2_test_pnl = periods_data["wf2_test"]["net_profit"]

        metrics[(act, dist)] = {
            "full_A_profit": full_a["net_profit"],
            "full_A_dd": full_a["max_dd"],
            "full_A_calmar": full_a["net_profit"] / full_a["max_dd"],
            "full_A_winrate": full_a["win_rate"],
            "full_B_profit": full_b["net_profit"],
            "full_B_dd": full_b["max_dd"],
            "full_B_calmar": full_b["net_profit"] / full_b["max_dd"],
            "full_B_winrate": full_b["win_rate"],
            "worst_year_pnl": worst_year_pnl,
            "best_year_pnl": best_year_pnl,
            "years_positive": years_positive,
            "worst_year_dd": worst_year_dd,
            "wr_std": wr_std,
            "wf1_train": wf1_train_pnl,
            "wf1_test": wf1_test_pnl,
            "wf2_train": wf2_train_pnl,
            "wf2_test": wf2_test_pnl,
            "annual": annual_profits,
        }

    # === 랭킹 ===
    def rank_by(metric, reverse=True):
        sorted_keys = sorted(metrics.keys(), key=lambda k: metrics[k][metric], reverse=reverse)
        return {k: i + 1 for i, k in enumerate(sorted_keys)}

    r_profit_a = rank_by("full_A_profit")
    r_profit_b = rank_by("full_B_profit")
    r_dd_a = rank_by("full_A_dd", reverse=False)
    r_dd_b = rank_by("full_B_dd", reverse=False)
    r_calmar_a = rank_by("full_A_calmar")
    r_calmar_b = rank_by("full_B_calmar")
    r_worst_year = rank_by("worst_year_pnl")
    r_years_positive = rank_by("years_positive")
    r_worst_year_dd = rank_by("worst_year_dd", reverse=False)
    r_wr_std = rank_by("wr_std", reverse=False)
    r_wf1_test = rank_by("wf1_test")
    r_wf2_test = rank_by("wf2_test")

    # === 민감도 ===
    # 각 조합의 이웃 4개 (위/아래 act, 위/아래 dist)와 얼마나 비슷한지
    def neighbor_sensitivity(key):
        act, dist = key
        neighbors = []
        # 인접 act
        act_idx = ACTIVATIONS.index(act)
        for di in [-1, 1]:
            if 0 <= act_idx + di < len(ACTIVATIONS):
                n = (ACTIVATIONS[act_idx + di], dist)
                if n in metrics:
                    neighbors.append(metrics[n]["full_B_profit"])
        # 인접 dist
        dist_idx = DISTANCES.index(dist)
        for di in [-1, 1]:
            if 0 <= dist_idx + di < len(DISTANCES):
                n = (act, DISTANCES[dist_idx + di])
                if n in metrics:
                    neighbors.append(metrics[n]["full_B_profit"])

        if not neighbors:
            return 0
        base = metrics[key]["full_B_profit"]
        avg_neighbor = mean(neighbors)
        return (avg_neighbor / base) if base > 0 else 0

    for key in metrics:
        metrics[key]["neighbor_ratio"] = neighbor_sensitivity(key)

    r_sensitivity = rank_by("neighbor_ratio")

    # === 복합 점수 ===
    # 가중치: 수익(30%) + DD(20%) + Calmar(15%) + WF 일관성(15%) + 최악연도(10%) + 민감도(10%)
    composite = {}
    for key in metrics:
        score = (
            r_profit_b[key] * 0.25 +        # 수동 감독 순이익
            r_profit_a[key] * 0.10 +        # 전체 순이익
            r_dd_b[key] * 0.15 +            # 수동 감독 DD
            r_calmar_b[key] * 0.15 +        # 수동 감독 Calmar
            r_wf1_test[key] * 0.075 +       # WF1 test
            r_wf2_test[key] * 0.075 +       # WF2 test
            r_worst_year[key] * 0.10 +      # 최악 연도
            r_wr_std[key] * 0.05 +          # 승률 안정성
            r_sensitivity[key] * 0.05       # 민감도 (이웃 일관성)
        )
        composite[key] = score

    # === 출력 ===
    print("\n" + "=" * 110)
    print("복합 점수 TOP 15 (낮을수록 좋음)")
    print("=" * 110)
    print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'점수':>7} "
          f"{'B순이익':>12} {'B_DD':>6} {'Calmar':>8} "
          f"{'WF1test':>10} {'WF2test':>10} {'최악연':>10} {'+연수':>5}")
    print("-" * 110)

    sorted_composite = sorted(composite.items(), key=lambda x: x[1])
    top_keys = []
    for i, (key, score) in enumerate(sorted_composite[:15]):
        m = metrics[key]
        act, dist = key
        top_keys.append(key)
        print(f"{i+1:>4} {act:>5} {dist:>5} {score:>7.2f} "
              f"${m['full_B_profit']:>+10,.0f} {m['full_B_dd']:>5.1f}% "
              f"{m['full_B_calmar']:>7.0f} "
              f"${m['wf1_test']:>+8,.0f} ${m['wf2_test']:>+8,.0f} "
              f"${m['worst_year_pnl']:>+8,.0f} {m['years_positive']:>4}/5")

    # 세부 top 5 연도별
    print("\n" + "=" * 110)
    print("TOP 5 연도별 상세")
    print("=" * 110)
    print(f"{'Act':>5} {'Dist':>5} {'2022':>11} {'2023':>11} {'2024':>11} "
          f"{'2025':>11} {'2026':>11} {'최악연':>11}")
    print("-" * 90)
    for key, _ in sorted_composite[:5]:
        m = metrics[key]
        act, dist = key
        a = m["annual"]
        worst = min(a)
        print(f"{act:>5} {dist:>5} "
              f"${a[0]:>+9,.0f} ${a[1]:>+9,.0f} ${a[2]:>+9,.0f} "
              f"${a[3]:>+9,.0f} ${a[4]:>+9,.0f} ${worst:>+9,.0f}")

    # 현재 설정
    cur = (1.5, 0.3)
    if cur in metrics:
        rank_cur = sorted_composite.index((cur, composite[cur])) + 1
        m = metrics[cur]
        print(f"\n[현재 1.5/0.3] 복합 순위 {rank_cur}/{len(metrics)}")
        print(f"  B 순이익: ${m['full_B_profit']:+,.0f} (순위 {r_profit_b[cur]})")
        print(f"  B DD: {m['full_B_dd']}% (순위 {r_dd_b[cur]})")
        print(f"  연도별 +연수: {m['years_positive']}/5")

    # === 최종 추천 ===
    winner_key = sorted_composite[0][0]
    winner_metrics = metrics[winner_key]

    print("\n" + "=" * 110)
    print("최종 추천")
    print("=" * 110)
    print(f"  ★ 최적 설정: activation={winner_key[0]}, distance={winner_key[1]}")
    print(f"  복합 점수: {sorted_composite[0][1]:.2f}")
    print(f"\n  [성과]")
    print(f"  전체 4년 순이익: ${winner_metrics['full_A_profit']:+,.0f} (DD {winner_metrics['full_A_dd']}%)")
    print(f"  수동 감독 순이익: ${winner_metrics['full_B_profit']:+,.0f} (DD {winner_metrics['full_B_dd']}%)")
    print(f"  Calmar (수익/DD): {winner_metrics['full_B_calmar']:.0f}")
    print(f"\n  [WF 검증]")
    print(f"  WF1 train(22-23): ${winner_metrics['wf1_train']:+,.0f} / test(24-26): ${winner_metrics['wf1_test']:+,.0f}")
    print(f"  WF2 train(22-24): ${winner_metrics['wf2_train']:+,.0f} / test(25-26): ${winner_metrics['wf2_test']:+,.0f}")
    print(f"\n  [연도별]")
    for y, pnl in zip(annual_keys, winner_metrics["annual"]):
        marker = "✓" if pnl > 0 else "✗"
        print(f"  {y}: ${pnl:+,.0f} {marker}")
    print(f"  플러스 연도: {winner_metrics['years_positive']}/5")
    print(f"  최악 연도: ${winner_metrics['worst_year_pnl']:+,.0f}")
    print(f"\n  [민감도]")
    print(f"  이웃 조합 일관성: {winner_metrics['neighbor_ratio']:.2f} (1.0 근처 = 이웃도 비슷)")

    # 2순위, 3순위 보조 옵션
    print(f"\n  [대안 후보]")
    for i in range(1, 4):
        k, s = sorted_composite[i]
        m = metrics[k]
        print(f"  {i+1}. {k[0]}/{k[1]}: 점수 {s:.2f}, B순이익 ${m['full_B_profit']:+,.0f}, DD {m['full_B_dd']}%, 최악연 ${m['worst_year_pnl']:+,.0f}")

    # === 저장 ===
    output = {
        "timestamp": datetime.now().isoformat(),
        "grid": {"activations": ACTIVATIONS, "distances": DISTANCES, "combos": len(combos)},
        "seed": SEED,
        "weights": {
            "profit_B": 0.25, "profit_A": 0.10,
            "dd_B": 0.15, "calmar_B": 0.15,
            "wf1_test": 0.075, "wf2_test": 0.075,
            "worst_year": 0.10, "wr_std": 0.05,
            "sensitivity": 0.05,
        },
        "winner": {
            "activation": winner_key[0],
            "distance": winner_key[1],
            "composite_score": sorted_composite[0][1],
            "metrics": winner_metrics,
        },
        "top_15": [
            {
                "rank": i + 1,
                "activation": k[0],
                "distance": k[1],
                "composite_score": s,
                **metrics[k],
            }
            for i, (k, s) in enumerate(sorted_composite[:15])
        ],
        "current_1.5_0.3": {
            "rank": sorted_composite.index(((1.5, 0.3), composite[(1.5, 0.3)])) + 1 if (1.5, 0.3) in metrics else None,
            "metrics": metrics.get((1.5, 0.3)),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "v5_final_optimization.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
