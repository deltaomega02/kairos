#!/usr/bin/env python3
"""
백테스팅 v3.0 — 종합 테스트
============================
Stage 1: 각 기능별 효과 검증 (펀딩/MTF/시간대/동적리스크)
Stage 2: 코인별 최적 파라미터
Stage 3: 통합 최적 조합 탐색
Stage 4: 트레일링 재최적화 (v3 기준)
"""

import os
import sys
import json
import time
import itertools
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data, run_multi_v3, SYMBOLS, DATA_DIR
from v3_engine import run_backtest_v3, evaluate_signal_v3
from v3_engine import align_funding_to_entry, get_15m_confirm
from comprehensive_backtest import (
    DEFAULT_PARAMS, compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry, INITIAL_BALANCE,
)

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1위 파라미터 (v2에서 검증)
BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}

# 트레일링 (v2 검증)
BEST_TRAILING = {"trailing_activation": 3.0, "trailing_distance": 0.5}


def print_result(name, r):
    print(f"  {name:<40} | {r['total']:>5}거래 | 승률{r['win_rate']:>5.1f}% | "
          f"{r['return_pct']:>+8.1f}% | DD{r['max_dd']:>5.1f}%")


# ================================================================
# Stage 1: 기능별 효과 검증
# ================================================================

def stage1_feature_validation(data):
    print("\n" + "=" * 90)
    print("Stage 1: 각 기능별 효과 검증")
    print("=" * 90)

    results = {}

    # 베이스라인 (v2 1위 + 트레일링)
    print("\n[베이스라인] v2 1위 + 트레일링 3%/0.5%")
    r = run_multi_v3(BEST_PARAMS, data, use_funding=False, use_mtf=False, **BEST_TRAILING)
    results["baseline"] = r
    print_result("베이스라인", r)

    # 펀딩 반영
    print("\n[펀딩] 펀딩 레이트 점수 반영")
    r = run_multi_v3(BEST_PARAMS, data, use_funding=True, use_mtf=False, **BEST_TRAILING)
    results["funding"] = r
    print_result("펀딩 반영", r)

    # 15분봉 MTF
    print("\n[MTF] 15분봉 RSI 확인")
    r = run_multi_v3(BEST_PARAMS, data, use_funding=False, use_mtf=True, **BEST_TRAILING)
    results["mtf"] = r
    print_result("15분봉 MTF", r)

    # 동적 리스크
    print("\n[동적리스크] 연패 시 사이즈 축소")
    r = run_multi_v3(BEST_PARAMS, data, use_funding=False, use_mtf=False,
                     use_dynamic_risk=True, **BEST_TRAILING)
    results["dynamic_risk"] = r
    print_result("동적 리스크", r)

    # 시간대 필터 - 아시아장 (0-8 UTC)
    print("\n[시간대] 아시아장만 (0-8 UTC)")
    r = run_multi_v3(BEST_PARAMS, data, session_filter=list(range(0, 8)), **BEST_TRAILING)
    results["asia_only"] = r
    print_result("아시아장만", r)

    # 시간대 필터 - 유럽장 (8-16 UTC)
    print("\n[시간대] 유럽장만 (8-16 UTC)")
    r = run_multi_v3(BEST_PARAMS, data, session_filter=list(range(8, 16)), **BEST_TRAILING)
    results["europe_only"] = r
    print_result("유럽장만", r)

    # 시간대 필터 - 미국장 (16-24 UTC)
    print("\n[시간대] 미국장만 (16-24 UTC)")
    r = run_multi_v3(BEST_PARAMS, data, session_filter=list(range(16, 24)), **BEST_TRAILING)
    results["us_only"] = r
    print_result("미국장만", r)

    # 전부 조합
    print("\n[전부] 펀딩 + MTF + 동적리스크")
    r = run_multi_v3(BEST_PARAMS, data, use_funding=True, use_mtf=True,
                     use_dynamic_risk=True, **BEST_TRAILING)
    results["all_features"] = r
    print_result("전부 적용", r)

    return results


# ================================================================
# Stage 2: 코인별 최적 파라미터
# ================================================================

def stage2_per_coin_optimization(data):
    print("\n" + "=" * 90)
    print("Stage 2: 코인별 개별 최적화")
    print("=" * 90)

    # 코인별 그리드
    ema_combos = [(5, 15), (5, 18), (5, 21), (7, 18), (7, 21), (9, 21), (9, 26)]
    sl_vals = [1.0, 1.5, 2.0, 2.5]
    tp_vals = [2.5, 3.0, 4.0, 5.0]
    adx_vals = [25, 30, 35]
    pb_vals = [0.8, 1.0, 1.5, 2.0]

    per_coin_best = {}

    for sym in SYMBOLS:
        print(f"\n  ▶ {sym} 최적화 중...")
        best = None

        for (ef, es), sl, tp, adx, pb in itertools.product(ema_combos, sl_vals, tp_vals, adx_vals, pb_vals):
            params = {
                **DEFAULT_PARAMS,
                "ema_fast": ef, "ema_slow": es,
                "sl_atr_mult": sl, "tp_rr_ratio": tp,
                "adx_enter_trending": adx,
                "pullback_ema_dist_pct": pb,
                "entry_score_threshold": 40,
            }

            # 단일 코인 백테스트
            entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), params)
            regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
            re = BacktestRegimeEngine(params)
            regimes = [re.update(row) for _, row in regime_df.iterrows()]
            regime_df["regime"] = regimes
            rm = align_regime_to_entry(regime_df, entry_df)

            funding_df = data.get(f"{sym}_funding", pd.DataFrame())
            fm = align_funding_to_entry(funding_df, entry_df)
            rsi_15_series = pd.Series([50.0] * len(entry_df), index=entry_df.index)

            trades = run_backtest_v3(
                entry_df, rm, fm, rsi_15_series, params,
                symbol=sym, **BEST_TRAILING,
            )

            if not trades:
                continue

            total_pnl = sum(t["pnl"] for t in trades)
            ret = total_pnl / INITIAL_BALANCE * 100
            wins = sum(1 for t in trades if t["pnl"] > 0)
            wr = wins / len(trades) * 100

            if best is None or ret > best["return_pct"]:
                best = {
                    "params": {"ema_fast": ef, "ema_slow": es, "sl": sl, "tp": tp, "adx": adx, "pb": pb},
                    "total": len(trades),
                    "win_rate": round(wr, 1),
                    "pnl": round(total_pnl, 2),
                    "return_pct": round(ret, 1),
                }

        per_coin_best[sym] = best
        if best:
            p = best["params"]
            print(f"    {sym} 최적: EMA{p['ema_fast']}/{p['ema_slow']} SL{p['sl']} TP{p['tp']} ADX{p['adx']} PB{p['pb']} → {best['return_pct']:+.1f}% ({best['total']}거래, {best['win_rate']}%)")

    return per_coin_best


# ================================================================
# Stage 3: 트레일링 재최적화
# ================================================================

def stage3_trailing_reoptimization(data):
    print("\n" + "=" * 90)
    print("Stage 3: 트레일링 재최적화 (펀딩 반영 기준)")
    print("=" * 90)

    activations = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    distances = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    results = []
    for act in activations:
        for dist in distances:
            if dist >= act:
                continue
            r = run_multi_v3(
                BEST_PARAMS, data,
                use_funding=True,  # 펀딩 포함
                trailing_activation=act, trailing_distance=dist,
            )
            r["activation"] = act
            r["distance"] = dist
            results.append(r)

    # 트레일링 없음 비교
    r_none = run_multi_v3(BEST_PARAMS, data, use_funding=True, use_trailing=False)
    r_none["activation"] = "없음"
    r_none["distance"] = "-"
    results.append(r_none)

    results.sort(key=lambda x: x["return_pct"], reverse=True)

    print(f"\n  {'순위':>4} {'Act':>6} {'Dist':>6} {'거래':>6} {'승률':>7} {'수익률':>9} {'DD':>7}")
    print("  " + "-" * 55)
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:>4} {str(r['activation']):>6} {str(r['distance']):>6} "
              f"{r['total']:>6} {r['win_rate']:>6.1f}% {r['return_pct']:>+8.1f}% {r['max_dd']:>6.1f}%")

    return results


# ================================================================
# Stage 4: 통합 최적
# ================================================================

def stage4_integrated_best(data, stage1_results, stage3_results):
    print("\n" + "=" * 90)
    print("Stage 4: 통합 최적 조합")
    print("=" * 90)

    # Stage 3에서 최고 트레일링 찾기
    best_trailing = next((r for r in stage3_results if r.get("activation") != "없음"), None)

    if best_trailing:
        act = best_trailing["activation"]
        dist = best_trailing["distance"]
        print(f"\n  최적 트레일링: Act={act}, Dist={dist}")
    else:
        act, dist = 3.0, 0.5
        print(f"\n  기본 트레일링 유지: Act={act}, Dist={dist}")

    # 펀딩 / MTF / 동적리스크 조합 테스트
    combos = [
        ("baseline", {"use_funding": False, "use_mtf": False, "use_dynamic_risk": False}),
        ("+funding", {"use_funding": True, "use_mtf": False, "use_dynamic_risk": False}),
        ("+mtf", {"use_funding": False, "use_mtf": True, "use_dynamic_risk": False}),
        ("+dynamic", {"use_funding": False, "use_mtf": False, "use_dynamic_risk": True}),
        ("+funding+mtf", {"use_funding": True, "use_mtf": True, "use_dynamic_risk": False}),
        ("+funding+dynamic", {"use_funding": True, "use_mtf": False, "use_dynamic_risk": True}),
        ("+mtf+dynamic", {"use_funding": False, "use_mtf": True, "use_dynamic_risk": True}),
        ("ALL", {"use_funding": True, "use_mtf": True, "use_dynamic_risk": True}),
    ]

    results = {}
    for name, opts in combos:
        r = run_multi_v3(
            BEST_PARAMS, data,
            trailing_activation=act, trailing_distance=dist,
            **opts,
        )
        results[name] = r
        print_result(name, r)

    # 최고 선정
    best_name = max(results.keys(), key=lambda k: results[k]["return_pct"])
    print(f"\n  ✓ 최고 조합: {best_name}")
    print(f"    → {results[best_name]['return_pct']:+.1f}% | {results[best_name]['total']}거래 | DD {results[best_name]['max_dd']}%")

    return results


# ================================================================
# 메인
# ================================================================

def main():
    print("=" * 90)
    print("HERMES 백테스팅 v3.0 — 종합 테스트")
    print("=" * 90)

    start_time = time.time()

    print("\n[데이터 로드]")
    data = load_all_data()
    print(f"  로드된 데이터셋: {len(data)}")
    for k in sorted(data.keys()):
        print(f"    {k}: {len(data[k])}개")

    # Stage 1
    stage1 = stage1_feature_validation(data)

    # Stage 2
    stage2 = stage2_per_coin_optimization(data)

    # Stage 3
    stage3 = stage3_trailing_reoptimization(data)

    # Stage 4
    stage4 = stage4_integrated_best(data, stage1, stage3)

    elapsed = time.time() - start_time

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed / 60, 1),
        "stage1_features": {k: {kk: vv for kk, vv in v.items() if kk != "coin_stats"}
                            for k, v in stage1.items()},
        "stage2_per_coin": stage2,
        "stage3_trailing_top15": [{k: v for k, v in r.items() if k != "coin_stats"}
                                   for r in stage3[:15]],
        "stage4_integrated": {k: {kk: vv for kk, vv in v.items() if kk != "coin_stats"}
                              for k, v in stage4.items()},
    }

    path = os.path.join(RESULTS_DIR, "v3_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {path}")
    print(f"소요 시간: {elapsed/60:.1f}분")

    print("\n" + "=" * 90)
    print("v3.0 종합 테스트 완료")
    print("=" * 90)


if __name__ == "__main__":
    main()
