#!/usr/bin/env python3
"""V13 점수 임계값 백테스트.

ENTRY_SCORE_THRESHOLD 40 → 50 → 60 → 70 비교.
약신호 차단 시 거래 품질 / 수익률 변화 측정.

운영자 현재 BTC SHORT는 점수 50 진입 (임계 40 통과). 만약 임계 60이면 차단됨.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v13_vs_v14_full as engine

PERIODS = [
    ("2022 베어",       "2022-01-01", "2022-12-31"),
    ("2023 횡보",       "2023-01-01", "2023-12-31"),
    ("2024 강세",       "2024-01-01", "2024-12-31"),
    ("2025 H2 베어",    "2025-07-01", "2025-12-31"),
    ("2026 현재",       "2026-01-01", "2026-04-26"),
    ("4년 통합",        "2022-01-01", "2026-04-26"),
]

THRESHOLDS = [40, 50, 60, 70]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run(threshold: int, start: str, end: str):
    engine.ENTRY_SCORE_THRESHOLD = threshold
    engine.START_DATE = start
    engine.END_DATE = end
    results = {}
    for sym in SYMBOLS:
        results[sym] = engine.backtest(sym, mode="V13")
    total_t = sum(r.get("trades", 0) for r in results.values())
    total_w = sum(r.get("wins", 0) for r in results.values())
    total_pct = sum(r.get("total_pct", 0) for r in results.values())
    wr = total_w / total_t * 100 if total_t > 0 else 0
    avg_per_trade = total_pct / total_t if total_t > 0 else 0
    return {
        "trades": total_t,
        "wins": total_w,
        "win_rate": wr,
        "total_pct": total_pct,
        "avg_per_trade_pct": avg_per_trade,
    }


def main():
    print("=" * 90)
    print("V13 점수 임계값 백테스트 (V13 모드, RANGE_REVERSION 비활성)")
    print("=" * 90)
    print()

    all_results = {}

    for label, start, end in PERIODS:
        print(f"[{label}] {start} ~ {end}")
        period_data = {}
        for thr in THRESHOLDS:
            r = run(thr, start, end)
            period_data[thr] = r
        all_results[label] = period_data

    print()
    for label, period_data in all_results.items():
        print("=" * 90)
        print(f"{label}")
        print("-" * 90)
        print(f"{'Threshold':>10} | {'거래':>6} {'승':>5} {'승률':>6} | {'총%':>10} | {'평균/거래':>9}")
        print("-" * 90)

        baseline = period_data[40]
        for thr in THRESHOLDS:
            r = period_data[thr]
            trade_diff = r["trades"] - baseline["trades"]
            wr_diff = r["win_rate"] - baseline["win_rate"]
            pct_diff = r["total_pct"] - baseline["total_pct"]
            print(f"{thr:>10} | {r['trades']:>6} {r['wins']:>5} {r['win_rate']:>5.1f}% | "
                  f"{r['total_pct']:>+9.2f}% | {r['avg_per_trade_pct']:>+8.4f}%")
        # 추천
        best_thr = max(THRESHOLDS, key=lambda t: period_data[t]["total_pct"])
        best_avg_thr = max(THRESHOLDS, key=lambda t: period_data[t]["avg_per_trade_pct"])
        print(f"  → 총수익 최대: 임계 {best_thr} (총 {period_data[best_thr]['total_pct']:+.2f}%)")
        print(f"  → 거래당 평균 최대: 임계 {best_avg_thr} (평균 {period_data[best_avg_thr]['avg_per_trade_pct']:+.4f}%)")
        print()

    # 종합 결론
    print("=" * 90)
    print("종합 — 4년 통합 데이터 기준")
    print("=" * 90)
    integ = all_results.get("4년 통합", {})
    if integ:
        for thr in THRESHOLDS:
            r = integ[thr]
            print(f"임계 {thr}: {r['trades']}거래, {r['win_rate']:.1f}% 승률, 총 {r['total_pct']:+.2f}%, 거래당 {r['avg_per_trade_pct']:+.4f}%")

    # 2026 현재 (운영자 운영 환경) 별도 강조
    print()
    print("=" * 90)
    print("2026 현재 환경 (운영자 운영) — 가장 중요")
    print("=" * 90)
    cur = all_results.get("2026 현재", {})
    if cur:
        for thr in THRESHOLDS:
            r = cur[thr]
            print(f"임계 {thr}: {r['trades']}거래, {r['win_rate']:.1f}% 승률, 총 {r['total_pct']:+.2f}%, 거래당 {r['avg_per_trade_pct']:+.4f}%")

    out = "/Users/sue/Projects/HERMES/backtest/v13_score_threshold_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
