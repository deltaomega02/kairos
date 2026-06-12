#!/usr/bin/env python3
"""V13 vs V14 환경별 분리 백테스트.

기간:
- 2022 베어 (1/1 ~ 12/31): BTC 47k→16k 대폭락
- 2023 횡보+회복 (1/1 ~ 12/31): 저변동 후 반등
- 2024 강세 (1/1 ~ 12/31): BTC 30k→100k+
- 2025 H1 (1/1 ~ 6/30): 강세 지속
- 2025 H2 (7/1 ~ 12/31): 베어 시작
- 2026 (1/1 ~ 4/26): 현재 베어+저변동 (운영자 운영)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# v13_vs_v14_full.py 모듈 import
import v13_vs_v14_full as engine

PERIODS = [
    ("2022 베어 폭락",     "2022-01-01", "2022-12-31"),
    ("2023 회복 횡보",     "2023-01-01", "2023-12-31"),
    ("2024 강세장",        "2024-01-01", "2024-12-31"),
    ("2025 H1 강세",       "2025-01-01", "2025-06-30"),
    ("2025 H2 베어",       "2025-07-01", "2025-12-31"),
    ("2026 베어+저변동",   "2026-01-01", "2026-04-26"),
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_period(start: str, end: str):
    """특정 기간 V13 vs V14 백테스트."""
    engine.START_DATE = start
    engine.END_DATE = end
    results_v13 = {}
    results_v14 = {}
    for sym in SYMBOLS:
        results_v13[sym] = engine.backtest(sym, mode="V13")
        results_v14[sym] = engine.backtest(sym, mode="V14")
    return results_v13, results_v14


def summarize(results: dict) -> dict:
    """3코인 합산."""
    total_t = sum(r.get("trades", 0) for r in results.values())
    total_w = sum(r.get("wins", 0) for r in results.values())
    total_pct = sum(r.get("total_pct", 0) for r in results.values())
    wr = total_w / total_t * 100 if total_t > 0 else 0
    return {
        "trades": total_t,
        "wins": total_w,
        "win_rate": wr,
        "total_pct": total_pct,
    }


def main():
    print("=" * 90)
    print("V13 vs V14 환경별 분리 백테스트")
    print("=" * 90)
    print()

    all_results = {}

    for label, start, end in PERIODS:
        print(f"[{label}] {start} ~ {end} 진행 중...")
        v13, v14 = run_period(start, end)
        s13 = summarize(v13)
        s14 = summarize(v14)
        all_results[label] = {
            "period": f"{start} ~ {end}",
            "v13": s13,
            "v14": s14,
            "v13_per_coin": v13,
            "v14_per_coin": v14,
        }

    print()
    print("=" * 90)
    print(f"{'기간':>20} | {'MODE':>5} | {'거래':>5} {'승':>4} {'승률':>6} | {'총%':>10} | {'V14-V13':>8}")
    print("-" * 90)

    for label, r in all_results.items():
        s13 = r["v13"]
        s14 = r["v14"]
        diff = s14["total_pct"] - s13["total_pct"]
        print(f"{label:>20} | {'V13':>5} | {s13['trades']:>5} {s13['wins']:>4} {s13['win_rate']:>5.1f}% | {s13['total_pct']:>+9.2f}% |")
        print(f"{'':>20} | {'V14':>5} | {s14['trades']:>5} {s14['wins']:>4} {s14['win_rate']:>5.1f}% | {s14['total_pct']:>+9.2f}% | {diff:>+7.2f}%")
        print()

    # 환경별 V14 우위 분석
    print("=" * 90)
    print("V14가 V13보다 나은 환경 (총% 기준)")
    print("=" * 90)
    win_periods = []
    lose_periods = []
    for label, r in all_results.items():
        diff = r["v14"]["total_pct"] - r["v13"]["total_pct"]
        if diff > 0:
            win_periods.append((label, diff))
        else:
            lose_periods.append((label, diff))

    print("V14 우위 환경:")
    for label, diff in sorted(win_periods, key=lambda x: -x[1]):
        print(f"  ✅ {label}: +{diff:.2f}%")

    print("\nV14 열위 환경:")
    for label, diff in sorted(lose_periods, key=lambda x: x[1]):
        print(f"  ❌ {label}: {diff:.2f}%")

    # 결론
    print()
    print("=" * 90)
    print("판정")
    print("=" * 90)
    bear_2022 = all_results.get("2022 베어 폭락", {})
    bear_2025 = all_results.get("2025 H2 베어", {})
    current_2026 = all_results.get("2026 베어+저변동", {})

    if bear_2022:
        d = bear_2022["v14"]["total_pct"] - bear_2022["v13"]["total_pct"]
        print(f"2022 베어 (BTC -65%): V14 vs V13 차이 {d:+.2f}%")
    if bear_2025:
        d = bear_2025["v14"]["total_pct"] - bear_2025["v13"]["total_pct"]
        print(f"2025 H2 베어:         V14 vs V13 차이 {d:+.2f}%")
    if current_2026:
        d = current_2026["v14"]["total_pct"] - current_2026["v13"]["total_pct"]
        print(f"2026 현재 (운영):     V14 vs V13 차이 {d:+.2f}%")

    out_path = "/Users/sue/Projects/HERMES/backtest/v13_v14_by_period.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
