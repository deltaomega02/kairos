#!/usr/bin/env python3
"""
v5 트레일링 스탑 전수 그리드 서치
===================================
현재: activation 1.5%, distance 0.3% (v3 확정)
문제 관찰: 2026-04-13 BTC SHORT, 평가익 최대 +1.4%에서 트레일링 미작동 → SL 반납

목표: activation을 더 빠르게/다양하게 테스트
- activation 14 종류 × distance 11 종류 = 154 조합 (dist < act 필터)
- 유효 ~110 조합
- 2개 시나리오: 전체 기간 / 수동 감독 (2023 skip)
- 시드 $600, 실제 HERMES shared-balance 구조
"""
import os
import sys
import json
import time
from datetime import datetime
from itertools import product

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

# 트레일링 그리드
ACTIVATIONS = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
DISTANCES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]


def run_one(data, act, dist, skip_years):
    """단일 조합 실행"""
    trades_result = run_shared_backtest(
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

    trades = trades_result.get("_trades", [])
    if not trades:
        return None

    # 상세 통계
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    # 트레일링 청산 비율
    trailing_exits = sum(1 for t in trades if t.get("reason") == "TRAILING")
    sl_exits = sum(1 for t in trades if t.get("reason") == "SL")
    tp_exits = sum(1 for t in trades if t.get("reason") == "TP")

    # 방향별
    long_trades = [t for t in trades if t.get("direction") == "LONG"]
    short_trades = [t for t in trades if t.get("direction") == "SHORT"]
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)
    long_wr = sum(1 for t in long_trades if t["pnl"] > 0) / len(long_trades) * 100 if long_trades else 0
    short_wr = sum(1 for t in short_trades if t["pnl"] > 0) / len(short_trades) * 100 if short_trades else 0

    # 최대 연패
    max_consec_loss = 0
    cur = 0
    for t in trades:
        if t["pnl"] <= 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

    return {
        "activation": act,
        "distance": dist,
        "final_balance": trades_result["final_balance"],
        "net_profit": trades_result["net_profit"],
        "net_pct": trades_result["net_pct"],
        "max_dd": trades_result["max_dd"],
        "total_trades": trades_result["total_trades"],
        "win_rate": trades_result["win_rate"],
        "sl_count": sl_exits,
        "tp_count": tp_exits,
        "trailing_count": trailing_exits,
        "trailing_pct": round(trailing_exits / len(trades) * 100, 1),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "long_wr": round(long_wr, 1),
        "short_wr": round(short_wr, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_consec_loss": max_consec_loss,
        "ruined": trades_result["ruined"],
    }


def run_baseline_no_trailing(data, skip_years):
    """트레일링 없음 (baseline)"""
    r = run_shared_backtest(
        data, BEST_PARAMS, SEED,
        use_funding=True,
        trailing_activation=999,  # 사실상 비활성화
        trailing_distance=1.0,
        block_sol_long=True,
        skip_years=skip_years,
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        use_trailing=False if False else True,  # v4 엔진은 항상 켜져 있음 → 999로 무력화
    )
    return {
        "activation": "OFF",
        "distance": "-",
        "final_balance": r["final_balance"],
        "net_profit": r["net_profit"],
        "net_pct": r["net_pct"],
        "max_dd": r["max_dd"],
        "total_trades": r["total_trades"],
        "win_rate": r["win_rate"],
    }


def print_top(results, label, metric="net_profit", n=20):
    valid = [r for r in results if r is not None and not r["ruined"]]
    sorted_r = sorted(valid, key=lambda x: x[metric], reverse=True)
    print(f"\n{'=' * 110}")
    print(f"[{label}] 상위 {n} — 정렬: {metric}")
    print(f"{'=' * 110}")
    print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'거래':>6} {'승률':>6} "
          f"{'순이익':>11} {'배수':>7} {'DD':>7} {'TP%':>6} {'TR%':>6} "
          f"{'LONG':>10} {'SHORT':>10}")
    print("-" * 110)
    for i, r in enumerate(sorted_r[:n]):
        print(f"{i+1:>4} {r['activation']:>5} {r['distance']:>5} "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"${r['net_profit']:>+9,.0f} {r['final_balance']/SEED:>5.1f}x "
              f"{r['max_dd']:>6.1f}% "
              f"{r['tp_count']/r['total_trades']*100:>5.1f}% "
              f"{r['trailing_pct']:>5.1f}% "
              f"${r['long_pnl']:>+8,.0f} ${r['short_pnl']:>+8,.0f}")


def main():
    print("=" * 110)
    print("HERMES v5 — 트레일링 스탑 전수 그리드 서치")
    print(f"시드 ${SEED:,.0f} | shared-balance | 4년 (2022~2026)")
    print(f"그리드: Act {len(ACTIVATIONS)}종 × Dist {len(DISTANCES)}종 = {len(ACTIVATIONS)*len(DISTANCES)} 조합")
    print(f"필터: distance < activation 만 유효")
    print("=" * 110)

    print("\n[데이터 로드]")
    data = load_all_data()
    print(f"  로드 완료: {len(data)} 데이터셋")

    # 유효 조합
    valid_combos = [(a, d) for a, d in product(ACTIVATIONS, DISTANCES) if d < a]
    print(f"\n유효 조합: {len(valid_combos)}")
    print(f"예상 시간: 조합당 ~8초 × 2시나리오 = ~{len(valid_combos) * 16}초")

    # 시나리오 1: 전체 기간
    print(f"\n{'=' * 110}")
    print(f"[시나리오 A] 전체 4년 (2022~2026)")
    print(f"{'=' * 110}")
    t0 = time.time()
    results_full = []
    for i, (act, dist) in enumerate(valid_combos):
        r = run_one(data, act, dist, skip_years=())
        if r:
            results_full.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(valid_combos) - i - 1)
            print(f"  진행: {i+1}/{len(valid_combos)} | 경과 {elapsed:.0f}s | 예상 남은 {eta:.0f}s")

    # 시나리오 2: 수동 감독 (2023 skip)
    print(f"\n{'=' * 110}")
    print(f"[시나리오 B] 수동 감독 (2023 skip)")
    print(f"{'=' * 110}")
    t0 = time.time()
    results_claude = []
    for i, (act, dist) in enumerate(valid_combos):
        r = run_one(data, act, dist, skip_years=(2023,))
        if r:
            results_claude.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(valid_combos) - i - 1)
            print(f"  진행: {i+1}/{len(valid_combos)} | 경과 {elapsed:.0f}s | 예상 남은 {eta:.0f}s")

    # 출력: 여러 기준으로 상위
    print_top(results_full, "A. 전체 4년 — 순이익 기준", "net_profit", 20)
    print_top(results_full, "A. 전체 4년 — DD 최저 기준",
              "max_dd", 15)

    # DD는 낮을수록 좋으므로 역정렬 필요
    valid_full = [r for r in results_full if not r["ruined"]]
    print(f"\n{'=' * 110}")
    print(f"[A. 전체 4년] DD 낮은 순 TOP 15")
    print(f"{'=' * 110}")
    print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'거래':>6} {'승률':>6} "
          f"{'순이익':>11} {'DD':>7}")
    print("-" * 70)
    for i, r in enumerate(sorted(valid_full, key=lambda x: x["max_dd"])[:15]):
        print(f"{i+1:>4} {r['activation']:>5} {r['distance']:>5} "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"${r['net_profit']:>+9,.0f} {r['max_dd']:>6.1f}%")

    # Sharpe-like: PnL / DD
    print(f"\n{'=' * 110}")
    print(f"[A. 전체 4년] PnL/DD 효율 TOP 15")
    print(f"{'=' * 110}")
    print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'순이익':>11} {'DD':>7} {'효율':>10}")
    print("-" * 60)
    eff = [(r["net_profit"] / r["max_dd"] if r["max_dd"] > 0 else 0, r) for r in valid_full]
    for i, (score, r) in enumerate(sorted(eff, key=lambda x: x[0], reverse=True)[:15]):
        print(f"{i+1:>4} {r['activation']:>5} {r['distance']:>5} "
              f"${r['net_profit']:>+9,.0f} {r['max_dd']:>6.1f}% {score:>+9,.0f}")

    print_top(results_claude, "B. 수동 감독 — 순이익 기준", "net_profit", 20)

    valid_claude = [r for r in results_claude if not r["ruined"]]
    print(f"\n{'=' * 110}")
    print(f"[B. 수동 감독] DD 낮은 순 TOP 15")
    print(f"{'=' * 110}")
    print(f"{'순위':>4} {'Act':>5} {'Dist':>5} {'거래':>6} {'승률':>6} "
          f"{'순이익':>11} {'DD':>7}")
    print("-" * 70)
    for i, r in enumerate(sorted(valid_claude, key=lambda x: x["max_dd"])[:15]):
        print(f"{i+1:>4} {r['activation']:>5} {r['distance']:>5} "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"${r['net_profit']:>+9,.0f} {r['max_dd']:>6.1f}%")

    # 현재 시스템 (1.5/0.3) 위치
    print(f"\n{'=' * 110}")
    print(f"현재 시스템 (act=1.5, dist=0.3) 위치 확인")
    print(f"{'=' * 110}")
    for label, rset in [("A. 전체", valid_full), ("B. 감독", valid_claude)]:
        cur = next((r for r in rset if r["activation"] == 1.5 and r["distance"] == 0.3), None)
        if cur:
            rank_profit = sorted(rset, key=lambda x: x["net_profit"], reverse=True).index(cur) + 1
            rank_dd = sorted(rset, key=lambda x: x["max_dd"]).index(cur) + 1
            eff_score = cur["net_profit"] / cur["max_dd"] if cur["max_dd"] > 0 else 0
            rank_eff = sorted(rset, key=lambda x: x["net_profit"] / x["max_dd"] if x["max_dd"] > 0 else 0, reverse=True).index(cur) + 1
            print(f"  [{label}] 순이익 ${cur['net_profit']:+,.0f} (순위 {rank_profit}/{len(rset)})"
                  f" | DD {cur['max_dd']}% (순위 {rank_dd})"
                  f" | 효율 {eff_score:+,.0f} (순위 {rank_eff})")

    # JSON 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "activations": ACTIVATIONS,
            "distances": DISTANCES,
            "valid_combos": len(valid_combos),
        },
        "scenario_A_full": sorted(results_full, key=lambda x: x["net_profit"], reverse=True),
        "scenario_B_claude": sorted(results_claude, key=lambda x: x["net_profit"], reverse=True),
    }

    out_path = os.path.join(RESULTS_DIR, "v5_trailing_exhaustive.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
