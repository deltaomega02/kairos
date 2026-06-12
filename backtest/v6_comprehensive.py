#!/usr/bin/env python3
"""
v6 완전 종합 백테스트
=======================
모든 가능한 파라미터를 계획적으로 검증.

Phase 1: 포지션 수 (2 vs 3) — 핵심 요청
Phase 2: 시드 × 포지션 그리드 (현재 vs 30만원 입금 후)
Phase 3: 리스크/거래 민감도
Phase 4: 최대 레버리지
Phase 5: 진입 스코어 임계값
Phase 6: 일일 거래 한도
Phase 7: 풀백 거리
Phase 8: ADX 진입 임계값
Phase 9: SL ATR 배수
Phase 10: TP R:R 비율
Phase 11: 코인 조합 ablation
Phase 12: 트레일링 미세조정 (포지션 3 환경)
Phase 13: 최종 best + 슬리피지 stress test
Phase 14: 최종 best 연도별 / Walk-forward
"""
import os
import sys
import json
import time
from datetime import datetime
from itertools import product
from copy import deepcopy

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 현재 최적 기준 파라미터
BASE_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
BASE_TRAILING_ACT = 1.2
BASE_TRAILING_DIST = 0.1
BASE_SEED_CURRENT = 580.0       # 현재 잔고 근사
BASE_SEED_AFTER_DEPOSIT = 780.0  # 30만원(~$200) 입금 후
BASE_SLIPPAGE_REALISTIC = 0.05
DAILY_COST_USD = 1150 / 1470


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


def run(data, seed, **overrides):
    """단일 실행 — overrides로 모든 파라미터 제어"""
    params = overrides.pop("params", BASE_PARAMS)
    trail_act = overrides.pop("trailing_activation", BASE_TRAILING_ACT)
    trail_dist = overrides.pop("trailing_distance", BASE_TRAILING_DIST)
    slippage = overrides.pop("slippage_pct", BASE_SLIPPAGE_REALISTIC)
    skip_years = overrides.pop("skip_years", (2023,))  # 기본 수동 감독

    r = run_shared_backtest(
        data, params, float(seed),
        use_funding=True,
        trailing_activation=trail_act,
        trailing_distance=trail_dist,
        block_sol_long=True,
        skip_years=skip_years,
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        slippage_pct=slippage,
        **overrides,
    )
    return {
        "net_profit": r["net_profit"],
        "net_pct": r["net_pct"],
        "max_dd": r["max_dd"],
        "final_balance": r["final_balance"],
        "total_trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "ruined": r["ruined"],
    }


def phase_header(n, title):
    print(f"\n{'='*110}\nPhase {n}: {title}\n{'='*110}")


def print_row(label, r, extra=""):
    if r["ruined"]:
        print(f"  {label:<40} RUINED {extra}")
    else:
        print(f"  {label:<40} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% "
              f"| 순이익 ${r['net_profit']:>+10,.0f} | DD {r['max_dd']:>5.1f}% {extra}")


def main():
    t_start = time.time()
    print("="*110)
    print("HERMES v6 — 완전 종합 백테스트")
    print(f"기준: trailing 1.2/0.1, 수동 감독(2023 skip), 슬리피지 0.05%, shared-balance")
    print(f"현재 시드: ${BASE_SEED_CURRENT:.0f} | 입금 후: ${BASE_SEED_AFTER_DEPOSIT:.0f}")
    print("="*110)

    print("\n[데이터 로드]")
    data = load_all_data()
    print(f"  로드 완료: {len(data)} 데이터셋")

    results = {}

    # ================================
    # Phase 1: 포지션 수 (2 vs 3)
    # ================================
    phase_header(1, "동시 포지션 수 (2 vs 3) — 현재 $580 기준")
    results["phase1"] = {}
    for max_sim in [2, 3]:
        for slip in [0.0, 0.05]:
            r = run(data, BASE_SEED_CURRENT, max_simultaneous=max_sim, slippage_pct=slip)
            k = f"pos{max_sim}_slip{slip}"
            results["phase1"][k] = r
            print_row(f"포지션 {max_sim}개 / slip {slip}%", r)

    # ================================
    # Phase 2: 시드 × 포지션 그리드
    # ================================
    phase_header(2, "시드 × 포지션 그리드 (슬리피지 0.05%)")
    seeds = [500, 580, 680, 780, 880, 1000]
    positions = [2, 3]
    results["phase2"] = {}
    print(f"  {'':5} " + "".join(f"{p}pos           " for p in positions))
    for seed in seeds:
        row = f"  ${seed:>4}:"
        for max_sim in positions:
            r = run(data, seed, max_simultaneous=max_sim, slippage_pct=0.05)
            k = f"seed{seed}_pos{max_sim}"
            results["phase2"][k] = r
            if r["ruined"]:
                val = "RUINED       "
            else:
                val = f"${r['net_profit']:>+8,.0f} {r['max_dd']:>4.1f}%"
            row += f" {val}"
        print(row)

    # ================================
    # Phase 3: 리스크/거래 민감도
    # ================================
    phase_header(3, "리스크/거래 민감도 ($780 + 3포지션)")
    results["phase3"] = {}
    for risk in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, risk_per_trade=risk)
        results["phase3"][f"risk{risk}"] = r
        print_row(f"risk {risk*100:.1f}%/거래", r)

    # ================================
    # Phase 4: 최대 레버리지
    # ================================
    phase_header(4, "최대 레버리지 ($780 + 3포지션)")
    results["phase4"] = {}
    for lev in [3, 5, 7, 10, 15, 20]:
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, max_leverage=lev)
        results["phase4"][f"lev{lev}"] = r
        print_row(f"max_leverage {lev}x", r)

    # ================================
    # Phase 5: 진입 스코어 임계값
    # ================================
    phase_header(5, "진입 스코어 임계값 ($780 + 3포지션)")
    results["phase5"] = {}
    for score in [30, 40, 50, 60, 70, 80]:
        p = deepcopy(BASE_PARAMS)
        p["entry_score_threshold"] = score
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, params=p)
        results["phase5"][f"score{score}"] = r
        print_row(f"score threshold {score}", r)

    # ================================
    # Phase 6: 일일 거래 한도
    # ================================
    phase_header(6, "일일 거래 한도 ($780 + 3포지션)")
    results["phase6"] = {}
    for cap in [3, 5, 7, 10, 15, 20, 30]:
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, max_daily_trades=cap)
        results["phase6"][f"daily{cap}"] = r
        print_row(f"daily cap {cap}", r)

    # ================================
    # Phase 7: 풀백 거리
    # ================================
    phase_header(7, "풀백 거리 ($780 + 3포지션)")
    results["phase7"] = {}
    for pb in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        p = deepcopy(BASE_PARAMS)
        p["pullback_ema_dist_pct"] = pb
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, params=p)
        results["phase7"][f"pb{pb}"] = r
        print_row(f"pullback dist {pb}%", r)

    # ================================
    # Phase 8: ADX 진입 임계값
    # ================================
    phase_header(8, "ADX 진입 임계값 ($780 + 3포지션)")
    results["phase8"] = {}
    for adx in [20, 25, 30, 35, 40, 45]:
        p = deepcopy(BASE_PARAMS)
        p["adx_enter_trending"] = adx
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, params=p)
        results["phase8"][f"adx{adx}"] = r
        print_row(f"ADX enter {adx}", r)

    # ================================
    # Phase 9: SL ATR 배수
    # ================================
    phase_header(9, "SL ATR 배수 ($780 + 3포지션)")
    results["phase9"] = {}
    for sl in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        p = deepcopy(BASE_PARAMS)
        p["sl_atr_mult"] = sl
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, params=p)
        results["phase9"][f"sl{sl}"] = r
        print_row(f"SL ATR x{sl}", r)

    # ================================
    # Phase 10: TP R:R 비율
    # ================================
    phase_header(10, "TP R:R 비율 ($780 + 3포지션)")
    results["phase10"] = {}
    for tp in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]:
        p = deepcopy(BASE_PARAMS)
        p["tp_rr_ratio"] = tp
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, params=p)
        results["phase10"][f"tp{tp}"] = r
        print_row(f"TP R:R {tp}", r)

    # ================================
    # Phase 11: 코인 조합 ablation
    # ================================
    phase_header(11, "코인 조합 ablation ($780 + 3포지션)")
    results["phase11"] = {}
    combos = [
        ("BTC only", ["BTCUSDT"]),
        ("ETH only", ["ETHUSDT"]),
        ("SOL only", ["SOLUSDT"]),
        ("BTC+ETH", ["BTCUSDT", "ETHUSDT"]),
        ("BTC+SOL", ["BTCUSDT", "SOLUSDT"]),
        ("ETH+SOL", ["ETHUSDT", "SOLUSDT"]),
        ("BTC+ETH+SOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
    ]
    for label, syms in combos:
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, enabled_symbols=syms)
        results["phase11"][label] = r
        print_row(label, r)

    # ================================
    # Phase 12: 트레일링 미세조정 (3포지션 환경)
    # ================================
    phase_header(12, "트레일링 미세조정 ($780 + 3포지션 + 슬리피지 0.05%)")
    results["phase12"] = {}
    for act in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5]:
        for dist in [0.08, 0.10, 0.12, 0.15]:
            if dist >= act:
                continue
            r = run(data, BASE_SEED_AFTER_DEPOSIT,
                    max_simultaneous=3,
                    trailing_activation=act, trailing_distance=dist)
            k = f"a{act}_d{dist}"
            results["phase12"][k] = r
    # 정렬 출력
    valid = sorted([(k, v) for k, v in results["phase12"].items() if not v["ruined"]],
                   key=lambda x: x[1]["net_profit"], reverse=True)
    print(f"  {'순위':>4} {'조합':<15} {'거래':>6} {'승률':>7} {'순이익':>12} {'DD':>7}")
    for i, (k, v) in enumerate(valid[:10]):
        print(f"  {i+1:>4} {k:<15} {v['total_trades']:>6} {v['win_rate']:>6.1f}% "
              f"${v['net_profit']:>+10,.0f} {v['max_dd']:>6.1f}%")

    # ================================
    # Phase 13: 슬리피지 stress test (기본 설정)
    # ================================
    phase_header(13, "슬리피지 stress test (3포지션 기본 설정)")
    results["phase13"] = {}
    for slip in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
        r = run(data, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, slippage_pct=slip)
        results["phase13"][f"slip{slip}"] = r
        print_row(f"slippage {slip}%", r)

    # ================================
    # Phase 14: 연도별 성과 (3포지션 기본)
    # ================================
    phase_header(14, "연도별 성과 (3포지션 + 기본 설정)")
    results["phase14"] = {}
    for y_start, y_end, label in [
        ("2022-01-01", "2023-01-01", "2022"),
        ("2023-01-01", "2024-01-01", "2023 저변동"),
        ("2024-01-01", "2025-01-01", "2024"),
        ("2025-01-01", "2026-01-01", "2025"),
        ("2026-01-01", "2026-04-12", "2026 YTD"),
    ]:
        yd = filter_data_by_date(data, y_start, y_end)
        r = run(yd, BASE_SEED_AFTER_DEPOSIT,
                max_simultaneous=3, skip_years=())
        results["phase14"][label] = r
        print_row(label, r)

    # ================================
    # Phase 15: Walk-forward (2포지션 vs 3포지션)
    # ================================
    phase_header(15, "Walk-forward: 2포지션 vs 3포지션")
    results["phase15"] = {}
    wf_splits = [
        ("train_22-23", "2022-01-01", "2024-01-01"),
        ("test_24-26",  "2024-01-01", "2026-04-12"),
    ]
    for label, start, end in wf_splits:
        sd = filter_data_by_date(data, start, end)
        for max_sim in [2, 3]:
            r = run(sd, BASE_SEED_AFTER_DEPOSIT,
                    max_simultaneous=max_sim, skip_years=())
            k = f"{label}_pos{max_sim}"
            results["phase15"][k] = r
            print_row(f"{label} / {max_sim}포지션", r)

    # ================================
    # Phase 16: 3포지션 + 최적 파라미터 조합
    # ================================
    phase_header(16, "3포지션 + 여러 조합 동시 적용")
    results["phase16"] = {}
    # 찾은 최적 조합들 조합
    scenarios = [
        ("baseline 3pos", {}),
        ("risk 1.0%", {"risk_per_trade": 0.010}),
        ("risk 2.0%", {"risk_per_trade": 0.020}),
        ("lev 7x", {"max_leverage": 7}),
        ("lev 10x", {"max_leverage": 10}),
        ("daily 10", {"max_daily_trades": 10}),
        ("score 50", {"params": {**BASE_PARAMS, "entry_score_threshold": 50}}),
        ("risk 2% + lev 7x", {"risk_per_trade": 0.020, "max_leverage": 7}),
        ("risk 2% + daily 10", {"risk_per_trade": 0.020, "max_daily_trades": 10}),
        ("lev 7x + daily 10", {"max_leverage": 7, "max_daily_trades": 10}),
        ("ALL: risk 2% + lev 7x + daily 10",
         {"risk_per_trade": 0.020, "max_leverage": 7, "max_daily_trades": 10}),
    ]
    for label, kw in scenarios:
        r = run(data, BASE_SEED_AFTER_DEPOSIT, max_simultaneous=3, **kw)
        results["phase16"][label] = r
        print_row(label, r)

    # ================================
    # 최종 정리
    # ================================
    elapsed = time.time() - t_start
    print(f"\n{'='*110}")
    print(f"총 소요 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"{'='*110}")

    # 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "base_config": {
            "seed_current": BASE_SEED_CURRENT,
            "seed_after_deposit": BASE_SEED_AFTER_DEPOSIT,
            "trailing_act": BASE_TRAILING_ACT,
            "trailing_dist": BASE_TRAILING_DIST,
            "slippage": BASE_SLIPPAGE_REALISTIC,
            "skip_years": [2023],
        },
        "results": results,
    }

    out_path = os.path.join(RESULTS_DIR, "v6_comprehensive.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
