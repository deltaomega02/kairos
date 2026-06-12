#!/usr/bin/env python3
"""
v6 2포지션 상세 분석 + 일일 캡 과적합 검증
==============================================
사용자 요구:
1. 2포지션 고정으로 모든 파라미터 상세 분석
2. 일일 거래 캡의 과적합 여부 철저히 검증 (walk-forward + 연도별)

시드 $580 (현재) 기준, 슬리피지 0.05% (현실).
"""
import os
import sys
import json
import time
from datetime import datetime
from copy import deepcopy

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
SEED = 580.0
SLIP_REAL = 0.05
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


def run(data, **kw):
    params = kw.pop("params", BASE_PARAMS)
    trail_act = kw.pop("trailing_activation", 1.2)
    trail_dist = kw.pop("trailing_distance", 0.1)
    slip = kw.pop("slippage_pct", SLIP_REAL)
    skip = kw.pop("skip_years", (2023,))
    max_sim = kw.pop("max_simultaneous", 2)  # 2포지션 고정!
    seed = kw.pop("seed", SEED)

    r = run_shared_backtest(
        data, params, float(seed),
        use_funding=True,
        trailing_activation=trail_act,
        trailing_distance=trail_dist,
        block_sol_long=True,
        skip_years=skip,
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        slippage_pct=slip,
        max_simultaneous=max_sim,
        **kw,
    )
    return {
        "net_profit": r["net_profit"],
        "max_dd": r["max_dd"],
        "final_balance": r["final_balance"],
        "total_trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "ruined": r["ruined"],
    }


def phase(n, title):
    print(f"\n{'='*110}\nPhase {n}: {title}\n{'='*110}")


def prow(label, r, extra=""):
    if r["ruined"]:
        print(f"  {label:<40} RUINED {extra}")
    else:
        print(f"  {label:<40} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% "
              f"| 순이익 ${r['net_profit']:>+10,.0f} | DD {r['max_dd']:>5.1f}% {extra}")


def main():
    t0 = time.time()
    print("="*110)
    print(f"HERMES v6 — 2포지션 상세 분석 + 일일 캡 과적합 검증")
    print(f"기준: 2포지션 고정, 시드 ${SEED:.0f}, trailing 1.2/0.1, 슬리피지 {SLIP_REAL}%")
    print("="*110)

    print("\n[데이터 로드]")
    data = load_all_data()

    all_results = {}

    # ================================================================
    # Phase A: 2포지션 기본 파라미터 민감도
    # ================================================================
    phase("A", "2포지션 기본 파라미터 민감도")
    all_results["phaseA"] = {}

    # A.1 리스크/거래
    print("\n[A.1] risk_per_trade")
    for risk in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        r = run(data, risk_per_trade=risk)
        all_results["phaseA"][f"risk{risk}"] = r
        prow(f"risk {risk*100:.1f}%", r)

    # A.2 레버리지
    print("\n[A.2] max_leverage")
    for lev in [3, 5, 7, 10, 15]:
        r = run(data, max_leverage=lev)
        all_results["phaseA"][f"lev{lev}"] = r
        prow(f"leverage {lev}x", r)

    # A.3 진입 스코어
    print("\n[A.3] entry score threshold")
    for score in [30, 40, 50, 60, 70]:
        p = {**BASE_PARAMS, "entry_score_threshold": score}
        r = run(data, params=p)
        all_results["phaseA"][f"score{score}"] = r
        prow(f"score {score}", r)

    # A.4 풀백 거리
    print("\n[A.4] pullback distance")
    for pb in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        p = {**BASE_PARAMS, "pullback_ema_dist_pct": pb}
        r = run(data, params=p)
        all_results["phaseA"][f"pb{pb}"] = r
        prow(f"pullback {pb}%", r)

    # A.5 ADX 진입
    print("\n[A.5] ADX enter threshold")
    for adx in [20, 25, 30, 35, 40, 45]:
        p = {**BASE_PARAMS, "adx_enter_trending": adx}
        r = run(data, params=p)
        all_results["phaseA"][f"adx{adx}"] = r
        prow(f"ADX {adx}", r)

    # A.6 SL ATR
    print("\n[A.6] SL ATR multiplier")
    for sl in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        p = {**BASE_PARAMS, "sl_atr_mult": sl}
        r = run(data, params=p)
        all_results["phaseA"][f"sl{sl}"] = r
        prow(f"SL ATR x{sl}", r)

    # A.7 TP R:R
    print("\n[A.7] TP R:R ratio")
    for tp in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]:
        p = {**BASE_PARAMS, "tp_rr_ratio": tp}
        r = run(data, params=p)
        all_results["phaseA"][f"tp{tp}"] = r
        prow(f"TP R:R {tp}", r)

    # ================================================================
    # Phase B: 일일 거래 캡 과적합 철저 검증
    # ================================================================
    phase("B", "일일 거래 캡 과적합 검증 (사용자 의심 포인트)")
    all_results["phaseB"] = {}

    # B.1: 전체 4년에서 캡별 성과
    print("\n[B.1] 전체 4년 (수동 감독) — 캡별 성과")
    for cap in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 999]:
        r = run(data, max_daily_trades=cap)
        all_results["phaseB"][f"full_cap{cap}"] = r
        prow(f"cap {cap}", r)

    # B.2: 연도별 캡 효과 분리
    print("\n[B.2] 연도별 — 캡 5 vs 캡 10 vs 캡 999")
    years = [
        ("2022-01-01", "2023-01-01", "2022"),
        ("2023-01-01", "2024-01-01", "2023 저변동"),
        ("2024-01-01", "2025-01-01", "2024"),
        ("2025-01-01", "2026-01-01", "2025"),
        ("2026-01-01", "2026-04-12", "2026 YTD"),
    ]
    print(f"  {'연도':<15} {'캡 5':>18} {'캡 10':>18} {'캡 999':>18} {'차이(10-5)':>15}")
    print("  " + "-" * 95)
    for start, end, label in years:
        yd = filter_data_by_date(data, start, end)
        r5 = run(yd, max_daily_trades=5, skip_years=())
        r10 = run(yd, max_daily_trades=10, skip_years=())
        r999 = run(yd, max_daily_trades=999, skip_years=())
        all_results["phaseB"][f"year_{label}_cap5"] = r5
        all_results["phaseB"][f"year_{label}_cap10"] = r10
        all_results["phaseB"][f"year_{label}_cap999"] = r999

        v5 = "RUINED" if r5["ruined"] else f"${r5['net_profit']:+,.0f}"
        v10 = "RUINED" if r10["ruined"] else f"${r10['net_profit']:+,.0f}"
        v999 = "RUINED" if r999["ruined"] else f"${r999['net_profit']:+,.0f}"
        diff = ""
        if not r5["ruined"] and not r10["ruined"]:
            d = r10["net_profit"] - r5["net_profit"]
            diff = f"${d:+,.0f}"
        print(f"  {label:<15} {v5:>18} {v10:>18} {v999:>18} {diff:>15}")

    # B.3: Walk-forward — 각 기간에서 best cap이 일관되는가
    print("\n[B.3] Walk-forward: 캡이 각 기간에서 같은 답을 주는가")
    wf_periods = [
        ("WF-A 22-23 train", "2022-01-01", "2024-01-01"),
        ("WF-A 24-26 test", "2024-01-01", "2026-04-12"),
        ("WF-B 22-24 train", "2022-01-01", "2025-01-01"),
        ("WF-B 25-26 test", "2025-01-01", "2026-04-12"),
    ]
    for label, start, end in wf_periods:
        pdata = filter_data_by_date(data, start, end)
        print(f"\n  [{label}]")
        best = None
        for cap in [5, 7, 10, 15, 20, 999]:
            r = run(pdata, max_daily_trades=cap, skip_years=())
            all_results["phaseB"][f"{label}_cap{cap}"] = r
            marker = ""
            if best is None or (not r["ruined"] and r["net_profit"] > best[1]):
                if not r["ruined"]:
                    best = (cap, r["net_profit"])
            if not r["ruined"]:
                print(f"    cap {cap:>3}: ${r['net_profit']:>+10,.0f} DD {r['max_dd']:>5.1f}%")
        if best:
            print(f"    → 이 기간 최적 cap: {best[0]} (${best[1]:+,.0f})")

    # B.4: 같은 캡 설정이 각 WF 기간에서 얼마나 일관되는가 (순위 변동 체크)
    print("\n[B.4] 캡 설정별 WF 기간 간 일관성")
    caps_to_check = [5, 7, 10, 15]
    print(f"  {'캡':>5} {'22-23 순위':>12} {'24-26 순위':>12} {'22-24 순위':>12} {'25-26 순위':>12}")

    def compute_rank(period_key, cap_val, all_caps=[5, 7, 10, 15, 20, 999]):
        vals = []
        for c in all_caps:
            key = f"{period_key}_cap{c}"
            if key in all_results["phaseB"]:
                r = all_results["phaseB"][key]
                if not r["ruined"]:
                    vals.append((c, r["net_profit"]))
        vals.sort(key=lambda x: x[1], reverse=True)
        for i, (c, _) in enumerate(vals):
            if c == cap_val:
                return i + 1
        return "-"

    for cap in caps_to_check:
        r1 = compute_rank("WF-A 22-23 train", cap)
        r2 = compute_rank("WF-A 24-26 test", cap)
        r3 = compute_rank("WF-B 22-24 train", cap)
        r4 = compute_rank("WF-B 25-26 test", cap)
        print(f"  {cap:>5} {r1:>12} {r2:>12} {r3:>12} {r4:>12}")

    # ================================================================
    # Phase C: 안전 업그레이드 패키지 (사용자 의심 고려)
    # ================================================================
    phase("C", "업그레이드 조합 — 2포지션, daily cap 유지(5)")
    all_results["phaseC"] = {}

    scenarios = [
        ("baseline 현재 (v5)", {}),
        ("+leverage 7x", {"max_leverage": 7}),
        ("+TP 5.0", {"params": {**BASE_PARAMS, "tp_rr_ratio": 5.0}}),
        ("+TP 6.0", {"params": {**BASE_PARAMS, "tp_rr_ratio": 6.0}}),
        ("+ADX 30", {"params": {**BASE_PARAMS, "adx_enter_trending": 30}}),
        ("+risk 2%", {"risk_per_trade": 0.020}),
        ("+SL 1.75", {"params": {**BASE_PARAMS, "sl_atr_mult": 1.75}}),
        ("lev7 + TP5", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0},
        }),
        ("lev7 + TP6", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0},
        }),
        ("lev7 + TP5 + ADX30", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0, "adx_enter_trending": 30},
        }),
        ("lev7 + TP6 + ADX30", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30},
        }),
        ("lev7 + TP5 + SL1.75", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0, "sl_atr_mult": 1.75},
        }),
        ("lev7 + TP6 + SL1.75 + ADX30", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "sl_atr_mult": 1.75, "adx_enter_trending": 30},
        }),
        ("risk2 + lev7", {
            "risk_per_trade": 0.020,
            "max_leverage": 7,
        }),
        ("risk2 + lev7 + TP5", {
            "risk_per_trade": 0.020,
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0},
        }),
        ("risk2 + lev7 + TP5 + ADX30", {
            "risk_per_trade": 0.020,
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0, "adx_enter_trending": 30},
        }),
    ]
    for label, kw in scenarios:
        r = run(data, **kw)
        all_results["phaseC"][label] = r
        prow(label, r)

    # ================================================================
    # Phase D: 최상위 조합의 연도별 검증 (과적합 재체크)
    # ================================================================
    phase("D", "최상위 조합들의 연도별 검증 (과적합 재체크)")
    all_results["phaseD"] = {}

    # Phase C에서 상위 5개 선정
    valid_c = sorted([(k, v) for k, v in all_results["phaseC"].items() if not v["ruined"]],
                     key=lambda x: x[1]["net_profit"], reverse=True)
    top_c = valid_c[:5]

    print(f"\n  상위 5개 조합:")
    for label, r in top_c:
        print(f"    {label}: ${r['net_profit']:+,.0f} (DD {r['max_dd']}%)")

    # 각 조합을 연도별로
    print(f"\n  연도별 성과 (2포지션, 해당 설정):")
    for label, _ in top_c:
        scenario_kw = next(kw for l, kw in scenarios if l == label)
        print(f"\n  [{label}]")
        yearly = []
        for start, end, y_label in years:
            yd = filter_data_by_date(data, start, end)
            r = run(yd, skip_years=(), **scenario_kw)
            yearly.append((y_label, r))
            all_results["phaseD"][f"{label}_{y_label}"] = r
            if r["ruined"]:
                print(f"    {y_label:<12}: RUINED")
            else:
                print(f"    {y_label:<12}: ${r['net_profit']:>+10,.0f} DD {r['max_dd']:>5.1f}% 거래 {r['total_trades']}")

        # 플러스 연도 카운트
        pos = sum(1 for _, rr in yearly if not rr["ruined"] and rr["net_profit"] > 0)
        worst = min((rr["net_profit"] for _, rr in yearly if not rr["ruined"]), default=0)
        print(f"    → 플러스 {pos}/5 연도 | 최악 연도 ${worst:+,.0f}")

    # ================================================================
    # Phase E: Walk-forward 최종 검증 (최상위 3개)
    # ================================================================
    phase("E", "Walk-forward 최종 검증 (상위 3 조합)")
    all_results["phaseE"] = {}

    top3 = valid_c[:3]
    print(f"\n  {'조합':<30} {'22-24 train':>15} {'25-26 test':>15} {'22-23 train':>15} {'24-26 test':>15}")
    for label, _ in top3:
        scenario_kw = next(kw for l, kw in scenarios if l == label)
        train1 = run(filter_data_by_date(data, "2022-01-01", "2025-01-01"), skip_years=(), **scenario_kw)
        test1 = run(filter_data_by_date(data, "2025-01-01", "2026-04-12"), skip_years=(), **scenario_kw)
        train2 = run(filter_data_by_date(data, "2022-01-01", "2024-01-01"), skip_years=(), **scenario_kw)
        test2 = run(filter_data_by_date(data, "2024-01-01", "2026-04-12"), skip_years=(), **scenario_kw)

        all_results["phaseE"][f"{label}_train1"] = train1
        all_results["phaseE"][f"{label}_test1"] = test1
        all_results["phaseE"][f"{label}_train2"] = train2
        all_results["phaseE"][f"{label}_test2"] = test2

        def fmt(r):
            if r["ruined"]:
                return "RUINED"
            return f"${r['net_profit']:+,.0f}"

        print(f"  {label[:28]:<30} {fmt(train1):>15} {fmt(test1):>15} {fmt(train2):>15} {fmt(test2):>15}")

    # ================================================================
    # 정리
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*110}")
    print(f"완료 | 소요 {elapsed:.0f}초")
    print(f"{'='*110}")

    out_path = os.path.join(RESULTS_DIR, "v6_2pos_detailed.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "slippage": SLIP_REAL,
            "max_simultaneous": 2,
            "elapsed_sec": round(elapsed, 1),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
