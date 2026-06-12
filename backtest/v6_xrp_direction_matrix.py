#!/usr/bin/env python3
"""
v6 XRP 방향성 매트릭스 + 전 경우의 수 테스트
==================================================
사용자 질문:
- SOL은 LONG 차단 중 (백테스트상 SOL LONG 약함)
- XRP는 어떤가? LONG/SHORT 방향별 성과?
- 변경 사항 적용 vs 미적용
- 3코인 vs 4코인
- XRP LONG만 / SHORT만 / 양방향
- 포지션 2/3/4
- 전 조합 비교

출력: 모든 경우의 수를 한 표로.
"""
import os
import sys
import json
import time
from datetime import datetime
from copy import deepcopy

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}

# 안전 패키지 파라미터
SAFE_PARAMS = {
    **BASE_PARAMS,
    "tp_rr_ratio": 6.0,
    "adx_enter_trending": 30,
}

SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470


def load_all_data_with_xrp():
    data = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        for iv in ["15", "60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
        f = os.path.join(DATA_DIR, f"{sym}_funding_long.csv")
        if os.path.exists(f):
            data[f"{sym}_funding"] = pd.read_csv(f)
    return data


def run(data, symbols, **kw):
    params = kw.pop("params", BASE_PARAMS)
    trail_act = kw.pop("trailing_activation", 1.2)
    trail_dist = kw.pop("trailing_distance", 0.1)
    slip = kw.pop("slippage_pct", SLIP)
    skip = kw.pop("skip_years", (2023,))
    max_sim = kw.pop("max_simultaneous", 2)
    max_lev = kw.pop("max_leverage", 5)
    seed = kw.pop("seed", SEED)
    blocked = kw.pop("blocked_directions", {})

    try:
        r = run_shared_backtest(
            data, params, float(seed),
            use_funding=True,
            trailing_activation=trail_act,
            trailing_distance=trail_dist,
            block_sol_long=True,   # SOL LONG 차단은 항상 유지 (기존 규칙)
            skip_years=skip,
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=slip,
            max_simultaneous=max_sim,
            max_leverage=max_lev,
            enabled_symbols=symbols,
            blocked_directions=blocked,
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
    except Exception as e:
        return {"error": str(e), "ruined": True, "net_profit": 0, "max_dd": 0,
                "final_balance": 0, "total_trades": 0, "win_rate": 0}


def fmt(r):
    if r.get("ruined") or "error" in r:
        return "RUINED".rjust(14)
    return f"${r['net_profit']:>+11,.0f}".rjust(14)


def phase(n, title):
    print(f"\n{'='*120}\nPhase {n}: {title}\n{'='*120}")


def main():
    t0 = time.time()
    print("="*120)
    print("HERMES v6 — XRP 방향성 매트릭스 + 전 경우의 수")
    print(f"기준: 시드 ${SEED:.0f}, 슬리피지 {SLIP}%, trailing 1.2/0.1, 쿨다운 제거, 수동 감독(2023 skip)")
    print("SOL LONG 차단은 기존대로 항상 적용 (이미 확정된 규칙)")
    print("="*120)

    print("\n[데이터 로드]")
    data = load_all_data_with_xrp()
    print(f"  로드 완료: {len(data)} 데이터셋")

    results = {}

    # ================================================================
    # Phase 1: XRP 단독 방향별
    # ================================================================
    phase(1, "XRP 단독 — 방향별 성과 (baseline + 안전 패키지)")
    results["p1"] = {}

    xrp_scenarios = [
        ("XRP 양방향", []),
        ("XRP LONG만",  ["SHORT"]),
        ("XRP SHORT만", ["LONG"]),
    ]
    params_sets = [
        ("baseline", BASE_PARAMS, 5),
        ("safe pkg", SAFE_PARAMS, 7),
    ]
    for param_label, p, lev in params_sets:
        print(f"\n  [{param_label}]")
        for direction_label, blocked_dirs in xrp_scenarios:
            blocked = {"XRPUSDT": blocked_dirs} if blocked_dirs else {}
            r = run(data, ["XRPUSDT"], params=p, max_leverage=lev, blocked_directions=blocked)
            results["p1"][f"{param_label}_{direction_label}"] = r
            print(f"    {direction_label:<15} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% | "
                  f"순이익 ${r['net_profit']:>+10,.0f} | DD {r['max_dd']:>5.1f}%")

    # ================================================================
    # Phase 2: 전 조합 매트릭스 — 3코인 / 4코인 / XRP 방향 / 파라미터 / 포지션
    # ================================================================
    phase(2, "전 조합 매트릭스")
    results["p2"] = {}

    THREE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    FOUR  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    scenarios = []
    # 3코인
    for params_label, p, lev in [("현재v5", BASE_PARAMS, 5), ("안전pkg", SAFE_PARAMS, 7)]:
        for sim in [2, 3]:
            scenarios.append((
                f"3코인 {params_label} {sim}pos",
                {"data": data, "symbols": THREE,
                 "params": p, "max_leverage": lev, "max_simultaneous": sim}
            ))
    # 4코인 — XRP 양방향
    for params_label, p, lev in [("현재v5", BASE_PARAMS, 5), ("안전pkg", SAFE_PARAMS, 7)]:
        for sim in [2, 3, 4]:
            scenarios.append((
                f"4코인 {params_label} {sim}pos (XRP 양방)",
                {"data": data, "symbols": FOUR,
                 "params": p, "max_leverage": lev, "max_simultaneous": sim}
            ))
    # 4코인 — XRP SHORT만
    for params_label, p, lev in [("현재v5", BASE_PARAMS, 5), ("안전pkg", SAFE_PARAMS, 7)]:
        for sim in [2, 3, 4]:
            scenarios.append((
                f"4코인 {params_label} {sim}pos (XRP SHORT만)",
                {"data": data, "symbols": FOUR,
                 "params": p, "max_leverage": lev, "max_simultaneous": sim,
                 "blocked_directions": {"XRPUSDT": ["LONG"]}}
            ))
    # 4코인 — XRP LONG만
    for params_label, p, lev in [("현재v5", BASE_PARAMS, 5), ("안전pkg", SAFE_PARAMS, 7)]:
        for sim in [2, 3, 4]:
            scenarios.append((
                f"4코인 {params_label} {sim}pos (XRP LONG만)",
                {"data": data, "symbols": FOUR,
                 "params": p, "max_leverage": lev, "max_simultaneous": sim,
                 "blocked_directions": {"XRPUSDT": ["SHORT"]}}
            ))

    print(f"\n  총 {len(scenarios)}개 시나리오 실행 중...\n")
    print(f"  {'시나리오':<40} {'거래':>6} {'승률':>7} {'순이익':>14} {'DD':>7}")
    print("  " + "-" * 80)

    for label, kw in scenarios:
        d = kw.pop("data")
        syms = kw.pop("symbols")
        r = run(d, syms, **kw)
        results["p2"][label] = r
        if r.get("ruined") or "error" in r:
            print(f"  {label:<40} RUINED")
        else:
            print(f"  {label:<40} {r['total_trades']:>6} {r['win_rate']:>6.1f}% "
                  f"${r['net_profit']:>+12,.0f} {r['max_dd']:>6.1f}%")

    # ================================================================
    # Phase 3: 정렬된 TOP 결과
    # ================================================================
    phase(3, "전체 랭킹 (순이익 기준)")
    valid = sorted([(k, v) for k, v in results["p2"].items() if not v.get("ruined")],
                   key=lambda x: x[1]["net_profit"], reverse=True)
    print(f"\n  {'순위':>4} {'시나리오':<40} {'순이익':>14} {'DD':>7} {'거래':>7} {'승률':>7}")
    print("  " + "-" * 85)
    for i, (label, r) in enumerate(valid):
        marker = ""
        if "현재v5" in label and "2pos" in label and "3코인" in label:
            marker = " ← v5 현재"
        print(f"  {i+1:>4} {label:<40} ${r['net_profit']:>+12,.0f} "
              f"{r['max_dd']:>6.1f}% {r['total_trades']:>7} {r['win_rate']:>6.1f}%{marker}")

    # ================================================================
    # Phase 4: DD 오름차순 TOP 10 (안정성 관점)
    # ================================================================
    phase(4, "DD 낮은 순 TOP 10 (안정성 관점)")
    by_dd = sorted([(k, v) for k, v in results["p2"].items() if not v.get("ruined")],
                   key=lambda x: x[1]["max_dd"])
    print(f"\n  {'순위':>4} {'시나리오':<40} {'DD':>7} {'순이익':>14}")
    for i, (label, r) in enumerate(by_dd[:10]):
        print(f"  {i+1:>4} {label:<40} {r['max_dd']:>6.1f}% ${r['net_profit']:>+12,.0f}")

    # ================================================================
    # Phase 5: 효율 (수익/DD) TOP 10
    # ================================================================
    phase(5, "효율 순위 (순이익/DD)")
    eff = [(k, v, v["net_profit"]/v["max_dd"] if v["max_dd"] > 0 else 0)
           for k, v in results["p2"].items() if not v.get("ruined")]
    eff.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {'순위':>4} {'시나리오':<40} {'효율':>10} {'순이익':>14} {'DD':>7}")
    for i, (label, r, score) in enumerate(eff[:10]):
        print(f"  {i+1:>4} {label:<40} {score:>+9,.0f} ${r['net_profit']:>+12,.0f} {r['max_dd']:>6.1f}%")

    # ================================================================
    # Phase 6: XRP SHORT만 vs XRP 양방 vs XRP LONG만 비교
    # ================================================================
    phase(6, "XRP 방향 분리 분석 — 같은 조건에서 순이익 차이")
    print(f"\n  {'조건':<30} {'양방향':>14} {'SHORT만':>14} {'LONG만':>14} {'권장':>10}")
    print("  " + "-" * 85)
    for params_label in ["현재v5", "안전pkg"]:
        for sim in [2, 3, 4]:
            base_label = f"4코인 {params_label} {sim}pos"
            both = results["p2"].get(f"{base_label} (XRP 양방)", {})
            short_only = results["p2"].get(f"{base_label} (XRP SHORT만)", {})
            long_only = results["p2"].get(f"{base_label} (XRP LONG만)", {})

            def getval(r):
                if not r or r.get("ruined"):
                    return 0
                return r.get("net_profit", 0)

            b, s, l = getval(both), getval(short_only), getval(long_only)

            best = max([("양방", b), ("SHORT", s), ("LONG", l)], key=lambda x: x[1])[0]

            print(f"  {base_label:<30} ${b:>+12,.0f} ${s:>+12,.0f} ${l:>+12,.0f} {best:>10}")

    # ================================================================
    # 저장
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*120}\n완료 | 소요 {elapsed:.0f}초\n{'='*120}")

    out_path = os.path.join(RESULTS_DIR, "v6_xrp_direction_matrix.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": SEED, "slippage": SLIP,
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
