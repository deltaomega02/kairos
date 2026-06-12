#!/usr/bin/env python3
"""
v6 XRP Out-of-Sample 검증
============================
XRP는 4년 최적화 과정에서 전혀 보지 못한 코인.
여기서 좋으면 진짜 로버스트. 아니면 과적합.

테스트:
1. XRP 단독: 주요 파라미터 설정들로 백테스트
2. BTC+ETH+SOL 단독 (train): 기존 결과 확인
3. 전 4코인 (BTC+ETH+SOL+XRP): 일반화 성능
4. 일일 캡 민감도 (XRP 단독) — 사용자 핵심 관심사
5. 안전 패키지 vs 공격 패키지 XRP에서 검증
"""
import os
import sys
import json
import time
from datetime import datetime
from copy import deepcopy

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data, DATA_DIR
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
SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470


def load_all_data_with_xrp():
    """BTC/ETH/SOL + XRP 전부 로드"""
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


def run(data, symbols, **kw):
    params = kw.pop("params", BASE_PARAMS)
    trail_act = kw.pop("trailing_activation", 1.2)
    trail_dist = kw.pop("trailing_distance", 0.1)
    slip = kw.pop("slippage_pct", SLIP)
    skip = kw.pop("skip_years", (2023,))
    max_sim = kw.pop("max_simultaneous", 2)
    seed = kw.pop("seed", SEED)

    try:
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
            enabled_symbols=symbols,
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


def prow(label, r, extra=""):
    if r.get("ruined") or "error" in r:
        err = r.get("error", "RUINED")
        print(f"  {label:<40} {err[:30]} {extra}")
    else:
        print(f"  {label:<40} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% "
              f"| 순이익 ${r['net_profit']:>+10,.0f} | DD {r['max_dd']:>5.1f}% {extra}")


def phase(n, title):
    print(f"\n{'='*110}\nPhase {n}: {title}\n{'='*110}")


def main():
    t0 = time.time()
    print("="*110)
    print("HERMES v6 — XRP Out-of-Sample 검증")
    print("XRP는 파라미터 최적화에 전혀 사용되지 않은 코인 → 완전 independent validation")
    print("="*110)

    print("\n[데이터 로드 (BTC/ETH/SOL/XRP)]")
    data = load_all_data_with_xrp()
    print(f"  로드: {len(data)} 데이터셋")

    # XRP 데이터 검증
    xrp_1h = data.get("XRPUSDT_60")
    if xrp_1h is None or xrp_1h.empty:
        print("  ❌ XRP 데이터 없음"); return
    xrp_first = datetime.utcfromtimestamp(xrp_1h['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')
    xrp_last = datetime.utcfromtimestamp(xrp_1h['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')
    print(f"  XRP 1H: {len(xrp_1h)}개 ({xrp_first} ~ {xrp_last})")

    results = {}

    # ================================================================
    # Phase 1: XRP 단독 — 다양한 설정 (2023 skip)
    # ================================================================
    phase(1, "XRP 단독 — 파라미터 설정 비교 (수동 감독)")
    results["p1"] = {}
    XRP = ["XRPUSDT"]

    configs = [
        ("baseline (v5 1.2/0.1)", {}),
        ("+ leverage 7x", {"max_leverage": 7}),
        ("+ TP 5.0", {"params": {**BASE_PARAMS, "tp_rr_ratio": 5.0}}),
        ("+ TP 6.0", {"params": {**BASE_PARAMS, "tp_rr_ratio": 6.0}}),
        ("+ ADX 30", {"params": {**BASE_PARAMS, "adx_enter_trending": 30}}),
        ("+ risk 2%", {"risk_per_trade": 0.020}),
        ("안전 (lev7+TP6+ADX30)", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30},
        }),
        ("공격 (risk2+lev7+TP5+ADX30)", {
            "risk_per_trade": 0.020, "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 5.0, "adx_enter_trending": 30},
        }),
    ]
    for label, kw in configs:
        r = run(data, XRP, **kw)
        results["p1"][label] = r
        prow(label, r)

    # ================================================================
    # Phase 2: 일일 캡 민감도 — XRP 단독 (사용자 핵심 관심)
    # ================================================================
    phase(2, "일일 캡 민감도 — XRP 단독 (과적합 재검증)")
    results["p2"] = {}
    for cap in [3, 5, 7, 10, 15, 999]:
        r = run(data, XRP, max_daily_trades=cap)
        results["p2"][f"cap{cap}"] = r
        prow(f"XRP cap {cap}", r)

    # 연도별 XRP
    print("\n  [XRP 연도별 — cap 5 vs 999]")
    print(f"  {'연도':<15} {'cap 5':>15} {'cap 999':>15} {'차이':>12}")
    years = [
        ("2022-01-01", "2023-01-01", "2022"),
        ("2023-01-01", "2024-01-01", "2023 저변동"),
        ("2024-01-01", "2025-01-01", "2024"),
        ("2025-01-01", "2026-01-01", "2025"),
        ("2026-01-01", "2026-04-15", "2026 YTD"),
    ]
    for start, end, label in years:
        yd = filter_data_by_date(data, start, end)
        r5 = run(yd, XRP, max_daily_trades=5, skip_years=())
        r999 = run(yd, XRP, max_daily_trades=999, skip_years=())
        results["p2"][f"year_{label}_cap5"] = r5
        results["p2"][f"year_{label}_cap999"] = r999
        v5 = "RUIN" if r5["ruined"] else f"${r5['net_profit']:+,.0f}"
        v999 = "RUIN" if r999["ruined"] else f"${r999['net_profit']:+,.0f}"
        diff = ""
        if not r5["ruined"] and not r999["ruined"]:
            diff = f"${r999['net_profit']-r5['net_profit']:+,.0f}"
        print(f"  {label:<15} {v5:>15} {v999:>15} {diff:>12}")

    # ================================================================
    # Phase 3: BTC+ETH+SOL vs 4코인 (XRP 추가 효과)
    # ================================================================
    phase(3, "코인 조합 — BTC+ETH+SOL (기존) vs 전 4코인")
    results["p3"] = {}

    for label, syms, sim in [
        ("BTC+ETH+SOL 2pos", ["BTCUSDT","ETHUSDT","SOLUSDT"], 2),
        ("BTC+ETH+SOL 3pos", ["BTCUSDT","ETHUSDT","SOLUSDT"], 3),
        ("4코인 2pos", ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"], 2),
        ("4코인 3pos", ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"], 3),
        ("4코인 4pos", ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"], 4),
    ]:
        r = run(data, syms, max_simultaneous=sim)
        results["p3"][label] = r
        prow(label, r)

    # ================================================================
    # Phase 4: XRP 연도별 상세 (주요 설정별)
    # ================================================================
    phase(4, "XRP 연도별 성과 (주요 설정)")
    results["p4"] = {}

    configs_for_yearly = [
        ("baseline", {}),
        ("안전 (lev7+TP6+ADX30)", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30},
        }),
    ]
    print(f"\n  {'설정':<30} {'2022':>10} {'2023':>10} {'2024':>10} {'2025':>10} {'2026':>10}")
    for label, kw in configs_for_yearly:
        row = f"  {label:<30}"
        yearly_profits = []
        for start, end, y in years:
            yd = filter_data_by_date(data, start, end)
            r = run(yd, XRP, skip_years=(), **kw)
            results["p4"][f"{label}_{y}"] = r
            if r["ruined"]:
                row += f" {'RUIN':>9}"
            else:
                row += f" ${r['net_profit']:>+8,.0f}"
                yearly_profits.append(r["net_profit"])
        print(row)

    # ================================================================
    # Phase 5: 최적 설정 — BTC/ETH/SOL train, XRP test 비교
    # ================================================================
    phase(5, "Train/Test: BTC/ETH/SOL vs XRP (같은 설정 상대 성과)")
    results["p5"] = {}

    print(f"\n  {'설정':<30} {'3코인(train)':>18} {'XRP(test)':>18} {'XRP/3코인':>12}")
    for label, kw in configs:
        r_train = run(data, ["BTCUSDT","ETHUSDT","SOLUSDT"], **kw)
        r_test = run(data, XRP, **kw)
        results["p5"][f"{label}_train"] = r_train
        results["p5"][f"{label}_test"] = r_test

        v_train = "RUIN" if r_train["ruined"] else f"${r_train['net_profit']:+,.0f}"
        v_test = "RUIN" if r_test["ruined"] else f"${r_test['net_profit']:+,.0f}"
        ratio = ""
        if not r_train["ruined"] and not r_test["ruined"] and r_train["net_profit"] != 0:
            ratio = f"{r_test['net_profit']/r_train['net_profit']*100:+.1f}%"
        print(f"  {label[:28]:<30} {v_train:>18} {v_test:>18} {ratio:>12}")

    # ================================================================
    # Phase 6: XRP Walk-forward
    # ================================================================
    phase(6, "XRP Walk-forward")
    results["p6"] = {}

    wf_periods = [
        ("train 22-23", "2022-01-01", "2024-01-01"),
        ("test  24-26", "2024-01-01", "2026-04-15"),
        ("train 22-24", "2022-01-01", "2025-01-01"),
        ("test  25-26", "2025-01-01", "2026-04-15"),
    ]
    print(f"\n  {'설정':<30}" + "".join(f" {label:>14}" for label, _, _ in wf_periods))
    for lb, kw in [
        ("baseline", {}),
        ("안전 (lev7+TP6+ADX30)", {
            "max_leverage": 7,
            "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30},
        }),
    ]:
        row = f"  {lb[:28]:<30}"
        for wf_label, start, end in wf_periods:
            pdata = filter_data_by_date(data, start, end)
            r = run(pdata, XRP, skip_years=(), **kw)
            results["p6"][f"{lb}_{wf_label}"] = r
            if r["ruined"]:
                row += f" {'RUIN':>14}"
            else:
                row += f" ${r['net_profit']:>+12,.0f}"
        print(row)

    # ================================================================
    # Phase 7: 4코인 통합 + 안전 패키지 (최종)
    # ================================================================
    phase(7, "4코인 통합 + 안전 패키지 (최종 결정 후보)")
    results["p7"] = {}

    FOUR = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]
    scenarios = [
        ("현재 v5 (3코인 2pos)", {"enabled_symbols_": ["BTCUSDT","ETHUSDT","SOLUSDT"], "max_simultaneous": 2}),
        ("3코인 2pos + 안전", {"enabled_symbols_": ["BTCUSDT","ETHUSDT","SOLUSDT"], "max_simultaneous": 2,
                                "max_leverage": 7,
                                "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30}}),
        ("4코인 2pos baseline", {"enabled_symbols_": FOUR, "max_simultaneous": 2}),
        ("4코인 2pos + 안전", {"enabled_symbols_": FOUR, "max_simultaneous": 2,
                                "max_leverage": 7,
                                "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30}}),
        ("4코인 3pos + 안전", {"enabled_symbols_": FOUR, "max_simultaneous": 3,
                                "max_leverage": 7,
                                "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30}}),
        ("4코인 4pos + 안전", {"enabled_symbols_": FOUR, "max_simultaneous": 4,
                                "max_leverage": 7,
                                "params": {**BASE_PARAMS, "tp_rr_ratio": 6.0, "adx_enter_trending": 30}}),
    ]
    for label, kw in scenarios:
        syms = kw.pop("enabled_symbols_")
        r = run(data, syms, **kw)
        results["p7"][label] = r
        prow(label, r)

    elapsed = time.time() - t0
    print(f"\n{'='*110}\n완료 | 소요 {elapsed:.0f}초\n{'='*110}")

    out_path = os.path.join(RESULTS_DIR, "v6_xrp_validation.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "slippage": SLIP,
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
