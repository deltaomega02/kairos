#!/usr/bin/env python3
"""
v8 — v3부터 v6까지 점진적 풀 비교 (6년 데이터, 철저)
====================================================
사용자 요구: "V5부터 지금 버전까지 테스팅. 시간/자원 무관"

테스트 매트릭스:
- 14개 incremental configurations
- 6년 실제 데이터 (2020-03 ~ 2026-04)
- 슬리피지 0.05% (현실)
- 수동 감독 (2023 skip)
- 각 설정의 연도별 + 월별 + WF 검증

목표: 현재 v6 풀 패키지가 진짜 최적인지 의심의 여지 없이 확인
"""
import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v8"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470

# v3 원본 파라미터 (변경 전)
V3_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0,           # v3: 4.0
    "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5,
    "adx_enter_trending": 35,     # v3: 35
}

# v6 안전 패키지 파라미터 (TP6, ADX30 적용)
V6_SAFE_PARAMS = {
    **V3_PARAMS,
    "tp_rr_ratio": 6.0,
    "adx_enter_trending": 30,
}

THREE_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FOUR_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_all_data():
    data = {}
    for sym in FOUR_COINS:
        for iv in ["60", "240"]:
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


def run(data, label, **kw):
    params = kw.pop("params", V3_PARAMS)
    trail_act = kw.pop("trailing_activation", 1.2)
    trail_dist = kw.pop("trailing_distance", 0.1)
    syms = kw.pop("enabled_symbols", THREE_COINS)
    max_sim = kw.pop("max_simultaneous", 2)
    max_lev = kw.pop("max_leverage", 5)
    skip = kw.pop("skip_years", (2023,))
    risk = kw.pop("risk_per_trade", None)

    try:
        r = run_shared_backtest(
            data, params, SEED,
            use_funding=True,
            trailing_activation=trail_act,
            trailing_distance=trail_dist,
            block_sol_long=True,
            skip_years=skip,
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=SLIP,
            max_simultaneous=max_sim,
            max_leverage=max_lev,
            enabled_symbols=syms,
            risk_per_trade=risk,
        )
        trades = r.get("_trades", [])

        # 통계
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        avg_win = sum(t["pnl"] for t in trades if t["pnl"] > 0) / wins if wins else 0
        avg_loss = sum(t["pnl"] for t in trades if t["pnl"] <= 0) / losses if losses else 0
        pf = abs(sum(t["pnl"] for t in trades if t["pnl"] > 0) /
                 sum(t["pnl"] for t in trades if t["pnl"] <= 0)) if losses else 0

        # 최대 연패
        max_consec_loss = 0
        cur = 0
        for t in trades:
            if t["pnl"] <= 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0

        # 연도별
        yearly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in trades:
            y = datetime.utcfromtimestamp(t["timestamp"]/1000).year
            yearly[y]["trades"] += 1
            if t["pnl"] > 0: yearly[y]["wins"] += 1
            yearly[y]["pnl"] += t["pnl"]

        return {
            "label": label,
            "net_profit": r["net_profit"],
            "max_dd": r["max_dd"],
            "final_balance": r["final_balance"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "ruined": r["ruined"],
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "max_consec_loss": max_consec_loss,
            "calmar": round(r["net_profit"] / r["max_dd"], 0) if r["max_dd"] > 0 else 0,
            "yearly": dict(yearly),
        }
    except Exception as e:
        return {"label": label, "error": str(e), "ruined": True,
                "net_profit": 0, "max_dd": 0, "total_trades": 0, "win_rate": 0,
                "calmar": 0, "yearly": {}}


def prow(label, r):
    if r.get("ruined") or "error" in r:
        err = r.get("error", "RUINED")
        print(f"  {label:<55} RUINED {err[:30]}")
    else:
        print(f"  {label:<55} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% | "
              f"순이익 ${r['net_profit']:>+12,.0f} | DD {r['max_dd']:>5.1f}% | PF {r['profit_factor']:>4.2f} | Calmar {r['calmar']:>5.0f}")


def phase(n, title):
    print(f"\n{'='*130}\nPhase {n}: {title}\n{'='*130}")


def main():
    t0 = time.time()
    print("="*130)
    print("HERMES v8 — v3부터 v6까지 점진적 비교 (6년 실제 데이터)")
    print(f"시드 ${SEED:.0f} | 슬리피지 {SLIP}% | 수동 감독 (2023 skip)")
    print("="*130)

    print("\n[데이터 로드]")
    data = load_all_data()
    for sym in FOUR_COINS:
        df = data.get(f"{sym}_60")
        if df is None or df.empty: continue
        first = datetime.utcfromtimestamp(df['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')
        last = datetime.utcfromtimestamp(df['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')
        print(f"  {sym}: {len(df):>6}개 ({first} ~ {last})")

    all_results = {}

    # ================================================================
    # Phase 1: 점진적 변경 (v3 → v6)
    # ================================================================
    phase(1, "점진적 변경 (v3 베이스라인 → v6 풀 패키지)")
    print(f"\n  {'순서':<3} {'설정':<55} {'거래':>6} {'승률':>7} {'순이익':>14} {'DD':>7} {'PF':>5} {'Calmar':>7}")
    print("  " + "-"*120)

    configs = [
        # 베이스라인: v3 원본 (모든 게 옛날)
        ("v3 원본 (trail 1.5/0.3, lev5, TP4, ADX35, 3코인 2pos)", {
            "trailing_activation": 1.5, "trailing_distance": 0.3,
            "params": V3_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 5,
        }),
        # v5: 트레일링만 변경
        ("v5 (trail 1.2/0.1만 변경, 나머지 v3)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V3_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 5,
        }),
        # 단일 변경
        ("v5 + lev7만", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V3_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 7,
        }),
        ("v5 + TP6만", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": {**V3_PARAMS, "tp_rr_ratio": 6.0},
            "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 5,
        }),
        ("v5 + ADX30만", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": {**V3_PARAMS, "adx_enter_trending": 30},
            "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 5,
        }),
        ("v5 + 3pos만", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V3_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 3, "max_leverage": 5,
        }),
        ("v5 + XRP만 (4코인 2pos)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V3_PARAMS, "enabled_symbols": FOUR_COINS,
            "max_simultaneous": 2, "max_leverage": 5,
        }),
        # 누적 (안전 패키지 빌드업)
        ("v5 + lev7 + TP6", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": {**V3_PARAMS, "tp_rr_ratio": 6.0},
            "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 7,
        }),
        ("v5 + lev7 + TP6 + ADX30 (안전, 3코인 2pos)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 2, "max_leverage": 7,
        }),
        ("안전 + XRP (4코인 2pos)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": FOUR_COINS,
            "max_simultaneous": 2, "max_leverage": 7,
        }),
        ("v6 풀 패키지 (4코인 3pos, 현재 GCP 운영)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": FOUR_COINS,
            "max_simultaneous": 3, "max_leverage": 7,
        }),
        # 추가 비교: v6 변형
        ("v6 풀 + 4pos (오버피팅 체크)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": FOUR_COINS,
            "max_simultaneous": 4, "max_leverage": 7,
        }),
        ("v6 풀 - 3코인으로 (XRP 없음)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": THREE_COINS,
            "max_simultaneous": 3, "max_leverage": 7,
        }),
        ("v6 풀 + risk 1.0% (보수)", {
            "trailing_activation": 1.2, "trailing_distance": 0.1,
            "params": V6_SAFE_PARAMS, "enabled_symbols": FOUR_COINS,
            "max_simultaneous": 3, "max_leverage": 7,
            "risk_per_trade": 0.010,
        }),
    ]

    for i, (label, kw) in enumerate(configs):
        r = run(data, label, **kw)
        all_results[f"config_{i}"] = r
        marker = " ← 현재" if "v6 풀 패키지" in label else ""
        if r.get("ruined") or "error" in r:
            print(f"  {i+1:<3} {label[:55]:<55} RUINED")
        else:
            print(f"  {i+1:<3} {label[:55]:<55} {r['total_trades']:>6} {r['win_rate']:>6.1f}% "
                  f"${r['net_profit']:>+12,.0f} {r['max_dd']:>6.1f}% {r['profit_factor']:>5.2f} {r['calmar']:>7.0f}{marker}")

    # ================================================================
    # Phase 2: 상위 5개 조합 연도별 분석
    # ================================================================
    phase(2, "상위 5개 — 연도별 분석")

    valid = [(k, v) for k, v in all_results.items() if not v.get("ruined")]
    top5 = sorted(valid, key=lambda x: x[1]["net_profit"], reverse=True)[:5]

    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    print(f"\n  {'설정':<55}" + "".join(f" {y:>10}" for y in years))
    print("  " + "-"*125)
    for key, r in top5:
        row = f"  {r['label'][:55]:<55}"
        for y in years:
            yr = r["yearly"].get(int(y))
            if yr:
                row += f" ${yr['pnl']:>+8,.0f}"
            else:
                row += f" {'-':>9}"
        print(row)

    # ================================================================
    # Phase 3: Walk-Forward 검증 (top 3)
    # ================================================================
    phase(3, "Walk-Forward (top 3 조합)")

    wf_periods = [
        ("WF1 train 20-23 / test 24-26",
         ("2020-03-25", "2024-01-01"),
         ("2024-01-01", "2026-04-15")),
        ("WF2 train 20-24 / test 25-26",
         ("2020-03-25", "2025-01-01"),
         ("2025-01-01", "2026-04-15")),
    ]

    for key, r_full in top5[:3]:
        # 같은 설정 추출
        idx = int(key.split("_")[1])
        label, kw = configs[idx]
        print(f"\n  [{label}]")
        for wf_label, (s1, e1), (s2, e2) in wf_periods:
            td = filter_data_by_date(data, s1, e1)
            vd = filter_data_by_date(data, s2, e2)
            r_train = run(td, f"{wf_label} TRAIN", skip_years=(), **kw)
            r_test = run(vd, f"{wf_label} TEST", skip_years=(), **kw)
            t_str = "RUINED" if r_train.get("ruined") else f"${r_train['net_profit']:+,.0f}"
            te_str = "RUINED" if r_test.get("ruined") else f"${r_test['net_profit']:+,.0f}"
            print(f"    {wf_label:<35} train {t_str:>14}  | test {te_str:>14}")

    # ================================================================
    # Phase 4: 변경 효과 분리 (각 단일 변경의 marginal 효과)
    # ================================================================
    phase(4, "변경별 marginal 효과 분리")

    base = all_results["config_1"]  # v5 baseline
    changes = [
        ("lev7 (5→7)", "config_2"),
        ("TP6 (4→6)", "config_3"),
        ("ADX30 (35→30)", "config_4"),
        ("3pos (2→3)", "config_5"),
        ("XRP 추가", "config_6"),
    ]

    print(f"\n  {'변경':<25} {'순이익 변화':>15} {'DD 변화':>10} {'%수익 변화':>12}")
    print("  " + "-"*70)
    for label, key in changes:
        if key in all_results and not all_results[key].get("ruined"):
            r = all_results[key]
            delta_pnl = r["net_profit"] - base["net_profit"]
            delta_dd = r["max_dd"] - base["max_dd"]
            pct = delta_pnl / base["net_profit"] * 100 if base["net_profit"] != 0 else 0
            print(f"  {label:<25} ${delta_pnl:>+13,.0f} {delta_dd:>+9.1f}%p {pct:>+10.1f}%")

    # ================================================================
    # Phase 5: 핵심 비교 — v5 vs v6 풀 패키지
    # ================================================================
    phase(5, "v5 vs v6 풀 패키지 — 핵심 비교")

    v5 = all_results["config_1"]  # v5 baseline
    v6 = all_results["config_10"]  # v6 풀

    print(f"\n  {'항목':<25} {'v5 baseline':>20} {'v6 풀 패키지':>20} {'차이':>20}")
    print("  " + "-"*90)
    metrics = [
        ("거래 수", f"{v5['total_trades']:,}", f"{v6['total_trades']:,}",
         f"{v6['total_trades']-v5['total_trades']:+,}"),
        ("승률", f"{v5['win_rate']}%", f"{v6['win_rate']}%",
         f"{v6['win_rate']-v5['win_rate']:+.2f}%p"),
        ("순이익", f"${v5['net_profit']:+,.0f}", f"${v6['net_profit']:+,.0f}",
         f"x{v6['net_profit']/v5['net_profit']:.1f}" if v5['net_profit'] > 0 else ""),
        ("최대 DD", f"{v5['max_dd']}%", f"{v6['max_dd']}%",
         f"{v6['max_dd']-v5['max_dd']:+.1f}%p"),
        ("Profit Factor", f"{v5['profit_factor']}", f"{v6['profit_factor']}",
         f"{v6['profit_factor']-v5['profit_factor']:+.2f}"),
        ("Calmar", f"{v5['calmar']:,}", f"{v6['calmar']:,}", ""),
        ("평균 수익", f"${v5['avg_win']}", f"${v6['avg_win']}", ""),
        ("평균 손실", f"${v5['avg_loss']}", f"${v6['avg_loss']}", ""),
        ("최대 연패", f"{v5['max_consec_loss']}", f"{v6['max_consec_loss']}",
         f"{v6['max_consec_loss']-v5['max_consec_loss']:+}"),
    ]
    for name, va, vb, diff in metrics:
        print(f"  {name:<25} {va:>20} {vb:>20} {diff:>20}")

    # ================================================================
    # 저장
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*130}\n완료 | 소요 {elapsed:.0f}s ({elapsed/60:.1f}분)\n{'='*130}")

    out_path = os.path.join(RESULTS_DIR, "v8_full_comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "slippage": SLIP,
            "configurations": [{"label": label, "kw": str(kw)} for label, kw in configs],
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
