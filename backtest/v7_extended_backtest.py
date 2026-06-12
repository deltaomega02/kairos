#!/usr/bin/env python3
"""
v7 확장 백테스트 (최대 6년)
===========================
2020-03 ~ 2026-04 (BTC 기준 최대)
풀 패키지 설정 + 상세 분석

주요 분석:
1. 전체 기간
2. 연도별
3. Walk-forward (4가지 분할)
4. 스트레스 기간 (2022 크래시, 2023 저변동, 2021 bull)
5. 수동 감독 시나리오 (2023 skip)
"""
import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict
from statistics import mean

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v7"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 풀 패키지 설정
FULL_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30,
}
SEED = 580.0
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_extended_data():
    """v7 확장 데이터 로드"""
    data = {}
    for sym in SYMBOLS:
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


def run(data, label="", **overrides):
    max_sim = overrides.pop("max_simultaneous", 3)
    max_lev = overrides.pop("max_leverage", 7)
    skip = overrides.pop("skip_years", ())
    syms = overrides.pop("enabled_symbols", SYMBOLS)

    try:
        r = run_shared_backtest(
            data, FULL_PARAMS, SEED,
            use_funding=True,
            trailing_activation=1.2, trailing_distance=0.1,
            block_sol_long=True,
            skip_years=skip,
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            use_cooldown=False,
            slippage_pct=SLIP,
            max_simultaneous=max_sim,
            max_leverage=max_lev,
            enabled_symbols=syms,
        )

        trades = r.get("_trades", [])
        # 방향별/코인별 통계
        by_dir = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0})
        by_coin = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0})
        by_year = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0})
        by_reason = defaultdict(int)
        for t in trades:
            d = t.get("direction","?")
            c = t.get("symbol","?")
            y = datetime.utcfromtimestamp(t["timestamp"]/1000).year
            rn = t.get("reason","?")
            for bucket, key in [(by_dir,d),(by_coin,c),(by_year,y)]:
                bucket[key]["trades"] += 1
                if t["pnl"] > 0: bucket[key]["wins"] += 1
                bucket[key]["pnl"] += t["pnl"]
            by_reason[rn] += 1

        return {
            "label": label,
            "net_profit": r["net_profit"],
            "max_dd": r["max_dd"],
            "final_balance": r["final_balance"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "ruined": r["ruined"],
            "by_direction": dict(by_dir),
            "by_coin": dict(by_coin),
            "by_year": {str(k): v for k, v in by_year.items()},
            "by_reason": dict(by_reason),
        }
    except Exception as e:
        return {"label": label, "error": str(e), "ruined": True,
                "net_profit": 0, "max_dd": 0, "final_balance": 0,
                "total_trades": 0, "win_rate": 0,
                "by_direction": {}, "by_coin": {}, "by_year": {}, "by_reason": {}}


def prow(label, r):
    if r.get("ruined") or "error" in r:
        err = r.get("error","RUINED")
        print(f"  {label:<50} RUINED {err[:30]}")
    else:
        print(f"  {label:<50} 거래 {r['total_trades']:>5} | 승률 {r['win_rate']:>5.1f}% | "
              f"순이익 ${r['net_profit']:>+12,.0f} | DD {r['max_dd']:>5.1f}%")


def phase(n, title):
    print(f"\n{'='*120}\nPhase {n}: {title}\n{'='*120}")


def main():
    t0 = time.time()
    print("="*120)
    print("HERMES v7 — 확장 실제 데이터 백테스트 (최대 6년)")
    print(f"풀 패키지: 4코인, 3pos, lev7, TP6, ADX30, trailing 1.2/0.1, slip {SLIP}%")
    print(f"시드 ${SEED:.0f}")
    print("="*120)

    print("\n[데이터 로드]")
    data = load_extended_data()
    for sym in SYMBOLS:
        df = data.get(f"{sym}_60")
        if df is None or df.empty:
            continue
        first = datetime.utcfromtimestamp(df['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d')
        last = datetime.utcfromtimestamp(df['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d')
        print(f"  {sym}: {len(df):>6}개 ({first} ~ {last})")

    all_results = {}

    # ================================================================
    # Phase 1: 최대 기간 — 3가지 버전
    # ================================================================
    phase(1, "최대 기간 (2020-03 ~ 2026-04)")

    r = run(data, "전체 기간 (감독 없음)", skip_years=())
    all_results["full_no_skip"] = r
    prow("전체 (감독 없음)", r)

    r = run(data, "전체 기간 (2023 skip, 수동 감독)", skip_years=(2023,))
    all_results["full_skip_2023"] = r
    prow("전체 (2023 skip)", r)

    r = run(data, "전체 기간 (2023+2022H2 skip)",
            skip_years=(2022, 2023))
    all_results["full_skip_22_23"] = r
    prow("전체 (2022+2023 skip)", r)

    # ================================================================
    # Phase 2: 연도별 (누적 아님, 각 연도 독립)
    # ================================================================
    phase(2, "연도별 독립 백테스트 (각 연도 $580 시작)")

    years = [
        ("2020-03-25", "2021-01-01", "2020 (COVID + 회복)"),
        ("2021-01-01", "2022-01-01", "2021 (Bull: $30k→$69k)"),
        ("2022-01-01", "2023-01-01", "2022 (Crash: -70%)"),
        ("2023-01-01", "2024-01-01", "2023 (저변동 지옥)"),
        ("2024-01-01", "2025-01-01", "2024 (ETF Bull)"),
        ("2025-01-01", "2026-01-01", "2025 (Consolidation)"),
        ("2026-01-01", "2026-04-15", "2026 YTD"),
    ]
    for start, end, label in years:
        yd = filter_data_by_date(data, start, end)
        r = run(yd, label, skip_years=())
        all_results[f"year_{label}"] = r
        prow(label, r)

    # ================================================================
    # Phase 3: Walk-Forward (4가지 분할)
    # ================================================================
    phase(3, "Walk-Forward 검증")

    wf_configs = [
        ("WF1 train 20-22 / test 23-26",
         ("2020-03-25", "2023-01-01"),
         ("2023-01-01", "2026-04-15")),
        ("WF2 train 20-23 / test 24-26",
         ("2020-03-25", "2024-01-01"),
         ("2024-01-01", "2026-04-15")),
        ("WF3 train 21-23 / test 24-26",
         ("2021-01-01", "2024-01-01"),
         ("2024-01-01", "2026-04-15")),
        ("WF4 train 20-24 / test 25-26",
         ("2020-03-25", "2025-01-01"),
         ("2025-01-01", "2026-04-15")),
    ]
    for label, (s1, e1), (s2, e2) in wf_configs:
        td = filter_data_by_date(data, s1, e1)
        vd = filter_data_by_date(data, s2, e2)
        r_train = run(td, f"{label} [TRAIN]", skip_years=())
        r_test = run(vd, f"{label} [TEST]", skip_years=())
        all_results[f"{label}_train"] = r_train
        all_results[f"{label}_test"] = r_test
        prow(f"{label} [TRAIN]", r_train)
        prow(f"{label} [TEST]", r_test)
        print()

    # ================================================================
    # Phase 4: 극한 스트레스 기간 (특정 시기 집중)
    # ================================================================
    phase(4, "극한 스트레스 기간")

    stress = [
        ("COVID 크래시 2020-03~04", "2020-03-25", "2020-05-01"),
        ("LUNA 붕괴 2022-05", "2022-04-15", "2022-06-15"),
        ("BTC 피크→크래시 2021-11~2022-06", "2021-11-01", "2022-07-01"),
        ("FTX 붕괴 2022-11", "2022-10-15", "2022-12-15"),
        ("2023 저변동 6개월", "2023-04-01", "2023-10-01"),
        ("2024 ETF 승인기 1~3월", "2024-01-01", "2024-04-01"),
        ("2025-12 ~ 2026-01 (최근)", "2025-12-01", "2026-02-01"),
    ]
    for label, start, end in stress:
        sd = filter_data_by_date(data, start, end)
        r = run(sd, label, skip_years=())
        all_results[f"stress_{label}"] = r
        prow(label, r)

    # ================================================================
    # Phase 5: 3코인 vs 4코인 (긴 기간 버전)
    # ================================================================
    phase(5, "3코인 vs 4코인 비교 (최대 기간)")

    for label, syms in [
        ("3코인 (BTC+ETH+SOL)", ["BTCUSDT","ETHUSDT","SOLUSDT"]),
        ("4코인 (BTC+ETH+SOL+XRP)", SYMBOLS),
    ]:
        for sim in [2, 3]:
            r = run(data, f"{label} {sim}pos",
                    enabled_symbols=syms, max_simultaneous=sim,
                    skip_years=(2023,))
            all_results[f"coins_{label}_{sim}pos"] = r
            prow(f"{label} {sim}pos", r)

    # ================================================================
    # Phase 6: 리스크 조정 테스트 (MC에서 파산률 11.3% 우려)
    # ================================================================
    phase(6, "리스크 조정 (MC 파산률 완화)")

    risk_scenarios = [
        ("기본 (risk 1.5, lev7, 3pos)", {}),
        ("보수 risk 1.0% (유지: lev7, 3pos)", {"risk_per_trade": 0.010}),
        ("보수 2pos (유지: risk 1.5, lev7)", {"max_simultaneous": 2}),
        ("보수 lev5 (유지: risk 1.5, 3pos)", {"max_leverage": 5}),
        ("안전 risk1.0 + 2pos", {"risk_per_trade": 0.010, "max_simultaneous": 2}),
        ("안전 risk1.0 + lev5", {"risk_per_trade": 0.010, "max_leverage": 5}),
        ("극보수 risk1.0+2pos+lev5", {"risk_per_trade": 0.010, "max_simultaneous": 2, "max_leverage": 5}),
    ]
    for label, kw in risk_scenarios:
        r = run(data, label, skip_years=(2023,), **kw)
        all_results[f"risk_{label}"] = r
        prow(label, r)

    # ================================================================
    # 정리
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"완료 | 소요 {elapsed:.0f}s ({elapsed/60:.1f}분)")
    print(f"{'='*120}")

    # 저장
    out = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "slippage": SLIP,
        "data_ranges": {
            sym: {
                "first": datetime.utcfromtimestamp(data[f"{sym}_60"]['timestamp'].iloc[0]/1000).strftime('%Y-%m-%d'),
                "last": datetime.utcfromtimestamp(data[f"{sym}_60"]['timestamp'].iloc[-1]/1000).strftime('%Y-%m-%d'),
                "count": len(data[f"{sym}_60"]),
            }
            for sym in SYMBOLS if f"{sym}_60" in data
        },
        "results": all_results,
        "elapsed_sec": round(elapsed, 1),
    }

    out_path = os.path.join(RESULTS_DIR, "v7_extended_backtest.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
