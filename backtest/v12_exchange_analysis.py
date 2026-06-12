#!/usr/bin/env python3
"""
거래소별 유동성 시나리오 시뮬
==============================
같은 V12 전략, 같은 시드, 같은 기간. 유동성 캡만 다름.

시나리오:
  1. Bybit only (1.0x) — 현재 배포 환경
  2. Binance only (2.5x) — Bybit → Binance 이전
  3. Bybit + Binance 분산 (3.5x) — 2개 거래소
  4. 3-exchange 분산 (5.0x) — Bybit + Binance + OKX
  5. 무한 유동성 (10x) — 이론적 상한

각 시나리오별로 잔고 구간별 수익/스킵/슬리피지 추출.
"""
import os, sys, json, time
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
import v12_realistic_engine as v12
from v12_realistic_engine import BASE_LIQUIDITY_CAPS_USD

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v12"

V11_PARAMS = {**DEFAULT_PARAMS, "ema_fast": 3, "ema_slow": 15, "sl_atr_mult": 1.5,
              "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
              "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30}

SEED = 16500.0
START = "2020-05-01"
END = "2026-04-18"


def filter_d(data, s, e):
    s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
    e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and 'timestamp' in v.columns:
            m = (v['timestamp']>=s_ts) & (v['timestamp']<e_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


def run_scenario(data, label: str, cap_multiplier: float):
    """유동성 캡을 multiplier 배수로 늘려서 시뮬. 나머지는 V12 동일."""
    # 모듈 변수 임시 수정
    original_caps = dict(v12.BASE_LIQUIDITY_CAPS_USD)
    original_hard = dict(v12.POSITION_HARD_CAP_USD)

    v12.BASE_LIQUIDITY_CAPS_USD = {k: v * cap_multiplier for k, v in original_caps.items()}
    v12.POSITION_HARD_CAP_USD = {k: v * cap_multiplier for k, v in original_hard.items()}

    try:
        r = v12.run_realistic_backtest(
            data, V11_PARAMS, SEED,
            start_year=2020, skip_years=(2023,),
            daily_cost_usd=DAILY_COST,
            slippage_pct_base=0.05,
            max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
            enabled_symbols=SYMBOLS,
            d1_filter_enable=True, d1_ema_period=2, d1_mode="price_above_ema",
            use_realism=True,
            api_fail_rate=0.0,  # GCP 가정
            funding_enabled=True,
        )
    finally:
        # 원복
        v12.BASE_LIQUIDITY_CAPS_USD = original_caps
        v12.POSITION_HARD_CAP_USD = original_hard

    return r


def analyze_by_balance_phase(trades, seed=SEED):
    """잔고 구간별 통계 추출.
    - 시드 기반 누적 balance 계산
    - 구간별로 bucketize
    """
    sorted_t = sorted(trades, key=lambda t: t["timestamp"])
    bal = seed
    # 잔고 구간 정의 (USD)
    phases = [
        (0, 10_000, "≤$10k"),
        (10_000, 50_000, "$10k-$50k"),
        (50_000, 200_000, "$50k-$200k"),
        (200_000, 1_000_000, "$200k-$1M"),
        (1_000_000, 5_000_000, "$1M-$5M"),
        (5_000_000, 20_000_000, "$5M-$20M"),
        (20_000_000, float('inf'), ">$20M"),
    ]
    stats = {lbl: {"n": 0, "wins": 0, "pnl": 0, "slip_pct_sum": 0, "slip_n": 0}
             for _, _, lbl in phases}
    crossing_ts = {}  # 각 구간 처음 진입한 timestamp

    def phase_of(b):
        for low, high, lbl in phases:
            if low <= b < high:
                return lbl
        return ">$20M"

    for t in sorted_t:
        phase = phase_of(bal)
        st = stats[phase]
        st["n"] += 1
        if t["pnl"] > 0:
            st["wins"] += 1
        st["pnl"] += t["pnl"]
        if "slip_pct" in t:
            st["slip_pct_sum"] += t["slip_pct"]
            st["slip_n"] += 1
        bal += t["pnl"]
        # 구간 최초 진입 기록
        if phase not in crossing_ts:
            crossing_ts[phase] = t["timestamp"]

    # 정리
    out = []
    for low, high, lbl in phases:
        s = stats[lbl]
        if s["n"] == 0:
            continue
        avg_slip = s["slip_pct_sum"] / s["slip_n"] if s["slip_n"] else 0
        wr = s["wins"] / s["n"] * 100
        first_ts = crossing_ts.get(lbl)
        first_dt = datetime.utcfromtimestamp(first_ts/1000).strftime('%Y-%m-%d') if first_ts else "?"
        out.append({
            "phase": lbl, "n": s["n"], "wins": s["wins"], "wr": round(wr, 1),
            "pnl": round(s["pnl"], 0), "avg_slip_pct": round(avg_slip, 4),
            "first_entry_date": first_dt,
        })
    return out


def main():
    print(f"데이터 로드: {START} ~ {END}")
    data_full = _load_data()
    data = filter_d(data_full, START, END)

    scenarios = [
        ("1.0x (Bybit only)", 1.0),
        ("2.5x (Binance)", 2.5),
        ("3.5x (Bybit+Binance)", 3.5),
        ("5.0x (3-exchange)", 5.0),
        ("10.0x (무한)", 10.0),
    ]

    results = {}
    for label, mult in scenarios:
        print(f"\n[{label}] 시뮬 중...")
        t0 = time.time()
        r = run_scenario(data, label, mult)
        r['elapsed'] = round(time.time() - t0, 1)
        r['multiplier'] = mult
        phase_stats = analyze_by_balance_phase(r["_trades"])
        r['phase_stats'] = phase_stats
        results[label] = r
        print(f"  {label}: 최종 ${r['final_balance']:,.0f} ({r['final_balance']/SEED:.1f}x), "
              f"스킵 {r['liquidity_skipped']}, 펀딩 ${r['funding_paid_total']:,.0f}, "
              f"{r['elapsed']}s")

    # 비교표
    print(f"\n{'='*100}")
    print(f"🏆 거래소 유동성 시나리오 비교 (시드 $16,500 → 6년 후)")
    print(f"{'='*100}")
    print(f"{'시나리오':<25}{'최종 USD':>14}{'USD 배수':>10}{'XBT 배수':>10}"
          f"{'DD':>6}{'스킵':>7}{'펀딩':>11}")
    print("-" * 100)

    btc_end = float(data_full["BTCUSDT_60"].iloc[-1]["close"])
    btc_start_approx = 8782
    for label, _ in scenarios:
        r = results[label]
        usd_mult = r['final_balance'] / SEED
        xbt_mult = (r['final_balance'] / btc_end) / (SEED / btc_start_approx)
        print(f"{label:<25}${r['final_balance']:>13,.0f}{usd_mult:>9.1f}x{xbt_mult:>9.1f}x"
              f"{r['max_dd']:>5.1f}%{r['liquidity_skipped']:>7}${r['funding_paid_total']:>10,.0f}")

    # vs Bybit baseline 개선율
    baseline = results["1.0x (Bybit only)"]['final_balance']
    print(f"\n📊 Bybit 대비 개선율")
    for label, _ in scenarios:
        r = results[label]
        improvement = (r['final_balance'] - baseline) / baseline * 100
        print(f"  {label:<25} {improvement:+.1f}%")

    # 잔고 구간별 분석 (Bybit only)
    print(f"\n{'='*100}")
    print(f"📈 잔고 구간별 스킵/슬리피지 분석")
    print(f"{'='*100}")
    print(f"{'Scenario':<25}{'Phase':<20}{'거래':>6}{'승률':>6}{'PnL':>14}{'슬리피지':>10}{'첫 진입':>14}")
    print("-" * 100)
    for label, _ in scenarios:
        for p in results[label]['phase_stats']:
            print(f"{label:<25}{p['phase']:<20}{p['n']:>6}{p['wr']:>5.1f}%"
                  f"${p['pnl']:>13,.0f}{p['avg_slip_pct']:>9.3f}%{p['first_entry_date']:>14}")
        print()

    # 저장
    out_data = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED, "period": f"{START} ~ {END}",
        "scenarios": {
            label: {
                "multiplier": r['multiplier'],
                "final_balance": r['final_balance'],
                "net_profit": r['net_profit'],
                "max_dd": r['max_dd'],
                "trades": r['total_trades'],
                "win_rate": r['win_rate'],
                "liquidity_skipped": r['liquidity_skipped'],
                "funding_paid": r['funding_paid_total'],
                "phase_stats": r['phase_stats'],
            }
            for label, r in results.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, "v12_exchange_analysis.json"), "w") as f:
        json.dump(out_data, f, default=str, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/v12_exchange_analysis.json")


if __name__ == "__main__":
    main()
