#!/usr/bin/env python3
"""
V12 심층 시나리오 분석
=======================
1. Multi-start-date: 61개 시작 시점 (2020-05 ~ 2025-05)
2. Stress periods: 7개 역사적 악재 구간
3. 현재 regime 매칭
4. Forward projection
"""
import os, sys, json, time
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v12_realistic_engine import run_realistic_backtest

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v12"
os.makedirs(RESULTS_DIR, exist_ok=True)

V11_PARAMS = {**DEFAULT_PARAMS, "ema_fast": 3, "ema_slow": 15, "sl_atr_mult": 1.5,
              "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
              "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30}


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


def get_btc_1d_series(data_full):
    """BTC 1D OHLC 변환"""
    df = data_full["BTCUSDT_60"].copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    d = df[["open", "high", "low", "close", "volume"]].resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    return d


def run_v12(data, seed, start_year=2020):
    return run_realistic_backtest(
        data, V11_PARAMS, seed,
        start_year=start_year, skip_years=(2023,),
        daily_cost_usd=DAILY_COST,
        slippage_pct_base=0.05,
        max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
        enabled_symbols=SYMBOLS,
        d1_filter_enable=True, d1_ema_period=2, d1_mode="price_above_ema",
        use_realism=True, api_fail_rate=0.0, funding_enabled=True,
    )


# ============================================================
# Part 1: Multi-start-date backtest
# ============================================================
def part1_multi_start(data_full):
    """매월 시작 시점 바꿔서 V12 실행. 시작 타이밍에 따른 결과 분포."""
    print("=" * 80)
    print("🕰 Part 1: Multi-Start-Date Backtest (61개 시점)")
    print("=" * 80)

    results = []
    start_dates = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            if year == 2020 and month < 5:
                continue
            if year == 2025 and month > 5:
                continue
            start_dates.append(f"{year:04d}-{month:02d}-01")

    SEED = 16500.0

    for i, sd in enumerate(start_dates):
        # 6년 또는 최대 기간
        sd_dt = datetime.strptime(sd, "%Y-%m-%d")
        end_dt = min(sd_dt + timedelta(days=365*6), datetime(2026, 4, 18))
        ed = end_dt.strftime("%Y-%m-%d")
        data = filter_d(data_full, sd, ed)
        try:
            r = run_v12(data, SEED, start_year=sd_dt.year)
            months = (end_dt - sd_dt).days / 30.4
            results.append({
                "start_date": sd,
                "months": round(months, 1),
                "final_balance": r["final_balance"],
                "net_profit": r["net_profit"],
                "multiple": round(r["final_balance"] / SEED, 2),
                "dd": r["max_dd"],
                "trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "ruined": r["ruined"],
            })
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(start_dates)}: 시작={sd} → {r['final_balance']/SEED:.1f}x")
        except Exception as e:
            print(f"  {sd}: 실패 {e}")

    # 결과 분포
    alive = [r for r in results if not r["ruined"]]
    ruined = [r for r in results if r["ruined"]]
    multiples = [r["multiple"] for r in alive]
    multiples.sort()

    print(f"\n결과 분포 ({len(results)}개 시작 시점):")
    print(f"  파산: {len(ruined)}개 ({len(ruined)/len(results)*100:.0f}%)")
    print(f"  생존: {len(alive)}개")
    if multiples:
        print(f"  중앙값 배수: {multiples[len(multiples)//2]:.1f}x")
        print(f"  최저 생존: {multiples[0]:.1f}x")
        print(f"  최고: {multiples[-1]:.1f}x")
        print(f"  상위 25%: {multiples[int(len(multiples)*0.75)]:.1f}x")
        print(f"  하위 25%: {multiples[int(len(multiples)*0.25)]:.1f}x")

    # 시작 시점별 배수 출력
    print(f"\n[시작 시점별 최종 배수]")
    for r in results:
        print(f"  {r['start_date']}: {r['multiple']:>6.1f}x ({r['months']}개월) "
              f"DD {r['dd']:>4.1f}%  {'RUINED' if r['ruined'] else ''}")

    return results


# ============================================================
# Part 2: 7개 악재 구간 V12 성능
# ============================================================
def part2_stress(data_full):
    print("=" * 80)
    print("⛈️ Part 2: 7개 역사적 악재 구간 V12 성능")
    print("=" * 80)

    stress_periods = [
        ("COVID 크래시", "2020-03-01", "2020-05-01"),
        ("2021 5월 Flash Crash", "2021-05-01", "2021-07-01"),
        ("BTC 피크→바닥 (21-11 → 22-06)", "2021-11-01", "2022-07-01"),
        ("Luna 붕괴", "2022-04-15", "2022-06-15"),
        ("2022 Q2 Crash", "2022-04-01", "2022-07-01"),
        ("FTX 붕괴", "2022-10-15", "2022-12-31"),
        ("2023 저변동 지옥 (전년)", "2023-01-01", "2024-01-01"),
        ("2024-08 일본 carry unwind", "2024-07-15", "2024-09-15"),
        ("2025-12 ~ 2026-01 최근 크래시", "2025-12-01", "2026-02-15"),
        ("2026-02 ~ 현재 (이란 긴장 시기)", "2026-02-01", "2026-04-18"),
    ]

    SEED = 16500.0
    results = []
    for label, sd, ed in stress_periods:
        data = filter_d(data_full, sd, ed)
        try:
            r = run_v12(data, SEED, start_year=int(sd[:4]))
            days = (datetime.strptime(ed, "%Y-%m-%d") -
                    datetime.strptime(sd, "%Y-%m-%d")).days
            result = {
                "label": label, "start": sd, "end": ed, "days": days,
                "final": r["final_balance"],
                "profit": r["net_profit"],
                "multiple": round(r["final_balance"]/SEED, 2),
                "dd": r["max_dd"],
                "trades": r["total_trades"], "wr": r["win_rate"],
                "ruined": r["ruined"],
            }
            results.append(result)
            status = "💀 RUIN" if r["ruined"] else "✓"
            print(f"  {status} {label:<38} {days:>4}d | 배수 {result['multiple']:>5.2f}x | "
                  f"DD {r['max_dd']:>4.1f}% | 거래 {r['total_trades']:>4} | WR {r['win_rate']:>4.1f}%")
        except Exception as e:
            print(f"  ERROR {label}: {e}")
    return results


# ============================================================
# Part 3: 현재 시장 regime 매칭
# ============================================================
def regime_signature(btc_1d, end_date, lookback_days=60):
    """60일 시장 signature 벡터 계산."""
    end_dt = pd.Timestamp(end_date, tz="UTC")
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    window = btc_1d[(btc_1d.index >= start_dt) & (btc_1d.index <= end_dt)]
    if len(window) < lookback_days * 0.5:
        return None

    close = window["close"]
    high = window["high"]
    low = window["low"]

    # 1. 누적 수익률 (60일)
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    # 2. 변동성 (일일 수익률 표준편차, annualized)
    daily_ret = close.pct_change().dropna()
    volatility = daily_ret.std() * np.sqrt(365) * 100
    # 3. Max DD within window
    peak = close.cummax()
    dd = ((peak - close) / peak).max() * 100
    # 4. 평균 절대 일일 범위
    daily_range = ((high - low) / close).mean() * 100
    # 5. 최근 30일 추세 방향
    mid = len(close) // 2
    first_half = close.iloc[:mid].mean()
    second_half = close.iloc[mid:].mean()
    trend = (second_half / first_half - 1) * 100
    # 6. 현재가 vs 60일 고점 (얼마나 내려왔나)
    from_peak = ((close.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]) * 100

    return np.array([total_return, volatility, dd, daily_range, trend, from_peak])


def part3_regime_match(data_full):
    print("=" * 80)
    print("🔍 Part 3: 현재 시장 Regime 매칭")
    print("=" * 80)

    btc_1d = get_btc_1d_series(data_full)

    # 현재 signature (지금 기준)
    current_date = "2026-04-18"
    current_sig = regime_signature(btc_1d, current_date, lookback_days=60)
    if current_sig is None:
        print("현재 signature 계산 실패")
        return []

    labels = ["60일 수익률", "연환산 변동성", "60일 Max DD",
              "일평균 범위", "30일 추세", "피크 대비"]
    print(f"\n현재 시장 지표 ({current_date} 기준, 60일 lookback):")
    for lbl, val in zip(labels, current_sig):
        print(f"  {lbl}: {val:+.2f}%")

    # 2020-05 ~ 2025-10 각 월말마다 signature 계산 → 현재와 유사도 비교
    historical_sigs = []
    dates = pd.date_range("2020-07-01", "2025-11-01", freq="MS")
    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        sig = regime_signature(btc_1d, dt_str, lookback_days=60)
        if sig is not None:
            historical_sigs.append((dt_str, sig))

    # Normalize (standardize) using historical mean/std
    all_sigs = np.array([s for _, s in historical_sigs])
    mean = all_sigs.mean(axis=0)
    std = all_sigs.std(axis=0)
    std[std == 0] = 1

    def normalize(sig):
        return (sig - mean) / std

    current_norm = normalize(current_sig)
    # Euclidean distance 계산
    distances = []
    for dt_str, sig in historical_sigs:
        sig_norm = normalize(sig)
        dist = np.linalg.norm(current_norm - sig_norm)
        distances.append((dt_str, dist, sig))
    distances.sort(key=lambda x: x[1])

    print(f"\n🔑 가장 유사한 과거 시점 TOP 10:")
    print(f"  {'시점':<12}{'거리':>7}  {'수익률':>8}{'변동성':>8}{'Max DD':>8}{'일범위':>8}{'추세':>8}{'피크대비':>10}")
    for i, (dt_str, dist, sig) in enumerate(distances[:10], 1):
        tr, vol, dd, rng, trend, fp = sig
        print(f"  {dt_str:<12}{dist:>6.3f}  {tr:>+7.1f}%{vol:>7.1f}%{dd:>7.1f}%"
              f"{rng:>7.2f}%{trend:>+7.1f}%{fp:>+9.1f}%")

    return distances[:10]


# ============================================================
# Part 4: Forward Projection (매칭 시점 기반)
# ============================================================
def part4_forward(data_full, matched_periods):
    print("=" * 80)
    print("🔮 Part 4: Forward Projection (매칭 시점 이후 V12 성능)")
    print("=" * 80)

    if not matched_periods:
        print("  매칭 데이터 없음")
        return []

    SEED = 571.0   # 사용자 현재 잔고
    horizons = [(30, "1개월"), (90, "3개월"), (180, "6개월"), (365, "1년")]

    print(f"\n시드 ${SEED} (현재 잔고 기준), 각 매칭 시점 이후 N일간 V12 실행")

    results = []
    for dt_str, dist, sig in matched_periods[:5]:  # 상위 5개
        sd_dt = datetime.strptime(dt_str, "%Y-%m-%d")
        print(f"\n매칭 기준점: {dt_str} (거리 {dist:.3f})")
        for days, hlabel in horizons:
            ed_dt = sd_dt + timedelta(days=days)
            if ed_dt > datetime(2026, 4, 18):
                continue
            ed_str = ed_dt.strftime("%Y-%m-%d")
            data = filter_d(data_full, dt_str, ed_str)
            try:
                r = run_v12(data, SEED, start_year=sd_dt.year)
                row = {
                    "match_date": dt_str, "horizon": hlabel, "days": days,
                    "final": r["final_balance"],
                    "multiple": round(r["final_balance"]/SEED, 2),
                    "dd": r["max_dd"],
                    "trades": r["total_trades"], "wr": r["win_rate"],
                    "ruined": r["ruined"],
                }
                results.append(row)
                status = "💀" if r["ruined"] else ("🟢" if row["multiple"] >= 1.0 else "🔴")
                print(f"  {status} {hlabel:<6} → ${r['final_balance']:>8,.0f} "
                      f"({row['multiple']:>5.2f}x)  DD {r['max_dd']:>4.1f}%  거래 {r['total_trades']}")
            except Exception as e:
                print(f"  {hlabel}: ERROR {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Forward Projection 종합 (시드 $571)")
    print(f"{'='*80}")
    for hlabel in ["1개월", "3개월", "6개월", "1년"]:
        relevant = [r for r in results if r["horizon"] == hlabel]
        if not relevant:
            continue
        mults = [r["multiple"] for r in relevant if not r["ruined"]]
        ruined_n = sum(1 for r in relevant if r["ruined"])
        if mults:
            mults.sort()
            median = mults[len(mults)//2]
            best = mults[-1]
            worst = mults[0]
            ruin_rate = ruined_n / len(relevant) * 100
            print(f"\n  {hlabel}:")
            print(f"    파산 확률: {ruin_rate:.0f}% ({ruined_n}/{len(relevant)})")
            print(f"    중앙값:   {median:.2f}x → ${SEED*median:,.0f}")
            print(f"    최악:     {worst:.2f}x → ${SEED*worst:,.0f}")
            print(f"    최고:     {best:.2f}x → ${SEED*best:,.0f}")
            print(f"    사례들:   {[round(m,2) for m in mults]}")
    return results


def main():
    print(f"Starting {datetime.now().isoformat()}")
    t0 = time.time()
    data_full = _load_data()

    p1 = part1_multi_start(data_full)
    print(f"\nPart 1 완료: {time.time()-t0:.0f}s\n")

    p2 = part2_stress(data_full)
    print(f"\nPart 2 완료: {time.time()-t0:.0f}s\n")

    matched = part3_regime_match(data_full)
    print(f"\nPart 3 완료: {time.time()-t0:.0f}s\n")

    p4 = part4_forward(data_full, matched)
    print(f"\nPart 4 완료: {time.time()-t0:.0f}s\n")

    print(f"\n총 소요: {time.time()-t0:.0f}s")

    # Save all
    out = {
        "timestamp": datetime.now().isoformat(),
        "multi_start": p1,
        "stress_periods": p2,
        "regime_match": [{"date": d, "distance": float(dist), "signature": sig.tolist()}
                          for d, dist, sig in matched],
        "forward_projection": p4,
    }
    with open(os.path.join(RESULTS_DIR, "v12_deep_scenarios.json"), "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/v12_deep_scenarios.json")


if __name__ == "__main__":
    main()
