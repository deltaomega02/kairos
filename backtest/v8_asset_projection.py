#!/usr/bin/env python3
"""
v8 자산 추이 시뮬레이션 + 보고서 생성
=====================================
1. 6년 v6 풀 패키지 백테스트 → 월별 잔고 추이 추출
2. 4가지 시나리오 (낙관/평균/현실/비관)
3. 현재 $523에서 1년/3년/5년 forward projection
4. 모든 결과를 사용자가 읽기 쉬운 보고서로 저장
"""
import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from statistics import mean

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import DATA_DIR
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v8"
REPORT_DIR = "~/Projects/HERMES_백테스팅"
os.makedirs(RESULTS_DIR, exist_ok=True)

V6_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30,
}
SEED = 580.0
CURRENT_BALANCE = 523.0  # 사용자의 현재 잔고
SLIP = 0.05
DAILY_COST_USD = 1150 / 1470
USDKRW = 1470
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def krw(usd):
    """USD → KRW 문자열"""
    return f"₩{usd*USDKRW:,.0f}"


def load_data():
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


def run_v6_full_package(data, skip=(2023,)):
    """v6 풀 패키지 백테스트 실행"""
    r = run_shared_backtest(
        data, V6_PARAMS, SEED,
        use_funding=True,
        trailing_activation=1.2, trailing_distance=0.1,
        block_sol_long=True,
        skip_years=skip,
        daily_cost_usd=DAILY_COST_USD,
        ruin_threshold=15.0,
        use_cooldown=False,
        slippage_pct=SLIP,
        max_simultaneous=3,
        max_leverage=7,
        enabled_symbols=SYMBOLS,
    )
    return r


def replay_trades_monthly(trades, initial_balance, daily_cost):
    """거래를 시간순 재생하며 월별 잔고 스냅샷 생성"""
    balance = initial_balance
    monthly = {}
    monthly_start = {}
    monthly_end = {}
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    monthly_wins = defaultdict(int)

    # 일별 서버비 차감을 위해 timestamp 정렬
    sorted_trades = sorted(trades, key=lambda t: t["timestamp"])

    if not sorted_trades:
        return {}

    last_day = sorted_trades[0]["timestamp"] // 86400000

    for t in sorted_trades:
        ts = t["timestamp"]
        day = ts // 86400000
        # 일별 서버비 차감
        days_passed = day - last_day
        if days_passed > 0:
            balance -= days_passed * daily_cost
            last_day = day

        # 거래 적용
        balance += t["pnl"]

        # 월별 통계
        month_key = datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m")
        if month_key not in monthly_start:
            monthly_start[month_key] = balance - t["pnl"]
        monthly_end[month_key] = balance
        monthly_pnl[month_key] += t["pnl"]
        monthly_trades[month_key] += 1
        if t["pnl"] > 0:
            monthly_wins[month_key] += 1

    # 통합
    monthly = {}
    for k in sorted(monthly_end.keys()):
        monthly[k] = {
            "start": round(monthly_start[k], 2),
            "end": round(monthly_end[k], 2),
            "pnl": round(monthly_pnl[k], 2),
            "trades": monthly_trades[k],
            "wins": monthly_wins[k],
            "win_rate": round(monthly_wins[k]/monthly_trades[k]*100, 1) if monthly_trades[k] else 0,
        }
    return monthly


def compute_yearly_compound_rates(monthly):
    """월별 데이터에서 연도별 복리 성장률 계산"""
    yearly = defaultdict(lambda: {"start": None, "end": None, "trades": 0, "wins": 0})
    for month_key, m in monthly.items():
        year = month_key.split("-")[0]
        if yearly[year]["start"] is None:
            yearly[year]["start"] = m["start"]
        yearly[year]["end"] = m["end"]
        yearly[year]["trades"] += m["trades"]
        yearly[year]["wins"] += m["wins"]

    result = {}
    for y, data in sorted(yearly.items()):
        if data["start"] and data["start"] > 0:
            growth = (data["end"] / data["start"] - 1) * 100
        else:
            growth = 0
        wr = data["wins"]/data["trades"]*100 if data["trades"] else 0
        result[y] = {
            "start": round(data["start"], 2),
            "end": round(data["end"], 2),
            "growth_pct": round(growth, 1),
            "trades": data["trades"],
            "win_rate": round(wr, 1),
        }
    return result


def project_forward(current_balance, monthly_rates, n_months):
    """월별 성장률로 forward projection"""
    balance = current_balance
    progression = [(0, balance)]
    for i in range(n_months):
        rate = monthly_rates[i % len(monthly_rates)]
        balance *= (1 + rate / 100)
        # 서버비 차감
        balance -= DAILY_COST_USD * 30
        if balance < 15:
            balance = 0
            progression.append((i + 1, 0))
            break
        progression.append((i + 1, balance))
    return progression


def main():
    print("v8 자산 추이 시뮬레이션 + 보고서 생성")
    print("=" * 80)

    print("\n[데이터 로드]")
    data = load_data()
    print("  완료")

    print("\n[v6 풀 패키지 6년 백테스트 실행 (수동 감독)]")
    result = run_v6_full_package(data, skip=(2023,))
    trades = result.get("_trades", [])
    print(f"  총 거래: {len(trades)}")
    print(f"  최종 잔고: ${result['final_balance']:,.0f}")
    print(f"  순이익: ${result['net_profit']:+,.0f}")
    print(f"  최대 DD: {result['max_dd']}%")
    print(f"  승률: {result['win_rate']}%")

    print("\n[월별 잔고 추이 재생]")
    monthly = replay_trades_monthly(trades, SEED, DAILY_COST_USD)
    print(f"  월별 데이터: {len(monthly)}개월")

    yearly = compute_yearly_compound_rates(monthly)
    print(f"  연도별 데이터: {len(yearly)}년")

    # 월별 성장률 추출 (compound rate)
    monthly_growth_rates = []
    sorted_months = sorted(monthly.keys())
    for i, m in enumerate(sorted_months):
        if monthly[m]["start"] > 0:
            rate = (monthly[m]["end"] / monthly[m]["start"] - 1) * 100
            monthly_growth_rates.append(rate)

    avg_monthly = mean(monthly_growth_rates) if monthly_growth_rates else 0
    print(f"  평균 월간 성장률: {avg_monthly:+.2f}%")

    # 연도별 실제 성장률 (백테스트 기반)
    yearly_growth_pcts = {}
    for y, ydata in yearly.items():
        if ydata["start"] and ydata["start"] > 0:
            yearly_growth_pcts[y] = ydata["growth_pct"]

    print("\n[연도별 실제 성장률]")
    for y, g in sorted(yearly_growth_pcts.items()):
        print(f"  {y}: {g:+.1f}%")

    # 4가지 시나리오를 연도별 성장률 기반으로
    print("\n[Forward Projection 시나리오 — 연도 단위]")

    sorted_yearly = sorted(yearly_growth_pcts.values())
    n_years_data = len(sorted_yearly)

    # 시나리오 정의
    pessimistic_yearly = sorted_yearly[0] if sorted_yearly else 0  # 최악
    conservative_yearly = sorted_yearly[max(0, n_years_data//4)]  # 25% 분위
    realistic_yearly = sorted_yearly[n_years_data//2] if sorted_yearly else 0  # 중앙값
    optimistic_yearly = sorted_yearly[3*n_years_data//4] if sorted_yearly else 0  # 75% 분위

    # CAGR (전체 6년 평균)
    if SEED > 0:
        total_years = len(yearly) if yearly else 6
        cagr_yearly = (result["final_balance"] / SEED) ** (1/total_years) - 1
        cagr_yearly *= 100
    else:
        cagr_yearly = 0

    print(f"\n  비관 (최악 연):   {pessimistic_yearly:+.1f}%/년")
    print(f"  보수 (25% 분위): {conservative_yearly:+.1f}%/년")
    print(f"  현실 (중앙값):   {realistic_yearly:+.1f}%/년")
    print(f"  CAGR (전체):     {cagr_yearly:+.1f}%/년")
    print(f"  낙관 (75% 분위): {optimistic_yearly:+.1f}%/년")

    def project_yearly(annual_rate_pct, years):
        """연복리로 forward projection"""
        path = [(0, CURRENT_BALANCE)]
        bal = CURRENT_BALANCE
        annual_server_cost = DAILY_COST_USD * 365
        for i in range(years):
            bal *= (1 + annual_rate_pct / 100)
            bal -= annual_server_cost
            if bal < 15:
                bal = 0
            path.append((i + 1, round(bal, 2)))
            if bal == 0:
                break
        return path

    horizons_y = [1, 2, 3, 5, 10]

    projections = {}
    for label, rate in [
        ("pessimistic", pessimistic_yearly),
        ("conservative", conservative_yearly),
        ("realistic", realistic_yearly),
        ("cagr", cagr_yearly),
        ("optimistic", optimistic_yearly),
    ]:
        projections[label] = {
            "annual_rate_pct": round(rate, 1),
            "horizons": {},
        }
        for h in horizons_y:
            path = project_yearly(rate, h)
            final = path[-1][1] if path else 0
            projections[label]["horizons"][f"{h}y"] = {
                "final_balance": round(final, 2),
                "path": path,
            }
        print(f"\n  [{label}] {rate:+.1f}%/년")
        for h in horizons_y:
            f_b = projections[label]["horizons"][f"{h}y"]["final_balance"]
            print(f"    {h}년: ${f_b:>12,.0f} ({krw(f_b)})")

    # JSON 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "current_balance": CURRENT_BALANCE,
        "backtest_seed": SEED,
        "backtest_result": {
            "total_trades": result["total_trades"],
            "win_rate": result["win_rate"],
            "net_profit": result["net_profit"],
            "max_dd": result["max_dd"],
            "final_balance": result["final_balance"],
        },
        "monthly_history": monthly,
        "yearly_history": yearly,
        "monthly_avg_growth_pct": round(avg_monthly, 2),
        "projections": projections,
    }

    out_path = os.path.join(RESULTS_DIR, "v8_asset_projection.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n저장: {out_path}")

    return output


if __name__ == "__main__":
    main()
