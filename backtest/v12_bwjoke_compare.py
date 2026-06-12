#!/usr/bin/env python3
"""
V12 vs bwjoke 비교 시뮬레이션
=============================
동일 조건:
  - 시드: $16,500 (bwjoke 1.84 XBT × 2020-05 BTC ~$9,000)
  - 시작: 2020-05-01
  - 종료: 2026-04-17

bwjoke: 52.4x XBT (USD 환산 ~440배)
HERMES V12: ???
"""
import os, sys, json, time
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, "/Users/sue/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v12_realistic_engine import run_realistic_backtest

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v12"
os.makedirs(RESULTS_DIR, exist_ok=True)

V11_PARAMS = {**DEFAULT_PARAMS, "ema_fast": 3, "ema_slow": 15, "sl_atr_mult": 1.5,
              "tp_rr_ratio": 6.0, "entry_score_threshold": 40,
              "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 30}

SEED_USD = 16500.0
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


def get_btc_price_at(data, timestamp_ms):
    """주어진 시간의 BTC 종가 (가장 가까운 1H 봉)"""
    df = data.get("BTCUSDT_60")
    if df is None or df.empty:
        return None
    idx = (df["timestamp"] - timestamp_ms).abs().idxmin()
    return float(df.iloc[idx]["close"])


def load_bwjoke_equity():
    """bwjoke equity curve 파싱 (XBT 단위)"""
    df = pd.read_csv('/tmp/BTC-Trading-Since-2020/derived-equity-curve.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 월별 요약
    df['ym'] = df['timestamp'].dt.to_period('M')
    monthly = df.groupby('ym').agg(
        wealth_xbt=('adjustedWealthXBT', 'last'),
        ts=('timestamp', 'last'),
    ).reset_index()
    return df, monthly


def main():
    data_full = _load_data()
    print(f"데이터 로드 완료. 기간 필터링: {START} ~ {END}")
    data = filter_d(data_full, START, END)

    # BTC 시작 가격 (시드 환산용)
    first_ts = int(datetime.strptime(START, "%Y-%m-%d").timestamp() * 1000)
    btc_start = get_btc_price_at(data_full, first_ts)
    print(f"2020-05-01 BTC 가격: ${btc_start:,.0f}")
    # 1.84 XBT × 9000 ≈ 16,560. 우리는 $16,500 고정 사용.

    # ===== HERMES V12 (현실) 실행 =====
    print(f"\n[V12 Realistic] Seed=${SEED_USD:,.0f}, 2020-05~2026-04...")
    t0 = time.time()
    r = run_realistic_backtest(
        data, V11_PARAMS, SEED_USD,
        start_year=2020,
        skip_years=(2023,),
        daily_cost_usd=DAILY_COST,
        slippage_pct_base=0.05,
        max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
        enabled_symbols=SYMBOLS,
        d1_filter_enable=True, d1_ema_period=2, d1_mode="price_above_ema",
        use_realism=True, api_fail_rate=0.05, funding_enabled=True,
    )
    print(f"  소요: {time.time()-t0:.1f}s")
    print(f"\n--- V12 Realistic 결과 ---")
    print(f"  시작: ${SEED_USD:,.0f}")
    print(f"  최종: ${r['final_balance']:,.0f}")
    print(f"  순이익: ${r['net_profit']:,.0f}")
    print(f"  USD 배수: {r['final_balance']/SEED_USD:.1f}x")
    print(f"  거래수: {r['total_trades']:,}")
    print(f"  승률: {r['win_rate']}%")
    print(f"  최대 DD: {r['max_dd']}%")
    print(f"  유동성 스킵: {r['liquidity_skipped']}")
    print(f"  API 실패: {r['api_failures']}")
    print(f"  펀딩 총 지출: ${r['funding_paid_total']:,.0f}")

    # ===== bwjoke 비교 =====
    print(f"\n[bwjoke] 로딩...")
    bwjoke_full, bwjoke_monthly = load_bwjoke_equity()
    bw_start_xbt = bwjoke_full['adjustedWealthXBT'].iloc[0]
    bw_end_xbt = bwjoke_full['adjustedWealthXBT'].iloc[-1]
    bw_mult_xbt = bw_end_xbt / bw_start_xbt
    bw_start_usd = bw_start_xbt * btc_start
    btc_end_price = get_btc_price_at(data_full, int(datetime.strptime(END, "%Y-%m-%d").timestamp()*1000))
    bw_end_usd = bw_end_xbt * btc_end_price
    bw_mult_usd = bw_end_usd / bw_start_usd

    print(f"\n--- bwjoke 결과 ---")
    print(f"  시작: {bw_start_xbt:.2f} XBT (${bw_start_usd:,.0f})")
    print(f"  최종: {bw_end_xbt:.2f} XBT (${bw_end_usd:,.0f})")
    print(f"  XBT 배수: {bw_mult_xbt:.1f}x")
    print(f"  USD 배수: {bw_mult_usd:.1f}x (BTC 상승 포함)")

    # ===== 비교 =====
    print(f"\n{'='*80}")
    print(f"🏆 최종 비교 (2020-05 ~ 2026-04, 시드 ≈$16,500)")
    print(f"{'='*80}")
    print(f"  지표                    bwjoke (인간)         HERMES V12 (현실 시뮬)")
    print(f"  {'-'*76}")
    print(f"  최종 USD 자산            ${bw_end_usd:>12,.0f}     ${r['final_balance']:>12,.0f}")
    print(f"  USD 배수                 {bw_mult_usd:>14.1f}x     {r['final_balance']/SEED_USD:>14.1f}x")
    print(f"  XBT 배수 (BTC 상승 제거) {bw_mult_xbt:>14.1f}x     {(r['final_balance']/btc_end_price)/(SEED_USD/btc_start):>14.1f}x")

    hermes_end_xbt = r['final_balance'] / btc_end_price
    hermes_xbt_mult = hermes_end_xbt / (SEED_USD / btc_start)
    print(f"\n순 edge 비교 (BTC 가격 변동 제거):")
    print(f"  bwjoke XBT 배수:  {bw_mult_xbt:.1f}x")
    print(f"  HERMES XBT 배수:  {hermes_xbt_mult:.1f}x")
    winner = "HERMES" if hermes_xbt_mult > bw_mult_xbt else "bwjoke"
    ratio = hermes_xbt_mult / bw_mult_xbt if bw_mult_xbt > 0 else 0
    print(f"  승자: {winner} ({ratio:.2f}배 차이)")

    # Equity curve 저장
    ec_df = pd.DataFrame(r["equity_curve"])
    ec_df['dt'] = pd.to_datetime(ec_df['ts'], unit='ms')
    ec_df['ym'] = ec_df['dt'].dt.to_period('M')
    hermes_monthly = ec_df.groupby('ym').agg(
        balance=('balance', 'last'),
        dd=('dd', 'max'),
    ).reset_index()

    # 월별 XBT 환산 (각 시점 BTC 가격으로)
    def usd_to_xbt_at(ts_ms, usd):
        price = get_btc_price_at(data_full, int(ts_ms))
        return usd / price if price else 0

    # 월별 비교 테이블
    print(f"\n{'='*80}")
    print(f"📊 월별 잔고 추이 (USD + XBT 환산)")
    print(f"{'='*80}")
    print(f"  {'월':<10}{'bwjoke XBT':>13}{'bwjoke USD':>15}{'HERMES USD':>15}{'HERMES XBT':>13}")

    bw_dict = {str(r['ym']): (r['wealth_xbt'], r['ts']) for _, r in bwjoke_monthly.iterrows()}
    hm_dict = {str(r['ym']): r['balance'] for _, r in hermes_monthly.iterrows()}
    all_months = sorted(set(list(bw_dict.keys()) + list(hm_dict.keys())))
    # 3개월마다 샘플링
    for m in all_months[::3]:
        bw_xbt, bw_ts = bw_dict.get(m, (0, None))
        if bw_ts is not None:
            bw_price = get_btc_price_at(data_full, int(bw_ts.timestamp() * 1000))
            bw_usd = bw_xbt * bw_price
        else:
            bw_usd = 0
        hm_usd = hm_dict.get(m, 0)
        if hm_usd > 0:
            hm_ts = pd.Period(m).end_time.timestamp() * 1000
            hm_xbt = usd_to_xbt_at(hm_ts, hm_usd)
        else:
            hm_xbt = 0

        def _f(v, unit='$'):
            if unit == 'x':
                return f"{v:,.2f}" if v else "—"
            if v >= 1e6:
                return f"${v/1e6:,.2f}M" if unit == '$' else f"{v:,.2f}"
            if v >= 1e3:
                return f"${v/1e3:,.1f}k" if unit == '$' else f"{v:,.2f}"
            return f"${v:,.0f}" if unit == '$' else f"{v:,.2f}"

        print(f"  {m:<10}{_f(bw_xbt,'x'):>13}{_f(bw_usd):>15}{_f(hm_usd):>15}{_f(hm_xbt,'x'):>13}")

    # 저장
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed_usd": SEED_USD,
            "start": START, "end": END,
            "btc_start_price": btc_start, "btc_end_price": btc_end_price,
        },
        "hermes_v12": {
            "final_balance": r['final_balance'],
            "net_profit": r['net_profit'],
            "usd_multiple": round(r['final_balance']/SEED_USD, 2),
            "xbt_multiple": round(hermes_xbt_mult, 2),
            "max_dd": r['max_dd'],
            "trades": r['total_trades'], "win_rate": r['win_rate'],
            "liquidity_skipped": r['liquidity_skipped'],
            "api_failures": r['api_failures'],
            "funding_paid": r['funding_paid_total'],
        },
        "bwjoke": {
            "start_xbt": bw_start_xbt, "end_xbt": bw_end_xbt,
            "start_usd": bw_start_usd, "end_usd": bw_end_usd,
            "xbt_multiple": bw_mult_xbt,
            "usd_multiple": bw_mult_usd,
        },
        "comparison": {
            "hermes_xbt_mult": hermes_xbt_mult,
            "bwjoke_xbt_mult": bw_mult_xbt,
            "ratio": ratio,
            "winner": winner,
        },
    }
    with open(os.path.join(RESULTS_DIR, "v12_bwjoke_compare.json"), "w") as f:
        json.dump(out, f, default=str, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/v12_bwjoke_compare.json")


if __name__ == "__main__":
    main()
