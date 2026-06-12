#!/usr/bin/env python3
"""
HERMES 장기 검증 백테스트
=========================
2년 백테스트 Top 파라미터를 4년+ 장기 데이터로 검증.
2022-01 ~ 2026-04 (하락장 + 회복장 + 상승장)
"""

import os
import sys

# 기존 백테스트 엔진 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import (
    load_or_fetch, compute_regime_indicators, compute_entry_indicators,
    BacktestRegimeEngine, align_regime_to_entry, run_single_backtest,
    SYMBOLS, TAKER_FEE_PCT, DEFAULT_PARAMS, INITIAL_BALANCE,
    DATA_DIR, RESULTS_DIR,
)

import json
from datetime import datetime

# 장기 데이터 설정
LONG_START = "2022-01-01"
LONG_END = "2026-04-08"

# 테스트할 파라미터 셋
TEST_CONFIGS = {
    "현재시스템 (SL1.5/TP2.5/S60/EMA9-21)": {
        **DEFAULT_PARAMS,
    },
    "#1 (SL2.0/TP4.0/S40/EMA7-21)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0, "tp_rr_ratio": 4.0,
        "entry_score_threshold": 40, "ema_fast": 7, "ema_slow": 21,
    },
    "#2 (SL2.0/TP4.0/S40/EMA7-15)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0, "tp_rr_ratio": 4.0,
        "entry_score_threshold": 40, "ema_fast": 7, "ema_slow": 15,
    },
    "#4 (SL2.0/TP4.0/S40/EMA9-26)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0, "tp_rr_ratio": 4.0,
        "entry_score_threshold": 40, "ema_fast": 9, "ema_slow": 26,
    },
    "#12 (SL2.0/TP3.0/S40/EMA9-21)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0, "tp_rr_ratio": 3.0,
        "entry_score_threshold": 40,
    },
    "#18 (SL2.5/TP2.5/S60/EMA9-21)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.5, "tp_rr_ratio": 2.5,
        "entry_score_threshold": 60,
    },
    "SL만변경 (SL2.0/TP2.5/S60/EMA9-21)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0,
    },
    "보수적최적 (SL2.0/TP3.0/S60/EMA9-21)": {
        **DEFAULT_PARAMS,
        "sl_atr_mult": 2.0, "tp_rr_ratio": 3.0,
    },
}


def fetch_long_data():
    """장기 데이터 다운로드 (1H + 4H)"""
    # comprehensive_backtest의 START_DATE/END_DATE를 임시로 변경
    import comprehensive_backtest as cb
    orig_start = cb.START_DATE
    orig_end = cb.END_DATE
    cb.START_DATE = LONG_START
    cb.END_DATE = LONG_END

    data_cache = {}
    for symbol in SYMBOLS:
        for interval in ["60", "240"]:
            cache_file = os.path.join(DATA_DIR, f"{symbol}_{interval}_long.csv")

            if os.path.exists(cache_file):
                import pandas as pd
                df = pd.read_csv(cache_file)
                if len(df) > 100:
                    print(f"  ✓ 캐시: {symbol} {interval} ({len(df)}개)")
                    data_cache[f"{symbol}_{interval}"] = df
                    continue

            print(f"  ↓ 다운로드: {symbol} {interval} (장기)...")
            start_ts = int(datetime.strptime(LONG_START, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(LONG_END, "%Y-%m-%d").timestamp() * 1000)

            from comprehensive_backtest import fetch_kline
            df = fetch_kline(symbol, interval, start_ts, end_ts)
            if len(df) > 0:
                df.to_csv(cache_file, index=False)
                print(f"  ✓ 저장: {symbol} {interval} ({len(df)}개)")
                data_cache[f"{symbol}_{interval}"] = df

    cb.START_DATE = orig_start
    cb.END_DATE = orig_end
    return data_cache


def run_config(name, params, data_cache):
    """단일 설정 멀티코인 백테스트"""
    all_trades = []
    coin_results = {}

    for symbol in SYMBOLS:
        entry_key = f"{symbol}_60"
        regime_key = f"{symbol}_240"

        if entry_key not in data_cache or regime_key not in data_cache:
            continue

        entry_df = data_cache[entry_key].copy()
        regime_df = data_cache[regime_key].copy()

        if len(entry_df) < 100 or len(regime_df) < 50:
            continue

        entry_df = compute_entry_indicators(entry_df, params)
        regime_df = compute_regime_indicators(regime_df)

        re = BacktestRegimeEngine(params)
        regimes = []
        for _, rr in regime_df.iterrows():
            regimes.append(re.update(rr))
        regime_df["regime"] = regimes

        regime_mapped = align_regime_to_entry(regime_df, entry_df)

        result = run_single_backtest(
            entry_df, regime_mapped, params,
            symbol=symbol, block_sol_long=True
        )
        coin_results[symbol] = result
        all_trades.extend(result.get("trades", []))

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t["timestamp"])
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)
    final = INITIAL_BALANCE + total_pnl

    # 최대 드로다운
    running = INITIAL_BALANCE
    peak = running
    max_dd = 0
    for t in all_trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # 연도별 분석
    yearly = {}
    for t in all_trades:
        year = datetime.fromtimestamp(t["timestamp"] / 1000).strftime("%Y")
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "pnl": 0}
        yearly[year]["trades"] += 1
        if t["pnl"] > 0:
            yearly[year]["wins"] += 1
        yearly[year]["pnl"] += t["pnl"]

    long_trades = [t for t in all_trades if t["direction"] == "LONG"]
    short_trades = [t for t in all_trades if t["direction"] == "SHORT"]

    return {
        "name": name,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final, 2),
        "return_pct": round((final / INITIAL_BALANCE - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "total_fees": round(total_fees, 2),
        "long_trades": len(long_trades),
        "long_pnl": round(sum(t["pnl"] for t in long_trades), 2),
        "long_wr": round(sum(1 for t in long_trades if t["pnl"] > 0) / len(long_trades) * 100, 1) if long_trades else 0,
        "short_trades": len(short_trades),
        "short_pnl": round(sum(t["pnl"] for t in short_trades), 2),
        "short_wr": round(sum(1 for t in short_trades if t["pnl"] > 0) / len(short_trades) * 100, 1) if short_trades else 0,
        "yearly": yearly,
        "coin_results": {
            sym: {k: v for k, v in r.items() if k != "trades"}
            for sym, r in coin_results.items()
        },
    }


def main():
    print("=" * 80)
    print("HERMES 장기 검증 백테스트")
    print(f"기간: {LONG_START} ~ {LONG_END} (~4년, 하락장+회복장+상승장)")
    print(f"코인: {', '.join(SYMBOLS)} | 타임프레임: 1H 진입 / 4H 레짐")
    print(f"초기 잔액: ${INITIAL_BALANCE}")
    print("=" * 80)

    # 데이터 다운로드
    print("\n[1/2] 장기 데이터 다운로드...")
    data_cache = fetch_long_data()
    print(f"  총 {len(data_cache)}개 데이터셋")

    # 백테스트 실행
    print("\n[2/2] 파라미터 검증...")
    results = []

    for name, params in TEST_CONFIGS.items():
        print(f"\n  ▶ {name}")
        r = run_config(name, params, data_cache)
        if r:
            results.append(r)
            print(f"    거래: {r['total_trades']}회 | 승률: {r['win_rate']}%")
            print(f"    수익: ${r['total_pnl']:+.2f} ({r['return_pct']:+.1f}%)")
            print(f"    최대DD: {r['max_drawdown_pct']:.1f}% | 수수료: ${r['total_fees']:.2f}")
            print(f"    LONG: {r['long_trades']}회 {r['long_wr']}% ${r['long_pnl']:+.2f}")
            print(f"    SHORT: {r['short_trades']}회 {r['short_wr']}% ${r['short_pnl']:+.2f}")

            # 연도별
            print(f"    연도별:")
            for year, yd in sorted(r["yearly"].items()):
                wr = round(yd["wins"] / yd["trades"] * 100, 1) if yd["trades"] > 0 else 0
                print(f"      {year}: {yd['trades']}거래 승률{wr}% PnL ${yd['pnl']:+.2f}")

    # 요약 테이블
    print("\n" + "=" * 80)
    print("장기 검증 요약 (4년)")
    print("=" * 80)
    print(f"{'설정':<40} {'거래':>5} {'승률':>6} {'수익률':>9} {'PnL':>10} {'DD':>6}")
    print("-" * 80)

    results.sort(key=lambda x: x["return_pct"], reverse=True)
    for r in results:
        print(f"{r['name']:<40} {r['total_trades']:>5} {r['win_rate']:>5.1f}% "
              f"{r['return_pct']:>+8.1f}% ${r['total_pnl']:>+9.2f} {r['max_drawdown_pct']:>5.1f}%")

    # 2022년 (하락장) 성과 비교
    print("\n" + "-" * 80)
    print("2022년 하락장 성과 비교")
    print("-" * 80)
    print(f"{'설정':<40} {'거래':>5} {'승률':>6} {'PnL':>10}")
    print("-" * 80)
    for r in results:
        y = r["yearly"].get("2022", {"trades": 0, "wins": 0, "pnl": 0})
        wr = round(y["wins"] / y["trades"] * 100, 1) if y["trades"] > 0 else 0
        print(f"{r['name']:<40} {y['trades']:>5} {wr:>5.1f}% ${y['pnl']:>+9.2f}")

    # 2023년 (저변동성) 성과 비교
    print("\n" + "-" * 80)
    print("2023년 저변동성 성과 비교")
    print("-" * 80)
    print(f"{'설정':<40} {'거래':>5} {'승률':>6} {'PnL':>10}")
    print("-" * 80)
    for r in results:
        y = r["yearly"].get("2023", {"trades": 0, "wins": 0, "pnl": 0})
        wr = round(y["wins"] / y["trades"] * 100, 1) if y["trades"] > 0 else 0
        print(f"{r['name']:<40} {y['trades']:>5} {wr:>5.1f}% ${y['pnl']:>+9.2f}")

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "period": f"{LONG_START} ~ {LONG_END}",
        "results": [{k: v for k, v in r.items() if k != "coin_results"} for r in results],
    }
    path = os.path.join(RESULTS_DIR, "longterm_validation.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {path}")

    print("\n" + "=" * 80)
    print("장기 검증 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
