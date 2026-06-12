#!/usr/bin/env python3
"""
V자 반등 이벤트 분석 (2026-04-13 유사 시점 찾기)
==================================================
현재 상황: 강한 상승장 중 뉴스 기반 급락 → 4H 레짐 TRENDING_UP→TRENDING_DOWN 전환
          → 시스템이 SHORT 진입 → V자 반등에 SL → 반복

4년 데이터에서 "레짐 flip downs" 이벤트를 찾고:
1. TRENDING_UP → TRENDING_DOWN 전환 이벤트 전부 식별
2. 각 전환 후 N 봉(예: 48봉=2일) 내 다시 TRENDING_UP 복귀하면 "V자 반등" 라벨
3. 각 케이스에서 HERMES가 SHORT 거래를 몇 건, 승률 얼마, PnL 얼마
4. V자 반등 vs 진짜 하락장 비교
"""
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest, prepare_symbol
from comprehensive_backtest import (
    DEFAULT_PARAMS, compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry,
)

BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}

# V자 반등 기준: 레짐 TRENDING_UP → TRENDING_DOWN 이후 N 봉(1H) 내 재전환
V_BOUNCE_WINDOW_HOURS = 48  # 2일 내 복귀 = V자 반등


def find_regime_flips(data, symbol):
    """특정 심볼의 4H 레짐 flip 시점들 + 각 flip 이후 재전환까지 걸린 시간"""
    entry_df = compute_entry_indicators(data[f"{symbol}_60"].copy(), BEST_PARAMS)
    regime_df = compute_regime_indicators(data[f"{symbol}_240"].copy())
    re = BacktestRegimeEngine(BEST_PARAMS)
    regimes = [re.update(row) for _, row in regime_df.iterrows()]
    regime_df["regime"] = regimes
    rm = align_regime_to_entry(regime_df, entry_df)
    rm = rm.values if hasattr(rm, "values") else rm

    timestamps = entry_df["timestamp"].values
    closes = entry_df["close"].values

    flips = []  # 각 flip: {type, from, to, ts_flip, ts_return, candles_to_return, is_vbounce, crash_pct}
    prev_regime = None
    current_flip_start = None

    for i in range(len(rm)):
        cur = rm[i]
        if prev_regime is not None and cur != prev_regime:
            # 전환 발생
            if prev_regime == "TRENDING_UP" and cur == "TRENDING_DOWN":
                # 하락 전환 시작
                current_flip_start = {
                    "i_flip": i,
                    "ts_flip": int(timestamps[i]),
                    "price_at_flip": float(closes[i]),
                    "pre_high": float(max(closes[max(0, i-24):i+1])),  # 24봉 고점
                }
            elif current_flip_start is not None and cur == "TRENDING_UP":
                # 재전환
                i_flip = current_flip_start["i_flip"]
                candles_to_return = i - i_flip
                low_during = float(min(closes[i_flip:i+1]))
                crash_pct = (low_during - current_flip_start["pre_high"]) / current_flip_start["pre_high"] * 100
                recovery_pct = (closes[i] - low_during) / low_during * 100

                flips.append({
                    "ts_flip": current_flip_start["ts_flip"],
                    "date_flip": datetime.utcfromtimestamp(current_flip_start["ts_flip"]/1000).strftime("%Y-%m-%d %H:%M"),
                    "i_flip": i_flip,
                    "i_return": i,
                    "candles_to_return": candles_to_return,
                    "pre_high": current_flip_start["pre_high"],
                    "low_during": low_during,
                    "return_price": float(closes[i]),
                    "crash_pct": round(crash_pct, 2),
                    "recovery_pct": round(recovery_pct, 2),
                    "is_vbounce": candles_to_return <= V_BOUNCE_WINDOW_HOURS,
                })
                current_flip_start = None

        prev_regime = cur

    return flips, timestamps, rm


def analyze_short_trades_in_windows(trades, flips, symbol):
    """V자 반등 구간 vs 정상 하락장 구간의 SHORT 거래 비교"""
    sym_trades = [t for t in trades if t.get("symbol") == symbol and t.get("direction") == "SHORT"]

    vbounce_trades = []
    bear_trades = []

    for t in sym_trades:
        ts = t["timestamp"]
        # 이 거래가 어느 flip 구간에 속하는지
        in_vbounce = False
        in_bear = False
        for f in flips:
            if ts >= f["ts_flip"]:
                # flip 이후 발생
                if f["is_vbounce"]:
                    # V바운스 윈도우 안에 있는 거래
                    ts_return = None
                    if f["i_return"]:
                        # i_return의 timestamp 찾아야 하지만 없음 — candles_to_return로 대체
                        ts_return_ms = f["ts_flip"] + f["candles_to_return"] * 3600000
                        if ts <= ts_return_ms + 24 * 3600000:  # flip부터 재전환+24h 내
                            vbounce_trades.append(t)
                            in_vbounce = True
                            break
                else:
                    # 진짜 하락장 (V바운스 아님)
                    ts_end = f["ts_flip"] + f["candles_to_return"] * 3600000
                    if ts <= ts_end:
                        bear_trades.append(t)
                        in_bear = True
                        break

        if not in_vbounce and not in_bear:
            # 어떤 구간에도 속하지 않음 (혹시 정상 TRENDING_DOWN 장기간)
            pass

    return vbounce_trades, bear_trades


def stats(trades, label=""):
    if not trades:
        return {"label": label, "count": 0}
    wins = sum(1 for t in trades if t["pnl"] > 0)
    pnl = sum(t["pnl"] for t in trades)
    avg_win = sum(t["pnl"] for t in trades if t["pnl"] > 0) / wins if wins else 0
    losses = len(trades) - wins
    avg_loss = sum(t["pnl"] for t in trades if t["pnl"] <= 0) / losses if losses else 0
    return {
        "label": label,
        "count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins/len(trades)*100, 1),
        "total_pnl": round(pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_pnl_per_trade": round(pnl/len(trades), 2),
    }


def main():
    print("=" * 100)
    print("V자 반등 이벤트 분석 — 2026-04-13 같은 상황 찾기")
    print(f"기준: 4H 레짐 TRENDING_UP→TRENDING_DOWN 후 {V_BOUNCE_WINDOW_HOURS}시간(2일) 내 복귀")
    print("=" * 100)

    print("\n[데이터 로드]")
    data = load_all_data()

    # 전체 백테스트 실행 (trades 추출)
    print("[백테스트 실행 — 전체 4년]")
    r = run_shared_backtest(
        data, BEST_PARAMS, 600.0,
        use_funding=True,
        trailing_activation=1.5, trailing_distance=0.3,
        block_sol_long=True,
        skip_years=(),  # 2023 포함해서 전체 보기
        daily_cost_usd=1150/1470,
        ruin_threshold=15.0,
        use_cooldown=False,
    )
    all_trades = r.get("_trades", [])
    print(f"  전체 거래: {len(all_trades)}")

    # 각 코인별 flip 이벤트 찾기
    all_flips = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"\n[{sym}] 레짐 flip 분석 중...")
        flips, ts_arr, regimes = find_regime_flips(data, sym)
        vbounce_count = sum(1 for f in flips if f["is_vbounce"])
        bear_count = len(flips) - vbounce_count
        print(f"  총 UP→DOWN flip: {len(flips)}회")
        print(f"  그중 V자 반등 (2일 내 복귀): {vbounce_count}회")
        print(f"  진짜 하락장 전환: {bear_count}회")
        all_flips[sym] = flips

    # V자 반등 이벤트 상세
    print("\n" + "=" * 100)
    print("V자 반등 이벤트 목록 (전 코인)")
    print("=" * 100)
    print(f"{'코인':<10} {'발생일':<18} {'크래시%':>8} {'복귀시간':>10} {'반등%':>8}")
    print("-" * 60)

    all_vbounces = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        for f in all_flips[sym]:
            if f["is_vbounce"]:
                all_vbounces.append((sym, f))
                print(f"{sym:<10} {f['date_flip']:<18} {f['crash_pct']:>+7.2f}% "
                      f"{f['candles_to_return']:>6}h {f['recovery_pct']:>+7.2f}%")

    print(f"\n총 V자 반등 이벤트: {len(all_vbounces)}회 (4년 기준)")

    # SHORT 거래 분류
    print("\n" + "=" * 100)
    print("SHORT 거래 구간별 성과 비교")
    print("=" * 100)

    total_vbounce_trades = []
    total_bear_trades = []

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        vb, br = analyze_short_trades_in_windows(all_trades, all_flips[sym], sym)
        total_vbounce_trades.extend(vb)
        total_bear_trades.extend(br)
        print(f"\n  {sym}: V자 반등 구간 SHORT {len(vb)}건, 진짜 하락장 SHORT {len(br)}건")

    vb_stats = stats(total_vbounce_trades, "V자 반등 구간")
    br_stats = stats(total_bear_trades, "진짜 하락장 구간")

    print("\n" + "-" * 70)
    print(f"{'구간':<25} {'거래':>6} {'승률':>7} {'평균수익':>10} {'평균손실':>10} {'거래당PnL':>11}")
    print("-" * 70)
    for s in [vb_stats, br_stats]:
        if s["count"] > 0:
            print(f"{s['label']:<25} {s['count']:>6} {s['win_rate']:>6.1f}% "
                  f"${s['avg_win']:>+8.2f} ${s['avg_loss']:>+8.2f} ${s['avg_pnl_per_trade']:>+9.2f}")

    # 전체 SHORT 거래 평균
    all_shorts = [t for t in all_trades if t.get("direction") == "SHORT"]
    all_stats = stats(all_shorts, "전체 SHORT")
    if all_stats["count"] > 0:
        print(f"{all_stats['label']:<25} {all_stats['count']:>6} {all_stats['win_rate']:>6.1f}% "
              f"${all_stats['avg_win']:>+8.2f} ${all_stats['avg_loss']:>+8.2f} ${all_stats['avg_pnl_per_trade']:>+9.2f}")

    # 현재 상황과 가장 유사한 이벤트 3개
    print("\n" + "=" * 100)
    print("현재(2026-04-13)와 가장 유사한 과거 V자 반등 TOP 5")
    print("=" * 100)
    # 현재: BTC ~-3.2% 크래시, 12~24h 내 반등 시작
    target_crash = -3.2
    candidates = []
    for sym, f in all_vbounces:
        if sym != "BTCUSDT":
            continue
        similarity = abs(f["crash_pct"] - target_crash) + abs(f["candles_to_return"] - 18)
        candidates.append((similarity, sym, f))

    candidates.sort(key=lambda x: x[0])
    for sim, sym, f in candidates[:5]:
        print(f"  {f['date_flip']:<18} 크래시 {f['crash_pct']:+.2f}% → {f['candles_to_return']}h 내 복귀 "
              f"(반등 {f['recovery_pct']:+.2f}%)")

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "current_situation": {
            "date": "2026-04-13",
            "crash_pct_approx": -3.2,
            "hours_to_recovery_approx": 18,
            "cause": "US-Iran peace talks failed, Trump Hormuz blockade",
        },
        "vbounce_events_by_coin": {
            sym: {
                "total_flips": len(flips),
                "vbounce_count": sum(1 for f in flips if f["is_vbounce"]),
                "bear_count": sum(1 for f in flips if not f["is_vbounce"]),
                "vbounce_details": [f for f in flips if f["is_vbounce"]],
            }
            for sym, flips in all_flips.items()
        },
        "short_trade_stats": {
            "vbounce_zone": vb_stats,
            "bear_zone": br_stats,
            "all_shorts": all_stats,
        },
        "similar_historical_events": [
            {
                "date": f["date_flip"],
                "crash_pct": f["crash_pct"],
                "hours_to_recovery": f["candles_to_return"],
                "recovery_pct": f["recovery_pct"],
            }
            for sim, sym, f in candidates[:5]
        ],
    }

    out_path = "~/Projects/HERMES_백테스팅/v5/v5_vbounce_analysis.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
