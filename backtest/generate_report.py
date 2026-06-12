#!/usr/bin/env python3
"""
HERMES 백테스팅 상세 리포트 생성
=================================
Top 5 고유 조합의 4년 상세 데이터 + 차트 + 리포트 생성.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_backtest import (
    DEFAULT_PARAMS, INITIAL_BALANCE, TAKER_FEE_PCT,
    compute_entry_indicators, compute_regime_indicators,
    BacktestRegimeEngine, align_regime_to_entry, run_single_backtest,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = "~/Projects/HERMES_백테스팅_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Top 5 고유 조합 (그리드 서치 + 이전 검증 결과)
# ================================================================

CONFIGS = {
    "1위_EMA5_18_ADX35_PB1.5": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
    },
    "2위_EMA5_18_ADX35_PB2.0": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 2.0, "adx_enter_trending": 35,
    },
    "3위_EMA5_15_ADX25_PB1.5": {
        **DEFAULT_PARAMS,
        "ema_fast": 5, "ema_slow": 15, "sl_atr_mult": 1.5,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 25,
    },
    "4위_EMA7_21_ADX25_PB1.0": {
        **DEFAULT_PARAMS,
        "ema_fast": 7, "ema_slow": 21, "sl_atr_mult": 2.0,
        "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
        "pullback_ema_dist_pct": 1.0, "adx_enter_trending": 25,
    },
    "5위_기존시스템_EMA9_21": {
        **DEFAULT_PARAMS,
        # 기존 기본값 그대로
    },
}


def load_data():
    """장기 데이터 로드"""
    data = {}
    for sym in SYMBOLS:
        for iv in ["60", "240"]:
            f = os.path.join(DATA_DIR, f"{sym}_{iv}_long.csv")
            if os.path.exists(f):
                data[f"{sym}_{iv}"] = pd.read_csv(f)
    return data


def run_detailed_backtest(name, params, data):
    """상세 백테스트 — 모든 거래 기록 포함"""
    all_trades = []
    coin_results = {}

    for sym in SYMBOLS:
        entry_df = compute_entry_indicators(data[f"{sym}_60"].copy(), params)
        regime_df = compute_regime_indicators(data[f"{sym}_240"].copy())
        re = BacktestRegimeEngine(params)
        regimes = [re.update(row) for _, row in regime_df.iterrows()]
        regime_df["regime"] = regimes
        rm = align_regime_to_entry(regime_df, entry_df)
        r = run_single_backtest(entry_df, rm, params, symbol=sym, block_sol_long=True)
        coin_results[sym] = {k: v for k, v in r.items() if k != "trades"}
        all_trades.extend(r.get("trades", []))

    all_trades.sort(key=lambda t: t["timestamp"])

    # 통계 계산
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)

    # 잔고 곡선
    balance_curve = [INITIAL_BALANCE]
    running = INITIAL_BALANCE
    for t in all_trades:
        running += t["pnl"]
        balance_curve.append(running)

    # 최대 드로다운
    peak = INITIAL_BALANCE
    max_dd = 0
    dd_curve = [0]
    for b in balance_curve[1:]:
        peak = max(peak, b)
        dd = (peak - b) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        dd_curve.append(dd)

    # 연도별
    yearly = {}
    for t in all_trades:
        y = datetime.fromtimestamp(t["timestamp"] / 1000).strftime("%Y")
        if y not in yearly:
            yearly[y] = {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0}
        yearly[y]["trades"] += 1
        if t["pnl"] > 0:
            yearly[y]["wins"] += 1
        yearly[y]["pnl"] += t["pnl"]
        yearly[y]["fees"] += t["fee"]

    # 월별
    monthly = {}
    for t in all_trades:
        m = datetime.fromtimestamp(t["timestamp"] / 1000).strftime("%Y-%m")
        if m not in monthly:
            monthly[m] = {"trades": 0, "wins": 0, "pnl": 0.0}
        monthly[m]["trades"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1
        monthly[m]["pnl"] += t["pnl"]

    # 방향별
    long_t = [t for t in all_trades if t["direction"] == "LONG"]
    short_t = [t for t in all_trades if t["direction"] == "SHORT"]

    # 코인별
    coin_trades = {}
    for sym in SYMBOLS:
        ct = [t for t in all_trades if t["symbol"] == sym]
        coin_trades[sym] = {
            "trades": len(ct),
            "wins": sum(1 for t in ct if t["pnl"] > 0),
            "pnl": sum(t["pnl"] for t in ct),
            "long": sum(1 for t in ct if t["direction"] == "LONG"),
            "short": sum(1 for t in ct if t["direction"] == "SHORT"),
        }

    # 연속 승/패
    max_streak_w = max_streak_l = streak_w = streak_l = 0
    for t in all_trades:
        if t["pnl"] > 0:
            streak_w += 1
            streak_l = 0
        else:
            streak_l += 1
            streak_w = 0
        max_streak_w = max(max_streak_w, streak_w)
        max_streak_l = max(max_streak_l, streak_l)

    avg_win = np.mean([t["pnl"] for t in all_trades if t["pnl"] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl"] for t in all_trades if t["pnl"] <= 0]) if (total - wins) > 0 else 0

    return {
        "name": name,
        "params": params,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "final_balance": round(INITIAL_BALANCE + total_pnl, 2),
        "return_pct": round((INITIAL_BALANCE + total_pnl) / INITIAL_BALANCE * 100 - 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "real_rr": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        "max_win_streak": max_streak_w,
        "max_loss_streak": max_streak_l,
        "yearly": yearly,
        "monthly": monthly,
        "long": {
            "trades": len(long_t),
            "wins": sum(1 for t in long_t if t["pnl"] > 0),
            "pnl": round(sum(t["pnl"] for t in long_t), 2),
        },
        "short": {
            "trades": len(short_t),
            "wins": sum(1 for t in short_t if t["pnl"] > 0),
            "pnl": round(sum(t["pnl"] for t in short_t), 2),
        },
        "coin_results": coin_trades,
        "balance_curve": balance_curve,
        "dd_curve": dd_curve,
        "trades": all_trades,
    }


def generate_text_report(results: List[Dict]):
    """텍스트 종합 리포트"""
    lines = []
    lines.append("=" * 80)
    lines.append("HERMES 종합 백테스팅 리포트")
    lines.append(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"기간: 2022-01-01 ~ 2026-04-08 (약 4년 3개월)")
    lines.append(f"데이터: Bybit USDT Perpetual 1H OHLCV (실제 시장 데이터)")
    lines.append(f"코인: BTC, ETH, SOL")
    lines.append(f"초기 잔액: ${INITIAL_BALANCE}")
    lines.append(f"수수료: Taker {TAKER_FEE_PCT*100:.3f}% (양방향)")
    lines.append(f"리스크: 거래당 1.5%, 최대 레버리지 5x")
    lines.append(f"전략: 4H 레짐 판독 + 1H EMA 풀백 진입 (추세 전략만)")
    lines.append(f"SOL LONG 차단 적용")
    lines.append("=" * 80)

    # 백테스팅 방법론
    lines.append("\n" + "=" * 80)
    lines.append("백테스팅 방법론")
    lines.append("=" * 80)
    lines.append("""
1. 데이터 수집
   - Bybit v5 API에서 4년치 1H/4H OHLCV 다운로드
   - BTC, ETH, SOL 각각 37,392개 1H 캔들, 9,348개 4H 캔들
   - 실제 시장 데이터 (시가/고가/저가/종가/거래량)

2. 레짐 판독 (4H)
   - ADX 히스테리시스: 진입 임계값 이상 → TRENDING_UP/DOWN
   - 방향 결정: +DI/-DI, EMA9/21, MACD 다수결
   - ATR 퍼센타일 85% 이상 → HIGH_VOL (거래 중단)
   - 디바운스 1봉 적용

3. 시그널 평가 (1H)
   - EMA 풀백: 가격이 EMA_fast 근처로 되돌아올 때 진입
   - EMA 정배열(LONG) / 역배열(SHORT) 필수
   - 스코어링: 기본 50 + RSI 보너스(±20) + 볼륨 보너스(±15)
   - 오더북/펀딩: 백테스트에서는 히스토리 없어 제외

4. SL/TP
   - SL = ATR × sl_atr_mult
   - TP = SL × tp_rr_ratio
   - 수수료 역산으로 실질 R:R ≥ 0.8 보장

5. 포지션 관리
   - 리스크 역산 사이징 (잔고 × 1.5% = 최대 손실)
   - 일일 최대 5거래
   - SL/TP 히트 시 각 캔들의 고가/저가로 체크
   - 동시 히트 시 보수적으로 SL 처리

6. 그리드 서치
   - 342,000개 파라미터 조합 전수 조사
   - EMA(4×5), SL(5), TP(5), Score(4), 풀백(5), ADX(4), RSI(3×3)
   - 소요시간: 232분 (3시간 52분)
""")

    # 순위 요약
    lines.append("\n" + "=" * 80)
    lines.append("순위 요약")
    lines.append("=" * 80)
    lines.append(f"\n{'순위':<6} {'거래':>6} {'승률':>7} {'수익률':>9} {'PnL':>11} {'DD':>7} {'R:R':>5} {'핵심 파라미터'}")
    lines.append("-" * 80)
    for i, r in enumerate(results):
        p = r["params"]
        pstr = f"EMA{int(p['ema_fast'])}/{int(p['ema_slow'])} SL{p['sl_atr_mult']} TP{p['tp_rr_ratio']} PB{p['pullback_ema_dist_pct']} ADX{int(p['adx_enter_trending'])}"
        lines.append(f"{i+1}위    {r['total_trades']:>6} {r['win_rate']:>6.1f}% {r['return_pct']:>+8.1f}% ${r['total_pnl']:>+10.2f} {r['max_drawdown_pct']:>6.1f}% {r['real_rr']:>4.1f} {pstr}")

    # 각 조합 상세
    for i, r in enumerate(results):
        p = r["params"]
        lines.append(f"\n\n{'='*80}")
        lines.append(f"{i+1}위 상세: {r['name']}")
        lines.append(f"{'='*80}")

        lines.append(f"\n파라미터:")
        lines.append(f"  EMA Fast/Slow: {int(p['ema_fast'])}/{int(p['ema_slow'])}")
        lines.append(f"  SL: ATR × {p['sl_atr_mult']}")
        lines.append(f"  TP: R:R {p['tp_rr_ratio']}")
        lines.append(f"  진입 스코어: {int(p['entry_score_threshold'])}")
        lines.append(f"  풀백 거리: {p['pullback_ema_dist_pct']}%")
        lines.append(f"  ADX 진입: {int(p['adx_enter_trending'])}")

        lines.append(f"\n전체 성과:")
        lines.append(f"  초기 잔액: ${INITIAL_BALANCE:.2f}")
        lines.append(f"  최종 잔액: ${r['final_balance']:.2f} ({r['return_pct']:+.1f}%)")
        lines.append(f"  총 거래: {r['total_trades']}회 (승 {r['wins']} / 패 {r['losses']})")
        lines.append(f"  승률: {r['win_rate']}%")
        lines.append(f"  평균 수익: ${r['avg_win']:.2f} / 평균 손실: ${r['avg_loss']:.2f}")
        lines.append(f"  실질 R:R: {r['real_rr']}")
        lines.append(f"  최대 드로다운: {r['max_drawdown_pct']}%")
        lines.append(f"  총 수수료: ${r['total_fees']:.2f}")
        lines.append(f"  최대 연승: {r['max_win_streak']}회 / 최대 연패: {r['max_loss_streak']}회")

        lines.append(f"\n방향별:")
        l = r["long"]
        s = r["short"]
        lwr = round(l["wins"]/l["trades"]*100,1) if l["trades"]>0 else 0
        swr = round(s["wins"]/s["trades"]*100,1) if s["trades"]>0 else 0
        lines.append(f"  LONG:  {l['trades']}거래 승률{lwr}% PnL ${l['pnl']:+.2f}")
        lines.append(f"  SHORT: {s['trades']}거래 승률{swr}% PnL ${s['pnl']:+.2f}")

        lines.append(f"\n코인별:")
        for sym, cd in r["coin_results"].items():
            cn = sym.replace("USDT", "")
            cwr = round(cd["wins"]/cd["trades"]*100,1) if cd["trades"]>0 else 0
            lines.append(f"  {cn}: {cd['trades']}거래 승률{cwr}% PnL ${cd['pnl']:+.2f} (L:{cd['long']} S:{cd['short']})")

        lines.append(f"\n연도별:")
        lines.append(f"  {'연도':<8} {'거래':>6} {'승률':>7} {'PnL':>12} {'수수료':>10}")
        lines.append(f"  {'-'*45}")
        for y, yd in sorted(r["yearly"].items()):
            wr = round(yd["wins"]/yd["trades"]*100,1) if yd["trades"]>0 else 0
            lines.append(f"  {y:<8} {yd['trades']:>6} {wr:>6.1f}% ${yd['pnl']:>+11.2f} ${yd['fees']:>9.2f}")

        lines.append(f"\n월별:")
        lines.append(f"  {'월':<10} {'거래':>5} {'승률':>7} {'PnL':>11}")
        lines.append(f"  {'-'*35}")
        for m, md in sorted(r["monthly"].items()):
            wr = round(md["wins"]/md["trades"]*100,1) if md["trades"]>0 else 0
            sign = "+" if md["pnl"] >= 0 else ""
            lines.append(f"  {m:<10} {md['trades']:>5} {wr:>6.1f}% ${md['pnl']:>+10.2f}")

    return "\n".join(lines)


def generate_trade_log(results: List[Dict]):
    """전체 거래 내역"""
    for i, r in enumerate(results):
        lines = []
        lines.append(f"거래 내역: {r['name']}")
        lines.append(f"{'#':>5} {'날짜':>12} {'코인':>6} {'방향':>6} {'진입':>12} {'청산':>12} {'PnL':>10} {'수수료':>8} {'사유':>6}")
        lines.append("-" * 85)
        for j, t in enumerate(r["trades"]):
            dt = datetime.fromtimestamp(t["timestamp"]/1000).strftime("%Y-%m-%d")
            sym = t["symbol"].replace("USDT","")
            lines.append(f"{j+1:>5} {dt:>12} {sym:>6} {t['direction']:>6} ${t['entry_price']:>11.2f} ${t['exit_price']:>11.2f} ${t['pnl']:>+9.2f} ${t['fee']:>7.2f} {t['reason']:>6}")

        path = os.path.join(OUTPUT_DIR, f"{i+1}위_거래내역.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))


def generate_charts(results: List[Dict]):
    """차트 생성"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("  matplotlib 없음 — 차트 스킵")
        return

    # 1. 잔고 곡선 비교 (Top 5)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax = axes[0]
    for i, r in enumerate(results):
        label = r["name"].split("_")[0] + f" ({r['return_pct']:+.1f}%)"
        ax.plot(r["balance_curve"], label=label, linewidth=1.5)
    ax.axhline(y=INITIAL_BALANCE, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("잔고 곡선 비교 (Top 5)")
    ax.set_ylabel("잔고 ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, r in enumerate(results):
        label = r["name"].split("_")[0]
        ax.plot(r["dd_curve"], label=label, linewidth=1.2)
    ax.set_title("드로다운 비교 (%)")
    ax.set_ylabel("드로다운 (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_잔고_드로다운_비교.png"), dpi=150)
    plt.close()
    print("  ✓ 01_잔고_드로다운_비교.png")

    # 2. 각 조합별 상세 차트
    for i, r in enumerate(results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{r['name']} | {r['return_pct']:+.1f}% | {r['total_trades']}거래", fontsize=14)

        # 잔고 곡선
        ax = axes[0][0]
        ax.plot(r["balance_curve"], color="blue", linewidth=1.2)
        ax.axhline(y=INITIAL_BALANCE, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("잔고 곡선")
        ax.set_ylabel("$")
        ax.grid(True, alpha=0.3)

        # 월별 PnL
        ax = axes[0][1]
        months = sorted(r["monthly"].keys())
        pnls = [r["monthly"][m]["pnl"] for m in months]
        colors = ["green" if p >= 0 else "red" for p in pnls]
        ax.bar(range(len(months)), pnls, color=colors, alpha=0.7)
        ax.set_title("월별 PnL")
        ax.set_ylabel("$")
        # 간격 표시
        tick_pos = list(range(0, len(months), 6))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([months[p] for p in tick_pos], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)

        # 연도별 바 차트
        ax = axes[1][0]
        years = sorted(r["yearly"].keys())
        ypnls = [r["yearly"][y]["pnl"] for y in years]
        ycolors = ["green" if p >= 0 else "red" for p in ypnls]
        bars = ax.bar(years, ypnls, color=ycolors, alpha=0.7)
        ax.set_title("연도별 PnL")
        ax.set_ylabel("$")
        for bar, val in zip(bars, ypnls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"${val:+.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
        ax.grid(True, alpha=0.3)

        # LONG vs SHORT
        ax = axes[1][1]
        labels = ["LONG", "SHORT"]
        pnl_vals = [r["long"]["pnl"], r["short"]["pnl"]]
        trade_counts = [r["long"]["trades"], r["short"]["trades"]]
        colors_ls = ["green" if p >= 0 else "red" for p in pnl_vals]
        bars = ax.bar(labels, pnl_vals, color=colors_ls, alpha=0.7)
        for bar, val, cnt in zip(bars, pnl_vals, trade_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"${val:+.0f}\n({cnt}거래)", ha="center",
                   va="bottom" if val >= 0 else "top", fontsize=10)
        ax.set_title("LONG vs SHORT")
        ax.set_ylabel("$")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{i+1:02d}위_상세차트.png"), dpi=150)
        plt.close()
        print(f"  ✓ {i+1:02d}위_상세차트.png")

    # 3. 타임프레임 비교 차트 (이전 결과)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", "comprehensive_results.json")) as f:
            comp = json.load(f)

        tf_data = comp.get("stage1_timeframe", {})
        if tf_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            tfs = ["5", "15", "60", "240"]
            tf_names = ["5분", "15분", "1시간", "4시간"]
            returns = [tf_data.get(t, {}).get("return_pct", 0) for t in tfs]
            colors_tf = ["green" if r >= 0 else "red" for r in returns]
            bars = ax.bar(tf_names, returns, color=colors_tf, alpha=0.7)
            for bar, val in zip(bars, returns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f"{val:+.1f}%", ha="center",
                       va="bottom" if val >= 0 else "top", fontsize=11, fontweight="bold")
            ax.set_title("타임프레임별 수익률 비교 (기본 파라미터, 2년)")
            ax.set_ylabel("수익률 (%)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "00_타임프레임_비교.png"), dpi=150)
            plt.close()
            print("  ✓ 00_타임프레임_비교.png")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("HERMES 백테스팅 상세 리포트 생성")
    print("=" * 60)

    print("\n[1] 데이터 로드...")
    data = load_data()
    print(f"  {len(data)}개 데이터셋")

    print("\n[2] Top 5 백테스트 실행...")
    results = []
    for name, params in CONFIGS.items():
        print(f"  ▶ {name}...")
        r = run_detailed_backtest(name, params, data)
        results.append(r)
        print(f"    {r['total_trades']}거래 | {r['win_rate']}% | {r['return_pct']:+.1f}%")

    print("\n[3] 텍스트 리포트 생성...")
    report = generate_text_report(results)
    path = os.path.join(OUTPUT_DIR, "00_종합리포트.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"  ✓ {path}")

    print("\n[4] 거래 내역 생성...")
    generate_trade_log(results)
    for i in range(len(results)):
        print(f"  ✓ {i+1}위_거래내역.txt")

    print("\n[5] 차트 생성...")
    generate_charts(results)

    print("\n[6] JSON 데이터 저장...")
    json_data = []
    for r in results:
        rd = {k: v for k, v in r.items() if k not in ("trades", "balance_curve", "dd_curve")}
        json_data.append(rd)
    with open(os.path.join(OUTPUT_DIR, "00_전체데이터.json"), "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 00_전체데이터.json")

    print(f"\n출력 폴더: {OUTPUT_DIR}")
    print("\n" + "=" * 60)
    print("리포트 생성 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
