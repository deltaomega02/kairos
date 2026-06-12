#!/usr/bin/env python3
"""
Equity Curve 비교 차트
======================
bwjoke 실전 6년 equity vs HERMES V12 현실 시뮬 equity
같은 시드 ($16,500), 같은 기간 (2020-05 ~ 2026-04)
"""
import os, sys, time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# Korean font setup
for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "~/Projects/HERMES/backtest")
from v9_mega_sweep import _load_data, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v12_realistic_engine import run_realistic_backtest

RESULTS_DIR = "~/Projects/HERMES_백테스팅/v12"
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def get_btc_close_series(data_full):
    """BTC 1H close 시계열 (시간 → 가격 매핑용)"""
    df = data_full["BTCUSDT_60"][["timestamp", "close"]].copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df["close"]


def main():
    print("데이터 로드 & V12 재실행...")
    data_full = _load_data()
    data = filter_d(data_full, START, END)
    btc_close = get_btc_close_series(data_full)

    t0 = time.time()
    r = run_realistic_backtest(
        data, V11_PARAMS, SEED,
        start_year=2020, skip_years=(2023,),
        daily_cost_usd=DAILY_COST,
        slippage_pct_base=0.05,
        max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
        enabled_symbols=SYMBOLS,
        d1_filter_enable=True, d1_ema_period=2, d1_mode="price_above_ema",
        use_realism=True, api_fail_rate=0.0, funding_enabled=True,
    )
    print(f"V12 실행 완료: {time.time()-t0:.1f}s | 최종 ${r['final_balance']:,.0f}")

    # V12 equity curve (일별 snapshot)
    v12_curve = pd.DataFrame(r["equity_curve"])
    v12_curve["dt"] = pd.to_datetime(v12_curve["ts"], unit="ms", utc=True)

    # bwjoke equity curve
    bw = pd.read_csv("/tmp/BTC-Trading-Since-2020/derived-equity-curve.csv")
    bw["dt"] = pd.to_datetime(bw["timestamp"], utc=True)
    bw = bw.sort_values("dt").reset_index(drop=True)

    # bwjoke XBT → USD 환산 (시간별 BTC 가격으로)
    def xbt_to_usd(row):
        dt = row["dt"]
        # 가장 가까운 1H close 찾기 (hourly floor)
        try:
            nearest = btc_close.asof(dt)
            if pd.isna(nearest):
                return np.nan
            return row["adjustedWealthXBT"] * float(nearest)
        except Exception:
            return np.nan

    bw["wealth_usd"] = bw.apply(xbt_to_usd, axis=1)
    bw = bw.dropna(subset=["wealth_usd"])

    # ===== 차트 1: 전체 6년 equity curve (log scale) =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax1 = axes[0]
    ax1.plot(v12_curve["dt"], v12_curve["balance"], label="HERMES V12 (봇, 현실 제약)",
             color="#2E86DE", linewidth=1.8)
    ax1.plot(bw["dt"], bw["wealth_usd"], label="bwjoke (인간 수동, USD 환산)",
             color="#EE5A24", linewidth=1.8, alpha=0.85)
    ax1.axhline(SEED, linestyle="--", color="gray", alpha=0.4, label=f"시드 ${SEED:,.0f}")
    ax1.set_yscale("log")
    ax1.set_title(f"HERMES V12 vs bwjoke — 6년 Equity Curve (Log Scale)\n"
                  f"시드 $16,500 | 2020-05 ~ 2026-04", fontsize=13, pad=12)
    ax1.set_ylabel("자산 (USD, 로그 스케일)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 연간 성장률 annotation
    for year in range(2020, 2027):
        year_end = datetime(year, 12, 31)
        ax1.axvline(year_end, color="black", alpha=0.05, linewidth=0.5)

    # 최종 수치 annotation
    hermes_final = v12_curve["balance"].iloc[-1]
    bw_final = bw["wealth_usd"].iloc[-1]
    ax1.annotate(f"HERMES 최종\n${hermes_final/1e6:.2f}M ({hermes_final/SEED:.0f}x)",
                 xy=(v12_curve["dt"].iloc[-1], hermes_final),
                 xytext=(10, 10), textcoords="offset points",
                 fontsize=10, color="#2E86DE", fontweight="bold")
    ax1.annotate(f"bwjoke 최종\n${bw_final/1e6:.2f}M ({bw_final/SEED:.0f}x)",
                 xy=(bw["dt"].iloc[-1], bw_final),
                 xytext=(10, -30), textcoords="offset points",
                 fontsize=10, color="#EE5A24", fontweight="bold")

    # ===== 차트 2: XBT 기준 비교 (BTC 가격 변동 제거) =====
    ax2 = axes[1]
    # HERMES → XBT 환산
    v12_curve["wealth_xbt"] = v12_curve.apply(
        lambda row: row["balance"] / float(btc_close.asof(row["dt"]))
        if not pd.isna(btc_close.asof(row["dt"])) else np.nan, axis=1
    )
    v12_curve_xbt = v12_curve.dropna(subset=["wealth_xbt"])

    ax2.plot(v12_curve_xbt["dt"], v12_curve_xbt["wealth_xbt"],
             label="HERMES V12 (XBT 환산)", color="#2E86DE", linewidth=1.8)
    ax2.plot(bw["dt"], bw["adjustedWealthXBT"],
             label="bwjoke (XBT 원본)", color="#EE5A24", linewidth=1.8, alpha=0.85)

    hermes_xbt_start = v12_curve_xbt["wealth_xbt"].iloc[0]
    ax2.axhline(hermes_xbt_start, linestyle="--", color="gray", alpha=0.4,
                label=f"시작 ~{hermes_xbt_start:.2f} XBT")
    ax2.set_yscale("log")
    ax2.set_title("순수 트레이딩 Edge 비교 — XBT 기준 (BTC 가격 변동 제거)",
                  fontsize=13, pad=12)
    ax2.set_ylabel("자산 (XBT, 로그 스케일)", fontsize=11)
    ax2.set_xlabel("날짜", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # XBT 배수 annotation
    hermes_xbt_final = v12_curve_xbt["wealth_xbt"].iloc[-1]
    bw_xbt_final = bw["adjustedWealthXBT"].iloc[-1]
    ax2.annotate(f"HERMES XBT\n{hermes_xbt_final:.1f} XBT ({hermes_xbt_final/hermes_xbt_start:.0f}x)",
                 xy=(v12_curve_xbt["dt"].iloc[-1], hermes_xbt_final),
                 xytext=(10, 10), textcoords="offset points",
                 fontsize=10, color="#2E86DE", fontweight="bold")
    ax2.annotate(f"bwjoke XBT\n{bw_xbt_final:.1f} XBT ({bw_xbt_final/bw['adjustedWealthXBT'].iloc[0]:.0f}x)",
                 xy=(bw["dt"].iloc[-1], bw_xbt_final),
                 xytext=(10, -30), textcoords="offset points",
                 fontsize=10, color="#EE5A24", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "v12_vs_bwjoke_equity.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"\n차트 저장: {out_path}")

    # 간단 요약 출력
    print(f"\n{'='*70}")
    print(f"📊 최종 비교")
    print(f"{'='*70}")
    print(f"  HERMES V12: ${hermes_final:,.0f} ({hermes_final/SEED:.1f}x USD, "
          f"{hermes_xbt_final/hermes_xbt_start:.1f}x XBT)")
    print(f"  bwjoke:     ${bw_final:,.0f} ({bw_final/SEED:.1f}x USD, "
          f"{bw_xbt_final/bw['adjustedWealthXBT'].iloc[0]:.1f}x XBT)")
    print(f"\n  ✓ XBT edge: HERMES {hermes_xbt_final/hermes_xbt_start / (bw_xbt_final/bw['adjustedWealthXBT'].iloc[0]):.2f}배 우위")


if __name__ == "__main__":
    main()
