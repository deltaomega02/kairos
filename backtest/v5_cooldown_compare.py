#!/usr/bin/env python3
"""
백테스팅 v5 — 쿨다운 유무 비교 (현재 시스템 기준)
===================================================
A) 현재 시스템 그대로 (2연패→60분, 3연패→당일중단)
B) 쿨다운 완전 제거

$600 시드, 수동 감독 (2023 skip), 서버비 차감.
연도별/월별/코인별/방향별 상세 비교.
"""
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_engine import load_all_data
from v4_shared_engine import run_shared_backtest
from comprehensive_backtest import DEFAULT_PARAMS

RESULTS_DIR = "/Users/sue/Projects/HERMES_백테스팅/v5"
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_PARAMS = {
    **DEFAULT_PARAMS,
    "ema_fast": 5, "ema_slow": 18, "sl_atr_mult": 1.5,
    "tp_rr_ratio": 4.0, "entry_score_threshold": 40,
    "pullback_ema_dist_pct": 1.5, "adx_enter_trending": 35,
}
SEED = 600.0
DAILY_COST_USD = 1150 / 1470

SCENARIOS = {
    "A_현재시스템": dict(
        use_cooldown=True, cooldown_after=2,
        cooldown_candles=1, daily_halt_after=3,
    ),
    "B_쿨다운제거": dict(
        use_cooldown=False,
    ),
}


def analyze_trades(trades, seed):
    """거래 리스트에서 상세 통계 추출"""
    if not trades:
        return {}

    # 기본 통계
    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)

    avg_win = sum(t["pnl"] for t in wins) / win_count if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / loss_count if losses else 0
    real_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 연승/연패
    max_consec_win = max_consec_loss = 0
    cur_win = cur_loss = 0
    for t in trades:
        if t["pnl"] > 0:
            cur_win += 1
            cur_loss = 0
            max_consec_win = max(max_consec_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_consec_loss = max(max_consec_loss, cur_loss)

    # 월별
    monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        m = datetime.utcfromtimestamp(t["timestamp"] / 1000).strftime("%Y-%m")
        monthly[m]["trades"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1
        monthly[m]["pnl"] += t["pnl"]

    # 코인별
    by_coin = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        sym = t.get("symbol", "?")
        by_coin[sym]["trades"] += 1
        if t["pnl"] > 0:
            by_coin[sym]["wins"] += 1
        by_coin[sym]["pnl"] += t["pnl"]

    # 방향별
    by_dir = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        d = t.get("direction", "?")
        by_dir[d]["trades"] += 1
        if t["pnl"] > 0:
            by_dir[d]["wins"] += 1
        by_dir[d]["pnl"] += t["pnl"]

    # 청산 사유별
    by_reason = defaultdict(int)
    for t in trades:
        by_reason[t.get("reason", "?")] += 1

    # 연도별 잔고 추이 (trade-by-trade replay)
    balance = seed
    peak = seed
    max_dd = 0
    yearly_balance = {}
    cur_year = None
    for t in trades:
        y = t.get("year", datetime.utcfromtimestamp(t["timestamp"] / 1000).year)
        if cur_year is None:
            cur_year = y
        elif y != cur_year:
            yearly_balance[cur_year] = round(balance, 2)
            cur_year = y
        balance += t["pnl"]
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    if cur_year is not None:
        yearly_balance[cur_year] = round(balance, 2)

    # 손실 월 비율
    loss_months = sum(1 for m in monthly.values() if m["pnl"] < 0)
    total_months = len(monthly)

    return {
        "total": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_count / total * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "real_rr": round(real_rr, 2),
        "max_consec_win": max_consec_win,
        "max_consec_loss": max_consec_loss,
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        "total_fee": round(sum(t["fee"] for t in trades), 2),
        "monthly": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                        for kk, vv in v.items()} for k, v in sorted(monthly.items())},
        "by_coin": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                        for kk, vv in v.items()} for k, v in sorted(by_coin.items())},
        "by_direction": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                             for kk, vv in v.items()} for k, v in sorted(by_dir.items())},
        "by_reason": dict(by_reason),
        "loss_months": loss_months,
        "total_months": total_months,
        "loss_month_pct": round(loss_months / total_months * 100, 1) if total_months else 0,
    }


def print_comparison(results):
    """두 시나리오 비교 출력"""
    a = results["A_현재시스템"]
    b = results["B_쿨다운제거"]
    ra = a["result"]
    rb = b["result"]
    sa = a["stats"]
    sb = b["stats"]

    print("\n" + "=" * 100)
    print("종합 비교")
    print("=" * 100)
    print(f"{'항목':<30} {'A. 현재시스템':>20} {'B. 쿨다운제거':>20} {'차이':>20}")
    print("-" * 100)

    rows = [
        ("거래 수", f"{ra['total_trades']}", f"{rb['total_trades']}", f"{rb['total_trades']-ra['total_trades']:+d}"),
        ("승률", f"{ra['win_rate']}%", f"{rb['win_rate']}%", f"{rb['win_rate']-ra['win_rate']:+.1f}%p"),
        ("순이익", f"${ra['net_profit']:+,.0f}", f"${rb['net_profit']:+,.0f}", f"${rb['net_profit']-ra['net_profit']:+,.0f}"),
        ("최종잔고", f"${ra['final_balance']:,.0f}", f"${rb['final_balance']:,.0f}", ""),
        ("수익배수", f"{ra['final_balance']/SEED:.1f}x", f"{rb['final_balance']/SEED:.1f}x", ""),
        ("최대 DD", f"{ra['max_dd']}%", f"{rb['max_dd']}%", f"{rb['max_dd']-ra['max_dd']:+.1f}%p"),
        ("평균 수익", f"${sa['avg_win']}", f"${sb['avg_win']}", ""),
        ("평균 손실", f"${sa['avg_loss']}", f"${sb['avg_loss']}", ""),
        ("실질 R:R", f"{sa['real_rr']}", f"{sb['real_rr']}", ""),
        ("최대 연승", f"{sa['max_consec_win']}", f"{sb['max_consec_win']}", ""),
        ("최대 연패", f"{sa['max_consec_loss']}", f"{sb['max_consec_loss']}", ""),
        ("총 수수료", f"${sa['total_fee']:,.0f}", f"${sb['total_fee']:,.0f}", ""),
        ("손실 월 비율", f"{sa['loss_month_pct']}%", f"{sb['loss_month_pct']}%", ""),
    ]
    for label, va, vb, diff in rows:
        print(f"{label:<30} {va:>20} {vb:>20} {diff:>20}")

    # 연도별
    print("\n" + "=" * 100)
    print("연도별 비교")
    print("=" * 100)
    ya = ra.get("yearly", {})
    yb = rb.get("yearly", {})
    all_years = sorted(set(list(ya.keys()) + list(yb.keys())))
    print(f"{'연도':<8} {'A거래':>7} {'A승률':>7} {'A_PnL':>12} {'B거래':>7} {'B승률':>7} {'B_PnL':>12} {'차이':>12}")
    print("-" * 80)
    for y in all_years:
        da = ya.get(y, {"trades": 0, "wins": 0, "pnl": 0})
        db = yb.get(y, {"trades": 0, "wins": 0, "pnl": 0})
        wr_a = round(da["wins"] / da["trades"] * 100, 1) if da["trades"] else 0
        wr_b = round(db["wins"] / db["trades"] * 100, 1) if db["trades"] else 0
        diff = db["pnl"] - da["pnl"]
        print(f"{y:<8} {da['trades']:>7} {wr_a:>6.1f}% ${da['pnl']:>+10,.0f} "
              f"{db['trades']:>7} {wr_b:>6.1f}% ${db['pnl']:>+10,.0f} ${diff:>+10,.0f}")

    # 코인별
    print("\n" + "=" * 100)
    print("코인별 비교")
    print("=" * 100)
    print(f"{'코인':<12} {'A거래':>7} {'A_PnL':>12} {'B거래':>7} {'B_PnL':>12} {'차이':>12}")
    print("-" * 60)
    all_coins = sorted(set(list(sa["by_coin"].keys()) + list(sb["by_coin"].keys())))
    for c in all_coins:
        da = sa["by_coin"].get(c, {"trades": 0, "pnl": 0})
        db = sb["by_coin"].get(c, {"trades": 0, "pnl": 0})
        diff = db["pnl"] - da["pnl"]
        print(f"{c:<12} {da['trades']:>7} ${da['pnl']:>+10,.0f} "
              f"{db['trades']:>7} ${db['pnl']:>+10,.0f} ${diff:>+10,.0f}")

    # 방향별
    print("\n" + "=" * 100)
    print("방향별 비교")
    print("=" * 100)
    print(f"{'방향':<8} {'A거래':>7} {'A승률':>7} {'A_PnL':>12} {'B거래':>7} {'B승률':>7} {'B_PnL':>12}")
    print("-" * 65)
    for d in ["LONG", "SHORT"]:
        da = sa["by_direction"].get(d, {"trades": 0, "wins": 0, "pnl": 0})
        db = sb["by_direction"].get(d, {"trades": 0, "wins": 0, "pnl": 0})
        wr_a = round(da["wins"] / da["trades"] * 100, 1) if da["trades"] else 0
        wr_b = round(db["wins"] / db["trades"] * 100, 1) if db["trades"] else 0
        print(f"{d:<8} {da['trades']:>7} {wr_a:>6.1f}% ${da['pnl']:>+10,.0f} "
              f"{db['trades']:>7} {wr_b:>6.1f}% ${db['pnl']:>+10,.0f}")

    # 청산 사유
    print("\n" + "=" * 100)
    print("청산 사유 비교")
    print("=" * 100)
    all_reasons = sorted(set(list(sa["by_reason"].keys()) + list(sb["by_reason"].keys())))
    print(f"{'사유':<12} {'A':>7} {'B':>7} {'차이':>7}")
    print("-" * 35)
    for r in all_reasons:
        ca = sa["by_reason"].get(r, 0)
        cb = sb["by_reason"].get(r, 0)
        print(f"{r:<12} {ca:>7} {cb:>7} {cb-ca:>+7}")

    # 손실 월 상세
    print("\n" + "=" * 100)
    print("월별 PnL 비교 (손실 월 하이라이트)")
    print("=" * 100)
    all_months = sorted(set(list(sa["monthly"].keys()) + list(sb["monthly"].keys())))
    print(f"{'월':<10} {'A_PnL':>10} {'B_PnL':>10} {'차이':>10} {'판정':>10}")
    print("-" * 55)
    a_better = b_better = 0
    for m in all_months:
        da = sa["monthly"].get(m, {"pnl": 0})
        db = sb["monthly"].get(m, {"pnl": 0})
        diff = db["pnl"] - da["pnl"]
        if diff > 0:
            verdict = "B승"
            b_better += 1
        elif diff < 0:
            verdict = "A승"
            a_better += 1
        else:
            verdict = "동일"
        print(f"{m:<10} ${da['pnl']:>+8,.0f} ${db['pnl']:>+8,.0f} ${diff:>+8,.0f} {verdict:>10}")

    print(f"\n월별 승수: A가 {a_better}개월, B가 {b_better}개월 우위")


def main():
    print("=" * 100)
    print("HERMES v5 — 쿨다운 유무 상세 비교")
    print(f"시드: ${SEED:,.0f} | 파라미터: v3 BEST + trailing 1.5/0.3")
    print(f"수동 감독 (2023 skip) | 서버비 ₩1,150/일")
    print("=" * 100)

    print("\n[데이터 로드]")
    data = load_all_data()
    print(f"  로드 완료: {len(data)} 데이터셋")

    results = {}
    for name, opts in SCENARIOS.items():
        print(f"\n  ▶ {name} 백테스트 중...")
        r = run_shared_backtest(
            data, BEST_PARAMS, SEED,
            use_funding=True,
            trailing_activation=1.5, trailing_distance=0.3,
            block_sol_long=True,
            skip_years=(2023,),
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            **opts,
        )

        # 상세 분석용으로 trades 다시 추출
        # (run_shared_backtest는 trades를 직접 안 줌 — result에 stats만 있음)
        # → 엔진에서 trades 반환하도록 수정 필요. 일단 result 기반으로 진행
        results[name] = {"result": r, "opts": opts}

    # trades가 필요하므로 엔진에서 반환하도록 재실행
    # v4_shared_engine의 return에 trades 추가
    print("\n  ▶ 상세 분석용 재실행 (trades 포함)...")
    for name, opts in SCENARIOS.items():
        r = run_shared_backtest(
            data, BEST_PARAMS, SEED,
            use_funding=True,
            trailing_activation=1.5, trailing_distance=0.3,
            block_sol_long=True,
            skip_years=(2023,),
            daily_cost_usd=DAILY_COST_USD,
            ruin_threshold=15.0,
            **opts,
        )
        # trades는 result 내에 없으므로 stats는 result에서 구함
        results[name]["result"] = r
        results[name]["stats"] = analyze_trades(
            r.get("_trades", []), SEED
        ) if "_trades" in r else {"avg_win": 0, "avg_loss": 0, "real_rr": 0,
                                   "max_consec_win": 0, "max_consec_loss": 0,
                                   "total_fee": 0, "monthly": {}, "by_coin": {},
                                   "by_direction": {}, "by_reason": {},
                                   "loss_months": 0, "total_months": 0, "loss_month_pct": 0}

    # trades 반환이 안 되면 직접 비교
    if not results["A_현재시스템"]["stats"].get("total", 0):
        print("\n  ⚠ trades 미반환 — 엔진에 _trades 반환 추가 필요")
        # 기본 비교만 출력
        for name in SCENARIOS:
            r = results[name]["result"]
            print(f"\n  [{name}]")
            print(f"    거래: {r['total_trades']} | 승률: {r['win_rate']}% | "
                  f"순이익: ${r['net_profit']:+,.0f} | DD: {r['max_dd']}%")
            print(f"    연도별: {r.get('yearly', {})}")

    print_comparison(results)

    # JSON 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "params": {k: v for k, v in BEST_PARAMS.items()
                       if isinstance(v, (int, float, str, bool))},
            "trailing": {"activation": 1.5, "distance": 0.3},
            "daily_cost_usd": round(DAILY_COST_USD, 4),
            "skip_years": [2023],
        },
        "A_현재시스템": {
            "cooldown": "2연패→60분, 3연패→당일중단",
            **{k: v for k, v in results["A_현재시스템"]["result"].items() if k != "_trades"},
        },
        "B_쿨다운제거": {
            "cooldown": "없음",
            **{k: v for k, v in results["B_쿨다운제거"]["result"].items() if k != "_trades"},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "v5_cooldown_compare.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
