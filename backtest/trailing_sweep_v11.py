#!/usr/bin/env python3
"""V11 엔진에서 트레일링 조합 스윕 — 8개월 + 6년 양쪽.

Grid:
  activation: 0.8, 1.0, 1.2, 1.5, 2.0, 2.5
  distance: 0.1, 0.2, 0.3, 0.5, 0.8

각 조합에서 V11 풀 실행 → 총 PnL, 승률, 평균/거래.
현재 1.2/0.1 vs 나머지 비교.
"""
import sys, json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, '~/Projects/HERMES/backtest')
from v9_mega_sweep import _load_data, SEED, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

OUT = Path('~/Projects/HERMES_백테스팅/v12/trailing_sweep.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


def filter_d(data, s, e):
    s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
    e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and 'timestamp' in v.columns:
            m = (v['timestamp']>=s_ts) & (v['timestamp']<e_ts)
            out[k] = v[m].reset_index(drop=True)
        else: out[k] = v
    return out


def run_combo(data, window_s, window_e, activation, distance):
    V11 = {'params': {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
           'tp_rr_ratio':6.0, 'entry_score_threshold':40, 'pullback_ema_dist_pct':1.5,
           'adx_enter_trending':30},
           'kw': {'d1_filter_enable':True, 'd1_ema_period':2, 'd1_mode':'price_above_ema'}}
    base = dict(trailing_activation=activation, trailing_distance=distance,
                skip_years=(), daily_cost_usd=DAILY_COST, slippage_pct=0.05,
                max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                enabled_symbols=SYMBOLS, block_sol_long=True)
    r = run_shared_backtest_v11(data, V11['params'], SEED, **base, **V11['kw'])
    all_trades = r.get('_trades', [])
    ws = int(datetime.strptime(window_s, '%Y-%m-%d').timestamp()*1000)
    we = int(datetime.strptime(window_e, '%Y-%m-%d').timestamp()*1000)
    window = [t for t in all_trades if ws <= t['timestamp'] < we]
    return window


def stats(trades):
    n = len(trades)
    if n == 0: return {'n':0, 'wr':0, 'pnl':0, 'trail':0, 'sl':0}
    w = sum(1 for t in trades if t['pnl']>0)
    p = sum(t['pnl'] for t in trades)
    trail = sum(1 for t in trades if t.get('reason') == 'TRAILING')
    sl = sum(1 for t in trades if t.get('reason') == 'SL')
    return {'n':n, 'wr':w/n*100, 'pnl':p, 'avg':p/n, 'trail':trail, 'sl':sl}


ACTIVATIONS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
DISTANCES = [0.1, 0.2, 0.3, 0.5, 0.8]


def main():
    print('='*110)
    print('V11 엔진 트레일링 스윕 (8개월 오더북 구간)')
    print('='*110)

    data = _load_data()
    d_8mo = filter_d(data, '2025-07-01', '2026-04-22')

    # Header
    header = "act / dist"
    print(f'{header:<10}', end='')
    for dst in DISTANCES:
        print(f'{f"{dst}%":>14}', end='')
    print()
    print('-'*110)

    results_8mo = {}
    best = (None, None, -1e99)

    for act in ACTIVATIONS:
        print(f'{f"{act}%":<10}', end='')
        for dst in DISTANCES:
            w = run_combo(d_8mo, '2025-08-22', '2026-04-22', act, dst)
            st = stats(w)
            marker = ' *' if (act, dst) == (1.2, 0.1) else ''
            print(f' ${st["pnl"]:>+8.0f} {st["wr"]:>4.1f}%', end='')
            results_8mo[(act, dst)] = st
            if st['pnl'] > best[2]:
                best = (act, dst, st['pnl'])
        print()

    print()
    print(f'현재 설정 (1.2/0.1): ${results_8mo[(1.2, 0.1)]["pnl"]:+.0f}, 승률 {results_8mo[(1.2, 0.1)]["wr"]:.1f}%, 거래 {results_8mo[(1.2, 0.1)]["n"]}')
    print(f'최고 조합 ({best[0]}/{best[1]}): ${best[2]:+.0f}, 승률 {results_8mo[(best[0], best[1])]["wr"]:.1f}%, 거래 {results_8mo[(best[0], best[1])]["n"]}')
    print(f'차이: ${best[2] - results_8mo[(1.2, 0.1)]["pnl"]:+.0f} ({(best[2] / max(results_8mo[(1.2, 0.1)]["pnl"], 1) - 1)*100:+.1f}%)')

    # Top 10
    print()
    print('상위 10 조합 (8개월 PnL 기준):')
    sorted_combos = sorted(results_8mo.items(), key=lambda x: -x[1]['pnl'])
    print(f'  {"순위":<4}{"act":<6}{"dist":<6}{"거래":>6}{"승률":>8}{"PnL":>14}{"TRAIL/SL":>12}{"평균":>10}')
    for i, ((act, dst), st) in enumerate(sorted_combos[:10], 1):
        marker = ' ← 현재' if (act, dst) == (1.2, 0.1) else ''
        trail_ratio = f'{st["trail"]}/{st["sl"]}'
        print(f'  {i:<4}{act:<6}{dst:<6}{st["n"]:>6}{st["wr"]:>6.1f}%  ${st["pnl"]:>+10.0f}{trail_ratio:>12}  ${st["avg"]:>+7.2f}{marker}')

    # 6년 샘플도 (가장 중요한 조합 몇 개만)
    print()
    print('='*110)
    print('6년 전체 확장 검증 (상위 5개 조합)')
    print('='*110)
    d_6yr = filter_d(data, '2020-03-01', '2026-04-22')
    print(f'  {"act":<6}{"dist":<6}{"거래":>7}{"승률":>8}{"PnL (6yr 무한복리)":>24}{"평균":>12}')
    top5 = [(act, dst) for (act, dst), _ in sorted_combos[:5]]
    # Include current if not in top 5
    if (1.2, 0.1) not in top5:
        top5.append((1.2, 0.1))
    for act, dst in top5:
        w = run_combo(d_6yr, '2020-05-01', '2026-04-22', act, dst)
        st = stats(w)
        marker = ' ← 현재' if (act, dst) == (1.2, 0.1) else ''
        print(f'  {act:<6}{dst:<6}{st["n"]:>7}{st["wr"]:>6.1f}%  ${st["pnl"]:>+20.0f}  ${st["avg"]:>+9.2f}{marker}')

    # 실전 2주 구간
    print()
    print('='*110)
    print('실전 2주 구간 (2026-04-09 ~ 04-22)')
    print('='*110)
    print(f'  {"act":<6}{"dist":<6}{"거래":>6}{"승률":>8}{"PnL":>10}{"평균":>10}')
    for act, dst in [(0.8, 0.1), (1.0, 0.1), (1.2, 0.1), (1.5, 0.1), (2.0, 0.1),
                     (1.2, 0.2), (1.5, 0.2), (1.5, 0.3), (2.0, 0.3)]:
        w = run_combo(d_6yr, '2026-04-09', '2026-04-22', act, dst)
        # d_6yr로 돌렸으므로 warmup OK
        st = stats(w)
        marker = ' ← 현재' if (act, dst) == (1.2, 0.1) else ''
        print(f'  {act:<6}{dst:<6}{st["n"]:>6}{st["wr"]:>6.1f}%  ${st["pnl"]:>+8.0f}  ${st["avg"]:>+7.2f}{marker}')

    # Save
    out = {
        'ran_at': datetime.now().isoformat(),
        '8mo_results': {f'{a}/{d}': s for (a, d), s in results_8mo.items()},
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n✓ Saved: {OUT}')


if __name__ == '__main__':
    main()
