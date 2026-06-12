#!/usr/bin/env python3
"""V11 백테 전체 구간 (2025-08-22 ~ 2026-04-21) + 오더북 필터 비교.

오더북 필터 on/off 두 시나리오 실행 후 지표 비교:
- 총 PnL, 거래수, 승률
- 월별 PnL 분포
- MC 중앙값 재추정 (간단 bootstrap)
- 실전 구간 (04-09~) 분리 통계
"""
import os, sys, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '/Users/sue/Projects/HERMES/backtest')
from v9_mega_sweep import _load_data, SEED, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

OB_DIR = Path('/Users/sue/Projects/HERMES/backtest/data/orderbook_hourly')
ORDERBOOK_THRESHOLD = 0.55


def load_orderbook_lookup():
    lookup = {}
    for f in OB_DIR.glob('*_hourly.json'):
        sym = f.stem.split('_')[0]
        try:
            data = json.loads(f.read_text())
        except:
            continue
        for rec in data:
            lookup[(sym, rec['hour_ts_ms'])] = rec['bid_ratio']
    return lookup


def nearest_hourly(ts_ms: int) -> int:
    dt = datetime.utcfromtimestamp(ts_ms / 1000)
    floor = dt.replace(minute=0, second=0, microsecond=0)
    return int(floor.timestamp() * 1000)


def orderbook_allows(lookup, symbol, ts_ms, direction):
    hour = nearest_hourly(ts_ms)
    br = lookup.get((symbol, hour))
    if br is None:
        return None
    if direction == 'LONG':
        return br >= ORDERBOOK_THRESHOLD
    return (1 - br) >= ORDERBOOK_THRESHOLD


def filter_d(data, s, e):
    s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp() * 1000)
    e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp() * 1000)
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and 'timestamp' in v.columns:
            m = (v['timestamp'] >= s_ts) & (v['timestamp'] < e_ts)
            out[k] = v[m].reset_index(drop=True)
        else:
            out[k] = v
    return out


def stats_line(trades, label):
    n = len(trades)
    if n == 0:
        return f'{label:<28} 0건'
    w = sum(1 for t in trades if t['pnl'] > 0)
    p = sum(t['pnl'] for t in trades)
    wr = w / n * 100
    avg = p / n
    return f'{label:<28} {n:>4}건  승률 {wr:>5.1f}%  PnL ${p:>+9.2f}  평균 ${avg:>+6.2f}/거래'


def monthly_breakdown(trades, lookup, filter_ob=False):
    monthly = defaultdict(lambda: {'pnl': 0, 'n': 0, 'w': 0})
    for t in trades:
        if filter_ob:
            allow = orderbook_allows(lookup, t['symbol'], t['timestamp'], t['direction'])
            if allow is False:
                continue
        dt = datetime.utcfromtimestamp(t['timestamp'] / 1000)
        key = dt.strftime('%Y-%m')
        monthly[key]['n'] += 1
        monthly[key]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            monthly[key]['w'] += 1
    return dict(monthly)


def main():
    print('오더북 룩업 로딩...', flush=True)
    lookup = load_orderbook_lookup()
    print(f'  {len(lookup)}개 시간 스냅샷 (코인별 균등 분배)')

    print('\nV11 백테 실행 (full 8개월)...', flush=True)
    data = _load_data()
    d = filter_d(data, '2025-07-01', '2026-04-22')  # warmup 포함
    V11 = {
        'params': {**DEFAULT_PARAMS, 'ema_fast': 3, 'ema_slow': 15, 'sl_atr_mult': 1.5,
                   'tp_rr_ratio': 6.0, 'entry_score_threshold': 40,
                   'pullback_ema_dist_pct': 1.5, 'adx_enter_trending': 30},
        'kw': {'d1_filter_enable': True, 'd1_ema_period': 2, 'd1_mode': 'price_above_ema'},
    }
    base_args = dict(trailing_activation=1.2, trailing_distance=0.1,
                     skip_years=(), daily_cost_usd=DAILY_COST, slippage_pct=0.05,
                     max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                     enabled_symbols=SYMBOLS, block_sol_long=True)
    r = run_shared_backtest_v11(d, V11['params'], SEED, **base_args, **V11['kw'])
    all_trades = r.get('_trades', [])

    window_start = int(datetime.strptime('2025-08-22', '%Y-%m-%d').timestamp() * 1000)
    window_end = int(datetime.strptime('2026-04-22', '%Y-%m-%d').timestamp() * 1000)
    window = [t for t in all_trades if window_start <= t['timestamp'] < window_end]

    # Classify trades
    kept = []
    rejected = []
    no_data = []
    for t in window:
        allow = orderbook_allows(lookup, t['symbol'], t['timestamp'], t['direction'])
        if allow is None:
            no_data.append(t)
            kept.append(t)
        elif allow:
            kept.append(t)
        else:
            rejected.append(t)

    print('\n' + '='*95)
    print('V11 백테 8개월 (2025-08-22 ~ 2026-04-21): 오더북 필터 비교')
    print('='*95)
    print(stats_line(window, '원본 (오더북 OFF)'))
    print(stats_line(kept, '오더북 ON (통과)'))
    print(stats_line(rejected, '오더북 ON (차단)'))
    print(stats_line(no_data, '오더북 데이터 없음'))

    # Month-by-month breakdown
    mon_off = monthly_breakdown(window, lookup, filter_ob=False)
    mon_on = monthly_breakdown(window, lookup, filter_ob=True)

    print('\n' + '='*95)
    print(f'월별 PnL ({"OFF":>12} / {"ON (오더북)":>12})')
    print('='*95)
    print(f'  {"월":<10}{"OFF 거래":>10}{"OFF PnL":>12}{"ON 거래":>10}{"ON PnL":>12}{"개선":>12}')
    print('  ' + '-' * 80)
    all_months = sorted(set(list(mon_off.keys()) + list(mon_on.keys())))
    off_total = 0
    on_total = 0
    for m in all_months:
        off = mon_off.get(m, {'pnl': 0, 'n': 0, 'w': 0})
        on = mon_on.get(m, {'pnl': 0, 'n': 0, 'w': 0})
        diff = on['pnl'] - off['pnl']
        off_total += off['pnl']
        on_total += on['pnl']
        print(f'  {m:<10}{off["n"]:>10}{off["pnl"]:>+12.2f}{on["n"]:>10}{on["pnl"]:>+12.2f}{diff:>+12.2f}')
    print('  ' + '-' * 80)
    print(f'  {"합계":<10}{len(window):>10}{off_total:>+12.2f}{len(kept):>10}{on_total:>+12.2f}{on_total-off_total:>+12.2f}')

    # 실전 구간 분리
    live_start = int(datetime.strptime('2026-04-09', '%Y-%m-%d').timestamp() * 1000)
    pre_live = [t for t in window if t['timestamp'] < live_start]
    live_ok = [t for t in kept if t['timestamp'] >= live_start]
    live_all = [t for t in window if t['timestamp'] >= live_start]
    pre_live_kept = [t for t in kept if t['timestamp'] < live_start]

    print('\n' + '='*95)
    print('실전 vs 과거 분리')
    print('='*95)
    print(stats_line(pre_live, '과거 OFF (2025-08 ~ 2026-04-08)'))
    print(stats_line(pre_live_kept, '과거 ON (오더북 필터 통과)'))
    print(stats_line(live_all, '실전 구간 OFF (2026-04-09+)'))
    print(stats_line(live_ok, '실전 구간 ON (오더북 통과)'))

    # 실전 실제 비교
    print('\n' + '='*95)
    print('실전 실제 -$32 vs 모델 예측')
    print('='*95)
    live_off_pnl = sum(t['pnl'] for t in live_all)
    live_on_pnl = sum(t['pnl'] for t in live_ok)
    print(f'  실전 실제:                     -$32.20  (29거래, 42.9%)')
    print(f'  V11 백테 OFF (가상):           ${live_off_pnl:+.2f}  ({len(live_all)}거래)')
    print(f'  V11 백테 ON  (실전과 동일):    ${live_on_pnl:+.2f}  ({len(live_ok)}거래)')

    # 간단 MC: 거래별 PnL을 bootstrap 해서 distribution 추정 (N=500)
    import random
    random.seed(42)
    def bootstrap_distribution(trades, n_samples=500, seed=580):
        if len(trades) == 0:
            return None
        pnls = [t['pnl'] for t in trades]
        results = []
        for _ in range(n_samples):
            # 4년 지속 기준: 거래수를 4x로 확장 (8개월 * 6 = 4년)
            n_resample = len(trades) * 6
            sample = random.choices(pnls, k=n_resample)
            results.append(sum(sample))
        results.sort()
        return {
            'p5': results[int(0.05 * n_samples)],
            'p25': results[int(0.25 * n_samples)],
            'median': results[int(0.50 * n_samples)],
            'p75': results[int(0.75 * n_samples)],
            'p95': results[int(0.95 * n_samples)],
            'n': n_samples,
        }

    off_mc = bootstrap_distribution(window)
    on_mc = bootstrap_distribution(kept)
    print('\n' + '='*95)
    print('Bootstrap MC (8개월 데이터 → 4년 확장, N=500)')
    print('='*95)
    print(f'  오더북 OFF 4년 기대치:')
    print(f'    p5={off_mc["p5"]:>+10.0f}  p25={off_mc["p25"]:>+10.0f}  '
          f'median={off_mc["median"]:>+10.0f}  p75={off_mc["p75"]:>+10.0f}  '
          f'p95={off_mc["p95"]:>+10.0f}')
    print(f'  오더북 ON  4년 기대치:')
    print(f'    p5={on_mc["p5"]:>+10.0f}  p25={on_mc["p25"]:>+10.0f}  '
          f'median={on_mc["median"]:>+10.0f}  p75={on_mc["p75"]:>+10.0f}  '
          f'p95={on_mc["p95"]:>+10.0f}')
    print()
    print('  (주의: bootstrap은 단순 순 PnL 합. 복리 시드성장/DD 미반영)')

    # Save
    save_path = Path('/Users/sue/Projects/HERMES_백테스팅/v11/orderbook_full_8mo.json')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'ran_at': datetime.now().isoformat(),
        'window': '2025-08-22 ~ 2026-04-21',
        'off': {'n': len(window), 'wins': sum(1 for t in window if t['pnl']>0),
                 'pnl': round(sum(t['pnl'] for t in window), 2)},
        'on': {'n': len(kept), 'wins': sum(1 for t in kept if t['pnl']>0),
                'pnl': round(sum(t['pnl'] for t in kept), 2)},
        'rejected': {'n': len(rejected), 'wins': sum(1 for t in rejected if t['pnl']>0),
                     'pnl': round(sum(t['pnl'] for t in rejected), 2)},
        'no_orderbook_data': len(no_data),
        'mc_off_4yr': off_mc,
        'mc_on_4yr': on_mc,
        'monthly': {m: {'off': mon_off.get(m, {}), 'on': mon_on.get(m, {})}
                    for m in all_months},
    }
    save_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n✓ Saved: {save_path}')


if __name__ == '__main__':
    main()
