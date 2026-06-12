#!/usr/bin/env python3
"""V11 백테: 오더북 필터 on/off 비교.

post-filter 방식: V11 백테를 그대로 돌려서 전체 거래 얻은 뒤,
오더북이 진입 방향을 막았을 거래들을 제거하여 비교한다.

오더북 데이터: data/orderbook_hourly/{SYMBOL}_{DATE}_hourly.json
필터 기준: bid_ratio >= 0.55 (LONG) / (1-bid_ratio) >= 0.55 (SHORT)
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
    """시간별 bid_ratio 룩업: {(symbol, hour_ts_ms): bid_ratio}"""
    lookup = {}
    for f in OB_DIR.glob('*_hourly.json'):
        parts = f.stem.split('_')
        symbol = parts[0]
        try:
            data = json.loads(f.read_text())
        except:
            continue
        for rec in data:
            lookup[(symbol, rec['hour_ts_ms'])] = rec['bid_ratio']
    return lookup


def nearest_hourly(ts_ms: int) -> int:
    """ts를 시간 내림 정렬."""
    dt = datetime.utcfromtimestamp(ts_ms / 1000)
    floor = dt.replace(minute=0, second=0, microsecond=0)
    return int(floor.timestamp() * 1000)


def orderbook_allows(lookup, symbol, ts_ms, direction):
    """오더북이 방향 진입 허용하는가? 데이터 없으면 True (후하게)."""
    hour = nearest_hourly(ts_ms)
    br = lookup.get((symbol, hour))
    if br is None:
        return None  # 데이터 없음
    if direction == 'LONG':
        return br >= ORDERBOOK_THRESHOLD
    else:
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


def main():
    print('오더북 룩업 로딩...')
    lookup = load_orderbook_lookup()
    print(f'  총 {len(lookup)}개 시간별 스냅샷 로드')
    sym_counts = defaultdict(int)
    for (sym, _), _ in lookup.items():
        sym_counts[sym] += 1
    for s, c in sorted(sym_counts.items()):
        print(f'    {s}: {c}')

    if not lookup:
        print('오더북 데이터 없음. 다운로드 먼저.')
        return

    print('\nV11 백테 실행 (오더북 없이)...')
    data = _load_data()
    d = filter_d(data, '2026-03-01', '2026-04-22')
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

    live_start = int(datetime.strptime('2026-04-09', '%Y-%m-%d').timestamp() * 1000)
    live_end = int(datetime.strptime('2026-04-22', '%Y-%m-%d').timestamp() * 1000)
    window = [t for t in all_trades if live_start <= t['timestamp'] < live_end]

    # Classify each trade
    kept = []
    rejected = []
    no_data = []
    for t in window:
        allow = orderbook_allows(lookup, t['symbol'], t['timestamp'], t['direction'])
        if allow is None:
            no_data.append(t)
            kept.append(t)  # 보수적으로 둠
        elif allow:
            kept.append(t)
        else:
            rejected.append(t)

    def stats(trades, label):
        n = len(trades); w = sum(1 for t in trades if t['pnl'] > 0)
        p = sum(t['pnl'] for t in trades)
        wr = w / max(n, 1) * 100
        return f'{label:<25} {n:>3}건 승률 {wr:>5.1f}%  PnL ${p:>+7.2f}'

    print('\n' + '=' * 80)
    print('V11 백테: 오더북 필터 전후 비교 (04-09 ~ 04-21)')
    print('=' * 80)
    print(stats(window, '오더북 OFF (전체)'))
    print(stats(kept, '오더북 ON (통과)'))
    print(stats(rejected, '오더북 ON (차단)'))
    print(stats(no_data, '데이터 없음 (참고)'))

    # Trade by trade
    print('\n' + '=' * 80)
    print('거래별 오더북 판정 상세')
    print('=' * 80)
    for t in window:
        dt_kst = datetime.utcfromtimestamp(t['timestamp']/1000 + 9*3600)
        hour = nearest_hourly(t['timestamp'])
        br = lookup.get((t['symbol'], hour))
        if br is None:
            status = '❓ 데이터X'
            br_str = '   -'
        else:
            allow = orderbook_allows(lookup, t['symbol'], t['timestamp'], t['direction'])
            status = '✅ 통과' if allow else '❌ 차단'
            br_str = f'{br:.3f}'
        print(f'  {dt_kst.strftime("%m-%d %H:%M KST"):<16} {t["symbol"][:3]} {t["direction"]:<5}  '
              f'bid_ratio={br_str}  {status}  PnL ${t["pnl"]:>+7.2f}  {t["reason"]}')

    # 요약
    print('\n' + '=' * 80)
    print('📊 요약')
    print('=' * 80)
    rej_pnl = sum(t['pnl'] for t in rejected)
    kept_pnl = sum(t['pnl'] for t in kept)
    total_pnl = sum(t['pnl'] for t in window)
    print(f'  거래 필터링: {len(rejected)}/{len(window)}건 제거')
    print(f'  차단된 거래 PnL 합: ${rej_pnl:+.2f}')
    print(f'    → 필터 유효성: {"✅ 손실 회피" if rej_pnl < 0 else "⚠️ 수익 기회 놓침"}')
    print(f'  필터 적용 후 예상 PnL: ${kept_pnl:+.2f}  (원래 ${total_pnl:+.2f})')
    print(f'  개선: ${kept_pnl - total_pnl:+.2f}')


if __name__ == '__main__':
    main()
