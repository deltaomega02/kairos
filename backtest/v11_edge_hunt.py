#!/usr/bin/env python3
"""V11 + 오더북 종합 edge hunt (exhaustive).

Phase 1-6 순차 실행:
  1. 방향별 threshold 스윕
  2. 코인별 threshold 스윕
  3. ADX 조건부 (TRENDING vs RANGING)
  4. 시간대별 (hour of day)
  5. Score threshold 변화
  6. Walk-forward (월별 분할)

결과: JSON + 콘솔 요약
"""
import sys, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, '~/Projects/HERMES/backtest')
from v9_mega_sweep import _load_data, SEED, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

OB_DIR = Path('~/Projects/HERMES/backtest/data/orderbook_hourly')
OUT = Path('~/Projects/HERMES_백테스팅/v11/edge_hunt.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_orderbook_lookup():
    lookup = {}
    for f in OB_DIR.glob('*_hourly.json'):
        sym = f.stem.split('_')[0]
        try:
            data = json.loads(f.read_text())
        except: continue
        for rec in data:
            lookup[(sym, rec['hour_ts_ms'])] = rec['bid_ratio']
    return lookup


def nearest_hourly(ts):
    dt = datetime.utcfromtimestamp(ts/1000)
    return int(dt.replace(minute=0, second=0, microsecond=0).timestamp()*1000)


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


def run_v11(data, score_thresh=40):
    """기본 V11 설정으로 백테 → 전체 거래 반환."""
    V11_params = {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
                  'tp_rr_ratio':6.0, 'entry_score_threshold':score_thresh,
                  'pullback_ema_dist_pct':1.5, 'adx_enter_trending':30}
    kw = {'d1_filter_enable':True, 'd1_ema_period':2, 'd1_mode':'price_above_ema'}
    base = dict(trailing_activation=1.2, trailing_distance=0.1, skip_years=(),
                daily_cost_usd=DAILY_COST, slippage_pct=0.05,
                max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                enabled_symbols=SYMBOLS, block_sol_long=True)
    r = run_shared_backtest_v11(data, V11_params, SEED, **base, **kw)
    return r.get('_trades', [])


def pnl_stats(trades):
    n = len(trades)
    if n == 0:
        return {'n':0, 'wr':0, 'pnl':0, 'avg':0}
    w = sum(1 for t in trades if t['pnl']>0)
    p = sum(t['pnl'] for t in trades)
    return {'n':n, 'wr':w/n*100, 'pnl':p, 'avg':p/n}


def apply_ob_filter(trades, lookup, rules):
    """rules: dict like {('LONG','BTCUSDT'): 0.55, ...} or ('LONG','*'): 0.55.
    None threshold = no filter.
    """
    kept = []
    for t in trades:
        direction = t['direction']
        sym = t['symbol']
        # 특정 (dir, coin) 우선, 없으면 (dir, '*')
        thresh = rules.get((direction, sym)) or rules.get((direction, '*'))
        if thresh is None:
            kept.append(t)
            continue
        hour = nearest_hourly(t['timestamp'])
        br = lookup.get((sym, hour))
        if br is None:
            kept.append(t)
            continue
        if direction == 'LONG':
            if br >= thresh:
                kept.append(t)
        else:
            if (1 - br) >= thresh:
                kept.append(t)
    return kept


def main():
    print('='*90)
    print('V11 Edge Hunt — 8개월 exhaustive analysis')
    print('='*90)

    lookup = load_orderbook_lookup()
    print(f'오더북 룩업: {len(lookup)}개 스냅샷 로드')

    print('\nV11 백테 실행 (baseline, score=40)...', flush=True)
    data = _load_data()
    d = filter_d(data, '2025-07-01', '2026-04-22')
    all_trades = run_v11(d, score_thresh=40)

    # 윈도우 필터
    ws = int(datetime.strptime('2025-08-22','%Y-%m-%d').timestamp()*1000)
    we = int(datetime.strptime('2026-04-22','%Y-%m-%d').timestamp()*1000)
    window = [t for t in all_trades if ws <= t['timestamp'] < we]
    print(f'기본 V11: {len(window)}건 거래 (full 8mo)')

    results = {'baseline': {}, 'phase1': {}, 'phase2': {}, 'phase3': {},
               'phase4': {}, 'phase5': {}, 'phase6': {}}

    # Baseline refs
    baseline_off = pnl_stats(window)
    baseline_on_055 = pnl_stats(apply_ob_filter(window, lookup,
                                                  {('LONG','*'):0.55, ('SHORT','*'):0.55}))
    baseline_long_only = pnl_stats(apply_ob_filter(window, lookup, {('LONG','*'):0.55}))
    results['baseline'] = {
        'A_all_off': baseline_off,
        'B_all_on_055': baseline_on_055,
        'C_long_only_055': baseline_long_only,
    }
    print(f'\n[Baseline]')
    print(f'  A (OFF)          : {baseline_off}')
    print(f'  B (ON 0.55)       : {baseline_on_055}')
    print(f'  C (LONG 0.55)     : {baseline_long_only}')

    # ===== Phase 1: 방향별 threshold 스윕 =====
    print('\n' + '='*90)
    print('Phase 1: 방향별 오더북 threshold 스윕 (0.40~0.70)')
    print('='*90)
    print(f'{"LONG":>7}{"SHORT":>8}{"거래":>7}{"승률":>8}{"PnL":>12}{"평균":>9}')
    best_p1 = (None, None, -1e9, None)
    for l_th in [None, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        for s_th in [None, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            rules = {}
            if l_th is not None: rules[('LONG','*')] = l_th
            if s_th is not None: rules[('SHORT','*')] = s_th
            filtered = apply_ob_filter(window, lookup, rules)
            st = pnl_stats(filtered)
            results['phase1'][f'L{l_th}_S{s_th}'] = st
            if st['pnl'] > best_p1[2]:
                best_p1 = (l_th, s_th, st['pnl'], st)
    # 상위 10개 출력
    sorted_p1 = sorted(results['phase1'].items(), key=lambda x: -x[1]['pnl'])[:15]
    for key, st in sorted_p1:
        lt = key.split('_')[0][1:]; st2 = key.split('_')[1][1:]
        print(f'  {lt:>5}  {st2:>5}   {st["n"]:>5}  {st["wr"]:>5.1f}%  ${st["pnl"]:>+9.0f}  ${st["avg"]:>+6.2f}')
    print(f'  최고: LONG={best_p1[0]}, SHORT={best_p1[1]} → ${best_p1[2]:+.0f}')

    # ===== Phase 2: 코인별 threshold 스윕 =====
    print('\n' + '='*90)
    print('Phase 2: 코인별 오더북 threshold (방향별로)')
    print('='*90)
    for coin in SYMBOLS:
        coin_sh = coin[:3]
        coin_trades = [t for t in window if t['symbol']==coin]
        if not coin_trades:
            continue
        print(f'\n[{coin_sh}] 총 {len(coin_trades)}건')
        for direction in ['LONG','SHORT']:
            dir_trades = [t for t in coin_trades if t['direction']==direction]
            if not dir_trades:
                continue
            print(f'  {direction} ({len(dir_trades)}건):')
            print(f'    {"threshold":<12}{"거래":>6}{"승률":>8}{"PnL":>10}{"vs NoFilter":>14}')
            no_filter_pnl = sum(t['pnl'] for t in dir_trades)
            for th in [None, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
                if th is None:
                    st = pnl_stats(dir_trades)
                    diff = 0
                else:
                    filtered = []
                    for t in dir_trades:
                        hour = nearest_hourly(t['timestamp'])
                        br = lookup.get((coin, hour))
                        if br is None:
                            filtered.append(t); continue
                        if direction == 'LONG':
                            if br >= th: filtered.append(t)
                        else:
                            if (1-br) >= th: filtered.append(t)
                    st = pnl_stats(filtered)
                    diff = st['pnl'] - no_filter_pnl
                th_s = 'NoFilter' if th is None else f'{th:.2f}'
                print(f'    {th_s:<12}{st["n"]:>6}  {st["wr"]:>5.1f}%  ${st["pnl"]:>+6.0f}  ${diff:>+9.0f}')
                results['phase2'][f'{coin}_{direction}_{th}'] = st

    # ===== Phase 3: ADX 조건부 =====
    # Engine's trade dict doesn't include adx directly, but we can derive from regime.
    # 일단 regime 필드 활용: TRENDING_UP/DOWN vs RANGING
    print('\n' + '='*90)
    print('Phase 3: 레짐별 오더북 효과 (거래에 붙은 regime 필드가 UNKNOWN 이라 스킵 불가)')
    print('='*90)
    # Actually, we can infer regime from direction + market state
    # We have regime in engine but not in trade dict
    # Let's use 4H ADX indirectly by looking at the 4H regime at trade time
    # Actually, let's use price-based proxy: is price in an uptrend (EMA ordering on 4H)?
    # Skipping for now since requires re-engineering
    print('  → 엔진이 regime을 거래 dict에 저장 안 함. Phase 3 스킵.')
    print('  (별도 작업: 엔진 수정해서 regime 기록 후 재분석 필요)')

    # ===== Phase 4: 시간대별 =====
    print('\n' + '='*90)
    print('Phase 4: 시간대별 (KST) 승률/PnL')
    print('='*90)
    hour_buckets = defaultdict(lambda: {'n':0, 'w':0, 'pnl':0, 'long':[], 'short':[]})
    for t in window:
        dt_kst = datetime.utcfromtimestamp(t['timestamp']/1000 + 9*3600)
        h = dt_kst.hour
        hour_buckets[h]['n'] += 1
        hour_buckets[h]['pnl'] += t['pnl']
        if t['pnl'] > 0: hour_buckets[h]['w'] += 1
        if t['direction'] == 'LONG':
            hour_buckets[h]['long'].append(t['pnl'])
        else:
            hour_buckets[h]['short'].append(t['pnl'])
    print(f'{"시간 KST":>8}{"거래":>6}{"승률":>8}{"PnL":>10}{"LONG pnl":>12}{"SHORT pnl":>12}')
    for h in sorted(hour_buckets.keys()):
        b = hour_buckets[h]
        wr = b['w']/max(b['n'],1)*100
        lp = sum(b['long']); sp = sum(b['short'])
        print(f'  {h:>4}시  {b["n"]:>5}  {wr:>5.1f}%  ${b["pnl"]:>+6.0f}  ${lp:>+8.0f}  ${sp:>+8.0f}')
        results['phase4'][h] = {'n':b['n'], 'wr':wr, 'pnl':b['pnl'],
                                  'long_pnl':lp, 'short_pnl':sp}

    # ===== Phase 5: Score threshold 변화 =====
    print('\n' + '='*90)
    print('Phase 5: 진입 점수 threshold 변화')
    print('='*90)
    print(f'{"score":>7}{"거래":>7}{"승률":>8}{"PnL":>12}{"평균":>9}')
    for s_th in [30, 35, 40, 45, 50, 55, 60]:
        trades_s = run_v11(d, score_thresh=s_th)
        win = [t for t in trades_s if ws <= t['timestamp'] < we]
        st = pnl_stats(win)
        print(f'  {s_th:>5}  {st["n"]:>5}  {st["wr"]:>5.1f}%  ${st["pnl"]:>+9.0f}  ${st["avg"]:>+6.2f}')
        results['phase5'][s_th] = st

    # ===== Phase 6: Walk-forward (월별 분할 검증) =====
    # 최적 조합을 월별로 돌려서 일관성 있는지 확인
    print('\n' + '='*90)
    print('Phase 6: 월별 Walk-forward (상위 시나리오 일관성)')
    print('='*90)
    scenarios = {
        'A_off': {},
        'B_on055': {('LONG','*'):0.55, ('SHORT','*'):0.55},
        'C_long_only': {('LONG','*'):0.55},
        'D_long055_short045': {('LONG','*'):0.55, ('SHORT','*'):0.45},
        'D_best_p1': {('LONG','*'):best_p1[0], ('SHORT','*'):best_p1[1]} if best_p1[0] or best_p1[1] else {},
    }
    months = ['2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04']
    print(f'{"시나리오":<22}', end='')
    for m in months:
        print(f'{m[5:]:>9}', end='')
    print(f'{"합계":>10}')
    print('-' * (22 + 9*len(months) + 10))
    for name, rules in scenarios.items():
        row = f'{name:<22}'
        total = 0
        for m in months:
            ms = int(datetime.strptime(f'{m}-01','%Y-%m-%d').timestamp()*1000)
            # next month start
            if m == '2026-04':
                me = int(datetime.strptime('2026-04-22','%Y-%m-%d').timestamp()*1000)
            else:
                yr, mo = m.split('-'); mo = int(mo)+1; yr = int(yr)
                if mo > 12: mo=1; yr+=1
                me = int(datetime.strptime(f'{yr}-{mo:02d}-01','%Y-%m-%d').timestamp()*1000)
            m_trades = [t for t in window if ms <= t['timestamp'] < me]
            filtered = apply_ob_filter(m_trades, lookup, rules) if rules else m_trades
            pnl = sum(t['pnl'] for t in filtered)
            total += pnl
            row += f'{pnl:>+9.0f}'
        row += f'{total:>+10.0f}'
        print(row)
        results['phase6'][name] = {'monthly': {m: None for m in months}, 'total': total}

    # Save
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n✓ Saved: {OUT}')


if __name__ == '__main__':
    main()
