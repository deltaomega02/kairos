#!/usr/bin/env python3
"""V11 Mega Analysis — enriched trade data로 다차원 edge 탐색.

구성:
  A. 방향 × 레짐 × 오더북 매트릭스
  B. RSI 버킷 × 방향 × 승률
  C. ATR 버킷 × 방향 × 승률
  D. Volume ratio × 승률
  E. Funding sign × 방향
  F. 시간대 × 방향 × 코인
  G. 요일 × 승률
  H. 연패 패턴
  I. 최종 최적 조합 + 월별 walk-forward
"""
import sys, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/sue/Projects/HERMES/backtest')
from v9_mega_sweep import _load_data, SEED, DAILY_COST, SYMBOLS
from comprehensive_backtest import DEFAULT_PARAMS
from v11_all_features_engine import run_shared_backtest_v11

OB_DIR = Path('/Users/sue/Projects/HERMES/backtest/data/orderbook_hourly')
OUT = Path('/Users/sue/Projects/HERMES_백테스팅/v11/mega_edge.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_ob():
    lookup = {}
    for f in OB_DIR.glob('*_hourly.json'):
        sym = f.stem.split('_')[0]
        try: data = json.loads(f.read_text())
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


def stats(pnls):
    n = len(pnls)
    if n == 0: return (0, 0, 0, 0)
    w = sum(1 for p in pnls if p > 0)
    s = sum(pnls)
    return (n, w/n*100, s, s/n)


def main():
    ob = load_ob()
    print(f'오더북: {len(ob)}개 스냅샷')

    print('\nV11 백테 (enriched) 실행...', flush=True)
    data = _load_data()
    d = filter_d(data, '2025-07-01', '2026-04-22')
    V11 = {'params': {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
           'tp_rr_ratio':6.0, 'entry_score_threshold':40, 'pullback_ema_dist_pct':1.5,
           'adx_enter_trending':30},
           'kw': {'d1_filter_enable':True, 'd1_ema_period':2, 'd1_mode':'price_above_ema'}}
    base = dict(trailing_activation=1.2, trailing_distance=0.1, skip_years=(),
                daily_cost_usd=DAILY_COST, slippage_pct=0.05,
                max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                enabled_symbols=SYMBOLS, block_sol_long=True)
    r = run_shared_backtest_v11(d, V11['params'], SEED, **base, **V11['kw'])
    all_trades = r.get('_trades', [])

    ws = int(datetime.strptime('2025-08-22','%Y-%m-%d').timestamp()*1000)
    we = int(datetime.strptime('2026-04-22','%Y-%m-%d').timestamp()*1000)
    trades = [t for t in all_trades if ws <= t['timestamp'] < we]
    print(f'총 거래: {len(trades)}건')

    # Add orderbook info
    for t in trades:
        br = ob.get((t['symbol'], nearest_hourly(t['timestamp'])))
        t['bid_ratio'] = br

    # Enrich datetime info
    for t in trades:
        dt_kst = datetime.utcfromtimestamp(t['timestamp']/1000 + 9*3600)
        t['kst_hour'] = dt_kst.hour
        t['kst_dow'] = dt_kst.weekday()  # 0=Mon, 6=Sun

    results = {}

    # ==========  A. 방향 × 레짐 × 오더북 ==========
    print('\n' + '='*90)
    print('A. 방향 × 레짐 × 오더북 매트릭스')
    print('='*90)
    print(f'{"regime":<20}{"direction":<10}{"OB":<12}{"거래":<6}{"승률":<8}{"PnL":<10}{"평균":<8}')
    print('-'*90)
    A = defaultdict(list)
    for t in trades:
        reg = t.get('entry_regime') or 'UNKNOWN'
        br = t.get('bid_ratio')
        ob_ok = 'N/A'
        if br is not None:
            if t['direction'] == 'LONG':
                ob_ok = 'pass' if br >= 0.55 else 'reject'
            else:
                ob_ok = 'pass' if (1-br) >= 0.55 else 'reject'
        key = (reg, t['direction'], ob_ok)
        A[key].append(t['pnl'])

    for key in sorted(A.keys()):
        n, wr, s, avg = stats(A[key])
        print(f'  {key[0]:<18}{key[1]:<10}{key[2]:<12}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}  ${avg:>+6.2f}')
    results['A'] = {str(k): stats(v) for k,v in A.items()}

    # ==========  B. RSI 버킷 ==========
    print('\n' + '='*90)
    print('B. RSI 버킷 × 방향 × 승률')
    print('='*90)
    B = defaultdict(list)
    for t in trades:
        rsi = t.get('entry_rsi')
        if rsi is None or pd.isna(rsi): continue
        if rsi < 25: b = '<25 (극과매도)'
        elif rsi < 35: b = '25-35'
        elif rsi < 45: b = '35-45'
        elif rsi < 55: b = '45-55'
        elif rsi < 65: b = '55-65'
        elif rsi < 75: b = '65-75'
        else: b = '>75 (극과매수)'
        B[(b, t['direction'])].append(t['pnl'])
    print(f'{"RSI":<16}{"방향":<8}{"거래":<6}{"승률":<8}{"PnL":<10}')
    for key in sorted(B.keys()):
        n, wr, s, avg = stats(B[key])
        print(f'  {key[0]:<14}{key[1]:<8}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}')
    results['B'] = {str(k): stats(v) for k,v in B.items()}

    # ==========  C. ATR 버킷 ==========
    print('\n' + '='*90)
    print('C. ATR% 버킷 (저변동 위험 구역)')
    print('='*90)
    C = defaultdict(list)
    for t in trades:
        atr = t.get('entry_atr_pct')
        if atr is None or pd.isna(atr): continue
        if atr < 0.3: b = '<0.3 (극저)'
        elif atr < 0.5: b = '0.3-0.5'
        elif atr < 0.8: b = '0.5-0.8'
        elif atr < 1.2: b = '0.8-1.2'
        elif atr < 2.0: b = '1.2-2.0'
        else: b = '>2.0 (극고)'
        C[(b, t['direction'])].append(t['pnl'])
    print(f'{"ATR":<16}{"방향":<8}{"거래":<6}{"승률":<8}{"PnL":<10}')
    for key in sorted(C.keys()):
        n, wr, s, avg = stats(C[key])
        print(f'  {key[0]:<14}{key[1]:<8}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}')
    results['C'] = {str(k): stats(v) for k,v in C.items()}

    # ==========  D. Volume ratio ==========
    print('\n' + '='*90)
    print('D. Volume ratio × 승률')
    print('='*90)
    D = defaultdict(list)
    for t in trades:
        v = t.get('entry_vol_ratio')
        if v is None or pd.isna(v): continue
        if v < 0.5: b = '<0.5 (사각지대)'
        elif v < 1.0: b = '0.5-1.0'
        elif v < 1.5: b = '1.0-1.5'
        elif v < 2.5: b = '1.5-2.5'
        else: b = '>2.5 (폭증)'
        D[(b, t['direction'])].append(t['pnl'])
    for key in sorted(D.keys()):
        n, wr, s, avg = stats(D[key])
        print(f'  {key[0]:<14}{key[1]:<8}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}')
    results['D'] = {str(k): stats(v) for k,v in D.items()}

    # ==========  E. Funding ==========
    print('\n' + '='*90)
    print('E. Funding rate sign × 방향')
    print('='*90)
    E = defaultdict(list)
    for t in trades:
        f = t.get('entry_funding')
        if f is None or pd.isna(f): continue
        sign = 'POS(+)' if f > 0.0001 else ('NEG(-)' if f < -0.0001 else 'ZERO')
        E[(sign, t['direction'])].append(t['pnl'])
    for key in sorted(E.keys()):
        n, wr, s, avg = stats(E[key])
        print(f'  {key[0]:<14}{key[1]:<8}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}')
    results['E'] = {str(k): stats(v) for k,v in E.items()}

    # ==========  F. 시간대 × 방향 × 코인 ==========
    print('\n' + '='*90)
    print('F. 시간대(KST) × 방향 (상위/하위)')
    print('='*90)
    F = defaultdict(list)
    for t in trades:
        F[(t['kst_hour'], t['direction'])].append(t['pnl'])
    # 요약: 전체 PnL
    by_hour = defaultdict(list)
    for t in trades:
        by_hour[t['kst_hour']].append(t['pnl'])
    hour_pnl = [(h, sum(p), len(p), sum(1 for x in p if x>0)/max(len(p),1)*100) for h,p in by_hour.items()]
    hour_pnl.sort(key=lambda x: x[1])
    print(f'\n  최악 5시간대 KST (진입 회피 후보):')
    for h, s, n, wr in hour_pnl[:5]:
        print(f'    {h:>2}시  거래 {n:>3}  승률 {wr:>5.1f}%  PnL ${s:>+6.0f}')
    print(f'\n  최고 5시간대 KST:')
    for h, s, n, wr in hour_pnl[-5:][::-1]:
        print(f'    {h:>2}시  거래 {n:>3}  승률 {wr:>5.1f}%  PnL ${s:>+6.0f}')

    # 방향별 최악 시간대
    by_hour_dir = defaultdict(lambda: defaultdict(list))
    for t in trades:
        by_hour_dir[t['direction']][t['kst_hour']].append(t['pnl'])
    for d_ in ['LONG', 'SHORT']:
        pnl_list = [(h, sum(p), len(p), sum(1 for x in p if x>0)/max(len(p),1)*100)
                    for h, p in by_hour_dir[d_].items()]
        pnl_list.sort(key=lambda x: x[1])
        print(f'\n  {d_} 최악 3시간대:')
        for h, s, n, wr in pnl_list[:3]:
            print(f'    {h:>2}시  {n:>3}건 {wr:>5.1f}% ${s:>+6.0f}')

    # ==========  G. 요일 ==========
    print('\n' + '='*90)
    print('G. 요일 × 승률')
    print('='*90)
    dow_names = ['월','화','수','목','금','토','일']
    G = defaultdict(list)
    for t in trades:
        G[(t['kst_dow'], t['direction'])].append(t['pnl'])
    for dow in range(7):
        for d_ in ['LONG', 'SHORT']:
            pnls = G.get((dow, d_), [])
            if not pnls: continue
            n, wr, s, avg = stats(pnls)
            print(f'  {dow_names[dow]} {d_:<7} {n:>3}건 {wr:>5.1f}% ${s:>+6.0f}')
    results['G'] = {str(k): stats(v) for k,v in G.items()}

    # ==========  H. 연패 패턴 ==========
    print('\n' + '='*90)
    print('H. 연패 후 거래 승률 (ADX조정 / 멘탈 대용)')
    print('='*90)
    sorted_trades = sorted(trades, key=lambda t: t['timestamp'])
    streak_buckets = defaultdict(list)
    current_streak = 0
    for t in sorted_trades:
        # 이 거래 진입 시점의 연패 카운트는 지금까지의 연패
        bucket = min(current_streak, 5)
        streak_buckets[bucket].append(t['pnl'])
        # 업데이트
        if t['pnl'] < 0: current_streak += 1
        else: current_streak = 0
    print(f'  진입 시 연속 손실 수 → 이 거래 결과:')
    for k in sorted(streak_buckets.keys()):
        n, wr, s, avg = stats(streak_buckets[k])
        label = f'{k}회 연패 후' if k < 5 else '5+회 연패 후'
        print(f'    {label:<18}{n:<6}{wr:>5.1f}%  ${s:>+7.0f}  평균 ${avg:>+6.2f}')
    results['H'] = {str(k): stats(v) for k,v in streak_buckets.items()}

    # ==========  I. 최종 최적 조합 ==========
    print('\n' + '='*90)
    print('I. 최종 최적 조합 시나리오')
    print('='*90)
    def scenario_pnl(keep_fn, label):
        kept = [t for t in trades if keep_fn(t)]
        n, wr, s, avg = stats([t['pnl'] for t in kept])
        print(f'  {label:<50}{n:<6}{wr:>5.1f}%  ${s:>+9.0f}')
        return {'n':n, 'wr':wr, 'pnl':s, 'avg':avg}

    def ob_ok(t, thresh, direction=None):
        if direction and t['direction'] != direction:
            return True  # 이 방향 아니면 필터 안 적용
        br = t.get('bid_ratio')
        if br is None: return True
        if t['direction'] == 'LONG': return br >= thresh
        return (1 - br) >= thresh

    # 기준선
    scenarios = {}
    scenarios['0_OFF (베이스)'] = scenario_pnl(lambda t: True, '0. 전부 OFF (베이스)')
    scenarios['1_현재 (0.55 both)'] = scenario_pnl(
        lambda t: (ob_ok(t, 0.55, 'LONG') if t['direction']=='LONG' else ob_ok(t, 0.55, 'SHORT')),
        '1. 현재 (LONG/SHORT 모두 0.55)')
    scenarios['2_LONG0.55_SHORTno'] = scenario_pnl(
        lambda t: (ob_ok(t, 0.55, 'LONG') if t['direction']=='LONG' else True),
        '2. LONG 0.55 + SHORT 제거')
    scenarios['3_LONG0.45_SHORTno'] = scenario_pnl(
        lambda t: (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '3. LONG 0.45 + SHORT 제거')
    scenarios['4_19시차단_3추가'] = scenario_pnl(
        lambda t: t['kst_hour'] != 19 and (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '4. 3번 + 19시 차단')
    scenarios['5_저변동차단_4추가'] = scenario_pnl(
        lambda t: t['kst_hour'] != 19 and
                  (t.get('entry_atr_pct') is None or t.get('entry_atr_pct') >= 0.3) and
                  (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '5. 4번 + ATR 0.3% 미만 차단')

    # H에서 발견한 것: 연패 후 거래 성과 보고
    # 연패 3회 이상이 나쁘면 3회이상 연패 후 차단
    def in_high_streak(t, sorted_list, threshold):
        # 이 거래 전까지의 연패 계산
        idx = sorted_list.index(t)
        streak = 0
        for i in range(idx-1, -1, -1):
            if sorted_list[i]['pnl'] < 0: streak += 1
            else: break
        return streak >= threshold

    # 연패 3+ 이상 후 거래를 차단
    sorted_ = sorted(trades, key=lambda t: t['timestamp'])
    def filter_after_streak(t, thresh):
        # index
        idx = sorted_.index(t)
        streak = 0
        for i in range(idx-1, -1, -1):
            if sorted_[i]['pnl'] < 0: streak += 1
            else: break
        return streak < thresh

    # 시간이 오래 걸릴 수 있으니 캐시
    streak_at_idx = []
    current = 0
    for t in sorted_:
        streak_at_idx.append(current)
        if t['pnl'] < 0: current += 1
        else: current = 0

    def streak_filter(t, max_streak):
        idx = sorted_.index(t)
        return streak_at_idx[idx] < max_streak

    # 아래 비효율. trade에 index 추가.
    for i, t in enumerate(sorted_):
        t['_idx'] = i

    def streak_gt(t, n):
        return streak_at_idx[t['_idx']] >= n

    scenarios['6_연패3차단_5추가'] = scenario_pnl(
        lambda t: not streak_gt(t, 3) and t['kst_hour'] != 19 and
                  (t.get('entry_atr_pct') is None or t.get('entry_atr_pct') >= 0.3) and
                  (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '6. 5번 + 3연패 후 차단')

    results['I'] = scenarios

    # ==========  J. 월별 walk-forward for 핵심 시나리오 ==========
    print('\n' + '='*90)
    print('J. 월별 Walk-forward (핵심 4시나리오)')
    print('='*90)
    months = ['2025-08','2025-09','2025-10','2025-11','2025-12',
              '2026-01','2026-02','2026-03','2026-04']
    month_keep = {
        '1_현재': lambda t: (ob_ok(t, 0.55, 'LONG') if t['direction']=='LONG' else ob_ok(t, 0.55, 'SHORT')),
        '3_L0.45+SHORT제거': lambda t: (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '5_5번최적': lambda t: t['kst_hour'] != 19 and
                                 (t.get('entry_atr_pct') is None or t.get('entry_atr_pct') >= 0.3) and
                                 (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
        '6_6번최적': lambda t: not streak_gt(t, 3) and t['kst_hour'] != 19 and
                                 (t.get('entry_atr_pct') is None or t.get('entry_atr_pct') >= 0.3) and
                                 (ob_ok(t, 0.45, 'LONG') if t['direction']=='LONG' else True),
    }
    print(f'{"시나리오":<20}', end='')
    for m in months:
        print(f'{m[5:]:>9}', end='')
    print(f'{"합":>10}')
    print('-'*(20 + 9*len(months) + 10))
    for name, fn in month_keep.items():
        row = f'{name:<20}'
        total = 0
        monthly = []
        for m in months:
            ms_dt = datetime.strptime(f'{m}-01','%Y-%m-%d')
            ms = int(ms_dt.timestamp()*1000)
            # next month
            if m == '2026-04':
                me_dt = datetime.strptime('2026-04-22','%Y-%m-%d')
            else:
                y, mo = m.split('-'); mo = int(mo)+1; y = int(y)
                if mo > 12: mo=1; y+=1
                me_dt = datetime.strptime(f'{y}-{mo:02d}-01','%Y-%m-%d')
            me = int(me_dt.timestamp()*1000)
            m_trades = [t for t in sorted_ if ms <= t['timestamp'] < me]
            kept = [t for t in m_trades if fn(t)]
            pnl = sum(t['pnl'] for t in kept)
            total += pnl
            monthly.append(pnl)
            row += f'{pnl:>+9.0f}'
        row += f'{total:>+10.0f}'
        print(row)

    # Save
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n✓ Saved: {OUT}')


if __name__ == '__main__':
    main()
