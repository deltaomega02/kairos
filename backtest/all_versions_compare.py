#!/usr/bin/env python3
"""모든 V7~V12 버전을 여러 기간에 돌려서 과적합 감별.

기간:
  A. 6년 (2020-05 ~ 2026-04, 전체)
  B. 8개월 (2025-08-22 ~ 2026-04-20, 오더북 보유)
  C. 2주 (2026-04-09 ~ 2026-04-22, 실전)

과적합 감별:
  - 각 버전 연도별 walk-forward
  - 각 버전 최악 연도
  - 랭킹 일관성 (시기 바뀌어도 순위 유지?)
  - V12의 특별한 우위가 실재인지 노이즈인지
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
OUT = Path('~/Projects/HERMES_백테스팅/v12/all_versions_compare.json')
OUT.parent.mkdir(parents=True, exist_ok=True)


# ======== 버전 정의 ========
VERSIONS = {
    'V7': {
        'params': {**DEFAULT_PARAMS, 'ema_fast':5, 'ema_slow':18, 'sl_atr_mult':1.5,
                    'tp_rr_ratio':6.0, 'entry_score_threshold':40,
                    'pullback_ema_dist_pct':1.5, 'adx_enter_trending':30},
        'kw': {},
        'desc': 'EMA 5/18, 1D 필터 없음'
    },
    'V8': {
        'params': {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
                    'tp_rr_ratio':6.0, 'entry_score_threshold':40,
                    'pullback_ema_dist_pct':1.5, 'adx_enter_trending':30},
        'kw': {},
        'desc': 'V7 + EMA 3/15 교체'
    },
    'V9': {
        'params': {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
                    'tp_rr_ratio':6.0, 'entry_score_threshold':40,
                    'pullback_ema_dist_pct':1.5, 'adx_enter_trending':30},
        'kw': {'d1_filter_enable':True, 'd1_ema_period':10, 'd1_mode':'direction'},
        'desc': 'V8 + 1D EMA10 direction 필터'
    },
    'V11': {
        'params': {**DEFAULT_PARAMS, 'ema_fast':3, 'ema_slow':15, 'sl_atr_mult':1.5,
                    'tp_rr_ratio':6.0, 'entry_score_threshold':40,
                    'pullback_ema_dist_pct':1.5, 'adx_enter_trending':30},
        'kw': {'d1_filter_enable':True, 'd1_ema_period':2, 'd1_mode':'price_above_ema'},
        'desc': 'V8 + 1D EMA2 price_above_ema 필터'
    },
}


# V12 post-filter (V11 trades에 5개 추가 필터 적용)
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


def v12_filter(v11_trades, ob_lookup):
    """V12 = V11 + 5 filters:
    1. SHORT 오더북 스킵
    2. LONG 오더북 0.45 (데이터 있을 때만)
    3. 19시 KST 차단
    4. ATR < 0.3% 차단
    5. 3연패 후 차단
    """
    sorted_trades = sorted(v11_trades, key=lambda t: t['timestamp'])
    # 연패 계산
    streak = []
    current = 0
    for t in sorted_trades:
        streak.append(current)
        if t['pnl'] < 0: current += 1
        else: current = 0

    kept = []
    for i, t in enumerate(sorted_trades):
        # 연패 3+ 차단
        if streak[i] >= 3:
            continue
        # 시간 차단
        dt_kst = datetime.utcfromtimestamp(t['timestamp']/1000 + 9*3600)
        if dt_kst.hour == 19:
            continue
        # ATR 차단
        atr = t.get('entry_atr_pct')
        if atr is not None and atr < 0.3:
            continue
        # 오더북 필터
        if t['direction'] == 'LONG':
            br = ob_lookup.get((t['symbol'], nearest_hourly(t['timestamp'])))
            if br is not None and br < 0.45:
                continue  # 데이터 있고 통과 못하면 차단
            # 데이터 없으면 통과 (pre-2025-08-22)
        # SHORT는 오더북 무시
        kept.append(t)
    return kept


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


def stats(trades):
    n = len(trades)
    if n == 0: return {'n':0, 'wr':0, 'pnl':0, 'avg':0, 'wins':0}
    w = sum(1 for t in trades if t['pnl']>0)
    p = sum(t['pnl'] for t in trades)
    return {'n':n, 'wr':w/n*100, 'pnl':p, 'avg':p/n, 'wins':w}


def year_breakdown(trades):
    years = defaultdict(list)
    for t in trades:
        y = datetime.utcfromtimestamp(t['timestamp']/1000).year
        years[y].append(t['pnl'])
    return {y: {'n':len(p), 'pnl':sum(p), 'wr':sum(1 for x in p if x>0)/max(len(p),1)*100}
            for y, p in sorted(years.items())}


def main():
    print('='*100)
    print('전 버전 × 전 기간 비교 (과적합 감별)')
    print('='*100)

    ob_lookup = load_ob()
    print(f'오더북 룩업: {len(ob_lookup)}개 스냅샷 (2025-08-22 ~ 2026-04-20)')

    # 데이터 로드 (warmup 포함)
    data = _load_data()

    base_args = dict(trailing_activation=1.2, trailing_distance=0.1,
                     skip_years=(), daily_cost_usd=DAILY_COST, slippage_pct=0.05,
                     max_simultaneous=3, risk_per_trade=0.015, max_leverage=7,
                     enabled_symbols=SYMBOLS, block_sol_long=True)

    # ===== 각 버전 6년 실행 =====
    all_results = {}
    print('\n[6년 전체 백테스트 실행]')
    d_6yr = filter_d(data, '2020-03-01', '2026-04-22')

    for vname, vcfg in VERSIONS.items():
        print(f'  {vname} 실행... ', end='', flush=True)
        r = run_shared_backtest_v11(d_6yr, vcfg['params'], SEED, **base_args, **vcfg['kw'])
        trades = r.get('_trades', [])
        all_results[vname] = trades
        print(f'{len(trades)}건')

    # V12 = V11 + post-filter
    v12_trades = v12_filter(all_results['V11'], ob_lookup)
    all_results['V12'] = v12_trades
    print(f'  V12 (V11+post-filter): {len(v12_trades)}건')

    # ===== 기간별 비교 =====
    windows = {
        '6년 전체 (2020-05 ~ 2026-04)': ('2020-05-01', '2026-04-22'),
        '8개월 (오더북 구간)': ('2025-08-22', '2026-04-22'),
        '2주 실전': ('2026-04-09', '2026-04-22'),
    }

    summary = {}
    for win_name, (s, e) in windows.items():
        s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
        e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
        print(f'\n{"="*100}')
        print(f'{win_name}: {s} ~ {e}')
        print('='*100)
        print(f'{"버전":<6}{"거래":>6}{"승률":>10}{"PnL":>14}{"평균":>10}{"최악年/최고年":>22}')
        print('-'*80)
        for vname in ['V7','V8','V9','V11','V12']:
            trades_v = [t for t in all_results[vname] if s_ts <= t['timestamp'] < e_ts]
            st = stats(trades_v)
            # 연도별
            yb = year_breakdown(trades_v)
            worst_y = min(yb.items(), key=lambda x: x[1]['pnl']) if yb else (None, {'pnl':0})
            best_y = max(yb.items(), key=lambda x: x[1]['pnl']) if yb else (None, {'pnl':0})
            print(f'  {vname:<4}{st["n"]:>6}  {st["wr"]:>5.1f}%  ${st["pnl"]:>+10.0f}  ${st["avg"]:>+6.2f}  '
                  f'{worst_y[0]} ${worst_y[1]["pnl"]:>+5.0f} / {best_y[0]} ${best_y[1]["pnl"]:>+5.0f}')
            summary[(win_name, vname)] = st
            summary[(win_name + '_yearly', vname)] = yb

    # ===== 연도별 walk-forward =====
    print(f'\n{"="*100}')
    print('연도별 Walk-forward (과적합 감별 핵심)')
    print('='*100)
    years = range(2020, 2027)
    print(f'{"버전":<6}', end='')
    for y in years:
        print(f'{y:>9}', end='')
    print(f'{"총합":>12}{"양수年":>8}')
    print('-'*100)
    for vname in ['V7','V8','V9','V11','V12']:
        yb = year_breakdown(all_results[vname])
        print(f'  {vname:<4}', end='')
        pos_years = 0
        total = 0
        for y in years:
            pnl = yb.get(y, {'pnl':0})['pnl']
            total += pnl
            if pnl > 0: pos_years += 1
            color = '+' if pnl > 0 else ''
            print(f'{color}{pnl:>8.0f}', end='')
        print(f'  ${total:>+9.0f}  {pos_years}/{sum(1 for y in years if yb.get(y))}')

    # ===== 랭킹 일관성 =====
    print(f'\n{"="*100}')
    print('기간별 랭킹 (1위=최고 PnL)')
    print('='*100)
    print(f'{"기간":<40}{"1위":>8}{"2위":>8}{"3위":>8}{"4위":>8}{"5위":>8}')
    for win_name, (s, e) in windows.items():
        s_ts = int(datetime.strptime(s, '%Y-%m-%d').timestamp()*1000)
        e_ts = int(datetime.strptime(e, '%Y-%m-%d').timestamp()*1000)
        rankings = []
        for vname in ['V7','V8','V9','V11','V12']:
            trades_v = [t for t in all_results[vname] if s_ts <= t['timestamp'] < e_ts]
            pnl = sum(t['pnl'] for t in trades_v)
            rankings.append((vname, pnl))
        rankings.sort(key=lambda x: -x[1])
        row = f'  {win_name:<38}'
        for vname, _ in rankings:
            row += f'{vname:>8}'
        print(row)

    # ===== 실전 비교 =====
    print(f'\n{"="*100}')
    print('실전 기록 (31거래 + V12 3거래 = 34거래) 기반 비교')
    print('='*100)
    # 실전 수치
    actual = {'n': 34, 'wins': 15, 'wr': 44.1, 'pnl': -34.66}
    print(f'  실전 실제:      {actual["n"]}건  {actual["wr"]:.1f}%  PnL ${actual["pnl"]:+.2f}  (-₩50,839)')

    s_ts = int(datetime.strptime('2026-04-09', '%Y-%m-%d').timestamp()*1000)
    e_ts = int(datetime.strptime('2026-04-22', '%Y-%m-%d').timestamp()*1000)
    for vname in ['V7','V8','V9','V11','V12']:
        trades_v = [t for t in all_results[vname] if s_ts <= t['timestamp'] < e_ts]
        st = stats(trades_v)
        gap = st['pnl'] - actual['pnl']
        print(f'  {vname} 백테 (2주):  {st["n"]:>3}건  {st["wr"]:>5.1f}%  PnL ${st["pnl"]:>+7.2f}  '
              f'실전 대비 ${gap:>+7.2f}')

    # ===== 변동성 (과적합 지표) =====
    print(f'\n{"="*100}')
    print('변동성 분석 (연별 PnL 표준편차 → 낮을수록 안정)')
    print('='*100)
    for vname in ['V7','V8','V9','V11','V12']:
        yb = year_breakdown(all_results[vname])
        pnls = [y['pnl'] for y in yb.values()]
        if pnls:
            mean = np.mean(pnls)
            std = np.std(pnls)
            cv = std / abs(mean) if mean != 0 else 0
            worst = min(pnls)
            best = max(pnls)
            print(f'  {vname}: 평균 ${mean:>+7.0f}  표준편차 ${std:>6.0f}  CV {cv:.2f}  '
                  f'최악 ${worst:>+6.0f}  최고 ${best:>+6.0f}')

    # ===== 최종 판정 =====
    print(f'\n{"="*100}')
    print('🎯 과적합 진단')
    print('='*100)
    v11_6yr = sum(t['pnl'] for t in all_results['V11'] if int(datetime.strptime('2020-05-01','%Y-%m-%d').timestamp()*1000) <= t['timestamp'])
    v12_6yr = sum(t['pnl'] for t in all_results['V12'] if int(datetime.strptime('2020-05-01','%Y-%m-%d').timestamp()*1000) <= t['timestamp'])
    v11_8mo = sum(t['pnl'] for t in all_results['V11'] if int(datetime.strptime('2025-08-22','%Y-%m-%d').timestamp()*1000) <= t['timestamp'])
    v12_8mo = sum(t['pnl'] for t in all_results['V12'] if int(datetime.strptime('2025-08-22','%Y-%m-%d').timestamp()*1000) <= t['timestamp'])

    ratio_6yr = (v12_6yr / v11_6yr - 1) * 100 if v11_6yr > 0 else 0
    ratio_8mo = (v12_8mo / v11_8mo - 1) * 100 if v11_8mo > 0 else 0
    print(f'  V12 vs V11 개선율:')
    print(f'    8개월 (튜닝 구간): +{ratio_8mo:.1f}%')
    print(f'    6년 (out-of-sample): +{ratio_6yr:.1f}%')
    diff = ratio_8mo - ratio_6yr
    if abs(diff) < 20:
        verdict = '✅ 양호 (8mo와 6yr 개선율 일관)'
    elif diff > 40:
        verdict = '🚨 과적합 의심 (8mo에서만 급등)'
    elif diff > 20:
        verdict = '⚠️ 약한 과적합 신호'
    else:
        verdict = '✅ 오히려 out-of-sample 더 좋음'
    print(f'  차이 {diff:+.1f}%p → {verdict}')

    # Save
    out = {
        'ran_at': datetime.now().isoformat(),
        'summary': {str(k): v for k, v in summary.items()},
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n✓ Saved: {OUT}')


if __name__ == '__main__':
    main()
