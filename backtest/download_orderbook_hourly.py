#!/usr/bin/env python3
"""Bybit Linear Perp 오더북 1H 샘플링 (고속 버전).

메모리 효율: 델타 누적 없이 in-place 적용.
병렬: 코인별 ProcessPoolExecutor.
출력: 코인-일 조합별 JSON (시간당 bid_ratio).
"""
import os, sys, json, zipfile, io, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
URL_BASE = "https://quote-saver.bycsi.com/orderbook/linear"
OUT_DIR = Path("/Users/sue/Projects/HERMES/backtest/data/orderbook_hourly")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEPTH = 25  # 실전 엔진 기본값과 일치


def download_zip(symbol: str, date_str: str) -> Optional[bytes]:
    url = f"{URL_BASE}/{symbol}/{date_str}_{symbol}_ob200.data.zip"
    for _ in range(3):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"    retry ({e})", flush=True)
            time.sleep(2)
    return None


def compute_ratio_top25(bids: Dict[str, float], asks: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    """상위 25레벨 기준 bid_ratio + 요약 반환."""
    valid_bids = sorted(
        ((float(p), q) for p, q in bids.items() if q > 0),
        key=lambda x: -x[0]
    )[:DEPTH]
    valid_asks = sorted(
        ((float(p), q) for p, q in asks.items() if q > 0),
        key=lambda x: x[0]
    )[:DEPTH]
    bv = sum(q for _, q in valid_bids)
    av = sum(q for _, q in valid_asks)
    tot = bv + av
    ratio = bv / tot if tot > 0 else 0.5
    best_bid = valid_bids[0][0] if valid_bids else 0.0
    best_ask = valid_asks[0][0] if valid_asks else 0.0
    return ratio, bv, av, best_bid, best_ask


def process_zip_streaming(zip_bytes: bytes) -> List[Dict]:
    """ZIP을 열고 in-place로 오더북 업데이트하며 각 시간 경계 스냅샷 저장."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            return []
        raw = zf.read(names[0])

    text = raw.decode('utf-8', errors='ignore')
    del raw

    bids: Dict[str, float] = {}
    asks: Dict[str, float] = {}
    results: List[Dict] = []
    last_hour = None

    # Stream line-by-line from in-memory string without loading all
    pos = 0
    text_len = len(text)
    while pos < text_len:
        nl = text.find('\n', pos)
        if nl == -1:
            line = text[pos:]
            pos = text_len
        else:
            line = text[pos:nl]
            pos = nl + 1
        if not line:
            continue
        try:
            rec = json.loads(line)
        except:
            continue

        ts_ms = rec.get('ts', 0)
        rec_type = rec.get('type', '')
        data = rec.get('data', {})

        if rec_type == 'snapshot':
            # 새 snapshot → 전체 재구성
            bids = {p: float(q) for p, q in data.get('b', [])}
            asks = {p: float(q) for p, q in data.get('a', [])}
        else:
            # delta → in-place 업데이트
            for p, q in data.get('b', []):
                qf = float(q)
                if qf == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = qf
            for p, q in data.get('a', []):
                qf = float(q)
                if qf == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = qf

        # 시간 경계 확인
        if ts_ms == 0:
            continue
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        hr = dt.replace(minute=0, second=0, microsecond=0)
        if hr != last_hour:
            ratio, bv, av, bb, ba = compute_ratio_top25(bids, asks)
            results.append({
                'hour_ts_ms': int(hr.timestamp() * 1000),
                'hour_utc': hr.strftime('%Y-%m-%d %H:%M UTC'),
                'last_ts_ms': ts_ms,
                'bid_ratio': round(ratio, 4),
                'bid_vol_top25': round(bv, 3),
                'ask_vol_top25': round(av, 3),
                'best_bid': bb,
                'best_ask': ba,
            })
            last_hour = hr

    return results


def download_and_save(symbol: str, date_str: str) -> str:
    """한 코인-일 처리."""
    out_file = OUT_DIR / f"{symbol}_{date_str}_hourly.json"
    if out_file.exists():
        return f"{symbol} {date_str} (skip, exists)"
    t0 = time.time()
    zb = download_zip(symbol, date_str)
    if zb is None:
        return f"{symbol} {date_str} ❌ 404"
    dl_t = time.time() - t0
    t1 = time.time()
    recs = process_zip_streaming(zb)
    proc_t = time.time() - t1
    out_file.write_text(json.dumps(recs, indent=1))
    return (f"{symbol} {date_str} ✅ {len(recs)}snap "
            f"({len(zb)/1024/1024:.0f}MB, dl {dl_t:.0f}s, proc {proc_t:.0f}s)")


def main():
    # 기본값: 가능한 전체 기간 (2025-08-22 ~ 2026-04-21)
    # 오버라이드: 환경변수 OB_START, OB_END (YYYY-MM-DD)
    start_s = os.environ.get('OB_START', '2025-08-22')
    end_s = os.environ.get('OB_END', '2026-04-22')
    start = datetime.strptime(start_s, '%Y-%m-%d')
    end = datetime.strptime(end_s, '%Y-%m-%d')

    jobs = []
    day = start
    while day < end:
        date_str = day.strftime('%Y-%m-%d')
        for sym in SYMBOLS:
            jobs.append((sym, date_str))
        day += timedelta(days=1)

    print(f'구간: {start_s} ~ {end_s}', flush=True)
    print(f'총 작업: {len(jobs)}개 (코인-일). 병렬 실행...', flush=True)

    # 4 workers (코인 수 만큼) — CPU + 네트워크 병렬
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_and_save, s, d): (s, d) for s, d in jobs}
        done_n = 0
        for fut in as_completed(futures):
            msg = fut.result()
            done_n += 1
            if done_n % 10 == 0 or done_n == len(jobs):
                print(f'  [{done_n}/{len(jobs)}] {msg}', flush=True)

    print(f'\n✓ 완료. {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
