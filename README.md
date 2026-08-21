# KAIROS

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white) ![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)

HERMES 운영에서 얻은 학습을 검증하기 위한 단순화 실험 시스템. (아카이브)

같은 거래소(Bybit USDT 선물)에서 자금·텔레그램·DB를 완전히 분리해 운영하도록 설계했다.
핵심 가설: "지표와 필터를 더하는 것이 아니라 빼는 것이 성능을 개선한다."

## 배경

HERMES 초기 운영에서 두 가지 문제를 관찰했다.

1. 짧은 기간에 파라미터를 5회 변경 — 표본이 쌓이기 전에 시스템을 흔드는 안티패턴
2. 실전 승률이 백테스트 대비 크게 낮음 (R:R은 백테스트와 유사) — 복잡한 진입 조건이 과적합일 가능성

KAIROS는 이 관찰을 바탕으로 전략을 의도적으로 단순화하고, 운영 규칙을 강제하는 실험이었다.

## HERMES와의 설계 차이

| 항목 | HERMES | KAIROS |
|---|---|---|
| 전략 | 4H 추세 풀백 + EMA + RSI + ADX + 오더북 + 펀딩 | 변동성 돌파 후보로 단순화 |
| 시간봉 | 4H + 1H + 1D | 일봉 단일 후보 |
| 코인 | BTC/ETH/SOL/XRP | BTC 단일 후보 |
| 파라미터 변경 | 운영 중 수시 변경 (안티패턴) | 50거래까지 변경 금지 룰 |
| 텔레그램 알림 | 모든 이벤트 | 체결·셧다운·에러만 |

## 운영 원칙

1. 50거래까지 동일 파라미터 유지, 이후 Walk-Forward 자동 최적화 — **현재 비활성**(`backtest/optimizer.py:20` `MIN_TRADES = 99999`. v8 실전 검증 동안 껐고 되돌리지 않았다)
2. 지표 추가 충동 차단 — 단순화 우선
3. 추가 입금 금지, 드로다운 셧다운 — **현재 비활성**(`config/settings.py:77` `MAX_DRAWDOWN_PCT = 1.00`). 설계에는 있으나 코드에서 꺼져 있다
4. 백테스트 절대값은 무시하고 상대 비교만 사용
5. 알림 노이즈 최소화

## 구조

```
KAIROS/
├── main.py                  # 메인 루프
├── config/                  # 설정
├── core/                    # 전략 엔진
├── exchange/                # Bybit API
├── database/                # SQLite
├── backtest/                # 백테스트 (HERMES 엔진 재활용 + 자체 전략 비교)
└── utils/                   # 텔레그램
```

`backtest/`에는 HERMES 공용 엔진 외에 KAIROS 자체 실험(변동성 돌파, 모멘텀 비교 등) 스크립트가 추가되어 있다.

## 이 실험이 남긴 것

"복잡도가 성능을 보장하지 않는다"는 가설은 이후 세대(ATHENA) 설계에 반영됐다 — 전략 단순화, 버전 동결 룰, 알림 최소화는 모두 KAIROS에서 정립된 원칙이다.

## 면책

연구·학습 목적의 개인 프로젝트입니다. 암호화폐 선물은 고위험 상품입니다.
