#!/bin/bash
# KAIROS 설치 스크립트
# 사용법: 파일 전부 홈 디렉터리에 업로드 후 실행
# chmod +x setup.sh && ./setup.sh

set -e

echo "=== KAIROS 설치 시작 ==="

# 디렉터리 생성
echo "[1/5] 디렉터리 생성..."
mkdir -p ~/kairos/{config,core,exchange,database,backtest,utils,logs,docs}

# 파일 이동
echo "[2/5] 파일 이동..."

mv ~/main.py ~/kairos/
mv ~/requirements.txt ~/kairos/
mv ~/README.md ~/kairos/ 2>/dev/null || true
mv ~/CLAUDE.md ~/kairos/ 2>/dev/null || true

mv ~/config__init__.py ~/kairos/config/__init__.py
mv ~/settings.py ~/kairos/config/settings.py
mv ~/parameters.py ~/kairos/config/parameters.py
mv ~/tunable_params.json ~/kairos/config/tunable_params.json
mv ~/logging_config.py ~/kairos/config/logging_config.py

mv ~/core__init__.py ~/kairos/core/__init__.py
mv ~/breakout_strategy.py ~/kairos/core/breakout_strategy.py 2>/dev/null || true
mv ~/risk_manager.py ~/kairos/core/risk_manager.py
mv ~/position_manager.py ~/kairos/core/position_manager.py
mv ~/technical_analysis.py ~/kairos/core/technical_analysis.py

mv ~/exchange__init__.py ~/kairos/exchange/__init__.py
mv ~/bybit_client.py ~/kairos/exchange/bybit_client.py
mv ~/bybit_websocket.py ~/kairos/exchange/bybit_websocket.py 2>/dev/null || true

mv ~/database__init__.py ~/kairos/database/__init__.py
mv ~/db_manager.py ~/kairos/database/db_manager.py
mv ~/schema.sql ~/kairos/database/schema.sql

mv ~/utils__init__.py ~/kairos/utils/__init__.py
mv ~/telegram_bot.py ~/kairos/utils/telegram_bot.py

# .env (HERMES와 분리, 새 텔레그램 봇 토큰 필수)
echo "[3/5] 환경변수 설정..."
echo "  ⚠️ KAIROS는 HERMES와 별도 텔레그램 봇 권장."
echo "  ⚠️ ~/kairos/.env 직접 작성 필요:"
echo "      BYBIT_API_KEY=..."
echo "      BYBIT_SECRET=..."
echo "      BYBIT_USE_TESTNET=false"
echo "      TELEGRAM_BOT_TOKEN=...  (새 봇)"
echo "      TELEGRAM_CHAT_ID=..."

# 가상환경
echo "[4/5] 가상환경 생성..."
cd ~/kairos
python3 -m venv kairos
source kairos/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  가상환경: ~/kairos/kairos/"

# 완료
echo "[5/5] 설치 완료!"
echo ""
echo "=== 실행 방법 (HERMES와 충돌 회피) ==="
echo "cd ~/kairos && source kairos/bin/activate"
echo "nohup python3 -u main.py > ./logs/kairos.out 2>&1 &"
echo "tail -f ./logs/kairos.out"
echo ""
echo "⚠️ HERMES와 같은 서버에서 가동 시:"
echo "   pkill -f \"python3 -u main.py\" 사용 금지 (둘 다 죽음)"
echo "   대신 PID 직접 관리 또는 systemd 서비스 분리"
