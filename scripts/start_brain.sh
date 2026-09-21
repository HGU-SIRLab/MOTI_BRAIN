#!/usr/bin/env bash
# 모티 뇌 전체 기동 — 재부팅 후 이거 하나만 실행하면 된다.
#
# 이미 떠 있는 것은 건드리지 않으므로 몇 번을 돌려도 안전하다. vLLM이 이미 돌고 있으면 그 29분을 다시 내지
# 않고 넘어간다. 자세한 설명은 docs/brain_startup.md.
set -u

ROOT=/home/herobot/moti_brain
cd "$ROOT" || exit 1
mkdir -p logs

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()  { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# --- 1. 전원 -----------------------------------------------------------------
# MAXN(nvpmodel -m 0)은 저장되는 설정이라 재부팅해도 유지된다. jetson_clocks는 유지되지
# 않으므로 벤치마크 전에는 다시 걸어야 한다(스펙 §16). 일상 대화에는 없어도 돈다 —
# 실제로 2026-09-21의 2.53초는 jetson_clocks 없이 나온 값이다.
say "1/5  전원 모드"
MODE=$(nvpmodel -q 2>/dev/null | grep -i "power mode" | cut -d: -f2- | xargs)
if [ "$MODE" = "MAXN" ]; then ok "MAXN"; else
  no "MAXN이 아니다 ($MODE) — 'sudo nvpmodel -m 0' 후 재부팅"
fi

# --- 2. vLLM -----------------------------------------------------------------
say "2/5  vLLM (google/gemma-4-E4B-it)"
if curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
  ok "이미 응답 중 — 건드리지 않는다"
else
  if docker ps -a --format '{{.Names}}' | grep -qx moti_vllm; then
    echo "  컨테이너는 있으나 응답이 없다. 다시 만든다."
  fi
  printf "  기동한다. \033[33m약 29분\033[0m 걸린다 (대부분 'model loading' 구간, 원인 미규명=Q17).\n"
  bash scripts/run_vllm.sh >/dev/null || { no "run_vllm.sh 실패"; exit 1; }
  printf "  대기 중"
  START=$(date +%s)
  until curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx moti_vllm; then
      echo; no "컨테이너가 죽었다. 로그:"; docker logs moti_vllm 2>&1 | tail -30; exit 1
    fi
    printf "."; sleep 20
  done
  echo; ok "준비됨 ($(( ($(date +%s) - START) / 60 ))분 소요)"
fi

# --- 3. 뇌 서버 --------------------------------------------------------------
# PYTHONPATH= 가 반드시 붙어야 한다. ROS2/HARU setup이 전역 PYTHONPATH에 HARU venv를
# 넣어두어서, 빼면 젯슨 torch와 충돌하는 PyPI 패키지가 보인다(트러블슈팅 메모).
# pkill은 쓰지 않는다 — 패턴이 자기 셸 명령줄에 걸려 스크립트째로 죽는다(exit 144).
say "3/5  뇌 서버 (:8765)"
PID=$(pgrep -f "venv_tts/bin/python [b]rain/server.py" || true)
if [ -n "$PID" ]; then
  ok "이미 떠 있다 (PID $PID)"
else
  PYTHONPATH= nohup .venv_tts/bin/python brain/server.py >> logs/brain.log 2>&1 &
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ':8765' && break
    sleep 1
  done
  if ss -ltn 2>/dev/null | grep -q ':8765'; then
    ok "기동됨 (PID $(pgrep -f 'venv_tts/bin/python [b]rain/server.py'))  로그: logs/brain.log"
  else
    no "8765가 안 열렸다. logs/brain.log 확인"; tail -20 logs/brain.log; exit 1
  fi
fi

# --- 4. 예열 -----------------------------------------------------------------
say "4/5  페르소나 예열"
PYTHONPATH= .venv_tts/bin/python scripts/prewarm.py "$@" || no "예열 실패 (치명적이지 않음 — 첫 인사가 느려질 뿐)"

# --- 5. 확인 -----------------------------------------------------------------
say "5/5  최종 확인"
curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null && ok "vLLM :8000" || no "vLLM"
ss -ltn 2>/dev/null | grep -q ':8765' && ok "뇌 서버 :8765" || no "뇌 서버"
for ip in $(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[\d.]+(?=/)' | grep -Ev '^(127|172)\.'); do
  echo "  로봇에서 접속할 주소:  ws://$ip:8765"
done

say "준비 완료"
echo "  로그 보기:  tail -f $ROOT/logs/brain.log"
echo "  스모크:     PYTHONPATH= .venv_tts/bin/python client/test_local_live.py"
