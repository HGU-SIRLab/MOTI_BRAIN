#!/usr/bin/env bash
# 검사 12종을 한 번에. 뇌 서버와 vLLM이 떠 있어야 한다.
#
# 이 파일이 저장소에 있는 이유: 그동안 세션마다 임시 디렉토리에 같은 걸 다시 만들어 쓰고
# 있었고, 세션이 끝나면 사라졌다. 검사 목록 자체가 프로젝트 자산이라 여기 둔다.
#
# 사용:  bash scripts/check_all.sh [--exp8]
#        --exp8 을 붙이면 실제 페르소나로 지연까지 측정한다 (몇 분 더 걸린다)
set -u
cd "$(dirname "$0")/.." || exit 1

fail=0
run() {
  printf '\n\033[1m=== %s ===\033[0m\n' "$1"
  if timeout "${3:-900}" env PYTHONPATH= .venv_tts/bin/python "$2" 2>&1 \
       | grep -vE "onnxruntime|device_discovery"; then :
  else fail=$((fail+1)); printf '\033[31m!! %s 실패\033[0m\n' "$1"; fi
}

printf '\033[1m사전 조건\033[0m\n'
curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null \
  && echo "  vLLM   OK" || { echo "  vLLM   응답 없음 — bash scripts/start_brain.sh"; exit 1; }
ss -ltn 2>/dev/null | grep -q ':8765' \
  && echo "  뇌서버 OK" || { echo "  뇌서버 없음 — bash scripts/start_brain.sh"; exit 1; }

# 서버가 필요 없는 것부터 — 빠르게 실패하는 순서
run "툴 스키마 (서버 불필요)"          client/test_tool_schemas.py 120
run "shim 에러 경로 (서버 불필요)"     client/test_error_path.py   120
run "문장분할·툴파싱·파이프라인"        brain/test_pipeline.py
run "턴 감지 (분할 예산)"              brain/test_vad.py
run "smart-turn 분리도"                brain/test_smart_turn.py
run "게이트: 이모지 + <SILENT>"        brain/test_gates.py
run "주입 턴이 사용자 발화로 안 돌아오나" client/test_injected_turn.py
run "launcher 수신 루프 재현"           client/test_local_live.py
run "재연결 후 대화 복구"               client/test_reconnect.py
run "재생 중 끼어들기"                  client/test_barge_in_playing.py 400
run "툴만 부른 턴: 순서·후속 발화"      client/test_tool_turn_order.py 300
run "퀴즈 여러 문항: 툴 히스토리·침묵"   client/test_quiz_flow.py 300

if [ "${1:-}" = "--exp8" ]; then
  printf '\n\033[1m=== EXP-8 지연 (실제 페르소나 18,344토큰 + 툴) ===\033[0m\n'
  timeout 900 env PYTHONPATH= .venv_tts/bin/python client/exp8_latency.py --real 8 \
    || { fail=$((fail+1)); echo "!! EXP-8 실패"; }
  echo "  ⚠️ 이 지표는 같은 조건에서도 2.5~4.7초로 갈린다 (§13.9). 점이 아니라 범위로 읽을 것"
fi

printf '\n======================================\n'
if [ "$fail" -eq 0 ]; then printf '\033[32m전부 통과 (12종)\033[0m\n'
else printf '\033[31m실패한 항목: %d\033[0m\n' "$fail"; fi
exit "$fail"
