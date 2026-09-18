#!/usr/bin/env bash
# Gemma 4 E4B via vLLM — OpenAI-compatible server on :8000 (spec §5.3, Stage 1)
set -e

NAME=moti_vllm
MODEL=google/gemma-4-E4B-it
IMAGE=ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin
CACHE=/home/herobot/moti_brain/.vllm_compile_cache

mkdir -p "$CACHE"
docker rm -f "$NAME" 2>/dev/null || true

# 0.40: Jetson has unified CPU/GPU memory, so vLLM's 0.9 default starves the system —
# and this box must also host TTS alongside the LLM (§1 distributed brain).
# 32768: MOTI's real persona (build_persona_system_instruction) is 18,344 tokens on its
# own, so 16384 cannot serve a single request. Leaves room for audio (750/30s) + history.
docker run -d --name "$NAME" --runtime=nvidia --network host \
  -v /home/herobot/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/root/.cache/vllm/torch_compile_cache \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.40 \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4

echo "기동 중. 첫 실행은 torch.compile 때문에 오래 걸리고, 이후는 위 캐시를 재사용한다."
echo "준비 확인:  curl -sf http://localhost:8000/v1/models"
echo "로그:       docker logs -f $NAME"
