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
#
# --mm-processor-cache-gb 0 disables vLLM's multimodal processor cache. With it on, the
# engine crashed mid-run with `AssertionError: Expected a cached item for mm_hash=...`
# and returned HTTP 500. The trigger is our own pattern: the transcription call and the
# reply call go out concurrently carrying the *same* audio, so two in-flight requests
# share an mm_hash and race in that cache. Turning it off costs re-deriving audio
# features twice per turn, which is cheap — prefix caching, which handles the expensive
# part, is separate and unaffected.
#
# EXP-3 (speculative decoding) is BLOCKED by this container — see §13.6. Both paths fail
# at startup, so neither flag is here:
#   draft_model : transformers in this image does not know model type `gemma4_assistant`,
#                 which is exactly the drafter the spec names (§5.4 rule 5)
#   ngram       : `ModuleNotFoundError: No module named 'numba'`
# Reviving it needs a rebuilt image, not a config change.
#
# If that ever happens: do *not* also disable multimodal, which the recipe suggests.
# Audio input is the primary path since EXP-13 (§12.4); turning it off breaks the input.
docker run -d --name "$NAME" --runtime=nvidia --network host \
  -v /home/herobot/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/root/.cache/vllm/torch_compile_cache \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.40 \
    --mm-processor-cache-gb 0 \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4

# 기동에 ~33분 걸린다. 내역(실측): 가중치 읽기 3.8s, torch.compile 85s,
# 엔진 init 139s, 그리고 "model loading" 단계에서 설명 안 되는 1674s(28분). 원인 미규명(Q17).
# 캐시는 설정별로 키가 갈리므로 플래그를 바꾸면 컴파일분(85s)만 다시 낸다.
# 실질적 대응: 서버를 재시작하지 않는다 — 뇌는 원래 상시 가동 서버다(§1).
echo "기동 중 (~33분 소요, 대부분 model loading 단계)."
echo "준비 확인:  curl -sf http://localhost:8000/v1/models"
echo "로그:       docker logs -f $NAME"
