#!/usr/bin/env python3
"""EXP-2: TTFT and decode speed against the local vLLM server (spec §12).

Streaming only (§20 rule 3). Reports mean and p95 (§12 EXP-2).
Usage: python3 scripts/bench_llm.py [--runs N] [--warmup]
"""
import argparse
import json
import statistics
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"

# 공감 대화 도메인 (§1). 실제 사용자가 로봇에게 할 법한 발화.
PROMPTS = [
    "요즘 잠이 잘 안 와서 너무 피곤해",
    "오늘 시험 진짜 잘 봤어!",
    "친구랑 좀 다퉜는데 마음이 안 좋아",
    "발표 앞두고 너무 떨려",
    "아무것도 하기 싫은 날이야",
    "과제가 너무 많아서 숨이 막혀",
    "오랜만에 가족이랑 밥 먹었어",
    "요즘 내가 뭘 하고 싶은지 모르겠어",
    "칭찬 받아서 기분이 좋아",
    "괜찮아, 별일 아니야",
]

SHORT_SYSTEM = (
    "너는 공감 로봇 모티야. 사용자의 말에 따뜻하게 공감하며 대화해. "
    "답변은 2~3문장으로 짧게 해."  # §5.4 rule 3 — verbosity is latency
)

MOTI_HRI = "/home/herobot/MOTI-HRI"


def real_persona() -> str:
    """The persona launcher.py actually sends — 18k tokens. A short stand-in
    prompt makes TTFT look ~4x better than production, so measure with this."""
    import sys
    sys.path.insert(0, MOTI_HRI)
    from core.utils import build_persona_system_instruction
    return build_persona_system_instruction(
        name="형민", facts_summary="- 전공: 전산전자공학부\n- 학년: 4학년")


def one(prompt: str, system: str) -> tuple[float, float, int]:
    """Returns (ttft_sec, total_sec, completion_tokens)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
        "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                completion_tokens = chunk["usage"]["completion_tokens"]
            choices = chunk.get("choices") or []
            if ttft is None and choices and choices[0]["delta"].get("content"):
                ttft = time.perf_counter() - start
    return ttft, time.perf_counter() - start, completion_tokens


def report(name: str, values: list[float], unit: str) -> None:
    ordered = sorted(values)
    p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    print(f"  {name:<16} mean {statistics.mean(values):6.3f} {unit}   "
          f"p95 {p95:6.3f} {unit}   min {ordered[0]:6.3f}   max {ordered[-1]:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=len(PROMPTS))
    ap.add_argument("--short-prompt", action="store_true",
                    help="실제 페르소나 대신 짧은 대역 프롬프트 사용 (비교용)")
    args = ap.parse_args()

    system = SHORT_SYSTEM if args.short_prompt else real_persona()
    print(f"시스템 프롬프트: {len(system):,}자 "
          f"({'짧은 대역' if args.short_prompt else '실제 MOTI 페르소나'})\n")

    ttfts, rates = [], []
    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        ttft, total, tokens = one(prompt, system)
        decode = total - ttft
        rate = (tokens - 1) / decode if decode > 0 and tokens > 1 else 0.0
        ttfts.append(ttft)
        rates.append(rate)
        tag = " <- cold (프리필 전량)" if i == 0 else ""
        print(f"[{i+1:2d}] ttft {ttft:5.3f}s  total {total:5.2f}s  "
              f"{tokens:3d} tok  {rate:5.2f} tok/s  | {prompt}{tag}")

    print(f"\ncold TTFT (1회차, prefix cache 미적중): {ttfts[0]:.3f}s")
    print("warm 구간 (2회차 이후, 페르소나 캐시 적중):")
    report("TTFT", ttfts[1:], "s")
    report("decode", rates[1:], "tok/s")


if __name__ == "__main__":
    main()
