# Moti Robot — Local Voice AI Implementation Spec (v6)
### Target: Gemini Live-class conversational experience, fully local
### Hardware: Jetson AGX Orin 64GB (brain) + Jetson Orin Nano Super 8GB (robot) · Ubuntu 22.04 LTS · JetPack 6.x (L4T r36.x)
### Revised: 2026-09-18 — supersedes all prior versions

> **For Claude Code**: This is the authoritative build spec. Read fully before writing code.
> Sections marked `[VERIFY-FIRST]` gate dependent work.

---

## 0-A. WHAT CHANGED IN v6 (2026-09-18, user-approved)

Six decisions, all made after reading the existing robot codebase (`HGU-SIRLab/MOTI-HRI`, branch `jetson-moti`).

| # | Change | Where | Why |
|---|---|---|---|
| 1 | **Distributed topology confirmed; standalone MVP phase dropped** | §1, §4 | The AGX has **no microphone** (verified: only APE/ADMAIF virtual routing endpoints, playback is HDMI-only). The robot already owns mic/speaker/AEC and is hardware-validated. A standalone AGX phase would mean building throwaway audio I/O. |
| 2 | **Pipecat removed; direct asyncio assembly** | §11, §20 | In a distributed split, Pipecat requires a custom transport, which re-implements by hand most of the wiring it was chosen for. Reduces the aarch64 install surface to `websockets` (already a google-genai transitive dep). Silero VAD and smart-turn-v3 are independent ONNX packages, usable without Pipecat. |
| 3 | **EXP-13 added: Gemma 4 E4B audio input as STT (+SER) replacement** | §12, §6, §7 | E4B accepts audio natively (`audio_config` + `audio_token_id` present in its config.json). If it holds, the two highest/mid risks (ctranslate2 aarch64 source build; unverified Korean SER) both disappear and two models are deleted. Approved as a time-boxed gamble with 4 kill criteria. |
| 4 | **Research/academic use confirmed → GPL is not a blocker; Piper reinstated** | §8.2, §15 | Piper was demoted on GPL-3.0 source-disclosure grounds. This project is research/thesis-only with no public or commercial distribution, so that constraint does not apply. Piper is the **most Jetson-proven** TTS available, which materially reduces the project's #1 risk. |
| 5 | **AEC status corrected** | §9.2, §18 | The 30–36dB validation in the robot repo was done **on Windows**. On Jetson the `aec-audio-processing` wheel does not exist and the repo documents a PulseAudio `module-echo-cancel` fallback configured on a Xavier board whose work was later reverted. **User states AEC is already solved on the Orin Nano** — the robot repo's docs are stale on this point and should be updated there. |
| 6 | **No schedule; no paper currently planned** | §12, §17 | Steady incremental progress is preferred over the Day1–Week2 sprint. Experiments remain **engineering decision gates**, not publication artifacts — run the lightest version that decides the question (e.g. EXP-1's "3+ raters" can be relaxed). |

**Transport note (§4, §13)**: phase 1 is same-LAN. Phase 2 exposes the AGX over **Tailscale** so the robot can connect from anywhere. This revives reconnection handling — see §11.5.

---

## 0. WHAT CHANGED IN v5

Re-verification found the landscape moved. Key deltas from the previous spec:

| Change | Impact |
|---|---|
| **A working Gemma 4 + Whisper + Piper voice assistant on Jetson Orin Nano 8GB now exists publicly** `[REPORTED]` | The core architecture is **de-risked by an existing reference implementation**. AGX Orin 64GB has far more headroom. |
| **Gemini Live moved to Gemini 3.8 Live** `[OFFICIAL]` | The parity target is now higher: proactive audio, async function calling, background reasoning, 24 languages. |
| **New lightweight TTS options** `[REPORTED]` | NeuTTS Air (0.5B, GGUF, on-device), VibeVoice-Realtime-0.5B (~300ms), KittenTTS (15M, <25MB) — reduce dependence on the unverified CosyVoice-on-Jetson path. |
| **Jetson AI Lab published per-device Gemma 4 guidance** `[OFFICIAL]` | On AGX Orin, "both small models fit well and give good performance, and that is where the larger models start to become realistic too." |
| **E2E alternatives formally evaluated and rejected** (§4.1) | NVIDIA's own open full-duplex model (VoiceChat 11B) is English-only and not supported on Jetson. Korean + Jetson eliminates every open E2E option — **cascade is confirmed as the only path, not a compromise.** |
| **Full-duplex behavior can be imported into a cascade** (§4.2) | `[REPORTED]` Interruption reaction in a cascade is already ~198ms; only response generation is slow. Backchanneling added as EXP-12. |
| **`[VERIFY-FIRST]` faster-whisper needs a source build on Jetson** (§6.1) | Newly identified Day-1 blocker — the PyPI ctranslate2 wheel is CPU-only on aarch64. |
| **NVIDIA official sizing guidance** `[OFFICIAL]` | AGX Orin 64GB targets the **4B–20B range**. Also: 64GB "makes it far easier to combine vision, language, and speech (ASR and TTS) models on a single device without constantly running into memory limits." |

---

## 1. PROJECT CONTEXT

**Goal**: Replace the Gemini Live API with a fully local Korean voice conversation pipeline for the companion robot "Moti".
**Why**: API cost is unsustainable and usage is rate-limited.
**Domain**: Empathetic Korean conversation (공감 대화). NOT reasoning-heavy, NOT coding, NOT tool-heavy.
**Use**: Research / thesis only. No public or commercial distribution → copyleft licenses are acceptable (§15).

**Scope of this phase (v6 — changed)**: **distributed.** The AGX Orin 64GB runs the brain as an always-on
server; the robot (Jetson Orin Nano Super 8GB, repo `HGU-SIRLab/MOTI-HRI` branch `jetson-moti`) is the client.

```
┌─ Robot: Jetson Orin Nano Super 8GB ──┐        ┌─ Brain: Jetson AGX Orin 64GB ──┐
│ mic + AEC + speaker (existing)        │◄──────►│ VAD → turn → STT/SER → LLM →   │
│ camera, face ID, motors, display      │  audio │ TTS  (this spec)               │
│ tool execution (remember_fact, …)     │  +json │ conversation + session state   │
│ launcher.py — unchanged except one    │        │                                │
│ line: the connect() call              │        │                                │
└───────────────────────────────────────┘        └────────────────────────────────┘
        phase 1: same LAN          phase 2: Tailscale (robot may be remote)
```

**Why not standalone first**: the AGX has no microphone (verified — capture devices are APE/ADMAIF virtual
routing endpoints only; playback is HDMI). Building audio I/O there would be throwaway work, and the robot's
mic/speaker/AEC path is already hardware-validated. Development without the robot uses **file injection**
(§12), which is also the more rigorous way to run EXP-4/8/11 since every model sees identical input.

**What stays on the robot, always**: mic capture, AEC, speaker playback + interrupt flush, camera/face
recognition, motor control, and **execution of all tool calls** (the brain only says *what* to do). AEC must
be co-located with the mic and speaker in any topology — it needs the speaker reference signal.

---

## 2. EVIDENCE LEVELS — ENFORCE THESE

| Tag | Meaning | Treatment |
|---|---|---|
| `[OFFICIAL]` | Vendor/primary documentation | Trust |
| `[REPORTED]` | Secondary source, community benchmark, third-party repo | Trust direction, verify magnitude |
| `[UNVERIFIED]` | No public data found | **Must measure. Never state as fact.** |

---

## 3. PARITY TARGET — GEMINI 3.8 LIVE

`[OFFICIAL]` Current Gemini Live feature set:
- High audio quality, natural realistic speech
- **Multilingual: 24 supported languages**
- **Barge-in**: users can interrupt the model at any time
- **Affective dialog**: adapts response style and tone to match the user's input expression
- **Tool use**: function calling and Google Search
- **Audio transcriptions**: text transcripts of both user input and model output

`[REPORTED]` Recent additions: async function calling (long-running tools execute in background while conversation continues), **proactive audio** (agent responds only when directly addressed or contextually relevant), context injection without forcing a turn, and background reasoning brought into the native audio path.

`[OFFICIAL]` Gemini 3.8 Live is based on Gemini 3 Pro, natively multimodal, 128K context, "optimized for high-volume, latency-sensitive tasks like real-time dialogue."

**Realistic position**: we target *functional* parity on barge-in, affective dialog, proactive audio, and transcription. We do **not** target parity on raw latency or on 24-language coverage (Moti is Korean-only).

---

## 4. ARCHITECTURE

```
[Mic + AEC]
   | 16kHz mono PCM
[Silero VAD] ──────────> speech start/stop (CPU)
   | on silence
[smart-turn-v3] ───────> semantic turn-end (ONNX, ~8MB)
   | turn confirmed
   +──────────────┬──────────────┐   PARALLEL
   v              v              v
 [STT]          [SER]          [AED]
 text         emotion      audio events
   +──────────────┴──────────────┘
   | text + emotion
[Gemma 4 LLM] ─────────> response OR <SILENT>
   | streaming, sentence-chunked
[TTS + emotion instruct]
   v
[Speaker]
```

**v6 — where each box runs (§1):** `[Mic + AEC]` and `[Speaker]` are on the **robot** and already exist.
Everything between them runs on the **AGX brain**, including **Silero VAD and smart-turn-v3**.

That placement is deliberate: the robot currently has **no VAD at all** — `MOTI-HRI/docs/architecture.md`
§09 records that Gemini Live's server-side VAD replaced the old mouth-shape VAD, and `launcher.py`'s
`send_loop()` streams every mic chunk unconditionally. Keeping VAD on the brain therefore means the robot's
audio path needs **no change**: it keeps streaming exactly as it does to Gemini today.

Consequence to hold onto: echo-induced false barge-in used to be Google's problem and is now ours (§18).

Side note: `should_send_while_sleeping()`'s RMS gate on the robot exists only to avoid paying Gemini for
silence. Locally that cost is zero, so the gate is inert — harmless, but no longer load-bearing.

**EXP-13 (§12.1) would collapse `[STT]` and `[SER]` into `[Gemma 4 LLM]`**, since E4B takes audio directly.
Treat the diagram above as the specified fallback if EXP-13 is killed.

### 4.1 Why cascade, not end-to-end — re-verified 2026-09-18

Gemini Live and GPT Realtime **are** end-to-end native audio models. That is not the argument against cascade.

`[REPORTED]` **Industry reality**: "Most production deployments in 2026 use pipelines because teams need to control which LLM handles reasoning, which voice the user hears, and what business logic runs between transcription and response." End-to-end is what hyperscalers sell as an API; teams shipping their own products mostly run cascades.

**Every open E2E model fails at least one hard requirement for Moti:**

| Model | Capability | Why it fails here |
|---|---|---|
| **NVIDIA NemotronLabs VoiceChat 11B** (2026-08-03) | `[OFFICIAL]` 11B end-to-end real-time full-duplex; single unified architecture removes ASR→LLM→TTS handoffs; turn-taking latency **448ms**, user-interruption latency **480ms**, interruption TOR **1.0**; first open FD model with tool calling; #2 on VoiceBench and FullDuplexBench 1.0 among open models | ❌ `[OFFICIAL]` **English-only** (encoder is `nemotron-speech-streaming-en-0.6b`; system prompt specifies English). ❌ `[OFFICIAL]` **Supported hardware is A100/H100/H200/B100/B200/RTX-6000 — Jetson is not listed.** ❌ `[OFFICIAL]` Prompts must be **ASCII-only** |
| **Moshi** (Kyutai) | `[REPORTED]` ~200ms end-to-end, runs on a single GPU or iPhone 15 Pro, open-source, "nothing else matches the speed" | ❌ `[REPORTED]` **English-only**, 7B reasoning |
| **MiniCPM-o 4.5** | `[REPORTED]` matches Gemini 2.5 Flash on vision and speech; full-duplex live streaming where output (speech/text) and input (video/audio) streams do not block each other — sees, listens and speaks simultaneously, with proactive interaction; local Docker image for low-latency full-duplex on a Mac | ⚠️ Korean quality `[UNVERIFIED]`. The only E2E worth a future look |
| Qwen3-Omni | Speech in/out | ⚠️ Korean speech-output coverage `[UNVERIFIED]`; Jetson tooling immature |

**Conclusion: Korean + Jetson eliminates every open E2E option.** The good ones are English-first and datacenter-class. Cascade is not a compromise here — it is the only path. Re-check MiniCPM-o quarterly.

### 4.2 But import full-duplex *behavior* into the cascade

`[REPORTED]` Active research/engineering exists on giving cascades full-duplex feel:
- **DuplexCascade** — full-duplex speech-to-speech via a **VAD-free** cascaded ASR-LLM-TTS pipeline with **micro-turn optimization**
- **duet** — an open-source "full-duplex naturalness layer" as a **drop-in for cascaded ASR→LLM→TTS stacks**: natural backchanneling, mid-sentence interruption recovery, sub-300ms responsiveness

`[REPORTED]` **duet's measured profile is the key insight**: 2.169s from final speech end to first server audio, but **198ms from caller-audio start to playback cancellation**.

**Read this carefully: interruption reaction is already fast in a cascade. What is slow is response generation.** Much of perceived liveness comes from the former. This is the cheapest available path toward Gemini-like naturalness — see EXP-12 (backchanneling).

---

## 5. LLM — GEMMA 4

### 5.1 Jetson support matrix

`[OFFICIAL]` Jetson AI Lab: "Gemma 4 was released in four practical variants for Jetson: E2B, E4B, 26B-A4B, and 31B. The E2B and E4B models support audio, text, and image input with text output. 26B-A4B is the MoE model, and 31B is the larger dense model. The full family is supported on Jetson through both vLLM and llama.cpp."

**12B is NOT a Jetson variant. Do not use it.**

`[OFFICIAL]` Per-device guidance: "So far, E2B is the one that fits best on Orin Nano. On Orin NX, E2B and E4B are the natural choices. **On AGX Orin, both small models fit well and give you good performance for different use cases, and that is where the larger models start to become realistic too.**"

| Variant | Params | Q4_0 memory `[OFFICIAL]` | Audio input | Context |
|---|---|---|---|---|
| E2B | 2B effective | 2.9 GB | YES ⚠️ | — |
| **E4B** | 4B effective | **4.5 GB** | YES | — |
| 26B-A4B | 25.2B total / 3.8B active MoE | 14.4 GB | NO | 256K `[OFFICIAL]` |
| 31B | 31B dense | 17.5 GB | NO | — |

**Memory figures are weights only. KV cache is separate and scales with context.** `[OFFICIAL]`

`[OFFICIAL]` NVIDIA sizing: **AGX Orin 64GB targets the 4B–20B parameter range.** This puts E4B comfortably inside the sweet spot and 26B-A4B at the edge (though its 3.8B active params make it behave lighter than total size suggests).

### 5.2 PRIMARY: E4B

Rationale:
- `[OFFICIAL]` Sits in NVIDIA's stated 4B–20B sweet spot for this device; officially "fits well and gives good performance" on AGX Orin
- `[REPORTED]` RTX 4070 Ti llama.cpp benchmark, E4B vs 26B-A4B: **prompt processing 9.49x faster, generation 2.03x faster**. Prompt processing maps directly to TTFT — the metric that decides whether conversation feels live.
- `[REPORTED]` Academic deployment evaluation: 26B-A4B zero-shot ranked 1st (0.794) but **E4B variants took ranks 2–4 (0.761/0.759/0.758); gap is 0.033 with overlapping bootstrap intervals**, and the paper concludes deployment cost remains central to model choice.
- **Counter-evidence, stated honestly** `[REPORTED]`: paired permutation tests still favor 26B-A4B (p≈0.015–0.024), and a different comparison reports 82.6% vs 69.4%. **The gap is real but small and benchmark-dependent.**
- E4B supports audio input; 26B-A4B does not. Useful as a secondary emotion channel.

**DECISION RULE**: Build on E4B. Run EXP-1 (§12). Escalate to 26B-A4B only if E4B measurably fails Korean empathetic conversation quality.

### 5.3 Serving

`[OFFICIAL]` "In practice, vLLM tends to deliver better serving performance, while llama.cpp remains a good option if you want the GGUF path."
BUT `[REPORTED]` a DGX Spark benchmark found llama.cpp (51.57 tok/s) beat vLLM (30 tok/s) on 26B-A4B. **Benchmark both (EXP-2). Do not assume.**

```bash
# vLLM (NVIDIA official path)
sudo docker run -it --rm --pull always --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin \
  vllm serve google/gemma-4-E4B-it \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4

# llama.cpp (GGUF path)
sudo docker run -it --rm --pull always --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-server -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
```

`[OFFICIAL]` Checkpoint IDs:
| Variant | vLLM | llama.cpp GGUF |
|---|---|---|
| E2B | `google/gemma-4-E2B-it` | `unsloth/gemma-4-E2B-it-GGUF:Q4_K_S` |
| E4B | `google/gemma-4-E4B-it` | `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` |
| 26B-A4B | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | `ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M` |
| 31B | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | `ggml-org/gemma-4-31B-it-GGUF:Q4_K_M` |

`[OFFICIAL]` **"Do not mix formats casually."** vLLM checkpoints → vLLM containers; GGUF → llama.cpp.

`[OFFICIAL]` **Startup failures on large models are usually memory, not the model.** Clear page cache before retrying, and confirm the previous container released memory:
```bash
sudo sysctl -w vm.drop_caches=3
```

### 5.4 Mandatory LLM settings

1. **`[OFFICIAL]` Thinking mode defaults OFF at request time**, requiring `chat_template_kwargs.enable_thinking=true` to turn on. **NEVER enable it.**
   Supporting `[REPORTED]`: a third-party analysis classes 26B-A4B as a reasoning model generating 74M tokens vs a 45M median — verbose.
2. **`[OFFICIAL]` Always stream.** Non-streaming can leak thought text into `content`.
3. **Cap responses at 2–3 sentences in the system prompt.** Verbosity is latency.
4. **`[OFFICIAL]` Gemma 4 has native system prompt support** — use it for the Moti persona.
5. **`[OFFICIAL]` Every Gemma 4 model ships an MTP draft model for speculative decoding** — faster inference, no quality loss. Checkpoint pattern: `{model}-qat-q4_0-unquantized` + `{model}-qat-q4_0-unquantized-assistant`.
   `[REPORTED]` Modal's vLLM Gemma 4 example confirms the practical recipe: **turn off multimodal features to save GPU RAM, and activate built-in MTP speculative decoding for improved throughput at low concurrency**, naming the drafter as a separate `-assistant` checkpoint (e.g. `google/gemma-4-26B-A4B-it-assistant`) pinned to a specific revision. **Moti is single-user, low-concurrency — exactly the regime where MTP helps most.** Disabling multimodal is safe for us (we use text-only; §7 SER is a separate model).
6. **`[OFFICIAL]` QAT checkpoints exist** (`google/gemma-4-qat-q4-0`) — quantization folded into training, near-baseline quality.
7. **`[OFFICIAL]` ⚠️ E2B audio is broken under llama.cpp on Orin.** If any audio path is used, serve via vLLM.
8. `[OFFICIAL]` Ollama does not work with Gemma 4 on Orin Nano (works elsewhere incl. AGX Orin). Prefer the containers above anyway.

---

## 6. STT

**PRIMARY: faster-whisper with a Korean fine-tuned checkpoint**

`[OFFICIAL]` (Korean phonetics journal) Whisper struggles with Korean since it was not a major training language; fine-tuning with ~1,000 hours of Korean speech significantly improved CER.

Candidates `[UNVERIFIED]` — benchmark in EXP-4:
- `o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2`
- `seastar105/Korean-Whisper` collection
- baseline `openai/whisper-large-v3` (quantifies the fine-tuning gain)

`[OFFICIAL]` **KNOWN SIDE EFFECT**: Korean fine-tuned Whisper degrades badly on other languages — even with English specified, output may come out Korean. Fine for Korean-only Moti; blocks multilingual plans.

### 6.1 `[VERIFY-FIRST]` ⚠️ faster-whisper DOES NOT WORK OUT OF THE BOX ON JETSON

**This was missing from prior revisions and is a Day-1 blocker.**

`[REPORTED]` The PyPI `ctranslate2` wheel (faster-whisper's backend) is **CPU-only on aarch64**. `pip install faster-whisper` then `device="cuda"` fails with an error stating faster-whisper and CTranslate2 were not compiled for CUDA. **CTranslate2 must be compiled from source on Jetson.**

`[REPORTED]` Build requirements documented in a Jetson Orin practical guide:
- **cuDNN is mandatory** — without it ctranslate2 compiles but **fails at runtime** when Whisper attempts Conv1D on GPU
- **MKL is x86-only**; on aarch64 substitute OpenBLAS (`libopenblas-dev` build, `libopenblas0` runtime)
- **Specify the CUDA architecture explicitly**: Orin family (Orin Nano / Orin NX / **AGX Orin**) is `sm_87` → `-DCMAKE_CUDA_ARCHITECTURES=87`
- **Build ordering matters**: the package manager must finish before the custom ctranslate2 install, or it reverts to the CPU-only PyPI version
- aarch64 also needs FFmpeg **dev headers** (not just runtime) because PyAV compiles from source

```bash
cmake -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON       -DCMAKE_CUDA_ARCHITECTURES=87 ...
```

`[OFFICIAL]` Version pinning: recent ctranslate2 supports **CUDA 12 + cuDNN 9** only. JetPack 6.2 ships CUDA 12.6 + cuDNN 9.3 `[OFFICIAL]`, so **we are on the supported combination** — no downgrade needed. (For CUDA 12 + cuDNN 8 one would pin `ctranslate2==4.4.0`.)

`[REPORTED]` **Shortcut worth trying first**: a prebuilt image `cbinckly/speaches:0.9.0-l4t-cuda-12.6.11-arch87` is reported to work on SM87 (Orin family) devices — matching our CUDA 12.6 / sm_87 exactly. **Try this before building from source.**

`[REPORTED]` Performance datapoint: faster-whisper-small on the much weaker Orin Nano transcribes 20s of audio in under 3s. AGX Orin with a larger model should be comfortably better, but measure (EXP-4).

**ALTERNATIVE: SenseVoiceSmall** — bundles STT+SER+AED, see §7.

### ❌ EXCLUSIONS — do not propose these
- **NVIDIA Riva**: `[OFFICIAL]` current Quick Start prerequisites require **Jetson Thor** and **JetPack 7.x**; embedded ARM64 is public beta and needs an NVIDIA AI Enterprise trial license. We are AGX Orin + JetPack 6. Out of scope.
- `[REPORTED]` An OOM report exists for canary-1b-v2 on Jetson AGX Orin ("memory not actually full"), reinforcing that the NeMo/Riva family is a poor fit here.

---

## 7. SER — CLOSING THE AFFECTIVE DIALOG GAP

**Why**: `[OFFICIAL]` Gemini Live's affective dialog "adapts response style and tone to match the user's input expression." A cascade loses tone at STT. Running SER **in parallel** with STT restores it at near-zero added latency.

| Option | Size `[OFFICIAL]` | Notes |
|---|---|---|
| emotion2vec+ base | ~90M | 9 classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown), 16kHz |
| emotion2vec+ large | ~300M | 42,526h training |
| SenseVoiceSmall | — | `[OFFICIAL]` ASR (incl. **Korean**) + SER + AED in one checkpoint, built for latency-sensitive realtime interaction |

**`[UNVERIFIED]` KOREAN GAP — do not paper over this**: emotion2vec's paper reports SOTA on Mandarin, French, German, Italian. **No Korean-specific evaluation found.** SenseVoice's SER evaluation sets are Chinese and English. Korean SER accuracy is unmeasured for both.

**Fallback**: `[OFFICIAL]` emotion2vec freezes the backbone and trains only a lightweight downstream model — so only the classifier needs Korean retraining. Data in §10.

**Prompt injection format**:
```
[system] 너는 공감 로봇 모티야. 사용자의 감정 상태가 함께 제공되니 그에 맞춰 반응해.
[user]   (감정: 슬픔) 괜찮아.
```

---

## 8. TTS — `[VERIFY-FIRST]` HIGHEST RISK, NOW WITH MORE OPTIONS

### 8.1 Candidates

| Option | Evidence | Korean | Streaming | License |
|---|---|---|---|---|
| **CosyVoice 2 (0.5B)** | `[OFFICIAL]` **150ms first-packet** streaming; Korean officially supported; emotion/dialect instruct; NVIDIA-contributed TensorRT-LLM runtime. `[REPORTED]` 2026 roundups still rank it top-3 and single it out for real-time + Korean. | YES | YES | Apache 2.0 |
| **Piper** | `[REPORTED]` **Proven on Jetson** — used in a working Orin Nano voice assistant and a JetPack 7.2 tutorial | Check voice availability | — | **⚠️ GPL-3.0** |
| **NeuTTS Air (0.5B)** | `[REPORTED]` "world's first on-device super-realistic TTS", GGUF/GGML, realtime on CPU and GPU, runs on laptops/phones/Raspberry Pi | Verify | — | Verify |
| **VibeVoice-Realtime-0.5B** | `[REPORTED]` audible speech in ~300ms, streaming text input, single-speaker | Verify | YES | Research-oriented |
| **KittenTTS** | `[REPORTED]` 15M params, <25MB, no GPU needed, emotion control, streaming | English+ | YES | Apache 2.0 |
| Supertonic 3 | `[REPORTED]` Korean vendor, CPU-capable | YES | — | OpenRAIL-M |
| MeloTTS | `[REPORTED]` supports Korean, optimized for realtime CPU inference | YES | — | MIT |

### 8.2 `[UNVERIFIED]` ⚠️ COSYVOICE ON JETSON ORIN STILL HAS NO PUBLIC PRECEDENT

Searched again: Jetson AI Lab, jetson-containers, NVIDIA forums, community blogs. **No aarch64/Jetson deployment record found.** This remains the single largest unknown.

**Day-1 protocol**:
1. Plain PyTorch path first (no TensorRT). If it runs, MVP is viable.
2. Then streaming mode. Then optimization.
3. `[REPORTED]` Jetson needs manually installed prebuilt PyTorch aarch64 wheels — pip PyTorch is incompatible.
4. **Run a fallback in parallel the same day.** Do not burn a day on one path.

**Fallback ladder (v6 — Piper reinstated, research use confirmed)**:
1. **Piper** — `[REPORTED]` **most proven on Jetson** of any candidate here. v5 demoted it to
   "prototype only" purely on GPL-3.0 source-disclosure risk; §1 now confirms research/thesis use with no
   distribution, so that risk does not apply (§15). **This is the de-risking option: try it first and the
   project's #1 risk stops being a blocker.** Verify Korean voice availability.
2. **MeloTTS-Korean** (MIT, Korean, realtime CPU) — closest thing to a no-strings fallback
3. **Supertonic 3** (Korean vendor, CPU) — OpenRAIL-M, acceptable for research use
4. **NeuTTS Air** (GGUF on-device) — verify Korean

**CosyVoice 2 remains the quality/streaming target** (§8.1: 150ms first-packet, Korean, emotion instruct) and
still has no Jetson precedent (§8.2). The v6 change is that **failing on CosyVoice is no longer expensive** —
Piper is a working floor, so CosyVoice can be pursued on its merits rather than under Day-1 pressure.

---

## 9. VAD, TURN DETECTION, BARGE-IN, PROACTIVE AUDIO

### 9.1 Components
`[OFFICIAL]` **Silero VAD** — Pipecat built-in, local CPU.
`[OFFICIAL]` **smart-turn-v3** — Whisper Tiny encoder + linear classifier, ~8M params, 8MB ONNX, BSD 2-clause; ~65ms on a standard 1-vCPU instance, as low as 12ms for the int8 CPU build; input 16kHz mono PCM up to 8s (truncate from the start if longer); runs only after VAD detects silence; default turn-stop strategy in current Pipecat.

### 9.2 Barge-in
`[OFFICIAL]` Pipecat: Silero VAD + SmartTurn together emit turn start/stop frames with high accuracy and very low latency, driving optimized interruption logic so the bot yields to interruptions but does not react prematurely to brief mid-sentence pauses.

`[OFFICIAL]` Mirror Gemini's interruption contract: on detected interruption, cancel and discard ongoing generation; retain only what was already sent; **the client stops playback and clears its queue.**

**AEC IS MANDATORY.** `[REPORTED]` An NVIDIA forum thread titled "Echo/Feedback Issue — Speaker Audio Being Captured by Microphone on Jetson Orin" confirms this is a real, recurring problem on this exact hardware.
- MVP: headset sidesteps it
- Production: hardware AEC mic array (speaker + AEC on the same device)
- Keep playback buffer 100–200ms and **flush on interrupt** — the classic bug is a cancelled generation still playing for seconds

### 9.3 Proactive audio
`[REPORTED]` Gemini's proactive audio "lets a developer instruct the agent to only respond when directly addressed or when a reply is contextually relevant, rather than responding to every utterance."

Implementation `[UNVERIFIED]` — validate in EXP-7:
```
사용자가 혼잣말을 하거나, 생각을 정리 중이거나, 너에게 말한 게 아니면
정확히 <SILENT> 만 출력해. 그 외에는 평소처럼 2-3문장으로 답해.
```
Pipeline skips TTS on `<SILENT>`. Later: a pre-LLM classifier so silence costs nothing.
Bonus: `[OFFICIAL]` SenseVoiceSmall's AED can distinguish laughter/cough/ambient noise from speech.

---

## 10. KOREAN EMOTION DATA (fallback for §7)

`[OFFICIAL]` AI-Hub:

| Dataset | Scale | Classes |
|---|---|---|
| 감정 음성합성 데이터셋 | 50 professional voice actors, **1,067 hours**, 5 speech styles × 3 vocal characters | 7: 기쁨·슬픔·분노·불안·상처·당황·중립 |
| 감정 분류를 위한 대화 음성 데이터셋 | Real users via emotion-dialogue app, 5 annotators | 7: happiness, angry, disgust, fear, neutral, sadness, surprise |

Maps near-1:1 onto emotion2vec+'s 9 classes. Only the classifier layer needs retraining.

`[OFFICIAL]` **ACCESS — START NOW, APPROVAL TAKES TIME**: download requires approval; API download unlocks after. Files ship split-compressed; merge on Linux:
```bash
find "<dir>" -name "<file>.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "<file>.zip"
```

---

## 11. PIPELINE IMPLEMENTATION CONTRACT (framework-agnostic)

**v6: Pipecat is not used.** Rationale in §0-A row 2. This section is now a contract on *behavior*, not on a
framework — the four items in §11.0 are things Pipecat provided for free and that we must now implement and
keep implemented. They are the easiest things in this project to forget.

Dependencies: `websockets` only (already pulled in transitively by `google-genai` on the robot side), plus
whatever the chosen STT/TTS/VAD models require. Silero VAD and smart-turn-v3 are standalone ONNX packages and
are called directly.

LLM leg: the vLLM server is OpenAI-compatible, so the LLM call is a plain HTTP request to
`http://127.0.0.1:8000/v1/chat/completions` with any placeholder API key. No adapter code is needed.

### 11.0 `[MANDATORY]` The four things Pipecat gave us for free

| # | Requirement | Failure if skipped |
|---|---|---|
| 1 | **Playback buffer 100–200ms, flushed immediately on interrupt** (§9.2) | The classic bug: a cancelled generation keeps playing for seconds after the user interrupts. |
| 2 | **Sentence-level TTS chunking** — cut the LLM stream at sentence boundaries and synthesize per sentence | Without it, TTS waits for the whole response and first-audio latency collapses. **Korean needs sentence-final endings (다/까/요/죠) checked together with punctuation** — punctuation alone under-segments Korean. |
| 3 | **VAD-detection-lag compensation ring buffer** — keep a small rolling buffer of audio from *before* VAD fired | The first syllable of every utterance is clipped. |
| 4 | **Cancellation propagation** — on interrupt, cancel *both* LLM generation and TTS synthesis | Orphaned tasks keep producing audio/tokens for a turn the user abandoned. Use asyncio task cancellation, and isolate TTS so a mid-synthesis abort cannot corrupt the next turn. |

### 11.1 STT leg
Run STT only on VAD-delimited segments (not continuously), and keep the §11.0-3 compensation buffer.

**`[MANDATORY]` Pass raw 16-bit PCM to the STT model — never a WAV container.** (v5 stated this as Pipecat's
`wants_wav_segments=False`; the underlying requirement is unchanged and framework-independent. WAV wrapping
exists for cloud upload APIs, which we do not use.)

### 11.2 TTS leg
Synthesize per sentence (§11.0-2) and stream audio frames out as they are produced. Must expose: start,
audio-chunk, and abort. Abort has to be immediate and leave no state that affects the next turn (§11.0-4).

### 11.3 Reference implementations
- `[REPORTED]` **`itsMustafamr/Jarvis-home`** — a fully local Jetson Orin Nano 8GB voice assistant: Gemma 4 E2B + whisper.cpp + Piper + Silero VAD + Python WebSocket server + ALSA/HID audio path, no cloud. **Closest existing analogue to Moti — study its audio path and VAD endpointing.**
- The robot side is its own reference: `MOTI-HRI/launcher.py` already implements the full client half of this contract against Gemini Live (playback interrupt, tool execution, transcript logging). We are replacing its server, not its client.

### 11.4 Architecture lesson
`[REPORTED]` The Modal 1-second voice-to-voice project found that **segmenting audio with local VAD + turn detection and passing it to a non-streaming STT was faster than the streaming STT implementations they tried** — for total voice-to-voice latency only the final transcript matters.

**Our design already matches this. Do not chase streaming STT.**

### 11.5 Wire protocol — derived, not invented

The message surface is **mechanically determined** by what the existing robot client already consumes. It is
not a design choice. `launcher.py` uses exactly four session methods and reads exactly these fields:

| Direction | Payload | Maps to launcher.py |
|---|---|---|
| robot → brain | binary frame: raw PCM16 16kHz mic audio | `send_realtime_input(audio=Blob(...))` |
| robot → brain | `{"t":"text", "text":…}` | `send_client_content(turns=Content(...), turn_complete=True)` |
| robot → brain | `{"t":"tool_result", "results":[{id,name,result}]}` | `send_tool_response(function_responses=[...])` |
| brain → robot | binary frame: output PCM (match the existing 24kHz playback path) | `message.data` |
| brain → robot | `{"t":"interrupted"}` | `server_content.interrupted` |
| brain → robot | `{"t":"transcript","role":"user"\|"model","text":…}` | `server_content.input_transcription` / `output_transcription` |
| brain → robot | `{"t":"tool_call","calls":[{id,name,args}]}` | `message.tool_call.function_calls` |
| brain → robot | `{"t":"turn_complete"}` | `server_content.turn_complete` |

`websockets` distinguishes binary and text frames natively — no framing code required.

**Client shim**: accept the `google.genai.types` objects (`Blob`, `Content`, `FunctionResponse`) that
`launcher.py` already constructs, and expose `receive()` as an async generator that **ends at each turn
boundary** (Gemini's semantics — `launcher.py` re-enters it in a `while` loop). Then the robot-side diff is
one line: the `connect()` call. Do not restructure `launcher.py`.

**Tool schemas**: `launcher.py:302-339` builds `tools` as a list of plain Python callables and relies on
google-genai deriving declarations from signatures. The brain needs those schemas, so the robot must send
them at handshake (derive from `inspect.signature` + docstring).

**Reconnection `[MANDATORY]` for phase 2**: over Tailscale the link can drop, so `launcher.py`'s existing
`connection_manager()` reconnect loop stays load-bearing — do **not** treat it as dead code. Gemini's
`go_away` / `session_resumption_update` fields may stay unused, but **the brain must hold conversation state
server-side and resume it on reconnect**, or a dropped link wipes the conversation. State lives on the AGX
anyway, so this is natural.

---

## 12. EXPERIMENTS — THE ACTUAL DELIVERABLE

No public data exists for any of these on this hardware in Korean.

| ID | Experiment | Method | Decision rule |
|---|---|---|---|
| EXP-1 | E4B vs 26B-A4B Korean empathetic quality | 30 blind scenarios, same system prompt, temp 0.7, 3+ raters | E4B not meaningfully worse → **lock E4B** |
| EXP-2 | TTFT + tok/s, **llama.cpp vs vLLM** | 20 empathy prompts, mean + p95 | Pick the faster runtime; vendor guidance is not decisive |
| EXP-3 | MTP speculative decoding on/off | Same prompts | Keep if faster and quality holds |
| EXP-4 | Korean STT: 3 Whisper checkpoints + SenseVoiceSmall | 50 self-recorded sentences, quiet + noisy, CER | Lowest CER wins; if SenseVoice is close, adopt (free SER+AED) |
| EXP-5 | emotion2vec+ Korean accuracy | 20 utterances per emotion | Poor → retrain classifier on AI-Hub data |
| EXP-6 | Empathy quality with vs without emotion label | A/B on emotional scenarios | **Decides whether the SER channel earns its place** |
| EXP-7 | `<SILENT>` gate accuracy | 20 self-talk + 20 direct utterances | Tune prompt, else move to classifier |
| EXP-8 | **End-to-end voice-to-voice latency** | Full pipeline, 10 runs, mean + p95 | **Primary success metric** |
| EXP-9 | Barge-in success + self-interruption count | 30 interruptions | Any self-interruption → AEC problem |
| EXP-10 | Peak memory / power / thermal | tegrastats continuous | Check throttling |
| EXP-11 | TTS shootout on Jetson | CosyVoice 2 vs MeloTTS vs NeuTTS Air: first-packet latency + Korean naturalness | Decides §8 |
| **EXP-12** | **Backchanneling / micro-turn** (§4.2) | Emit short Korean acknowledgements ("응", "그렇구나") from a pre-synthesized cache **without** a full LLM round trip, triggered on VAD-detected pauses. A/B against the plain pipeline on 15 emotional scenarios | **If perceived liveness improves, this is the cheapest gain in the project** — real latency unchanged, felt latency drops |
| **EXP-13** | **E4B audio input as STT (+SER) replacement** — see §12.1 | Feed Korean speech audio directly to E4B via vLLM; compare against the Whisper cascade | 4 kill criteria in §12.1. Survives all four → **delete faster-whisper and emotion2vec from the design** |

**v6 note (no paper currently planned, §0-A row 6)**: these are **engineering decision gates**, not
publication artifacts. Run the lightest version that actually decides the question — e.g. EXP-1's "3+ raters"
can be a single careful pass. If a paper is planned later, the experiments worth publishing get re-run
properly then.

### 12.1 EXP-13 — kill criteria (user-approved, time-boxed to ~1 day)

**Why worth trying**: success deletes two models and removes risk #1 (ctranslate2 aarch64 source build) and
the Korean-SER unknown at once. Failure costs one day and we return to the §6 Whisper path.

`[MANDATORY]` Serve via vLLM, never llama.cpp — §5.4 rule 7 documents broken E2B audio under llama.cpp on
Orin; apply the same caution to E4B's audio path.

| # | Kill criterion | Verdict if hit |
|---|---|---|
| 1 | Korean comprehension noticeably worse than the Whisper path | **Kill** |
| 2 | TTFT slower than the Whisper path | **Kill** |
| 3 | No workable strategy for utterances beyond the 30s ceiling — **including multi-clip segmentation** (§12.2) | **Kill, or hybrid → §12.2** |
| 4 | Context erosion severe *at the context length we can afford to serve* | **Kill** — but §12.2 shows this is unlikely (30s = 0.57% of 128K) |

### 12.2 Audio token budget — `[OFFICIAL]` resolved from the checkpoint itself

Read directly from the downloaded E4B checkpoint, so this is primary vendor data, not a report:

| Source | Field | Value | Meaning |
|---|---|---|---|
| `processor_config.json` | `audio_ms_per_token` | **40** | **25 audio tokens per second** |
| `processor_config.json` | `audio_seq_length` | **750** | **30.0s hard ceiling per audio clip** (750 × 40ms) |
| `processor_config.json` | `feature_extractor.hop_length` | 160 @ 16kHz | 10ms mel frames → 4× subsampling to reach 40ms/token |
| `config.json` | `text_config.max_position_embeddings` | **131072** | 128K context |
| `config.json` | `text_config.sliding_window` | 512 | hybrid attention — most layers are cheap on long context |

The "~25 tokens/sec, ~30s ceiling" figures are therefore **confirmed exactly**. Resolves Q13's first half.

**This reweights kill criteria 3 and 4:**

**Criterion 4 (context erosion) — much weaker than assumed. Likely will not trigger.**

| Utterance | Audio tokens | Share of 128K context |
|---|---|---|
| 5s | 125 | 0.10% |
| 10s | 250 | 0.19% |
| 30s (max) | 750 | 0.57% |

Fifty turns of 10s speech is ~12.5K tokens, under 10% of context. Audio tokens do **not** meaningfully crowd
out the persona or accumulated `facts`. The real constraint is not the model's ceiling but **the context
length we choose to serve and its KV-cache cost** (≈84 KB/token here: 42 layers × 2 KV heads × 256 head_dim
× 2 × 2 bytes). So criterion 4 should be re-read as: *does the context length we can actually afford to
serve leave room for both audio and history?* — a serving-config question, measured in Stage 1.

**Criterion 3 (30s ceiling) — this is the one that matters.** A person unloading for 30+ seconds is a
*normal* scenario in empathetic conversation, not an edge case.

But there is a mitigation worth testing before falling back to Whisper: **we control segmentation.** A long
turn can be split into consecutive ≤30s clips and passed as multiple audio parts in one prompt. If Gemma 4
accepts interleaved multi-audio input, the ceiling stops being a wall and the "delete two models" win
survives. **Test this explicitly in EXP-13 before invoking the hybrid/SER-only landing point.**

**Prefill grows → TTFT grows** remains true and is criterion 2 by another route: a 30s utterance adds 750
prefill tokens.

**Landing point if criterion 3 hits**: hybrid — short utterances go through audio directly, long ones fall
back to Whisper. But that means keeping Whisper, which erases the "delete two models" win. In that case the
cleaner outcome is to **use the audio path for SER only**: it resolves §7 (unverified Korean emotion2vec)
while leaving §6 (Whisper STT) as specified. Decide between hybrid and SER-only on measured numbers.

---

## 13. LATENCY BUDGET

### 13.0 `[MEASURED]` Stage 1 results — E4B on this AGX (2026-09-18)

vLLM 0.19.0, `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin`, bf16 (no quantization),
`--max-model-len 32768`, `--gpu-memory-utilization 0.40`, MAXN + `jetson_clocks`.
KV cache 67,712 tokens. No config patch of any kind was needed for E4B.

| Condition | TTFT | decode |
|---|---|---|
| Real MOTI persona (18,344 tok), **first turn of a session** | **17.26 s** | 13.2 tok/s |
| Real MOTI persona, **turns 2+** (prefix cache hit) | **0.209 s** | 13.2 tok/s |
| Short stand-in prompt (~50 tok) | 0.174 s | 14.4 tok/s |

Warm TTFT beats the §13 escalation threshold (>0.7s) by 3.3×, so **no quantization or MTP is
needed for TTFT** — decode speed is the remaining lever, not prefill.

### 13.1 `[RESOLVED]` Persona prefill — 17s cold, solved by pre-warming on the brain

`build_persona_system_instruction()` in `MOTI-HRI/core/utils.py` produces **31,845 chars =
18,344 tokens**, and the per-user part (the user's name) sits at **character 372 — 1.2% in**.
Everything user-specific comes *first*; the ~18K tokens of static instruction come *after* it.

Consequence: the cacheable shared prefix is ~200 tokens. Every session — and every different
user — pays the full 18,344-token prefill. `launcher.py` sends a "greet the user first" turn
immediately on connect, so **the robot stands silent for 17 seconds when someone walks up.**

This is a regression against the system being replaced: the robot repo measured Gemini's first
response at 0.49–0.66s. Gemini absorbs an 18K prefill on datacenter hardware; this AGX cannot.

**`[MEASURED]` Fix — pre-warm on the brain. No robot-side change at all.**

vLLM's prefix cache is GPU-resident and **survives across client processes for the life of the
server**. Verified: a fresh benchmark process issuing the *first* request with a persona that had
been prefilled minutes earlier by a different process got **0.178s**, not 17.26s.

So the brain simply issues a throwaway request per persona at startup — before anyone is standing
in front of the robot — and the 18K prefill is paid once, offline. Pre-warm:
1. the **unknown-user** persona variant (a single fixed prompt shared by *every* first-time user), and
2. the personas of known regular users.

Capacity: 67,712 KV tokens ÷ 18,344 per persona ≈ **3.7 personas** resident at once, minus room for
the active session. For more, raise `--gpu-memory-utilization` — E4B is small and 0.40 leaves
headroom. Cache eviction is LRU and load is a single robot, so pressure is low.

**Rejected alternative — reordering the prompt.** The earlier plan here was to move the name/facts
block to the end so the static body becomes a shared prefix. Measurement killed it: the user's name
is interpolated at **four** scattered points (1.2%, 1.2%, 9.5%, 46.9%), the last inside
`deep_context_block`'s rule that an omitted Korean subject always refers to the user — wording
written to fix a real observed failure. Moving it would mean rewriting tuned persona text, i.e. a
behavior change, not the free reordering first assumed. Pre-warming achieves the same result with
zero edits to the robot repo, so this stays rejected unless pre-warming proves insufficient.

**Also rejected — `QUIZ_EXPERIMENT_MODE=true`.** The robot repo already ships this switch and it
halves the prompt (31,835 → 15,982 chars), added because Gemini bills the system instruction every
turn. It only halves the cold cost (~8.6s, still too slow) and it drops the free-conversation blocks
(campus culture, deeper counselling, club suggestions) that *are* the 공감 대화 domain. Wrong trade here.

**Secondary lever, untouched**: 18K tokens is also 27% of the KV budget per session and costs some
decode speed (14.4 → 13.2 tok/s vs a short prompt). Shrinking it is a behavior change; leave it.

---

Everything below is `[UNVERIFIED]` — estimates only. Replace with EXP-8.

| Stage | Reference |
|---|---|
| Turn-end gating | smart-turn inference `[OFFICIAL]` 12–65ms; total gating delay `[UNVERIFIED]` |
| STT finalization | `[UNVERIFIED]` |
| LLM TTFT | `[REPORTED]` DGX Spark (much stronger than AGX Orin) ~428ms e2e TTFT for 26B-A4B at pp2048; cloud providers 0.68–5.51s. **E4B should be substantially better given 9.49x prompt processing.** |
| TTS first packet | `[OFFICIAL]` CosyVoice 2 claims 150ms; `[REPORTED]` VibeVoice-Realtime ~300ms; Jetson values `[UNVERIFIED]` |

**Reference points**:
- `[REPORTED]` A Jetson Orin Nano document-chat project reports "real-time STT, ~15 tok/s LLM, near-instant TTS, full conversation loop in 2–3 seconds" — a realistic floor for an unoptimized small-device pipeline. AGX Orin + E4B should beat this substantially.
- `[REPORTED]` A commercial Jetson voice reference design claims sub-180ms streaming ASR+TTS on Jetson — evidence the speech ends of the pipeline can be fast; the LLM is the long pole.

**v6 additions to the budget — both found by reading the robot code, both measurable in EXP-8:**

1. **Voice-shift buffer: 700ms.** The robot's `VOICE_SHIFT_BUFFER_MS` is **700ms** (raised from 500ms after
   reported crackling/dropouts), sitting on top of playback. §9.2 calls for a 100–200ms playback buffer, so
   this is 3.5–7× that guidance and it delays interrupt flush as well as first audio. It exists to support
   the pyworld pitch/formant shift (+3.5st/×1.12) that makes the voice sound younger — which was needed only
   because Gemini's `Zephyr` voice is an adult voice. **A local TTS lets us pick a young-sounding voice
   directly, and then the shifter (and its 700ms) can be switched off entirely.** Add
   `ENABLE_VOICE_SHIFT` on/off as an arm of EXP-8 and EXP-9, and treat "pick the voice at the TTS instead of
   post-processing it" as the preferred outcome (EXP-11 selection criterion).
2. **Network hop.** Phase 1 (same LAN, wired GbE) should be ~1ms and negligible, but audio buffering across
   the link is not — measure it, don't assume. Phase 2 (Tailscale) must be measured separately, and
   distinguish direct vs. relayed (§18). Note the tension: a lossy remote link wants *more* jitter buffer,
   while §9.2 wants *less* playback buffer. Resolve with measurement, and let the degraded mode be explicit.

Bandwidth, for reference: 16kHz mono PCM16 upstream ≈ 256 kbps, 24kHz downstream ≈ 384 kbps. Not a
constraint on any realistic link, including Tailscale.

**Escalation**: E4B TTFT consistently >700ms → apply MTP + QAT → shorten context → consider E2B.

---

## 14. GEMINI 3.8 LIVE PARITY MATRIX

| Feature `[OFFICIAL]` | Status | Path |
|---|---|---|
| Barge-in | ✅ Achievable | Pipecat + Silero + smart-turn (§9.2) |
| Interruption cancel/discard | ✅ Achievable | Mirror Gemini's contract |
| Audio transcription (both sides) | ✅ **Advantage** | Cascade produces text natively |
| High-quality natural speech | ✅ Achievable | §8 candidates |
| Affective dialog | ✅ Achievable | Parallel SER (§7) — **pending Korean verification** |
| Emotional speech output | ✅ Achievable | CosyVoice 2 emotion instruct |
| Proactive audio | ✅ Achievable | `<SILENT>` gate (§9.3) |
| Function calling | ✅ Available | `[OFFICIAL]` Gemma 4 native function calling; vLLM flags provided |
| Async function calling | ⚠️ Custom work | Not free; low priority for Moti |
| Background reasoning | ❌ Deliberately excluded | Thinking mode kills latency (§5.4) |
| 24 languages | ❌ Korean-only | Irrelevant for Moti |
| Natural conversational rhythm | ⚠️ Partial | Backchanneling / micro-turn (§4.2, EXP-12) closes part of the *felt* gap |
| Sub-500ms latency | ⚠️ **Not achievable** | Irreducible gap. `[OFFICIAL]` For reference, NVIDIA's own open E2E model reports ~448ms turn-taking — and it needs an H100-class GPU |

**Honest position**: every functional gap has a closing path. **Latency is the one irreducible difference.** In exchange Moti gets zero marginal cost, unlimited use, full privacy, a fixed persona, long-term memory, and robot-body integration — none of which the API offers.

---

## 15. LICENSE REGISTER

**v6 scope change**: §1 confirms **research / thesis use with no public or commercial distribution.** The
"Commercial" column is therefore not a gate for this project — the "Research use" column is. Keep the
commercial column filled in anyway, so that a future decision to release does not have to re-derive it.

| Component | License | Commercial | Research use (this project) |
|---|---|---|---|
| Gemma 4 (all) | Apache 2.0 `[OFFICIAL]` | YES | YES |
| CosyVoice 2 | Apache 2.0 `[OFFICIAL]` | YES | YES |
| MeloTTS | MIT `[REPORTED]` | YES | YES |
| KittenTTS | Apache 2.0 `[REPORTED]` | YES | YES |
| Silero VAD | MIT | YES | YES |
| smart-turn-v3 | BSD 2-clause `[OFFICIAL]` | YES | YES |
| faster-whisper | MIT | YES | YES |
| **Piper (`piper-tts`)** | **GPL-3.0 `[OFFICIAL]`** | ⚠️ NO — source-disclosure risk | **YES — reinstated (§8.2)** |
| Supertonic 3 | OpenRAIL-M | Verify terms | YES |
| VibeVoice | Research-oriented `[REPORTED]` | Verify | YES |
| EXAONE 4.0 | Non-commercial `[OFFICIAL]` | NO | **YES — no longer excluded** |

**On EXAONE 4.0**: v5 excluded it on licensing alone, so research use unblocks it, and it is a Korean-native
model — relevant to a Korean empathetic-conversation domain. **Not proposed as a change now**, for two
reasons: Gemma 4's native audio input is what EXP-13 is built on, and Jetson support for Gemma 4 is
`[OFFICIAL]`-documented per variant (§5.1) while EXAONE's is unknown here. Revisit only if EXP-1 shows E4B
failing on Korean empathetic quality.

**If distribution is ever reconsidered**: Piper and EXAONE are the two components that would have to be
swapped out, and re-verify every row at that point.

---

## 16. ENVIRONMENT

```bash
cat /etc/nv_tegra_release                      # expect R36 (JetPack 6)
sudo apt update && sudo apt install nvidia-jetpack
sudo nvpmodel -m 0                             # MAXN — before any benchmark
sudo jetson_clocks
sudo pip3 install jetson-stats && jtop         # no nvidia-smi on Jetson
sudo sysctl -w vm.drop_caches=3                # before relaunching large models
```

`[OFFICIAL]` NVMe SSD strongly recommended.
`[REPORTED]` Jetson traps: **never set the `cma=` kernel parameter** (breaks GPU detection); CMA fragmentation on model unload needs a reboot; pip PyTorch is incompatible with aarch64 — use prebuilt wheels.

---

## 17. BUILD ORDER

**v6: no day labels.** §0-A row 6 — no deadline, no paper currently planned, steady incremental progress.
The ordering below is by **dependency and risk**, not by calendar. Finish and verify one stage before
starting the next; do not run parallel blockers just to save days we do not need to save.

**Stage 0 — environment (mostly done)**
- [x] `nvpmodel` MAXN confirmed on the AGX (`jetson_clocks` still needed immediately before any benchmark; it resets on reboot)
- [x] NVMe space confirmed (583G free)
- [x] Gemma 4 E4B downloaded (`google/gemma-4-E4B-it`, bf16, HF cache)
- [ ] Skim `itsMustafamr/Jarvis-home` for its ALSA/VAD/WebSocket audio path
- [ ] AI-Hub request — **deferred**: only needed if EXP-13 is killed *and* Korean SER proves poor (§10). Approval latency is still the reason to submit early if EXP-13 looks shaky.

**Stage 1 — LLM leg standing up** (lowest risk, unblocks everything)
- [ ] Serve E4B via vLLM in the NVIDIA-supported container (§5.3), confirm streaming `curl`
- [ ] bf16 first — no quantization until measurement says otherwise (§13 escalation)
- [ ] Measure TTFT + tok/s → EXP-2

**Stage 2 — EXP-13, the fork in the road** (§12.1)
- [ ] Feed Korean audio directly to E4B via vLLM; measure audio token rate and clip ceiling empirically
- [ ] Run the four kill criteria
- [ ] **Survives → delete the Whisper and SER legs from the design. Killed → Stage 2b.**
- [ ] Stage 2b (only if killed): `[VERIFY-FIRST]` faster-whisper GPU on Jetson (§6.1) — prebuilt `l4t-cuda-12.6.11-arch87` image first, source build as fallback

**Stage 3 — TTS leg** (§8.2 ladder, Piper first now)
- [ ] Piper on Jetson + Korean voice availability — this is the de-risking step, not the ambition
- [ ] Voice selection with §13's finding in mind: prefer a young-sounding voice over pitch-shift post-processing, so `ENABLE_VOICE_SHIFT` can go off
- [ ] CosyVoice 2 on merit afterwards (plain PyTorch first) → EXP-11

**Stage 4 — the pipeline and the wire** (§11)
- [ ] VAD + smart-turn-v3 on the brain, with the §11.0-3 compensation buffer
- [ ] Brain server: wire protocol per §11.5, conversation state held server-side
- [ ] Robot-side client shim; `launcher.py` diff must be the `connect()` call only (§20 rule 17)
- [ ] Instrument every stage from the first commit (§20 rule 5)
- [ ] Fake-robot client for round-trip testing without the robot; then the real robot on the same LAN
- [ ] The four `[MANDATORY]` items in §11.0 — treat as acceptance criteria, not TODOs

**Stage 5 — behavior and measurement**
- [ ] Barge-in + playback buffer flush → EXP-9 (**on the robot**, not the AGX — §12 note)
- [ ] EXP-8 end-to-end latency, with `ENABLE_VOICE_SHIFT` on/off arms
- [ ] EXP-10 thermal/power, EXP-3 MTP
- [ ] `<SILENT>` gate → EXP-7
- [ ] EXP-1 (E4B vs 26B-A4B) only if Korean quality looks marginal
- [ ] EXP-12 backchanneling
- [ ] EXP-4/5/6 only in the branch where EXP-13 was killed

**Stage 6 — phase 2 transport**
- [ ] Tailscale; verify `direct` not `relay` (§18); re-measure EXP-8 over it
- [ ] Reconnect/resume across link drops (§20 rule 18)

---

## 18. RISK REGISTER

| Risk | Severity (v6) | Mitigation |
|---|---|---|
| CosyVoice fails on Jetson | 🟡 **MED — downgraded from 🔴** | Piper is reinstated as a working floor (§8.2), so this is no longer a blocker. Still no Jetson precedent; pursue on merit, not under deadline. |
| **faster-whisper CPU-only on Jetson** | 🟡 **MED — conditional** | `[REPORTED]` ctranslate2 PyPI wheel is CPU-only on aarch64; needs prebuilt sm_87 image or source build with cuDNN + OpenBLAS + `CUDA_ARCHITECTURES=87` (§6.1). **EXP-13 may remove this risk entirely** by deleting the Whisper leg. Do not start the source build until EXP-13 lands. |
| Korean SER accuracy poor | 🟡 **MED — conditional** | Retrain classifier on AI-Hub data (§10). **EXP-13 may remove this too** (E4B hears tone directly). |
| Latency above target | 🟡 MED | E4B → MTP → QAT → shorter context → E2B. New v6 contributors: voice-shift buffer (§13) and, in phase 2, Tailscale relay fallback. |
| **Tailscale falls back to DERP relay (phase 2)** | 🟡 **MED — new in v6** | Relayed WireGuard adds unpredictable latency, which lands directly in the voice loop. Require a **direct** connection (`tailscale status` shows `direct`, not `relay`) and treat relayed operation as a degraded mode. |
| Missing AEC breaks barge-in | 🟡 MED | **v6 correction**: the robot repo's 30–36dB validation was on **Windows**; on Jetson the wheel is absent and its documented PulseAudio fallback was configured on a since-reverted Xavier board. **User states this is already solved on the Orin Nano — the robot repo's docs are stale and should be updated there.** This matters more now: replacing Gemini's server-side VAD with ours makes echo-induced false barge-in *our* problem. |
| CMA fragmentation | 🟢 LOW | Never touch `cma=`; reboot if hit |
| ~~Piper GPL contamination~~ | ✅ **Closed in v6** | Research-only use (§1, §15). Keep the §20 rule 7 flag for a future release decision. |

---

## 19. OPEN QUESTIONS — MEASURE, DO NOT GUESS

| # | Question | Resolved by |
|---|---|---|
| Q1 | Does CosyVoice 2 run on Jetson Orin at all? | Day 1 / EXP-11 |
| Q2 | E4B or 26B-A4B for Korean empathetic conversation? | EXP-1 |
| Q3 | llama.cpp or vLLM faster here? | EXP-2 |
| Q4 | Is emotion2vec+ usable for Korean out of the box? | EXP-5 |
| Q5 | Does SenseVoiceSmall's Korean ASR match a Korean-tuned Whisper? | EXP-4 |
| Q6 | Does the emotion label actually improve perceived empathy? | EXP-6 |
| Q7 | Achievable end-to-end latency on this hardware? | EXP-8 |
| Q8 | Which Korean Whisper checkpoint is best? | EXP-4 |
| Q9 | Do NeuTTS Air / VibeVoice-Realtime support Korean adequately? | EXP-11 |
| Q10 | Does the prebuilt sm_87 ctranslate2 image work, or is a source build required? | Day 1 (§6.1) |
| Q11 | Does backchanneling measurably improve perceived liveness in Korean? | EXP-12 |
| Q12 | Has any open E2E model gained Korean + Jetson support? (re-check quarterly) | Quarterly review of MiniCPM-o, Qwen3-Omni |
| ~~Q13a~~ | ~~E4B audio token rate and clip ceiling?~~ | ✅ **Resolved from the checkpoint: 25 tok/s, 30s ceiling (§12.2)** |
| **Q13b** | Does E4B audio input beat the Whisper cascade on Korean, and does multi-clip segmentation clear the 30s ceiling? | **EXP-13 / §12.1–12.2** |
| **Q14** | Must the current young voice (Zephyr + pitch shift) be reproduced, or can a young-sounding local TTS voice replace it — letting the 700ms shift buffer go? | User decision + EXP-11 (§13) |
| **Q15** | Does Tailscale hold a `direct` connection in practice, and what does it add to EXP-8? | Stage 6 |
| **Q16** | How was AEC actually solved on the Orin Nano, and is that recorded anywhere? | Robot repo's docs are stale (§18) — update them there |

---

## 20. RULES FOR CLAUDE CODE

1. Never present `[UNVERIFIED]` items as measured fact — in code comments, commits, or reports.
2. Never enable Gemma 4 thinking mode in any code path.
3. Always stream LLM and TTS output.
4. **Pass raw 16-bit PCM to the STT model — never a WAV container.** (v6: generalized from the Pipecat-specific `wants_wav_segments=False`, since Pipecat is no longer used. The requirement is the same.)
5. Instrument latency at every stage from commit one — EXP-8 is the primary deliverable and retrofitting is painful.
6. When a benchmark contradicts vendor guidance, trust the measurement and record both.
7. Still flag GPL dependencies explicitly when introducing them. (v6: GPL no longer *blocks* anything — §1 confirms research-only use, so Piper is reinstated — but the flag must stay so that a future decision to distribute can find them. §15 names Piper and EXAONE as the two that would need swapping.)
8. If CosyVoice fails on Day 1, switch to the fallback rather than debugging past the day boundary.
9. Never propose NVIDIA Riva — it needs Jetson Thor + JetPack 7.
10. If any Gemma 4 audio path is used, serve via vLLM (E2B audio is broken under llama.cpp on Orin).
11. Do not mix model formats — vLLM checkpoints with vLLM, GGUF with llama.cpp.
12. Record §19 resolutions in commit messages or a decision log as they land.
13. **Never assume `pip install faster-whisper` gives GPU acceleration on Jetson** — verify `device="cuda"` actually works before building anything on top of it (§6.1).
14. **Do not propose switching to an end-to-end speech model** — §4.1 documents why every open option fails on Korean, Jetson support, or both. If that changes, it changes at the quarterly review (Q12), not mid-build.
15. **Optimize felt latency, not just measured latency** — backchanneling and fast interruption reaction (§4.2) buy more perceived liveness per engineering hour than shaving TTFT.
16. **Do not reintroduce Pipecat or any pipeline framework** without revisiting §0-A row 2. Instead, honor the four `[MANDATORY]` items in §11.0 — playback buffer + interrupt flush, sentence-level TTS chunking (Korean endings, not just punctuation), VAD-lag compensation buffer, and cancellation propagation. These are the things the framework used to guarantee; nothing guarantees them now except this list.
17. **Do not restructure `launcher.py`.** The robot-side change is the `connect()` call and nothing else (§11.5). Its reconnect loop stays live for phase 2 (Tailscale) — do not delete it as dead code.
18. **The brain holds conversation state server-side** and resumes it across reconnects (§11.5). A dropped link must not lose the conversation.

---

## SOURCES

NVIDIA Jetson AI Lab (Gemma 4 on Jetson tutorial; Gemma 4 26B-A4B model page) · NVIDIA Technical Blog / Edge AI and Vision Alliance (Jetson sizing guidance) · NVIDIA Riva documentation (Quick Start prerequisites) · NVIDIA Developer Forums (Jetson Orin echo/feedback, canary OOM) · Google AI for Developers (Gemma 4 overview, model card, audio understanding; Gemini API models & changelog) · Google Cloud (Gemini Live API overview) · Google DeepMind (Gemini 3.8 Audio model card) · daily.dev (Gemini Live API feature roundup) · Pipecat API reference (tts_service, stt_service, piper) · pipecat-ai/smart-turn · FunAudioLLM CosyVoice · SenseVoice · emotion2vec (ACL Findings 2024, model cards) · AI-Hub dataset pages · itsMustafamr/Jarvis-home · Modal engineering blog · SiliconFlow / BentoML / Northflank / Gladia 2026 TTS & STT roundups · community benchmarks (DGX Spark, RTX 4070 Ti) · NVIDIA NemotronLabs VoiceChat 11B model card · Kyutai Moshi (via Inworld S2S comparison) · OpenBMB MiniCPM-o 4.5 · DuplexCascade (arXiv) · mssharatchandra/duet · cbinckly Jetson STT practical guide · SYSTRAN/faster-whisper · Modal vLLM Gemma 4 example · Pipecat Ollama service reference
