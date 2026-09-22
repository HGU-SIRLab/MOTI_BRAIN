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

**✅ EXP-13 won (§12.4), so `[STT]` and `[SER]` are collapsed into `[Gemma 4 LLM]`** — E4B takes the audio
directly and hears tone. The live pipeline is therefore:

```
[Mic + AEC] -> [Silero VAD] -> [smart-turn-v3] -> split into <=30s clips -> [Gemma 4 E4B]
   (robot)  |                    (brain)                                  |  audio in, text out
            |                                                             v
            +<---------------------- [Speaker] <--------------------- [TTS]
```

The diagram above with separate `[STT]`/`[SER]`/`[AED]` boxes is retained as the documented fallback
if the audio path fails in live use.

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
   `[REPORTED]` Modal's vLLM Gemma 4 example confirms the practical recipe: **turn off multimodal features to save GPU RAM, and activate built-in MTP speculative decoding for improved throughput at low concurrency**, naming the drafter as a separate `-assistant` checkpoint (e.g. `google/gemma-4-26B-A4B-it-assistant`) pinned to a specific revision. **Moti is single-user, low-concurrency — exactly the regime where MTP helps most.**

   🔴 **v6 correction — do NOT disable multimodal.** v5 wrote that it was safe "because we use
   text-only". EXP-13 (§12.4) made **audio the primary input path**: the model hears the user
   directly, which is what deleted the Whisper and SER legs. Following the Modal recipe verbatim
   would break the pipeline's input entirely. Take the MTP half of that recipe and leave
   multimodal on.
6. **`[OFFICIAL]` QAT checkpoints exist** (`google/gemma-4-qat-q4-0`) — quantization folded into training, near-baseline quality.
7. **`[OFFICIAL]` ⚠️ E2B audio is broken under llama.cpp on Orin.** If any audio path is used, serve via vLLM.
8. `[OFFICIAL]` Ollama does not work with Gemma 4 on Orin Nano (works elsewhere incl. AGX Orin). Prefer the containers above anyway.

---

## 6. STT — ⚠️ NOT USED (fallback only, see §12.5)

**EXP-13 removed this leg from the pipeline.** E4B takes audio directly and matched a perfect transcript
(§12.4), so faster-whisper and its aarch64 `ctranslate2` source build are not needed. Everything below is
kept as the fallback plan if the audio path later fails in live use. **Do not start the source build.**


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

## 7. SER — ⚠️ NOT USED (fallback only, see §12.5)

**EXP-13 removed this leg too.** E4B hears prosody directly: given identical wording, it contradicted the
literal words on a suppressed-sadness reading (§12.4). The affective-dialog gap this section was written to
close is already closed, and the unmeasured Korean-SER risk is closed with it. Kept as fallback.


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

### 8.3 ✅ `[DECIDED]` TTS = Piper `ko_KR-kss-medium`, voice unmodified (2026-09-19)

Piper works on this AGX and is fast enough: **RTF 0.22–0.38** measured on real Moti lines (8.4s of speech
synthesized in 1.88s). With sentence-level chunking (§11.0-2) the first sentence lands well inside 0.5s.

Two constraints found while validating it, both accepted:
- **Korean has exactly one voice** — `ko_KR-kss-medium`, 1 speaker, 22,050 Hz. There is no voice to choose,
  so "pick a young-sounding voice instead of pitch-shifting" was never an option here.
- **Dataset licence is CC BY-NC-SA 4.0** (KSS). Fine for research use (§1, §15), and a second thing to swap
  alongside Piper's GPL if distribution is ever reconsidered.

User listened to the voice with and without a +3.5 semitone shift and **chose the unmodified voice.**
Consequences: `ENABLE_VOICE_SHIFT=false`, the 700ms buffer disappears (§13), and `pyworld` drops off the
robot's dependency list.

✅ **Resampling (was a Stage 4 TODO)**: Piper emits 22,050Hz; `MOTI-HRI/media/audio_manager.py`
hardcodes `OUTPUT_RATE = 24000` and opens the sounddevice stream at it. The brain converts before
sending — polyphase 160/147 (exact, since gcd(24000, 22050) = 150), ~3ms per 3s of audio.

**Why not just change the robot's constant?** It is a single well-factored constant, so it *could*
be changed. Keeping 24000 is better on four counts:
- **24000 is exactly half of 48000**, the rate most USB audio runs natively, so the OS resample
  downstream is a trivial 2×; 22050 → 48000 is 320/147. The AEC reference path is likewise an exact
  2:3 at 24000 → 16000 versus 320/441 at 22050. (Device native rate on their Jieli UAC unit is
  unverified — this is the general case.)
- **CosyVoice 2 is natively 24kHz**, so an upgrade later (§8.2) would mean changing it back.
- **`assets/audio/snore.wav` is 24kHz** and `launcher.py:145` *skips the sleepy sound with a warning*
  on a rate mismatch — a silent feature loss if the asset is not regenerated.
- The cost is one upsample that adds no information, on the AGX rather than the weaker Orin Nano.

It also preserves the one-line-change property (§11.5).

**Installation trap**: the official `piper_linux_aarch64` binary (1.2.0) **crashes** on this Korean model —
`"aɪ" is not a single codepoint`, a mismatch between its bundled espeak-ng and the model's phoneme map.
The maintained `piper-tts` pip package (1.4.2) works. Installed in `.venv_tts/` to keep it away from the
system Python.

**CosyVoice 2 remains the quality/streaming target** (§8.1: 150ms first-packet, Korean, emotion instruct) and
still has no Jetson precedent (§8.2). The v6 change is that **failing on CosyVoice is no longer expensive** —
Piper is a working floor, so CosyVoice can be pursued on its merits rather than under Day-1 pressure.

---

## 9. VAD, TURN DETECTION, BARGE-IN, PROACTIVE AUDIO

### 9.1 Components
**v6**: Pipecat is not used (§0-A row 2), so the *components* below stand but the *wiring* between them is ours to write — see the §11.0 checklist. Where v5 credited Pipecat for a behaviour, read it as a requirement on our own code.

`[OFFICIAL]` **Silero VAD** — standalone, local CPU (v5 noted it as a Pipecat built-in; it is an independent package).
`[OFFICIAL]` **smart-turn-v3** — Whisper Tiny encoder + linear classifier, ~8M params, 8MB ONNX, BSD 2-clause; ~65ms on a standard 1-vCPU instance, as low as 12ms for the int8 CPU build; input 16kHz mono PCM up to 8s (truncate from the start if longer); runs only after VAD detects silence; default turn-stop strategy in current Pipecat, which is why v5 chose it — the model is independent of the framework.

### 9.1a `[MEASURED]` Turn detection as built (2026-09-19)

**Silero VAD works; smart-turn-v3 does not, and is not wired in.**

Silero runs over ONNX Runtime with no torch (this machine exports a global `PYTHONPATH` into
HARU's ROS venv, so pip torch collides with the Jetson build — see `brain/vad.py`). One trap:
the v5 ONNX needs **64 samples of previous context prepended to each 512-sample window**.
Omitting it raises nothing and returns ~0.001 for everything, including obvious speech.

`stop_secs = 1.5` is set from measurement, not preference. Longest pause *inside* a single
utterance across our recordings:

| clip | longest internal pause |
|---|---|
| c1_neutral | 0.38s |
| a2_happy | 0.70s |
| a1_tired | 0.80s |
| a3_anxious | 0.93s |
| **c2_suppressed** | **1.38s** |

All five recordings now detect as exactly one turn (68–98% of audio retained).

**★ The domain finding here matters more than the number.** The longest pause belongs to the
*emotionally suppressed* delivery — the take where the speaker is holding something back. That
is not a coincidence, and it is precisely the user this robot exists for: someone fighting back
tears pauses longer than someone reporting their day. **A silence threshold tuned on ordinary
speech will cut off exactly the people the robot is meant to serve.** 1.5s is the price of not
doing that, and it lands on every turn.

**smart-turn-v3 attempt, and why it stopped.** The model (`models/smart_turn_v3.onnx`, Whisper
Tiny encoder + linear head, 80×800 log-mel) would let `stop_secs` drop back to ~0.2s, buying
back over a second on every turn. Hand-rolling Whisper's mel in numpy did not work:

- Speech, silence and white noise all score 0.50–0.73 with no consistent ordering.
- Tried front-padding and back-padding short audio, and realistic inputs that end in a pause
  (the shape the model actually sees in use). Neither separates finished from mid-word speech.
- The first validation *passed* this broken implementation, because it only compared a full clip
  against a truncated one and accepted differences of 0.001. A wrong mel produces confident
  numbers, not errors. `brain/test_smart_turn.py` is now the strict gate and fails.

Stopped deliberately rather than guessing further: without a reference implementation to compare
against, the search is unbounded, and a turn detector that is subtly wrong is worse than none.
Tracked as Q19.

### 9.1b `[MEASURED]` smart-turn — preprocessing solved, validation is not (2026-09-19)

Revisited because EXP-8 showed this is the only lever that matters (§13.4). Diffing against
the reference (`pipecat-ai/smart-turn`, `inference.py`) found two real bugs:

1. **Missing `do_normalize`** — Whisper's extractor normalizes the *waveform* to zero mean and
   unit variance before the mel. Without it the features sit on a scale the model never saw.
2. **A sigmoid on an output that is already a probability**, despite the ONNX tensor being
   named `logits`. This is what produced the earlier "no opinion" readings: everything landed
   in 0.50–0.73, which is exactly sigmoid(0)–sigmoid(1). The model had been answering 0.0 and
   1.0 all along, and the earlier conclusion that "silence scores higher than speech, so the
   mel is wrong" was drawn from doubly-squashed numbers.

After the fixes, `log_mel()` matches `WhisperFeatureExtractor(chunk_length=8, do_normalize=True)`
**exactly** — max absolute error 0.0000 over all 80×800 values, mel filterbank identical.
Front-padding was already correct (their `truncate_audio_to_last_n_seconds` pads at the start).

**It is still not wired in.** On our five recordings it separates finished from mid-word speech
only 2/5, and inverted on a1_tired (complete 0.062, mid-word 0.235). Published accuracy on
Korean is **96.96%, the best of 23 languages**, so the model is not the problem and neither is
the preprocessing. The remaining suspect is the test material: the recordings are scripted lines
read aloud by one speaker, while smart-turn is trained on spontaneous conversational audio
(`human_convcollector`, etc.). Read speech does not carry the same turn-final prosody.

Enabling it on these numbers would be reckless in a specific way: the failures point the
dangerous direction. A mid-word cut scored 0.893 — acting on that cuts a user off mid-sentence,
which is the exact failure §9.1a's 1.5s exists to prevent.

**Unblocked by data, not code**: a handful of spontaneous conversational recordings — someone
talking naturally, including pauses that are *not* turn ends. Q19b.

### 9.1c ✅ `[MEASURED]` Q19b resolved — and it overturns the latency plan (2026-09-19)

Six spontaneous recordings (25–32s each, ~3 minutes total) with self-labelling built in:
every pause where the speaker *continued* is ground truth "not finished", the end of each
file is "finished". No scripting — that was the flaw in the earlier set.

**Finding 1 — the headline, and it is not about smart-turn.** Natural thinking pauses reach
**2.98s**, more than double the 1.38s seen in scripted reads. `stop_secs = 1.5` therefore cuts
the speaker off at **19 of 22 pauses** — roughly one false interruption every ten seconds of
natural speech. The current setting is not slightly conservative, it is unusable, and this was
invisible until spontaneous audio existed.

**Finding 2 — smart-turn works on this material.** Median 0.019 for "still going" versus 0.880
for "finished". The earlier 2/5 was the test set, exactly as suspected (§9.1b).

**Finding 3 — direction matters more than accuracy.** The obvious wiring (end the turn *early*
when confident) measured **worse than no smart-turn at all**: 24 wrong splits against 19,
because it can only add endpoints, never prevent the timer from firing. Inverting it to a
**veto** — when the timer expires, ask, and extend the wait if the speaker is judged still
going — is what produced the gain.

**Finding 4 — the honest comparison, which shrinks the win.** A fixed longer timer does nearly
as well:

| configuration | wrong splits | wait at a real ending |
|---|---|---|
| timer 1.5s (current) | **19** | 0.80s |
| timer 3.2s, no smart-turn | 0 | 2.51s |
| **veto @0.7, max 4.0s** | **1** | **2.05s** |

smart-turn buys 0.46s over simply raising `stop_secs` to 3.2. **My earlier projection that this
would take perceived latency from 1.98s to ~0.65s was wrong** (§13.4). That assumed a working
smart-turn would permit a *short* timer; on spontaneous speech the early-end direction is what
fails, so the wait stays.

**Shipping the veto anyway**, for one reason: the fixed timer's clean score is an artifact of
this sample's longest pause being 2.98s. One longer pause and it interrupts; the veto adapts.
It also degrades to timer-only if the model fails to load.

**Latency now has to be re-measured** — EXP-8's 1.98s was taken on scripted clips whose pauses
were short. Expect roughly 3.2s on natural speech, whichever endpointing is used. That is the
real number, and it is worse than previously recorded.

### 9.1d `[IMPLEMENTED]` Speculative generation — take the brain off the critical path

Direction check after §9.1c. The wait cannot be shortened without interrupting people, and
smart-turn only bought 0.46s of it. But nothing says the brain has to be *idle* during that
wait.

So: at 0.5s into a pause the brain starts generating and synthesizing the reply **into a
buffer**. It is never sent early — the robot does not speak one millisecond sooner. When the
turn is confirmed, the answer is already made and goes out immediately, which removes the
brain's ~0.5s (§13.4) from what the user waits through. If the speaker resumes instead, the
work is thrown away.

Why this is safe where shortening the wait is not: being wrong costs GPU time, not an
interruption. And the machinery already exists — `Turn.cancel()` stops generation and
synthesis together, measured at 70ms (§13.3).

The buffered reply is only used when the confirmed turn's audio is **byte-identical** to what
was speculated on. Any resumed speech changes it, and then it is discarded rather than
answering a question the user did not finish asking.

Per-session switch `speculate` in `hello`, alongside `backchannel`.

### 9.1e ✅ `[MEASURED]` Fast path — real-time is the goal, so stop waiting out the timer

§9.1c left every turn waiting ~1.2s to be sure, which is polite and not real-time. §4.2's
argument applies directly: interruption reaction in a cascade is already fast (ours is 70ms,
§13.3), so **being wrong is cheap and being slow on every turn is not.**

smart-turn is therefore asked **twice** per pause, for opposite purposes:

| | when | threshold | effect |
|---|---|---|---|
| fast path | 0.6s into the pause | ≥ 0.95 → end now | most turns answer at ~0.77s |
| veto | when the 1.5s timer expires | < 0.7 → keep waiting, to 4.0s max | protects the pauses it is unsure about |

Measured over the six spontaneous recordings (~3 minutes):

| configuration | wrong splits | wait at a real ending |
|---|---|---|
| timer 1.5s (original) | **19** | 0.80s |
| veto only | 2 | 1.22s |
| **veto + fast path @0.95** | **3** | **0.77s** |

Same speed as the original timer with **six times fewer interruptions**, and with speculative
generation (§9.1d) the reply is already synthesized, so perceived latency is roughly that 0.77s.

**How much trailing silence smart-turn sees turned out to matter as much as the threshold.**
Strip it and the model has no pause to judge; hand it the whole accumulated silence and
everything reads as finished (the veto fired on 1 of 22 pauses with a 0.3s tail, 6 of 22 with
the lot). It is now given a fixed 0.3s tail, matching how the reference calls it.

**The cost, stated plainly**: at 0.95 the fast path splits `c2_suppressed` — the emotionally
suppressed take, which §9.1a identified as exactly the user who must not be cut off. Barge-in
recovers in 70ms, so it is brief rather than harmless. **`fast_confidence = 0.99` disables the
fast path** and returns to 1.22s with `c2_suppressed` intact; that is the one-line revert if
live use shows the interruptions matter more than the 0.45s.

`brain/test_vad.py` now asserts a **split budget** (4) rather than perfection, so a regression
toward the 19-split behaviour fails loudly while the accepted trade passes.

### 9.2 Barge-in
`[OFFICIAL]` Silero VAD + smart-turn-v3 together give accurate, low-latency turn start/stop signals. v5 relied on Pipecat to turn those signals into interruption logic that yields to a real interruption without reacting to brief mid-sentence pauses — **we now implement that logic ourselves** (§11.0 items 1 and 4).

`[OFFICIAL]` Mirror Gemini's interruption contract: on detected interruption, cancel and discard ongoing generation; retain only what was already sent; **the client stops playback and clears its queue.**

**AEC IS MANDATORY.** `[REPORTED]` An NVIDIA forum thread titled "Echo/Feedback Issue — Speaker Audio Being Captured by Microphone on Jetson Orin" confirms this is a real, recurring problem on this exact hardware.
- MVP: headset sidesteps it
- Production: hardware AEC mic array (speaker + AEC on the same device)
- Keep playback buffer 100–200ms and **flush on interrupt** — the classic bug is a cancelled generation still playing for seconds

### 9.3 Proactive audio
`[REPORTED]` Gemini's proactive audio "lets a developer instruct the agent to only respond when directly addressed or when a reply is contextually relevant, rather than responding to every utterance."

### 9.3a ✅ `[MEASURED]` EXP-7 — the wording in v5 does not work (2026-09-19)

v5 proposed this, tagged `[UNVERIFIED]`:
```
사용자가 혼잣말을 하거나, 생각을 정리 중이거나, 너에게 말한 게 아니면
정확히 <SILENT> 만 출력해. 그 외에는 평소처럼 2-3문장으로 답해.
```
Measured **4/6**. It answered all three addressed utterances correctly but stayed silent
on only one of three self-talk cases — and that one was literally prefixed "(혼잣말)",
which real audio will never carry. Left as written, the robot would interrupt someone
thinking out loud.

**Naming the concrete shapes of self-talk gets 6/6** (`brain/test_gates.py`):
```
너에게 직접 말을 건 것이 아니면 응답하지 마라. 다음은 모두 혼잣말이므로 정확히 <SILENT>만 출력한다:
- 할 일이나 순서를 스스로 정리하는 말 ("이걸 먼저 하고... 아니다")
- 뭔가를 떠올리거나 메모하듯 되뇌는 말 ("아 맞다, 우유 사야지")
- 스스로에게 묻는 말, 말끝을 흐리며 생각하는 말
너를 부르거나, 너에게 감정을 털어놓거나, 너에게 질문하면 평소처럼 2-3문장으로 답한다.
```
A "default to silence, when unsure stay quiet" variant also scored 6/6 and was
**rejected**: for a companion robot, staying silent when spoken to is a worse failure than
answering something not addressed to it.

`n=6`, so this is a direction, not a validated rate. Two caveats worth holding:
- Tested on **text**. Production is audio, where prosody should help — self-talk is
  quieter and trails off, and §12.4 showed E4B does hear tone. Untested: no self-talk
  recordings exist yet. Worth one when recordings are next made.
- The wording belongs to the **robot's persona**, not the brain. The brain only honors the
  token (`SILENT_TOKEN` in `brain/pipeline.py`): it strips it and synthesizes nothing, so
  the turn completes silently. Verified TTS is never called.

Later: a pre-LLM classifier so silence costs nothing.

### 9.3b ✅ `[MEASURED]` Emoji suppression works with one line

§12.4 flagged that E4B emits emoji, which TTS would read aloud or choke on. Measured:
**3/3 replies contained emoji without an instruction, 0/3 with `이모지는 절대 쓰지 마.`,
including through the audio input path.** One prompt line is enough; no output filtering
needed. Kept as a gate in `brain/test_gates.py` so a model or prompt change cannot
quietly undo it.
Bonus: `[OFFICIAL]` SenseVoiceSmall's AED can distinguish laughter/cough/ambient noise from speech.

---

## 10. KOREAN EMOTION DATA — ⚠️ NOT NEEDED (see §12.5)

This data existed only to retrain an SER classifier for §7, which EXP-13 removed. **Do not submit the
AI-Hub request** unless the audio path fails in live use and §7 is reinstated.


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
| 5 | **Split audio into ≤30s clips before sending** — unconditional, EXP-13 won (§12.4) | Measured in §12.3: a longer clip is truncated **silently** — no error, no warning, nothing in the response reveals the loss. A user talking for 45s loses their last 15s and nobody ever finds out. |

### 11.1 STT leg
Run STT only on VAD-delimited segments (not continuously), and keep the §11.0-3 compensation buffer.

**v6 correction — this rule no longer applies where v5 put it.** There is no separate STT model any more
(§12.5); audio goes to E4B through vLLM's OpenAI-compatible `input_audio` field, which *requires* a
container and takes base64 WAV. The rule's original point was to avoid pointless wrapping added for cloud
upload APIs — but that request shape is now our own local transport, so the wrapping is not optional.

Where the rule still means something is **the robot↔brain wire (§11.5): send raw PCM16 frames there, not
WAV.** The brain wraps into WAV once, at the vLLM call boundary. See §20 rule 4.

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
| robot → brain | `{"t":"hello", "system":…, "tools":[…]}` | the `LiveConnectConfig` equivalent: persona + tool schemas, sent once at session open |
| brain → robot | `{"t":"ready"}` | handshake ack; no Gemini counterpart |
| robot → brain | binary frame: raw PCM16 16kHz mic audio | `send_realtime_input(audio=Blob(...))` |
| robot → brain | `{"t":"text", "text":…}` | `send_client_content(turns=Content(...), turn_complete=True)` |
| robot → brain | `{"t":"tool_result", "results":[{id,name,result}]}` | `send_tool_response(function_responses=[...])` |
| brain → robot | `{"t":"audio", "rate":…, "bytes":…}` then a binary frame of PCM | `message.data`. The rate header has no Gemini counterpart and is needed because Piper emits 22,050Hz while the robot's playback path was built for Gemini's 24kHz (§8.3) — the client must be told, not assume. Resampling could remove it later. |
| brain → robot | `{"t":"interrupted"}` | `server_content.interrupted` |
| brain → robot | `{"t":"transcript","role":"user"\|"model","text":…}` | `server_content.input_transcription` / `output_transcription` |
| brain → robot | `{"t":"tool_call","calls":[{id,name,args}]}` | `message.tool_call.function_calls`. **`args` is a dict, not a JSON string** — `launcher.py` splats it as `fn(**(fc.args or {}))`. Emitting `arguments` as a string (as the OpenAI API does) would break the robot. |
| brain → robot | `{"t":"error","detail":…}` | no Gemini counterpart; one failed turn must not kill the session |
| brain → robot | `{"t":"turn_complete"}` | `server_content.turn_complete` |

`websockets` distinguishes binary and text frames natively — no framing code required.

**Client shim — `client/local_live.py`, written and verified without the robot.** It duck-types the
`google.genai.types` objects launcher constructs (reads `.data`, `.parts[0].text`, `.id/.name/.response`)
instead of importing the SDK, so the brain keeps no dependency on Google's client library — which is not
installed on this machine anyway. `receive()` is an async generator that **ends at each turn boundary**
(Gemini's semantics — launcher re-enters it in a `while` loop; get this wrong and the conversation stops
after one turn). `go_away` and `session_resumption_update` are present and always None, because launcher
checks them on every message and its reconnect loop stays load-bearing over Tailscale (§20 rule 17).

Robot-side diff stays one line:

```python
async with local_live.connect(BRAIN_URI, config=config) as session:   # was client.aio.live.connect
```

`client/test_local_live.py` runs launcher's recv_loop **structurally unchanged** against a live brain —
same fields, same `fn(**(fc.args or {}))` splat, same `turn_complete` exit — and passes.

**Tool schemas**: `launcher.py:302-339` builds `tools` as a list of plain Python callables and relies on
google-genai deriving declarations from signatures. The brain needs those schemas, so the robot must send
them at handshake (derive from `inspect.signature` + docstring).

**Reconnection `[MANDATORY]` for phase 2 — ✅ implemented and verified (2026-09-19).** Over Tailscale the
link will drop, so `launcher.py`'s existing `connection_manager()` reconnect loop stays load-bearing — do
**not** treat it as dead code. Gemini's `go_away` / `session_resumption_update` stay unused, but the brain
holds conversation state and resumes it, or a dropped link wipes the conversation.

How: the client sends a `session_id` in `hello`; the brain keys sessions by it and reuses on reconnect,
adopting the *new* persona from that handshake (launcher rebuilds it, and by then it may contain the user's
name learned via `remember_fact`). `ready` reports `resumed` and the number of turns carried over.

Two decisions worth keeping:
- The id is **one UUID per robot process**, generated at shim import. It cannot live on the connection
  object, since launcher calls `connect()` afresh each reconnect.
- It must **not** be derived from the persona. Hashing the system prompt looks tempting and fails exactly
  when it matters: the persona changes the moment `remember_fact` learns the user's name, so the key would
  rotate mid-conversation and drop the history.
- `SESSIONS` is bounded (8, oldest evicted). An always-on server otherwise accumulates them forever — the
  same unbounded-growth bug as the conversation history in §13.1.

Verified by `client/test_reconnect.py`: talk, disconnect, reconnect, then ask what was said before the
drop. The model answered "밤을 새워서 피곤하다는 내용이었죠" — the conversation survived.

---

### 11.6 `[MANDATORY]` Which side owns the bug — diagnose before editing either

Two Claude Code instances now work on this system: one on the AGX (this repo, the brain) and one on
the robot (`MOTI-HRI`). **Neither may edit the other side's code.** Added 2026-09-21 after the
robot-side instance began editing `client/local_live.py` in place — an understandable move that was
wrong for a reason worth writing down: that file *runs* on the robot but is *owned* here, and the
eight-test suite that covers it runs here too. A fix made there is untested, diverges from the repo,
and is destroyed by the next `scp` the integration doc instructs.

**Ownership is not negotiable per-bug:**

| file / concern | owner | the other side does what |
|---|---|---|
| `brain/`, `client/local_live.py`, `scripts/`, this spec | **brain (AGX)** | reports the symptom |
| `launcher.py`, `core/`, `media/`, robot `.env`, AEC, motors, camera | **robot** | reports the symptom |

`client/local_live.py` is the trap: it is deployed to the robot but belongs to this repo. Fixes go in
here, get tested here, get pushed, and the robot re-copies. Never the reverse.

#### The brain log is the arbiter

Most cases resolve mechanically rather than by judgement. **Did the brain log say it sent the thing?**
If yes, the fault is downstream (robot). If no, upstream (brain).

| symptom | side | why it is decided, not guessed |
|---|---|---|
| `client silent Ns — closing turn anyway` | **robot** | the brain stopped *receiving* audio. Nothing the brain does can cause that |
| `barge-in: turn cancelled` repeating while the user is silent | **robot** | echo re-entering the mic — AEC (§18.1) |
| `turn failed` + traceback | **brain** | an exception in our pipeline |
| tool markup spoken aloud | **brain** | `strip_tool_calls` (§13.9) |
| turn cut mid-sentence, or waits too long | **brain** | VAD thresholds (§9.1e) |
| speech ~9% fast and high-pitched | **brain** first (resample), then robot if `OUTPUT_RATE` was changed |
| tools never execute | brain log has `tool_call`? **yes → robot**; **no → brain** |
| user transcript missing from the saved log | brain log has `transcript role=user`? **yes → robot**; **no → brain** |
| robot hangs with no error anywhere | did the brain send `turn_complete` or `error`? **yes → robot**; **no → brain** (§13.7) |

#### When it is not clear: change nothing, ask

If the table does not decide it, **neither side edits anything.** Report to the user with:
1. the exact symptom (traceback verbatim, or the brain-log lines with timestamps),
2. what was already ruled out and how,
3. the one or two hypotheses, and which side each would live on.

The user routes it to the right side. A speculative fix on the wrong side is worse than waiting: it
adds an untested change to a system where, as of 2026-09-21, **nothing has been verified against real
hardware** — so a second variable makes the first one unfindable.

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

### 12.3 `[MEASURED]` EXP-13 partial result — 3 of 4 criteria cleared without recordings

Run with `scripts/probe_audio.py` using generated tones. Tones cannot test comprehension, but they fully
test the audio *path*, token accounting, and the ceiling — so criteria 2, 3 and 4 were decided before any
Korean speech existed. Server: E4B bf16, vLLM 0.19.0, persona already prefix-cached.

**The audio path works with zero extra code.** vLLM's OpenAI endpoint accepts `input_audio` (base64 WAV)
for E4B and the model describes the sound correctly. No patch, no custom serving.

| Clip | prompt_tokens | audio tokens | tok/s |
|---|---|---|---|
| 5s | 146 | 127 | 25.4 |
| 10s | 271 | 252 | 25.2 |
| 30s | 771 | 752 | 25.1 |
| **45s** | **771** | **752** | — |

**Criterion 2 — TTFT. Passes.** Measured with the real persona cached:

| Request | TTFT | prompt_tokens |
|---|---|---|
| text only | 0.182 s | 18,364 |
| + 10s audio | 0.473 s | 18,616 |
| + 30s audio | 1.008 s | 19,116 |

Audio prefill costs roughly **29 ms per second of speech**. A typical 5–10s turn adds 0.15–0.29s, keeping
TTFT under 0.5s. Only a worst-case 30s turn pushes past the 0.7s threshold, at 1.0s. For comparison the
Whisper path must *finish transcribing before the LLM can start at all*, and §6.1 cites faster-whisper
needing seconds for a 20s clip on Orin-class hardware — so audio-direct is very likely the faster path,
not the slower one. Confirm against a real Whisper run only if EXP-13 is otherwise killed.

**Criterion 3 — the 30s ceiling. Mitigated, but it hides a trap.**

Two findings, and the second is the important one:
1. **Multi-clip segmentation works.** Two 20s parts in one prompt produced **1,004 audio tokens** — exactly
   2 × 502, with no truncation. The ceiling is per clip, not per request, so splitting a long turn into
   ≤30s parts clears it. **The "delete two models" win survives.**
2. 🔴 **A single clip past 30s is truncated silently.** 45s produced the same 752 tokens and the same
   response as 30s. **No error, no warning, no field in the response indicates loss.** A user who talks
   for 45 seconds simply has their last 15 seconds discarded, invisibly — far more dangerous than a
   loud failure, because nothing in the system would ever report it.

`[MANDATORY]` The brain must split captured audio into ≤30s clips before sending. Never hand the model a
single clip longer than 30s and assume it was heard. This belongs with the §11.0 checklist.

**Criterion 4 — context erosion. Passes.** 30s of audio moved prompt_tokens from 18,364 to 19,116, i.e.
752 tokens against a 128K window and a 67,712-token KV budget. Confirms §12.2's analysis with live numbers.

### 12.4 ✅ `[MEASURED]` EXP-13 **WON** — criterion 1 cleared on real Korean speech (2026-09-19)

Seven recordings, `scripts/exp13_korean.py`, `temperature=0`, real persona. Transcripts and recording
conditions in `docs/exp13_recordings.md`. Recordings were uncompressed PCM16 48kHz, so no codec
processing could have flattened prosody; the c-set's RMS spread was 10.6 dB (c2 quietest at −33.6,
c3 loudest at −23.0), confirming the tone dynamics survived and comparison C was valid.

**Comprehension matches a perfect transcript.** Audio-direct replies were as accurate and specific as
replies to the reference text — e.g. for a1 it produced "며칠 동안 밤을 새우셨다니… 눈이 감길 정도로",
recovering the exact content. Note the baseline here is a *perfect* transcript, which is stronger than
any real Whisper would deliver, and audio-direct still matched it.

**The tone channel works, and it is the whole argument for this path.** Identical wording, prosody only:

| Input | Reply |
|---|---|
| text only (what Whisper would give) | "괜찮다고 말해주니 마음이 놓이네요…" — takes the words at face value |
| c1_neutral | similar to text — face value |
| **c2_suppressed** | **"괜찮다고 말했지만, 사실은 많이 신경 쓰이고 힘든 마음이 느껴져요"** |
| c3_bright | "많이 긴장하셨군요. 그래도 잘 해내셨다니 정말 다행이에요" |

c2 is the result that settles it: the model **contradicted the literal words** because it heard the
suppressed sadness. That is precisely the affective-dialog behaviour §7 was going to add a separate
emotion2vec model to recover, and it comes free.

⚠️ **Caveat — tone can drive fabrication.** c3_bright invented context that was never said
("잘 해내셨다니" — nothing in the utterance mentions succeeding at anything). The tone channel is real,
but the model will over-infer a *situation* from prosody. Watch for this in live use; it is a
hallucination risk the text-only path does not have. Not disqualifying, but it belongs in any write-up.

**30s truncation and the split fix, both confirmed on real speech.** `b1_long` is 52.5s:

| Send method | prompt_tokens | What the reply shows |
|---|---|---|
| single clip | 823 | only first-half content |
| split into 2 × ≤30s | 1,387 | reaches "작은 진전… 희망" and "이렇게 이야기 나눠주셔서" — the *end* of the utterance |

Token delta 564 ≈ 22.5 s × 25 tok/s — exactly the discarded tail. The split reply demonstrably heard
the second half; the single-clip reply did not, and nothing in the response said so.

**Sample size is small** — one speaker, one session, three utterances plus one long and one tone triple.
Enough to decide the architecture (which is what §12 experiments are for, §0-A row 6), not enough to
publish. Re-run properly if a paper happens.

### 12.7 ⚠️ `[MEASURED]` EXP-12 — backchanneling has almost no window (2026-09-19)

Built and measured. The conclusion is not the one expected: **EXP-12 is not independent of
Q19b, it is blocked by the same thing.**

Implementation is cheap and works — four short acknowledgements ("응", "응...", "어", "음")
pre-synthesized at startup (0.38s total), trimmed of Piper's padding from 0.81s down to
0.23–0.34s, never repeating consecutively, emitted with `kind:"backchannel"` so the robot
does not log them as things Moti said.

The problem is *when* to fire. Against the real pause distribution:

| trigger after | fires mid-utterance | fires at turn end |
|---|---|---|
| 0.6s | **6** | 5 |
| 0.8s | 3 | 5 |
| 1.0s | 1 | 5 |
| 1.2s | 1 | 5 |
| **1.4s** | **0** | 5 |

Six mid-utterance fires across five clips at 0.6s — the robot talking over someone who
paused to think. Overlap only vanishes at **1.4s**, because the longest intra-utterance
pause we have is 1.38s (§9.1a). And `stop_secs` is 1.5s. **The safe window is 0.1s wide.**

So the acknowledgement lands at 1.4s against a reply at ~1.98s: it covers 0.58s of a 1.98s
silence, not the 1.4s the idea promised. Real, but marginal.

The root cause is identical to Q19: **without semantic turn detection a pause is just a
pause**, and any threshold that avoids interrupting people is necessarily too late to help
much. Note the corollary — if Q19b lands, `stop_secs` drops to ~0.2s, the reply arrives
around 0.65s, and backchanneling becomes largely unnecessary. **Q19b is the answer to both
problems; EXP-12 is a workaround for a window that barely exists.**

Default set to 1.4s (safe). Per-session overrides (`backchannel`, `backchannel_after` in
`hello`) exist because restarting the brain costs ~33 minutes, which makes server-level
flags useless for an A/B.

### 12.6 `[MEASURED]` Two gaps the deletion opened, both closed

Removing the STT leg quietly broke something v5 had counted as a *strength*. Found by re-auditing
against §14, not by anything failing.

**Gap 1 — the user-side transcript.** `launcher.py` feeds `server_content.input_transcription` into
`turn_user` -> `session_history` -> `report_manager.save_conversation_log()`. Without an STT leg nothing
produces it, which would have silently broken three things: the conversation log, the 마음처방전 report,
and the quiz's `user_spoke` guard (added 2026-08-10 to stop the model calling `submit_guess` when the
user had not actually spoken).

Closed: **E4B transcribes it itself, word-for-word exact** on the a1 recording — output identical to the
reference transcript, character for character.

`[MANDATORY]` design note for Stage 4: order the prompt as **[system][audio][instruction]**, not
[system][instruction][audio]. The audio prefill is the expensive part (~29ms per second of speech,
§12.3); putting it before the instruction makes `[system][audio]` a shared prefix, so the reply call
and the transcription call hit the same cache entry instead of paying the audio prefill twice.

🔴 **That last clause is false as implemented, found 2026-09-19 by reading the code against this
paragraph.** The transcription call had to be given **its own system prompt and no history** — with the
persona it obeys "always call `set_emotion` first" and returns `set_emotion(tired)\n며칠 내내…`, putting
tool syntax into the conversation log as words the user never said. Necessary fix, but it means the two
calls are `[transcriber][audio]` and `[persona][history…][audio]`: **they diverge at token 0, so there is
no shared prefix and the KV cache cannot dedupe the audio between them.**

The ordering itself stays — it is still right within each call, and harmless. What dies is the reason
this paragraph gave for it, and one consequence is worth carrying to §13.7: **vLLM's multimodal processor
cache was the only thing deduplicating audio work across those two calls.** Disabling it therefore costs
a genuine second encode, which is consistent with the pipeline floor moving 1.47s → 1.61s (~0.26s
predicted for that 9s clip, §12.3).

Fourth instance of the pattern §14 now warns about: a later fix silently invalidated an earlier
paragraph's premise, and nothing failed.

**Gap 2 — function calling was never tested**, only asserted by §14. The entire robot depends on it
(`remember_fact`, `set_emotion`, `play_manual_motion`, `express_gesture`, quiz tools).

Closed: verified on this server. Text input produced three correct calls with correct arguments
(`set_emotion(happy)`, `remember_fact(name)`, `remember_fact(major)`), `finish_reason: tool_calls`.
**It also fires with audio input** — the a1 tired recording produced `set_emotion(sad)`, which is the
real production shape.

### 12.5 Consequences — two models leave the design

EXP-13 passing all four criteria means:
- **faster-whisper is out of the pipeline** (§6). The aarch64 `ctranslate2` source build — risk #1 in
  v5's register — never has to happen.
- **emotion2vec / SenseVoice SER is out** (§7). The unmeasured-Korean-SER risk is closed.
- **AI-Hub emotion datasets are no longer needed** (§10) — they existed only to retrain an SER classifier.
- EXP-4, EXP-5, EXP-6 are moot; they only existed to choose and validate those two legs.
- §11.0 item 5 (≤30s clip splitting) is now **unconditionally mandatory**, not conditional.

Keep §6 and §7 in this document as the documented fallback if the audio path later fails in live use.

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

1. ✅ **Voice-shift buffer: 700ms — removed.** The robot's `VOICE_SHIFT_BUFFER_MS` was **700ms** (raised
   from 500ms after reported crackling), sitting on top of playback — 3.5–7× the 100–200ms §9.2 calls for,
   delaying both first audio and interrupt flush. It existed only to support the pyworld pitch/formant shift
   (+3.5st/×1.12) that made Gemini's adult `Zephyr` voice sound younger. **Decision 2026-09-19: use Piper's
   voice as recorded, no pitch shift** (§8.3) → set `ENABLE_VOICE_SHIFT=false`, the 700ms is gone, and the
   robot no longer needs `pyworld` at all — one fewer aarch64 source build on that side.
2. **Network hop.** Phase 1 (same LAN, wired GbE) should be ~1ms and negligible, but audio buffering across
   the link is not — measure it, don't assume. Phase 2 (Tailscale) must be measured separately, and
   distinguish direct vs. relayed (§18). Note the tension: a lossy remote link wants *more* jitter buffer,
   while §9.2 wants *less* playback buffer. Resolve with measurement, and let the degraded mode be explicit.

Bandwidth, for reference: 16kHz mono PCM16 upstream ≈ 256 kbps, 24kHz downstream ≈ 384 kbps. Not a
constraint on any realistic link, including Tailscale.

### 13.2 `[MEASURED]` First end-to-end voice-to-voice (2026-09-19, Stage 4 spine)

Real recording in (`a1_tired.wav`, 9.0s of the developer's speech), synthesized speech out.
Persona cached, Piper loaded, `brain/test_pipeline.py`.

| Stage | Time from turn end |
|---|---|
| first LLM token | **0.20 s** |
| **first audio out** | **1.47 s** |
| user transcript (off the critical path) | 3.55 s |

**Beats the §4.2 reference point**: `duet` reports 2.169s from final speech end to first server
audio; this is 1.47s on a Jetson. Not comparable to Gemini Live, but the §13 reference floor for
small-device pipelines (2–3s full loop) is cleared.

**Found by measuring, not by reasoning**: the first version ran transcription *before* generating and
cost **3.96s** to first audio — 2.4s of pure silence waiting for a transcript that only the
conversation log needs. Running it alongside the reply and collecting it before `done` (which is when
`launcher.py` actually reads it) cut time-to-first-audio 2.7×. Transcription itself got *slower*
(2.38 → 3.55s, it now competes for the GPU) and that is the right trade: it is off the path the user
waits on.

### 13.3 `[MEASURED]` Full round trip over the wire (2026-09-19, Stage 4)

`brain/server.py` + `brain/fake_robot.py`: mic PCM streamed continuously in real time,
turn boundaries decided by the brain's VAD, reply audio and tool calls streamed back.

| | measured |
|---|---|
| silence-end -> first reply audio | **0.24–1.29 s** across clips |
| barge-in: user speaks -> `interrupted` at the robot | **0.07 s** |

Barge-in beats §4.2's reference badly — `duet` reports 198ms for the same reaction, and
that paper calls fast interruption the cheapest source of perceived liveness. Note what
is *not* in these numbers: `stop_secs = 1.5` (§9.1a) still elapses before the brain even
considers the turn finished, so felt latency is that plus the figure above.

### 13.4 ✅ `[MEASURED]` EXP-8 — voice-to-voice latency as a distribution (2026-09-19)

`client/exp8_latency.py`, n=8 over four clips, measured from the client through the wire
(what the user experiences), fresh session per run so history does not confound it.

| | mean | p95 | min | max |
|---|---|---|---|---|
| **perceived** — speaking stops → first audio | **1.98s** | 2.81s | 1.22s | 2.81s |
| **brain** — turn detected → first audio | **0.48s** | 1.31s | −0.28s | 1.31s |

**76% of what the user waits through is the `stop_secs` silence, not this code.** The
cascade — audio in, E4B, sentence chunking, Piper, out over a WebSocket — costs about
half a second. The other 1.5s is the VAD waiting to be sure the person stopped talking.

**This reprioritizes Q19.** Fixing smart-turn was filed as an optimization worth ~1.3s;
it is in fact *the* latency lever, and nothing else on the list comes close. Quantization,
MTP, a faster TTS — all of them optimize the 0.48s. Semantic turn detection attacks the
1.5s. Perceived latency would go from ~1.98s to roughly 0.65s, which is a different class
of conversation.

Two notes on reading the numbers:
- `brain` min is negative because the clips contain their own trailing silence, so the
  VAD's countdown starts before the speaker's last word. Legitimate, not a timing bug.
- An earlier version of this script measured from "all trailing silence sent" and reported
  negative numbers throughout. The brain endpoints at `stop_secs` and begins replying
  while the client is still sending the rest of its silence — an artifact of how the test
  fed audio, not a result. Worth remembering when writing the next measurement.

### 13.5 ✅ `[MEASURED]` EXP-8 on real conversational turns (2026-09-19)

Measured over the wire on six **short spontaneous turns**, cut from the tail of each
monologue (the stretch after its last long pause) so each is one natural turn rather than a
scripted read or a multi-turn narration.

| | mean | p95 | min | max |
|---|---|---|---|---|
| speaking stops → first audio | **2.02 s** | 2.82 s | 1.42 s | 2.82 s |

Breaks down roughly as **turn close ~0.6s + pipeline ~1.4s**, and the pipeline is
first token 0.2s + generating the first sentence + TTS ~0.2s.

**So the bottleneck moved.** It was turn detection (§13.4: 76% of the wait); the fast path
(§9.1e) cut that to ~0.6s. What dominates now is **decoding the first sentence at 13 tok/s** —
roughly a second for ~15 tokens. Nothing downstream can start until that sentence exists.

This makes two previously-dismissed items relevant again:
- **EXP-3 / MTP speculative decoding** (§5.4.5). Dismissed because TTFT was comfortable; TTFT
  was never the problem. MTP attacks decode rate, which is.
- **Quantization** (§5.4.6). Same reasoning — bf16 was kept because TTFT passed. Decode is
  memory-bandwidth bound, so a q4 checkpoint should decode faster.

**Speculative generation helps less than expected** and the measurement says why: it starts at
0.2s into the pause while the fast path closes the turn at 0.6s, so 4 of 6 turns adopt it with
"0 ready" — nothing has been produced yet. It pays off on the veto path, where the wait is
seconds. Kept, because that is also when the user is waiting longest.

An earlier version of it made things **worse**: it buffered the entire turn and released
everything at confirmation, so first audio waited for the *last* sentence — 3.25s mean instead
of 2.02s. It now flushes what is ready and streams the remainder.

### 13.6 ❌ `[BLOCKED]` EXP-3 speculative decoding — the container cannot run either path

§13.5 identified decode rate as the remaining bottleneck, which makes EXP-3 the natural next
lever. Both ways of enabling it fail **at startup** on
`ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin`:

| approach | failure |
|---|---|
| `draft_model` with `google/gemma-4-E4B-it-assistant` | `transformers` in this image does not recognise model type `gemma4_assistant` — vLLM refuses the config before the engine starts |
| `ngram` (no draft model at all) | `ModuleNotFoundError: No module named 'numba'` in `v1/spec_decode/ngram_proposer.py` |

The irony is that the drafter is exactly what §5.4 rule 5 names, it exists on the Hub, it is
tiny (4 layers, 256 hidden, 183MB), and it downloaded fine. The blocker is the image, not the
model. Upgrading `transformers` inside an image NVIDIA built specifically for Gemma 4 risks the
support the image exists to provide, and adding `numba` means maintaining a derived image.

**Reverted; EXP-3 needs a rebuilt image, not a config change.** Worth revisiting if the image
is rebuilt for another reason. Two cycles were spent learning this, at ~33 minutes per vLLM
restart — after the first failure the config was validated in a throwaway container before
restarting, which is the right order and should be the habit.

Remaining levers on the ~1s first-sentence decode, in rough order of appeal:
- a shorter opening sentence from the persona (prompt-side, no infrastructure risk)
- ~~a q4 QAT checkpoint (§5.4 rule 6)~~ — **investigated 2026-09-19 and deferred; see §13.10**

### 13.7 🔴 `[MEASURED]` Two ways a failed turn became an infinite hang (2026-09-19)

Found while re-running EXP-8, which sat for 27 minutes producing nothing. Both causes are worth
keeping because neither announces itself.

**The vLLM side.** The engine died mid-run with
`AssertionError: Expected a cached item for mm_hash=...` and returned HTTP 500. Fixed with
`--mm-processor-cache-gb 0`.

⚠️ **The mechanism is a hypothesis, not a measurement — this was first written as though it were
established.** What is certain: our design has the reply call and the transcription call carry the
**same audio** (§12.6, deliberately, to share a prefix-cache entry) and run **concurrently** (§13.2), so
two in-flight requests hold an identical `mm_hash`. What is *not* established is that they race. An
`Expected a cached item` assertion fits **eviction** at least as well — an entry dropped while a queued
request still referenced it — and EXP-8 walks 8 distinct audio clips through that cache twice each.
Both readings are fixed by turning the cache off, so the distinction was never worth a 29-minute
restart to settle; it is only worth not asserting.

Note the shape of it either way: the two optimizations are individually correct and interact badly.
Serializing the calls would also fix it and was rejected — that is the change §13.2 measured at 2.7×
on first audio.

**The client side, and this one is worse.** `client/local_live.py` handled `{"t":"error"}` with
`continue`. The brain reported the failure correctly; the shim swallowed it and waited for a
`turn_complete` that was never coming. **`launcher.py` would hang identically** — its `recv_loop` also
exits only on `turn_complete`, so on the real robot this is a wedged conversation with no error
anywhere, requiring a restart. An error now sets `turn_complete` and is recorded in `last_error`.

The general rule, since §11.5 lists `{"t":"error"}` as having *"no Gemini counterpart"*: **every brain
message that can end a turn must end the turn.** A protocol addition that the client merely logs is a
hang waiting for the first backend failure.

**Re-measured after both fixes** (vLLM restart took 29 min, consistent with Q17):

| | before | after |
|---|---|---|
| EXP-8 perceived, n=8 | mean 2.02s / p95 2.82s / min 1.42s | **mean 2.11s / p95 3.33s / min 1.42s** |
| `test_pipeline` first audio (a1_tired) | 1.47s (§13.2) | **1.61 / 1.73 / 1.95 / 2.27s** over 4 runs |
| 500s from the mm-cache race | the run that prompted this | **0 across 8 EXP-8 runs + 4 pipeline runs** |

The crash is gone — every one of those runs fires the exact trigger (transcribe and reply concurrent on
identical audio) and none failed. Turn detection (3 splits, budget 4), smart-turn separation, emoji
suppression (0/3) and the `<SILENT>` gate (6/6) all still pass.

⚠️ **But do not read "2.11 ≈ 2.02" as unchanged.** The wire figure is within noise, yet four pipeline
runs put the *floor* at 1.61s against §13.2's 1.47s — the best case got worse, which noise does not do.
The plausible mechanism is exactly what the fix trades away: with the processor cache off, the reply call
and the transcription call each encode the audio (~29 ms per second of speech, §12.3, so ~0.26s for this
9s clip) instead of the second reusing the first, and the transcription competes for the GPU while the
reply is still decoding.

**Not measured, and deliberately so.** Isolating it needs a 29-minute restart, and the result could not
change anything: the cache cannot be left on — it crashes the engine. The real alternative is serializing
the two calls, which §13.2 measured at 3.96s vs 1.47s to first audio. Paying ~0.2s to keep 2.4s is not a
close decision. Revisit only if a rebuilt image (§13.6) fixes the cache behaviour itself.

### 13.8 `[MEASURED]` Did the EXP-3 restore cause either bug? No — and that is the uncomfortable part

Asked directly, checked in git rather than reasoned about.

| claim | evidence |
|---|---|
| The EXP-3 revert restored the serving config faithfully | `diff` of `run_vllm.sh` at 5a479f0 (pre-EXP-3) vs 6f1c691 (post-revert): **comments only**, `docker run` arguments byte-identical |
| The shim bug predates the restore by six hours | `git log -S` puts `elif kind == "error": continue` in 76dfc89 (10:14), the shim's **first** commit. The restore was 15:15 |
| No Python changed between the last clean EXP-8 and the crash | `git diff --stat 6963254 adc7ca5~1` → `CLAUDE.md`, a new doc, and `run_vllm.sh` comments. **Zero lines of code** |

So neither bug was introduced. Both were **latent**, and each needed a condition that had never occurred:

- The shim bug needed the brain to fail a turn. In six hours of testing the brain had never failed one,
  so the handler that dropped the failure was never exercised.
- The crash needed whatever engine-internal state a fresh vLLM process happened to reach. Same code,
  same flags, clean at 14:43 and crashing at 16:15 — the only variable is that the engine was restarted
  twice in between.

**The lesson is about the test suite, not the restore.** A failure path with no test is not "probably
fine", it is untested, and it stays green for exactly as long as nothing fails upstream. The crash was
the first real backend failure this project ever had, and it immediately found the one path that had
never run. `client/test_error_path.py` now closes it: it drives a brain error through the shim and fails
if `receive()` does not terminate.

Worth recording precisely because it is counter-intuitive: that test file also contains a **structural**
check (every `send(t=...)` kind in the brain is matched by a `kind == ...` in the shim), and **that check
does not catch this bug** — the `error` branch existed, it just did the wrong thing. Only executing the
path catches it. Coverage of the message table is not coverage of the behaviour.

### 13.9 🔴 `[MEASURED]` Everything before this was measured with a toy persona

Asked whether `docs/robot_integration.md` was ready to hand to the robot, and checking rather than
answering, the benchmark's own config turned out not to be the robot's:

| | EXP-8 / `fake_robot` used | the robot actually sends |
|---|---|---|
| system prompt | ~50 tokens | **18,344 tokens** (31,874 chars, §13.1) |
| tools | **none** | nine declarations |

`client/exp8_latency.py --real` now builds the real persona via
`MOTI-HRI/core/utils.build_persona_system_instruction` and passes stand-ins for launcher's nine tools
(the real ones are closures over motors and a quiz UI, so only their declarations can be reproduced).
It found three things, and two of them would have reached the user as sound.

**1. Latency is 2.65s, not 2.02s.**

| config | mean | p95 | min |
|---|---|---|---|
| toy (every figure before today) | 2.11s | 3.33s | 1.42s |
| **real persona + tools** | **2.65s** | **5.10s** | 1.42s |

Consistent with §13.0's 13.2 tok/s under the real persona against 14.4 with a short prompt. The floor is
unchanged — short turns still answer in 1.42s — but the mean and the tail both stretch.

⚠️ **Amended 2026-09-21: this metric is far noisier than any single figure suggests.** Four runs of the
same eight clips, same config, same server: **2.65 / 2.53 / 4.73 / 3.63s**. The floor is rock steady at
1.42s every time and two clips always land there, while the same three clips swing between 2 and 7.7s.
The driver is visible in the replies — first audio waits for the first *sentence*, and at temperature
0.7 the model sometimes opens with a short one and sometimes with a long one. Nothing about the
pipeline changed between those runs.

So: **quote a range, not a point — roughly 2.5–4.7s, and the fast path is 1.4s.** An n=8 run cannot
detect a change smaller than about 2s, which means it could not have told us whether §13.11's barge-in
fix cost anything. If a latency change ever needs to be *proven*, instrument `first_token` separately
from `first_audio` (both already live in `turn.marks`) — the first should be stable and would isolate
generation length from everything else.

**2. The first turn of a fresh brain costs 14.17s.** Measured, once, before the persona entered the
prefix cache. §13.1 documents pre-warming as the fix and tags it `[MEASURED]`, but **no code does it** —
grep for it and there is nothing. `launcher.py` sends a greeting trigger the moment it connects, so as
written the robot stands silent for ~14s in front of whoever walked up. Not a regression; a fix that was
designed and never built.

**3. 🔴 Tool markup reached TTS again — three new ways.** Five of eight replies carried it, and
`_speak()` synthesizes exactly the string it emits as the transcript, so the robot would have read it
aloud. None of it was reachable with the toy config, which declares no tools.

| shape | why it got through |
|---|---|
| `set_emotion(emotion="tender")` | `_find_bare_call` only looked for `name{`. The parenthesised form is as common as the brace form under the real persona |
| a live `<\|tool_call>call:` released as speech | the unterminated-bare-call branch returned `text[:i]` directly, **skipping `_holdback`**, so an open tool region in the prefix went straight out |
| `remember_fact(...)` dribbling out one character at a time | `_holdback` tested marker prefixes **shortest-first and returned on the first match**: text ending in "r" held back one character, released `remembe`, and the held "r" could never grow into the marker |

All three are fixed and `brain/test_pipeline.py` now drives four shapes across seven chunk boundaries.
The third is the interesting one — a latent bug with no connection to personas at all, reachable
whenever a stream delta ends on a one-character prefix. It survived every previous test because the toy
config passes no tool names, so the marker list was effectively empty.

**The lesson generalises past this bug.** A benchmark config that is easier than production does not
merely give optimistic numbers; it removes whole code paths from the test. Tool parsing is ~60 lines of
the pipeline and no test had ever run it against a declared tool list.

### 13.10 ⛔ `[DROPPED]` q4 quantization — off the roadmap, kept as a watch item

§13.5 and §13.9 both end by pointing at quantization, and it was written up as if a vendor path
existed. It does not. Searching the Hub for `gemma-4-E4B` quantizations (2026-09-19):

| what exists | usable here? |
|---|---|
| MLX 4-bit (3) | ❌ Apple Silicon |
| GGUF, including the QAT ones (19) | ❌ **llama.cpp only, and §5.4 rule 7 forbids llama.cpp on any audio path** (E2B audio is broken there on Orin). Audio is the primary input since EXP-13 |
| `unsloth/...-unsloth-bnb-4bit` | ⚠️ bitsandbytes — limited vLLM support, routinely breaks on multimodal |
| `Chunity/gemma-4-E4B-it-AWQ-4bit` | ⚠️ the only plausible candidate, an unvetted community upload |

§5.3's checkpoint table lists AWQ for **26B and 31B only**. There is no E4B entry, and the
`{model}-qat-q4_0-unquantized` pattern in §5.4 rule 5 is, as the name says, *unquantized* — it is the
MTP drafter base, not a memory-bandwidth win.

Stacked on top of that:
1. **What happens to the audio tower is unknown.** AWQ quantizes the LLM's linear layers; how a given
   uploader handled the audio encoder is up to them. This is HARU's W4A4 lesson (§13 escalation note)
   in a place where it now costs more — audio is the input, not a side channel.
2. **This container may have no AWQ kernels for sm_87.** EXP-3 (§13.6) already established that the
   image is narrowly built: no numba, a transformers that does not know `gemma4_assistant`.
3. **29 minutes per attempt**, and EXP-3 spent two of those learning point 2 the hard way.

**The decisive objection is not any of those — it is that there is no quality baseline.** EXP-1 was
never run (§12, Q2). Everything measured so far is speed and transcription accuracy; nothing measures
Korean empathetic conversation quality. Changing the model's numerics before that exists means a
degradation could not be distinguished from ordinary variance. **Quantize after EXP-1, not before.**

#### Decision 2026-09-22: stop planning for it

Re-examined after three live sessions and moved from "deferred" to **not on the roadmap.** Four things
turned the balance, and the first two are specific to this project rather than general caution.

**1. The capability quantization degrades is the one the project is built on.** Quantization does not
dull a model evenly — benchmark scores move a point or two while particular capabilities fall over, and
the ones that fall over are non-dominant languages, precise token-level instruction following, and
**multimodal encoders**. EXP-13 deleted two models from the design because E4B hears prosody through
exactly that encoder. Risking it to save a second is a bad trade for a robot whose entire argument is
that it noticed someone was not fine.

**2. We are already marginal on precisely those capabilities**, measured, at bf16:

| behaviour | today, unquantized |
|---|---|
| emits `[대화종료]` when asked to end | **40%** (§13.13) |
| fills `set_emotion`'s required argument | fails ~1 turn in 3 (§13.14) |
| Korean proper nouns | 조형민 → 조효형민 |

These are not comfortable margins to spend.

**3. The free lever is about as strong as the risky one.** First audio waits for the first *sentence*,
so the term is `tokens × (1 / decode rate)`. Quantization attacks the second factor, maybe 3×. Telling
the persona to open with a short sentence attacks the first, and a 30-token opener versus a 12-token
one is 2–3×. **Comparable effect, zero quality risk, zero infrastructure, one line in a prompt that is
already being edited.** Try that before spending anything.

**4. For a thesis, bf16 is the more defensible number.** "Here is the honest latency of an unquantized
open model on this board" reproduces; "here is a number from an unvetted community AWQ whose audio
tower we did not audit" does not. Reproducibility is one of the stated reasons for going local at all
(§1).

**Reopen only if all three hold**: an official or vendor-supported quantized E4B for vLLM appears
(not a community upload); live use shows latency is the *top* complaint rather than one of several; and
a quality baseline exists to measure the damage against. Until then this is a watch item, not work.

**Knock-on**: EXP-1 was promoted partly because "no numeric change to the model can be judged without a
quality baseline". That rationale goes away with this decision. EXP-1 stays, for a better reason — it
is the project's actual research question (is a local model good enough for Korean empathetic
conversation?), not a gate on an optimization we are no longer pursuing.

### 13.11 🔴 `[MEASURED]` Barge-in did nothing on the real robot — the fake one never played audio

First live session (2026-09-21, 2.5 minutes, reported from the robot side). Everything completed:
handshake, ten tools, greeting, tool round trips, conversation log, clean shutdown. No `WARNING`, no
`turn failed`, no echo-induced false barge-in. And **barge-in never fired once** — the user said so out
loud during the session and the model transcribed the complaint.

The robot side ruled itself out mechanically before reporting, which is what §11.6 asks for:
`send_loop()` streams unconditionally and both of its gates (quiz, SLEEPY RMS) were inactive; and
decisively, **speech spoken over the reply reached the brain and was transcribed correctly** — so the
audio path was fine and PulseAudio AEC was not over-suppressing either.

**The cause is here.** Barge-in was keyed on `self.current`:

```python
turn = self.current
if turn is not None and not turn.cancelled and self.detector.speech_run >= BARGE_WINDOWS:
```

A turn ends when the last audio chunk has been **sent**, not when it has been **heard**. The brain
streams ~14s of speech down the socket in ~2s, so for the remaining ~12s — the part the user actually
listens to, and therefore the only part they can interrupt — there is no turn object and the condition
is dead. The interrupting speech became a *new turn* instead, which the log shows plainly: three
`speculating on …` lines in the seconds after `speculation adopted`, and zero cancellations.

**Fix**: track when the robot will run out of audio. `emit()` already funnels every reply chunk, so it
accumulates `_playing_until = max(_playing_until, now) + chunk_duration`, and barge-in now fires either
when a turn is live (as before) or when the robot is still playing. In the second case there is nothing
to cancel — the work is done — so it only sends `interrupted`, which is precisely what tells the robot
to drop its buffer (§11.0-1). Backchannels are excluded from the accounting: they fire *during* the
user's pause, and counting them would make the user's own continued speech look like an interruption.

**Why no test caught it**: `fake_robot.py` consumes audio as fast as the socket delivers it. Real-time
playback — and therefore this twelve-second window — has never existed in any test we wrote.
`client/test_barge_in_playing.py` now waits out the audio it received before cutting in. It fails
against the old code and reports **0.10s** against the new.

Third instance of the same lesson, now with the sharpest example: §13.9 was a test config easier than
production, this is a test *client* easier than production. The `[MEASURED]` 0.07s in §13.3 was never
wrong — it measured interrupting the brain mid-generation, which is a different and much rarer moment
than interrupting the robot mid-sentence.

### 13.12 🔴 `[MEASURED]` Every tool was declared without its parameters (2026-09-22)

Second live session. Barge-in (§13.11) and the SLEEPY reconnect both verified on the robot and closed.
A new one, reported with the cause already isolated on the robot side:

```
❌ 툴 호출 실패: set_emotion({}) -> TypeError("... missing 1 required positional argument: 'emotion'")
```

Every turn, 100% reproducible. **The robot's face never changed once in local-brain mode** — a plain
regression against Gemini.

`tool_schemas()` built each declaration's description as
`inspect.getdoc(fn).split("\n\n")[0][:300]` — the first paragraph only. Every tool in this robot
documents its valid values in an `Args:` block, which sits *after* a blank line. So the model was told a
tool existed and never told what to put in it. This was not specific to `set_emotion`: all four tool
modules (`emotion`, `memory`, `motion`, `quiz`) use `Args:`, so **every parameter of every tool was
undocumented.**

Google's SDK takes the callable whole and the model sees the entire docstring. The shim exists to
present that same surface (§11.5), and here it was quietly presenting less.

**Fix**: parse the `Args:` block into per-parameter `description`, and use everything above `Args:` as
the function description so the *when to call it* paragraph survives too.

⚠️ **The enum extraction had to be made deliberately timid.** The obvious rule — pull quoted strings out
and call them an enum — is wrong here, and would have broken a working tool:

| docstring | enum? | why |
|---|---|---|
| `emotion: one of "neutral", "happy", …` | ✅ 11 values | exhaustive by wording |
| `joint: one of "right_arm", …` | ✅ 4 values | same |
| `field: … (e.g. "name", "grade", …), **or any free-form label**` | ❌ none | quoted values are *examples*; an enum would reject everything else |
| `speed: "slow", "normal", or "fast"` | ❌ none | no "one of", so description only — safe, not maximal |

So: enum only when the line says **"one of"** and carries no hedge (`e.g.`, `any`, `free-form`, `etc`).
Everything else rides in the description, which the model reads anyway.

**Verified end to end**, real persona, real docstrings, four clips: `a1_tired → sad`,
`a2_happy → happy`, `a3_anxious → sad`, `turn_s2 → tender`. Four for four, all valid, all matching the
audio's tone. `client/test_tool_schemas.py` guards it with no server needed, including the negative
case — it fails if `remember_fact.field` ever gains an enum.

**Why nothing caught it**: the stand-in tools in `exp8_latency.py` and `fake_robot.py` were written by
hand with clean one-line docstrings and `Literal[...]` hints, so the `Literal` branch always fired and
the `Args:` path never ran. A fourth instance of §20 rule 20 — the test doubles were *tidier* than the
real thing, and tidiness is its own kind of unrealistic.

### 13.13 🔴 `[MEASURED]` The brain was saying the session-end tag out loud (2026-09-22)

Third live session. The robot reported that the user said "대화 종료" three times, Moti said goodbye
three times, and the session never ended — they had to kill the process. Non-deterministic: the
previous session had detected it correctly.

**First question: did the model not emit `[대화종료]`, or did we strip it?** Only the brain can tell,
and it is a five-minute experiment, so it was run before touching anything. Five phrasings, real
persona, raw vLLM deltas printed beside what the pipeline emitted:

| phrasing | tag in raw output | tag in spoken output |
|---|---|---|
| "이제 대화 그만할게" | yes | yes |
| "대화 종료할게요" | yes | yes |
| **"대화 종료"** | **no** | no |
| "오늘은 여기까지 할게. 잘 있어" | no | no |
| "그만 이야기하자" | no | no |

**The pipeline is not guilty** — every tag the model produced arrived intact. The model emits it about
40% of the time, and the phrasing it missed includes *the literal phrase the user said*. That is
persona wording, which is robot-owned, so it goes back with the report.

🔴 **But the experiment found a brain bug on the way through.** When the tag *does* arrive it reaches
`_speak()` like any other sentence, and `Tts().synth("[대화종료]")` measures **1.23s of clear speech at
full amplitude**. So on every successful exit, Moti announced "대화종료" out loud before hanging up.

The fix has to thread a needle: `launcher.py` scans `output_transcription` for the tag and that is its
**only** channel, so the tag must stay in the transcript while vanishing from the audio. `_speak()` now
splits the two — the transcript carries the sentence as written, TTS gets it with control tokens
removed, and a sentence that is *only* a control token emits the transcript and skips TTS entirely.

Matched by **shape, not by a list**: a bracketed run with no spaces (`[대화종료]`, `<SILENT>`). The brain
should not have to know the robot's vocabulary, and §9.3a already set the precedent that control tokens
are the robot's to define and the brain's to honour. Ordinary empathetic Korean does not contain
`[한단어]`, and being wrong costs one unspoken bracketed word.

### 13.14 `[MEASURED]` Two smaller ones from the same report

**A tool call missing its required arguments is now dropped, loudly.** `set_emotion({})` came back once
in three turns even with a correct schema and enum (§13.12) — the model just does this sometimes.
Forwarding it means the robot runs `set_emotion()`, its `try/except` swallows the `TypeError`, and the
face silently stays put. The brain knows `required` from the schema it was handed, so it drops the call
and logs `dropped set_emotion — model gave no emotion`. An invisible robot-side failure becomes a
visible brain-side line. Tools that legitimately take nothing (`end_quiz_early`) are unaffected.

**Description truncation no longer cuts mid-sentence.** §13.12's fix capped descriptions at a bare
`[:600]`, and the robot found it slicing `remember_fact` through the middle of its load-bearing
sentence — *"...confirming without calling lose"* — so the model received an unfinished instruction with
nothing marking it unfinished. Now the cap is 1200 (their longest real docstring is 550), the cut lands
on the last sentence boundary before it, and a truncated description ends in `[…]` so the reader can
see it happened. **A truncation the reader cannot detect is worse than a shorter description** — and
note that this was a bug *introduced by the previous fix*, found in one session.

**Escalation** (v5, now largely closed): E4B TTFT consistently >700ms → ~~MTP~~ (blocked, §13.6) →
~~QAT~~ (dropped, §13.10) → shorten context → consider E2B. In practice TTFT was never the problem
(0.209s warm); decode rate is, and the remaining lever there is prompt-side, not model-side.

---

## 14. GEMINI 3.8 LIVE PARITY MATRIX

| Feature `[OFFICIAL]` | Status | Path |
|---|---|---|
| Barge-in | ✅ **Verified on the robot 2026-09-22**, 0.10s | Silero VAD on the brain (§9.1a). 🔄 *Corrected 2026-09-19*: this row used to read "smart-turn is not in this path — it failed validation", written while Q19b was open. §9.1c/§9.1e resolved it and smart-turn **is** wired in, twice per pause (fast path + veto). No longer fake-robot only: confirmed on hardware, both while generating and while the robot is playing (§13.11). AEC turned out fine (§18.1). |
| Interruption cancel/discard | ✅ **Implemented** | Cancels LLM generation and TTS together; robot drops its buffer (§11.0-1/-4). |
| Audio transcription (both sides) | ✅ **Verified** | v5 credited the cascade's STT for this; EXP-13 deleted that leg, so the brain must ask E4B for the user transcript explicitly (§12.6). Measured word-for-word exact on Korean. |
| High-quality natural speech | ⚠️ **Adequate, not chosen** | Piper `ko_KR-kss-medium` is the *only* Korean voice that exists for Piper (§8.3). It works; it was not selected on quality. |
| Affective dialog | ✅ **Verified** | In-band: E4B hears prosody itself (§12.4). No parallel SER. |
| Emotional speech output | ⚠️ **At risk** | Piper's only Korean voice has no emotion control and no voice choice (§8.2). CosyVoice 2 instruct would restore it but has no Jetson precedent. |
| Proactive audio | ✅ **Implemented**, 6/6 on text | v5's wording scored 4/6 and had to be rewritten (§9.3a). Untested on audio — no self-talk recordings yet. |
| Function calling | ✅ **Verified end to end** | 3 correct calls with correct args against the server, and it still fires with audio input (§12.6). Since then the **full round trip** is verified over the wire: `tool_call` → robot executes → `tool_result` → follow-up reply, with `args` as a dict per §11.5. |
| Async function calling | ⚠️ Custom work | Not free; low priority for Moti |
| Background reasoning | ❌ Deliberately excluded | Thinking mode kills latency (§5.4) |
| 24 languages | ❌ Korean-only | Irrelevant for Moti |
| Natural conversational rhythm | ⚠️ **Built, and switched off in the field** | Backchanneling works (§12.7) but its safe trigger is 1.4s against a reply at 2.5s+, so it covers little. 🔄 *2026-09-22*: after live use the robot set `BRAIN_BACKCHANNEL=false` in its `.env` — users found "음..." mistimed and artificial. Not a defect, a felt-quality call, and reversible in one line. The brain's default stays on. |
| Sub-500ms latency | ❌ **2.5–4.7s** | 🔄 *Re-corrected 2026-09-22.* Was still quoting §13.4's 2.02s, a toy-persona figure (§13.9). Under the real persona the floor is a steady **1.42s** and the mean wanders between 2.5s and 4.7s run to run, because first audio waits for the first *sentence* and its length is sampled. Turn detection is ~0.6s of it; the rest is generating that sentence at 13 tok/s. Quantization is the obvious lever and is **deferred on evidence** (§13.10). A cheaper one is untried: telling the persona to open with a short sentence costs no quality at all. |

**Status as of 2026-09-19**: of 13 rows — **6 verified or implemented**, 4 partial (voice quality,
emotional speech output, async function calling, conversational rhythm), 2 deliberately out of scope,
and **1 genuine miss: sub-500ms latency**. Everything measured so far ran against a fake robot —
**nothing has touched real hardware**, so AEC, mic quality and playback timing are all still unknowns (§18).

⚠️ **This table is a status claim, not a log.** Three of its rows silently went stale between
2026-09-19 morning and evening because Q19b's resolution invalidated the premises they were written on —
the same failure pattern §0-A and the troubleshooting notes record twice already. **Re-read this table
whenever a §9/§13 measurement lands**, not only when a feature is added.

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

**Stage 1 — LLM leg standing up** ✅ **DONE (§13.0, 2026-09-18)** — boxes were left unticked
until 2026-09-22 although the measurements had been in §13.0 for four days (rule 19 again)
- [x] E4B served via vLLM in the NVIDIA container (§5.3); `scripts/run_vllm.sh`, streaming confirmed
- [x] bf16, no quantization — and §13.10 has since deferred q4 on evidence, not assumption
- [x] EXP-2: warm TTFT **0.209s**, decode **13.2 tok/s**, KV 67,712 tokens (`scripts/bench_llm.py`)

**Stage 2 — EXP-13 ✅ DONE (§12.4, 2026-09-19)**
- [x] Audio token rate and 30s ceiling measured empirically (§12.3)
- [x] All four kill criteria cleared
- [x] **Survived → Whisper and SER legs deleted (§12.5). Stage 2b cancelled.**

**Stage 3 — TTS leg ✅ DONE (§8.3, 2026-09-19)**
- [x] Piper runs on this AGX, Korean voice exists (exactly one), RTF 0.22–0.38
- [x] Voice decided: `ko_KR-kss-medium` unmodified, `ENABLE_VOICE_SHIFT=false` → 700ms buffer removed
- [ ] CosyVoice 2 — **deferred, not cancelled.** Piper is a working floor, so this is now an
      optional quality upgrade rather than a risk item. Revisit if the voice disappoints in live use,
      or if emotional speech output (§14) turns out to matter.

**Stage 4 — the pipeline and the wire** (§11) — **everything but the real robot is done**
- [x] VAD + smart-turn-v3 on the brain, with the §11.0-3 compensation buffer (§9.1a–e)
- [x] Brain server: wire protocol per §11.5, conversation state held server-side, resumed across
      reconnects (`client/test_reconnect.py`)
- [x] Robot-side client shim; the `launcher.py` diff **is** the `connect()` call only (§20 rule 17).
      Procedure for the robot side: `docs/robot_integration.md`
- [x] Instrument every stage from the first commit (§20 rule 5) — §13.2–13.5
- [x] Fake-robot client for round-trip testing (`brain/fake_robot.py`)
- [x] The four `[MANDATORY]` items in §11.0, plus item 5 (≤30s clip splitting)
- [ ] **The real robot on the same LAN** — the only Stage 4 item left, and it is the one that
      decides whether any of the above survives contact with a microphone (§14 status note)

**Stage 5 — behavior and measurement** (ordered 2026-09-19; everything here needs the robot)
- [x] **① AEC** ✅ **closed 2026-09-22** — PulseAudio `module-echo-cancel`, survives reboot, zero
      echo-induced barge-ins across three live sessions, and no doubletalk over-suppression either
      (§18.1). Q16 answered
- [ ] ② The five remaining checks in `docs/robot_integration.md` §4, in that order
- [ ] ③ EXP-9 barge-in + playback flush, with a real mic in the loop
- [ ] ④ EXP-8 again on the robot — §13.9's 2.65s is over a loopback socket with no mic, no AEC,
      no motors. Expect it to move
- [ ] ⑤ EXP-10 thermal/power under motors + camera + face UI together
- [ ] **⑥ EXP-1 (E4B vs 26B-A4B Korean quality)** — no longer a gate on quantization (§13.10 dropped
      that), but kept and arguably more important: it is the project's actual research question, and
      after live sessions there is finally an informed opinion about what to rate
- [x] `<SILENT>` gate → EXP-7 ✅ 6/6 on text (§9.3a); still untested on audio — needs self-talk
      recordings, worth making during the robot session
- [x] EXP-12 backchanneling ✅ built and measured (§12.7), window is narrow
- [x] ~~EXP-4/5/6~~ — cancelled, they only existed to validate the deleted legs (§12.5)
- [x] ~~EXP-3 MTP~~ — blocked by the container (§13.6); ~~q4 quantization~~ **dropped** (§13.10)

**Stage 6 — phase 2 transport**
- [ ] Tailscale; verify `direct` not `relay` (§18); re-measure EXP-8 over it
- [ ] Reconnect/resume across link drops (§20 rule 18)

---

## 18. RISK REGISTER

| Risk | Severity (v6) | Mitigation |
|---|---|---|
| CosyVoice fails on Jetson | 🟡 **MED — downgraded from 🔴** | Piper is reinstated as a working floor (§8.2), so this is no longer a blocker. Still no Jetson precedent; pursue on merit, not under deadline. |
| ~~faster-whisper CPU-only on Jetson~~ | ✅ **Closed by EXP-13 (§12.4)** | The Whisper leg is gone (§12.5). The aarch64 `ctranslate2` source build — v5's risk #1 — never has to happen. Reopens only if the audio path fails in live use. |
| ~~Korean SER accuracy poor~~ | ✅ **Closed by EXP-13 (§12.4)** | E4B hears prosody directly and contradicted literal words on a suppressed-sadness reading. No SER model, no AI-Hub retraining. |
| **Tone-driven fabrication** | 🟡 **MED — new in v6** | §12.4: on a bright-toned reading the model invented a situation that was never said. Prosody sensitivity cuts both ways. Watch in live use; the text-only path does not have this failure mode. |
| Latency above target | 🟡 MED | E4B → MTP → QAT → shorter context → E2B. New v6 contributors: voice-shift buffer (§13) and, in phase 2, Tailscale relay fallback. |
| **Tailscale falls back to DERP relay (phase 2)** | 🟡 **MED — new in v6** | Relayed WireGuard adds unpredictable latency, which lands directly in the voice loop. Require a **direct** connection (`tailscale status` shows `direct`, not `relay`) and treat relayed operation as a degraded mode. |
| Missing AEC breaks barge-in | ✅ **Closed 2026-09-21 on the real robot** | The mechanism is **PulseAudio `module-echo-cancel`** (webrtc), not the Python wheel. Verified working and reboot-surviving on the Orin Nano; a whole live session produced **zero** echo-induced barge-ins. §18.1 keeps the reasoning because the wheel-based path is still absent and would silently disable AEC if anyone ever re-enabled `ENABLE_AEC`. **Q16 answered.** |
| CMA fragmentation | 🟢 LOW | Never touch `cma=`; reboot if hit |

### 18.1 🔴 AEC — the robot disables it silently, and our barge-in is 5× more aggressive than Gemini's

Read out of `MOTI-HRI` directly rather than taken from the v6 note, because this is the single thing most
likely to make the first robot session look broken.

**The robot really does implement AEC**: `media/audio_manager.py` has `EchoCanceller`, a WebRTC AEC3
binding (`aec_audio_processing`) shared by the mic and speaker callbacks, feeding the far end in at the
near-end frame size. It is wired into `MicStreamer._callback` via `process_near()`. Good code.

**But it turns itself off without failing.** `audio_manager.py:25-32` probes the import at module load
and, on `ImportError`, prints a warning and sets `ENABLE_AEC = False` — added 2026-08-24 during the
Jetson port, because the robot used to *crash* on startup instead. And `requirements-jetson.txt:121`
still says, of that exact package, *"Linux 휠이 아예 없음(재확인)"* while `requirements.txt` (the
Windows/dev list) installs it normally.

So the failure mode is: **one warning line scrolls past at boot, the robot works fine, and echo
cancellation is simply absent.** Nothing later reports it.

Three things make this worse for us specifically than it was for Gemini:
1. **Our barge-in is 0.07s** (§13.3) against `duet`'s 0.198s reference. It is built to react to the
   faintest onset of speech, which is exactly what leaked speaker audio looks like.
2. **Turn detection moved to our side** (§4). With Gemini, echo-induced false barge-in was Google's
   problem; now every echo frame reaches our Silero VAD.
3. `AEC_STREAM_DELAY_MS` defaults to 100 and its own comment says it is **not measured**.

✅ **Resolved 2026-09-21 — and the answer was not the one this section assumed.** The wheel really is
absent and `ENABLE_AEC=false` really is correct on this robot, but AEC is **not** off: it runs one layer
down, as **PulseAudio `module-echo-cancel`** (webrtc) with the default sink and source set to
`echocancel_*`. That configuration survives reboot, and a full live session logged **zero** echo-induced
barge-ins. **Q16 is answered**: the mechanism is PulseAudio, not the Python binding.

Two things to keep from the original worry:
- The silent-disable path in `audio_manager.py:25-32` is still there. It is harmless while the app-level
  AEC is deliberately off, but it would hide a real failure if anyone ever set `ENABLE_AEC=true` again.
- PulseAudio AEC also **did not over-suppress**: during a reply the robot was still playing, the user
  spoke over it, and the brain transcribed that speech correctly. Doubletalk suppression was ruled out
  as a cause of the barge-in failure, which is how §13.11 ended up looking at the brain instead.
| ~~Piper GPL contamination~~ | ✅ **Closed in v6** | Research-only use (§1, §15). Keep the §20 rule 7 flag for a future release decision. |

---

## 19. OPEN QUESTIONS — MEASURE, DO NOT GUESS

| # | Question | Resolved by |
|---|---|---|
| Q1 | Does CosyVoice 2 run on Jetson Orin at all? | Day 1 / EXP-11 |
| Q2 | E4B or 26B-A4B for Korean empathetic conversation? | EXP-1 |
| Q3 | llama.cpp or vLLM faster here? | EXP-2 |
| ~~Q4~~ | ~~emotion2vec+ for Korean?~~ | ⬜ **Moot — SER leg removed (§12.5)** |
| ~~Q5~~ | ~~SenseVoiceSmall Korean ASR?~~ | ⬜ **Moot — STT leg removed (§12.5)** |
| ~~Q6~~ | ~~Does an emotion label help?~~ | ⬜ **Moot — no separate label; tone is in-band (§12.4)** |
| Q7 | Achievable end-to-end latency on this hardware? | EXP-8 |
| ~~Q8~~ | ~~Best Korean Whisper checkpoint?~~ | ⬜ **Moot — STT leg removed (§12.5)** |
| Q9 | Do NeuTTS Air / VibeVoice-Realtime support Korean adequately? | EXP-11 |
| Q10 | Does the prebuilt sm_87 ctranslate2 image work, or is a source build required? | Day 1 (§6.1) |
| Q11 | Does backchanneling measurably improve perceived liveness in Korean? | EXP-12 |
| Q12 | Has any open E2E model gained Korean + Jetson support? (re-check quarterly) | Quarterly review of MiniCPM-o, Qwen3-Omni |
| ~~Q13a~~ | ~~E4B audio token rate and clip ceiling?~~ | ✅ **Resolved from the checkpoint: 25 tok/s, 30s ceiling (§12.2)** |
| ~~Q13b~~ | ~~Does multi-clip segmentation clear the 30s ceiling?~~ | ✅ **Resolved: yes — 20s × 2 = 1,004 tokens, no truncation (§12.3)** |
| ~~Q13c~~ | ~~Comprehension and tone?~~ | ✅ **Resolved: matched a perfect transcript, and tone changed the reply (§12.4)** |
| **Q18** | How often does prosody make the model fabricate situations, and can the persona suppress it? | Live use; §12.4 caveat |
| ~~Q19a~~ | ~~Can smart-turn's Whisper mel be reproduced in numpy?~~ | ✅ **Yes — proven identical to `WhisperFeatureExtractor`, max error 0.0000 (§9.1b)** |
| ~~Q19b~~ | ~~Does smart-turn actually work on our Korean audio?~~ | ✅ **Resolved 2026-09-19 (§9.1c): yes — median 0.019 "still going" vs 0.880 "finished" on spontaneous audio. The 2/5 was the test material, not the model.** Two corrections came with it: the useful wiring is a **veto**, not an early end (the obvious direction measured *worse* than no smart-turn), and the "this takes 1.98s → 0.65s" projection was **wrong** — see §9.1c. |
| **Q20** | Where did 2.02s actually go, now that turn detection is only ~0.6s of it? Decode of the first sentence at 13 tok/s is the stated cause (§13.5) but has not been isolated end to end. | q4 W4A16 checkpoint + one vLLM restart (§13.6) |
| **Q17** | Why does vLLM's "model loading" stage take 1,674s (28 min) when weights read in 3.8s? | Unexplained. Mitigation is to not restart the server (§1 always-on). |
| ~~Q14~~ | ~~Reproduce the young voice, or drop the pitch shift?~~ | ✅ **Resolved 2026-09-19: Piper's voice as-is, no pitch shift. `ENABLE_VOICE_SHIFT=false` (§8.3)** |
| **Q15** | Does Tailscale hold a `direct` connection in practice, and what does it add to EXP-8? | Stage 6 |
| ~~Q16~~ | ~~How was AEC actually solved on the Orin Nano?~~ | ✅ **Resolved 2026-09-21 on the robot: PulseAudio `module-echo-cancel` (webrtc), default sink/source `echocancel_*`, survives reboot.** Not the Python wheel — that is still absent, so `ENABLE_AEC=false` is correct here. Zero echo-induced barge-ins in a live session, and no doubletalk over-suppression either (§18.1). The robot repo's docs should be updated to say this. |

---

## 20. RULES FOR CLAUDE CODE

1. Never present `[UNVERIFIED]` items as measured fact — in code comments, commits, or reports.
2. Never enable Gemma 4 thinking mode in any code path.
3. Always stream LLM and TTS output.
4. **Send raw PCM16 on the robot↔brain wire; wrap in WAV only at the vLLM call.** (v5 said "never a WAV container" for a local STT model that no longer exists — vLLM's `input_audio` requires the container. §11.1 explains the correction.)
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
19. **Re-read the status tables (§14, §19) whenever a §9/§13 measurement lands** — not when a feature is added. They are claims about the present, not a log, and they go stale silently because nothing fails when they do. Four occurrences by 2026-09-19: §14's barge-in / rhythm / latency rows, §12.6's prompt-ordering rationale, §11.1's raw-PCM rule, §5.4 rule 5. Each was invalidated by a *later fix being correct*, which is exactly why no test caught it.
20. **A benchmark or test config that is easier than production does not just give optimistic numbers — it deletes whole code paths from the test.** `fake_robot` and EXP-8 used a ~50-token persona with **no tools**, so the ~60 lines of tool parsing never ran against a declared tool name. Switching to the real 18,344-token persona surfaced four markup shapes that would all have been spoken aloud, plus a latent `_holdback` bug with no connection to personas at all (§13.9). Before trusting a measurement, ask what the real client sends; **making a test condition realistic finds more than adding a new test does.**
21. **Diagnose which side owns a bug before editing anything, and when it is unclear, edit nothing and ask** (§11.6). The brain and the robot are worked on by separate agents; neither edits the other's code, and `client/local_live.py` belongs to this repo even though it runs on the robot. The brain log usually decides it mechanically: if the brain logged sending something, the fault is downstream.

---

## SOURCES

NVIDIA Jetson AI Lab (Gemma 4 on Jetson tutorial; Gemma 4 26B-A4B model page) · NVIDIA Technical Blog / Edge AI and Vision Alliance (Jetson sizing guidance) · NVIDIA Riva documentation (Quick Start prerequisites) · NVIDIA Developer Forums (Jetson Orin echo/feedback, canary OOM) · Google AI for Developers (Gemma 4 overview, model card, audio understanding; Gemini API models & changelog) · Google Cloud (Gemini Live API overview) · Google DeepMind (Gemini 3.8 Audio model card) · daily.dev (Gemini Live API feature roundup) · Pipecat API reference (tts_service, stt_service, piper) · pipecat-ai/smart-turn · FunAudioLLM CosyVoice · SenseVoice · emotion2vec (ACL Findings 2024, model cards) · AI-Hub dataset pages · itsMustafamr/Jarvis-home · Modal engineering blog · SiliconFlow / BentoML / Northflank / Gladia 2026 TTS & STT roundups · community benchmarks (DGX Spark, RTX 4070 Ti) · NVIDIA NemotronLabs VoiceChat 11B model card · Kyutai Moshi (via Inworld S2S comparison) · OpenBMB MiniCPM-o 4.5 · DuplexCascade (arXiv) · mssharatchandra/duet · cbinckly Jetson STT practical guide · SYSTRAN/faster-whisper · Modal vLLM Gemma 4 example · Pipecat Ollama service reference
