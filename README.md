# MOTI_BRAIN

**공감 대화 로봇 모티의 뇌를, 클라우드 없이 젯슨 한 대 위에 올린다.**

로봇이 Gemini Live API에 하던 것과 **똑같은 방식으로** 말을 걸면, Jetson AGX Orin 위의 상시
서버가 그 오디오를 듣고 한국어로 공감하는 음성과 로봇 동작 지시를 돌려준다. 로봇 쪽 코드
변경은 `connect()` **한 줄**이다.

한동대학교 SIR Lab · 연구/논문용 · 진행 중

---

## 왜 만드는가

로봇 [MOTI-HRI](https://github.com/HGU-SIRLab/MOTI-HRI)가 Gemini Live API로 대화하고 있었다.
잘 돌지만 두 가지가 걸렸다.

- **비용** — 페르소나 시스템 프롬프트가 18,344토큰이고 세션마다 올라간다
- **rate limit** — 실험을 원하는 만큼 못 돌린다

여기에 연구용으로는 더 중요한 것들이 따라온다. **음성이 기기를 떠나지 않고**(대학생 상담
대화가 주제다), 모델 제공사가 모델을 바꿔도 **로봇 성격이 안 변하고**(재현성), 턴 감지 임계값 하나를
**한 줄로 고칠 수 있다**(API 뒤에서는 못 하는 일).

대가는 지연이다. Gemini Live가 0.5초, 이쪽은 **1.4~4.7초**다. §성능 참고.

---

## 아키텍처

```
┌─ 로봇: Jetson Orin Nano Super 8GB ─┐        ┌─ 뇌: Jetson AGX Orin 64GB ──────┐
│  마이크 · AEC · 스피커              │◄──────►│  Silero VAD → smart-turn-v3     │
│  카메라 · 얼굴인식 · 모터 · 표정     │ 오디오  │  → Gemma 4 E4B → Piper TTS      │
│  툴 실행 (remember_fact, …)         │ + JSON │  대화 상태 · 재연결 복구         │
│  launcher.py — connect() 한 줄      │        │                                 │
└─────────────────────────────────────┘        └─────────────────────────────────┘
                     같은 LAN, 웹소켓 :8765
                     (Stage 6에서 Tailscale로 원격)
```

**파이프라인**

```
마이크 16kHz PCM16 (끊김 없이 스트리밍)
  └→ Silero VAD ─── 음성/침묵
       └→ smart-turn-v3 ─── 침묵마다 두 번 질의 (빠른 경로 + 거부권)
            └→ ≤30초 클립으로 분할
                 └→ Gemma 4 E4B  (오디오 직접 입력 — STT가 없다)
                      ├→ 응답 텍스트 → 문장 단위 청킹 → Piper → 24kHz PCM ──→ 스피커
                      ├→ 툴 호출 ──────────────────────────────────────────→ 로봇이 실행
                      └→ 사용자 발화 전사 (병렬, 임계 경로 밖) ─────────────→ 대화록
```

### 설계에서 눈여겨볼 세 가지

**① STT가 없다.** Whisper를 쓸 계획이었는데 실험(EXP-13)에서 **Gemma 4 E4B가 오디오를 직접
받고 운율까지 듣는다**는 게 확인돼 두 모델(faster-whisper, emotion2vec)이 설계에서 통째로
빠졌다. 같은 문장을 톤만 바꿔 녹음했더니, 억눌린 슬픔으로 "괜찮아"라고 말한 쪽에서 모델이
**문자 그대로의 말을 반박**했다:

> "괜찮다고 말했지만, 사실은 많이 신경 쓰이고 힘든 마음이 느껴져요"

별도 감정 인식 모델이 하려던 일을 공짜로 얻었다. 대신 새 리스크가 생겼다 — 밝은 톤에서는
발화에 없던 상황을 지어내기도 한다.

**② 턴 경계는 뇌가 정한다.** Gemini Live가 서버 쪽 VAD를 했기 때문에 로봇은 마이크를 조건
없이 계속 보낸다. 그 계약을 그대로 유지했다. 덕분에 로봇의 오디오 경로를 한 줄도 안 건드렸고,
대신 에코로 인한 오탐이 우리 문제가 됐다.

**③ 로봇 변경이 한 줄인 이유.** `client/local_live.py`가 `google.genai` 라이브 세션의 겉면을
덕 타이핑한다 — `send_realtime_input`, `send_client_content`, `send_tool_response`, 그리고
턴 경계에서 끝나는 `receive()` 비동기 제너레이터까지. 그래서 `launcher.py`의 수신 루프·툴
실행·재연결 루프가 **수정 없이** 돈다. Gemini로 되돌리는 것도 한 줄이다.

---

## 사용 기술

| | 무엇 | 왜 이것인가 |
|---|---|---|
| **LLM** | Gemma 4 **E4B** (bf16) | 젯슨 공식 지원 variant, AGX 권장 4B~20B 구간. **오디오 입력을 받는다** — 이게 STT를 지웠다 |
| **서빙** | vLLM 0.19, `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin` | prefix 캐시가 GPU 상주라 18K 페르소나를 세션당 한 번만 문다. llama.cpp는 오디오 경로가 깨져서 못 쓴다 |
| **VAD** | Silero VAD v5 (ONNX) | torch 없이 ONNX Runtime만으로 동작 — 이 젯슨의 PYTHONPATH 사정상 중요했다 |
| **턴 감지** | smart-turn-v3 (Whisper Tiny 인코더 + 선형 헤드, 8MB ONNX) | 한국어 정확도 96.96%(23개 언어 중 1위). 침묵마다 두 번, 반대 목적으로 질의한다 |
| **TTS** | Piper `ko_KR-kss-medium` | **한국어 음성이 정확히 하나뿐**이라 선택지가 없었다. RTF 0.22~0.38 |
| **전송** | `websockets` (라이브러리 하나) | Pipecat을 쓰려다 뺐다 — 분산 구성에서는 커스텀 transport를 직접 써야 해서 프레임워크가 줄여주는 걸 다시 손으로 쓰게 된다 |

**쓰지 않은 것과 이유**: 종단간(end-to-end) 음성 모델 — 열린 모델은 전부 영어 전용이거나
데이터센터 GPU 전용이다(VoiceChat 11B, Moshi). **한국어 + 젯슨 조합이 모든 선택지를 지운다.**
캐스케이드는 타협이 아니라 유일한 길이다. 양자화 — 서빙 가능한 E4B q4 체크포인트가 사실상
없고, 무엇보다 **품질 기준선이 아직 없다**.

---

## 용어

음성 대화 시스템에서만 쓰는 말들이라 따로 모았다.

**VAD (Voice Activity Detection, 음성 구간 검출)**
지금 들어오는 소리가 **사람 말인지 아닌지**만 판단한다. 무슨 말인지는 모른다 — 그건 LLM이
한다. 마이크는 24시간 켜져 있고 대부분은 에어컨 소리와 정적이므로, 이걸 걸러내지 않으면
모든 소음을 LLM에 넣게 된다. 우리는 **Silero VAD v5**를 32ms 창마다 돌려 0~1 확률을 받고
0.5를 넘으면 "말하는 중"으로 본다.

**턴 감지 (turn detection) — VAD와 다르다**
VAD는 *"지금 소리가 나는가"*를 답하고, 턴 감지는 *"이 사람이 말을 **끝냈는가**"*를 답한다.
둘은 전혀 다른 질문이다. 1초의 침묵이 **생각 중**일 수도 **끝**일 수도 있는데, VAD는 둘을
구분하지 못한다. 실제로 자연스러운 대화에서 생각하다 멈추는 침묵이 **2.98초**까지 갔다 —
"1.5초 조용하면 끝"으로 잡으면 사람 말을 계속 자른다. 그래서 VAD 위에 **smart-turn-v3**를
얹어 "이 억양이 문장을 맺는 억양인가"를 따로 묻는다.

**barge-in (끼어들기)**
로봇이 말하는 도중에 사람이 끼어들면 로봇이 즉시 입을 다무는 것. 대화가 살아있게 느껴지는
데 가장 크게 기여하는 요소이고, 우리는 0.10초다.

**TTS / STT**
TTS는 글자를 소리로(Text-to-Speech), STT는 소리를 글자로(Speech-to-Text). **이 프로젝트에
STT는 없다** — LLM이 오디오를 직접 먹는다.

**TTFT (Time To First Token)**
질문을 넣고 **첫 글자**가 나오기까지의 시간. 전체 생성 시간과 구분해서 본다. 대화에서는
이쪽이 체감을 지배한다.

**프리필(prefill) / prefix 캐시**
LLM은 답을 만들기 전에 프롬프트 전체를 한 번 읽어야 하고(프리필), 우리 페르소나는
18,344토큰이라 그게 **17.9초**다. vLLM은 그 계산 결과를 GPU에 캐시해두고 같은 프롬프트가
다시 오면 재사용한다(prefix 캐시) — 두 번째부터는 0.2초다. 그래서 아무도 없을 때 미리
한 번 돌려두는 "예열"을 한다.

**shim**
겉모습은 기존 것과 똑같이 흉내 내면서 속만 바꿔치기하는 얇은 껍데기.
`client/local_live.py`가 Gemini Live 세션인 척해서 로봇 코드가 자기가 바뀐 줄 모르게 한다.

**RTF (Real-Time Factor)**
음성 합성 속도. 0.3이면 10초짜리 음성을 3초에 만든다는 뜻이고, 1보다 작아야 실시간이 된다.

---

## 시작하기

### 뇌 켜기 (AGX)

```bash
bash scripts/start_brain.sh
```

**몇 번을 돌려도 안전하다.** 이미 떠 있는 건 건드리지 않고 죽은 것만 골라 올리므로,
상태가 의심스러우면 그냥 다시 돌리면 된다.

| 상황 | 시간 |
|---|---|
| 전부 살아 있음 | 몇 초 |
| 뇌 서버만 죽음 | 약 10초 |
| **vLLM까지 죽음 / 재부팅 직후** | **약 29분** |

**vLLM은 함부로 끄지 않는다** — 29분의 대부분이 원인 미규명 구간이고, 이건 원래 상시 가동
서버로 설계됐다. 자세한 절차와 함정은 [`docs/brain_startup.md`](docs/brain_startup.md).

### 로봇 붙이기

[`docs/robot_integration.md`](docs/robot_integration.md) 하나면 된다. 요약하면
`launcher.py:540` 한 줄 + `.env` 두 개 + shim 파일 복사.

### 로봇 없이 말 걸어보기

```bash
# 노트북에서 (AGX 아님 — 브라우저 마이크가 필요하다)
scp herobot@<AGX>:~/moti_brain/web/test_client.html .
python3 -m http.server 8000
# http://localhost:8000/test_client.html
```

브라우저 마이크로 뇌에 직접 말을 건다. 에코 캔슬을 끄는 체크박스가 있어서 **AEC가 없을 때
무슨 일이 나는지 미리 겪어볼 수 있다**. 자세한 건 [`web/README.md`](web/README.md).

### 검사

```bash
PYTHONPATH= .venv_tts/bin/python brain/test_pipeline.py       # 문장 분할·툴 파싱·전체 왕복
PYTHONPATH= .venv_tts/bin/python brain/test_vad.py            # 턴 감지 (분할 예산)
PYTHONPATH= .venv_tts/bin/python brain/test_smart_turn.py     # 분리도 유지 여부
PYTHONPATH= .venv_tts/bin/python brain/test_gates.py          # 이모지 억제 · <SILENT>
PYTHONPATH= .venv_tts/bin/python client/test_error_path.py    # 실패한 턴이 턴을 끝내는가
PYTHONPATH= .venv_tts/bin/python client/test_local_live.py    # launcher 수신 루프 재현
PYTHONPATH= .venv_tts/bin/python client/test_reconnect.py     # 재연결 후 대화 복구
PYTHONPATH= .venv_tts/bin/python client/test_barge_in_playing.py   # 재생 중 끼어들기
PYTHONPATH= .venv_tts/bin/python client/exp8_latency.py --real 8   # 지연 (실제 페르소나)
```

🔴 **`PYTHONPATH=` 를 반드시 붙인다.** ROS2/HARU setup이 전역 `PYTHONPATH`에 다른 venv를
넣어두어서, 빼면 젯슨 torch와 충돌하는 패키지가 보이고 `undefined symbol`로 죽는다.

---

## 파일 구조

```
brain/                뇌 — AGX에서 돈다
  server.py           웹소켓 전송 계층. 턴 경계·끼어들기·세션 복구·투기적 생성
  pipeline.py         캐스케이드 본체. 오디오 → E4B → 문장 청킹 → Piper. 툴 마크업 파싱
  vad.py              Silero VAD + 하이브리드 종료 판단 (빠른 경로 + 거부권)
  smart_turn.py       smart-turn-v3 ONNX. Whisper mel을 numpy로 재현(참조와 오차 0.0000)
  backchannel.py      EXP-12. 미리 합성한 "응/어/음"
  fake_robot.py       하드웨어 없이 전체 루프를 돌리는 가짜 로봇
  test_*.py           단위 검사 4종

client/               로봇에 들어가는 것 + 실험 스크립트
  local_live.py       ★ Gemini Live 세션을 덕 타이핑하는 shim. 로봇으로 복사되는 유일한 파일
  test_*.py           shim 검사 4종
  exp8_latency.py     EXP-8 지연 분포 (--real 로 실제 페르소나)
  exp12_backchannel.py

scripts/
  start_brain.sh      ★ 전체 기동. 몇 번 돌려도 안전하다
  run_vllm.sh         vLLM 컨테이너. 인자마다 이유가 주석에 있다
  prewarm.py          페르소나 프리필 — 아무도 기다리지 않을 때 18K 값을 치른다
  bench_llm.py        EXP-2 TTFT·디코딩 속도
  exp13_korean.py     EXP-13 한국어 이해·운율
  probe_audio.py      오디오 토큰 회계와 30초 천장 확인

web/
  test_client.html    브라우저에서 말 걸어보는 페이지 (의존성 없음)

docs/
  brain_startup.md    재부팅 후 되살리는 법
  robot_integration.md ★ 로봇 쪽 작업 지시서 + 소유권 규칙
  exp13_recordings.md 녹음 대본과 검증 목적

CLAUDE.md             ★ authoritative 스펙 — 아키텍처·측정·리스크·규칙 21개
```

---

## 성능 (실측)

전부 **실제 페르소나 18,344토큰 + 툴 9개** 기준이다.

| | |
|---|---|
| 말 멈춤 → 첫 소리 | **1.42초(바닥) ~ 4.7초**, 평균 2.5~3.6초 사이에서 흔들림 |
| 끼어들기 반응 | **0.10초** |
| 첫 턴 (콜드 페르소나) | 17.9초 → 예열하면 **0.2초** |
| 디코딩 | 13.2 tok/s (메모리 대역폭 바운드: 204GB/s ÷ 15GB ≈ 13.6) |
| warm TTFT | 0.209초 |

⚠️ **지연은 점이 아니라 범위로 읽어야 한다.** 같은 조건 4회가 2.65 / 2.53 / 4.73 / 3.63초로
갈린다. 바닥은 매번 1.42초로 고정이고, 첫 오디오가 **첫 문장**을 기다리는데 그 길이가
샘플링에서 흔들리기 때문이다. **n=8로는 2초 미만의 변화를 판별할 수 없다.**

### Gemini Live와 비교

**기능은 대체로 따라잡았다** — 끼어들기, 취소·폐기, 양쪽 전사, 감정에 맞춘 반응, 함수 호출,
능동적 침묵, 세션 복구.

**못 따라잡은 것은 둘이다.**
- **지연** — 0.5초 대 2.5~4.7초
- **목소리의 감정** — 이건 최적화로 안 된다. Gemini는 오디오를 직접 생성해서 떨림·웃음·망설임을
  낼 수 있지만, 캐스케이드는 텍스트를 거치면서 감정을 버린다. 모티는 슬픔을 *알아듣고* 슬픔에
  *맞는 말*을 하지만 **목소리는 평평하다**. 얼굴 표정과 모션이 일부 메운다.

---

## 이 저장소가 일하는 방식

측정과 문서화에 대해 세게 잡은 규칙이 몇 개 있다. 전부 **실제로 물려서** 생겼다.

**직접 잰 값이 제조사 문서보다 우선한다.** 추측한 숫자는 `[UNVERIFIED]`로 표시하고 사실처럼 쓰지 않는다.

**문서는 조용히 거짓이 된다.** 나중 수정이 앞 문단의 전제를 무효화하는데 아무것도 실패하지
않는다. 지금까지 다섯 번 겪었고, 전부 **나중 수정이 옳았기 때문에** 생겼다. 그래서 상태를
주장하는 표는 측정이 들어올 때마다 다시 읽는다.

**테스트 조건이 프로덕션보다 쉬우면 코드 경로가 통째로 빠진다.** 숫자만 낙관적으로 나오는 게
아니다. 두 번 크게 물렸다:
- 벤치마크가 50토큰 페르소나에 **툴 0개**를 써서 툴 파싱 60줄이 한 번도 안 돌았다. 실제
  페르소나로 바꾸자 **로봇이 소리내어 읽었을 마크업 4종**이 쏟아졌다
- 가짜 로봇이 오디오를 소켓 속도로 받아서 **실시간 재생이라는 개념이 없었다.** 그래서
  "로봇이 말하는 중에 끼어들기" 창이 어떤 테스트에도 존재하지 않았고, 실물에서 barge-in이
  한 번도 안 먹었다

**양쪽을 동시에 고치지 않는다.** 뇌와 로봇을 각각 다른 Claude Code가 작업한다. 어느 쪽도
상대편 코드를 고치지 않고, **버그가 나오면 어느 쪽 책임인지부터 진단**한다. 애매하면 양쪽
다 건드리지 않고 사람에게 묻는다. 대부분은 기계적으로 갈린다 — **뇌 로그가 심판이다.**
(스펙 §11.6)

**대본 낭독으로는 대화를 평가할 수 없다.** 자연 발화의 생각 침묵은 2.98초인데 낭독은
1.38초다. 이걸 몰라서 턴 감지가 자연 발화 침묵 22개 중 19개를 끊는 걸 한참 못 봤다.

---

## 현재 상태

**Stage 0~4 완료**, 실물 로봇 1차 세션 완주(2026-09-21).

- ✅ 실시간 스트리밍, 턴 감지, 툴 왕복, 재연결 복구, 끼어들기, 맞장구, 능동적 침묵
- ✅ 실물 로봇 1차 세션 — 크래시 없이 완주, AEC 정상(PulseAudio `module-echo-cancel`)
- ⬜ 2차 세션 — 재생 중 끼어들기 확인, 장시간 세션, 모터·카메라 동시 구동 부하
- ⬜ EXP-1 품질 기준선 → 그다음이 양자화
- ⬜ Stage 6 Tailscale 원격

---

## 라이선스

코드는 저장소 라이선스를 따른다. 의존성 중 둘은 **연구용이라 쓰는 것**이므로 배포를
고려한다면 교체 대상이다:

- **Piper** — GPL-3.0
- **KSS 데이터셋**(`ko_KR-kss-medium`) — CC BY-NC-SA 4.0

Gemma 4, Silero VAD, smart-turn-v3는 각각 Apache 2.0 / MIT / BSD 2-clause다.

`testdata/`는 커밋하지 않는다 — 개발자 본인 목소리이고 이 저장소는 공개다.
