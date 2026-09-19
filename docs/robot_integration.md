# 로봇 쪽 작업 지시서

`MOTI-HRI` (`jetson-moti` 브랜치, Jetson Orin Nano Super)를 로컬 뇌에 붙이는 절차.
뇌는 AGX Orin에서 이미 돌고 있고, **로봇 코드 변경은 한 줄 + 설정 두 개**다.

전제: 뇌(AGX)와 로봇이 같은 LAN. 원격은 Stage 6(Tailscale)이고 아직 미착수.

---

## 0. 뇌가 떠 있는지 먼저 확인

AGX에서:

```bash
curl -sf http://localhost:8000/v1/models   # vLLM (없으면 moti_brain/scripts/run_vllm.sh)
ss -ltn | grep 8765                        # 뇌 서버
```

로봇에서:

```bash
curl -v --http1.1 --max-time 3 http://192.168.0.5:8765   # 연결만 확인 (426 응답이 정상)
```

426 Upgrade Required가 오면 도달한 것이다. 타임아웃이면 **유선 IP `192.168.0.5`와
WiFi `192.168.0.247`을 둘 다** 시도해볼 것 — WiFi는 AP의 클라이언트 격리에 막힐 수 있다.

---

## 1. shim 복사

AGX의 `moti_brain/client/local_live.py`를 로봇의 `MOTI-HRI/` 아래로 복사한다.

```bash
# 로봇에서
scp herobot@192.168.0.5:/home/herobot/moti_brain/client/local_live.py ~/MOTI-HRI/core/
```

의존성은 `websockets` 하나이고 `google-genai`가 이미 끌어온다. 추가 설치 없음.

---

## 2. `launcher.py` — 한 줄

**540행:**

```python
# 변경 전
async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:

# 변경 후
async with local_live.connect(BRAIN_URI, config=config) as session:
```

그리고 import 한 줄과 주소 한 줄을 추가한다 (파일 상단, 다른 `from core...` 옆):

```python
from core import local_live
BRAIN_URI = os.getenv("BRAIN_URI", "ws://192.168.0.5:8765")
```

**그 외에는 아무것도 바꾸지 않는다.** `recv_loop`, 툴 실행, 전사 처리, 재연결 루프,
`send_realtime_input` / `send_client_content` / `send_tool_response` 호출부는 모두 그대로
동작한다 — shim이 Gemini Live 세션과 같은 면을 제공한다. 구조를 고치기 시작하면
그때부터는 검증된 코드가 아니다.

---

## 3. `.env` — 두 가지

```ini
# 로컬 뇌를 쓰지만 launcher.py:211이 값이 비어 있으면 대화를 시작하지 않고 반환한다.
# 아무 값이나 넣으면 된다 — 어디에도 전송되지 않는다.
GOOGLE_API_KEY=local

# Q14 결정: Piper 음성을 가공 없이 쓴다. 이걸 켜두면 700ms 버퍼가 붙어
# 끼어들기 반응과 첫 소리가 그만큼 늦어진다.
ENABLE_VOICE_SHIFT=false
```

`BRAIN_URI`를 `.env`로 주고 싶으면 위 2번의 기본값 대신 거기에 적어도 된다.

### `ENABLE_AEC`는 반드시 확인할 것

`.env.example`은 `true`지만 `requirements-jetson.txt`는 *"aec-audio-processing: Linux 휠이
아예 없음 → `ENABLE_AEC=false` + PulseAudio `module-echo-cancel`로 우회"*라고 적고 있고,
그 PulseAudio 설정은 **이후 revert된 Xavier에서 확정**된 것이다. Orin Nano에서 어떻게
해결됐는지가 저장소에 기록돼 있지 않다(Q16).

**지금 실제로 어떤 경로로 AEC가 동작하는지 확인하고 저장소 문서를 갱신할 것.** 아래
4번의 첫 번째 확인 항목이 이것이다.

---

## 4. 붙이고 나서 볼 것 — 순서대로

### ① AEC (최대 위험)

로봇이 **자기 목소리에 스스로 끼어드는지**. 뇌의 barge-in은 70ms로 빠르고, 그만큼 잘
자해한다.

**증상**: 모티가 말하는 중에 AGX 서버 로그에 이 줄이 반복된다.

```
INFO barge-in: turn cancelled
```

모티가 말하는 동안 사용자가 조용한데도 이게 찍히면 에코가 마이크로 되돌아오는 것이다.

**응급 조치**: 확인만 하려면 이어폰을 꽂으면 에코 경로가 사라진다(로봇 저장소도 그렇게
격리해 원인을 규명한 적이 있다). 근본 해결은 AEC이고, 그게 §9.2의 `[MANDATORY]`다.

### ② 소리가 정상 속도·음높이로 나는가

뇌가 22,050Hz Piper 출력을 24,000Hz로 변환해 보낸다(로봇의 `OUTPUT_RATE`에 맞춤).
9% 빠르고 높게 들리면 변환이 안 된 것이다.

### ③ 턴이 제때 끊기는가

- **말하는 중에 끊긴다** → `stop_secs`/빠른 경로가 이 마이크·환경에서 너무 공격적이다.
  AGX의 `brain/vad.py`에서 `fast_confidence=0.99`로 바꾸면 빠른 경로가 꺼진다(한 줄).
- **말을 끝냈는데 한참 기다린다** → 거부권이 과하게 붙잡고 있다. `max_wait`를 줄인다.

측정된 기준값: 자연 발화에서 말 멈춤 → 첫 오디오 **약 2.0초**.

### ④ 대화 기록이 제대로 저장되는가

세션 종료 후 결과지(마음처방전)와 대화록에 **사용자 발화가 들어있는지** 확인한다.
E4B가 전사를 만들어 `input_transcription`으로 보내는데, 이 경로는 실물에서 확인된 적이
없다. 비어 있으면 연구 데이터가 유실된다.

### ⑤ 툴이 실제로 실행되는가

표정이 바뀌고(`set_emotion`), 제스처가 나오고(`play_gesture`), 이름을 기억하는지
(`remember_fact`). AGX 로그에 `tool_call`이 찍히고 로봇이 결과를 돌려주면 정상이다.

### ⑥ 동시 구동 부하

모터·카메라·표정 UI가 같이 도는 상태에서 오디오가 끊기거나 지연이 늘어나는지.
Orin Nano는 8GB이고 뇌와 달리 여유가 없다.

---

## 5. 되돌리는 법

| 문제 | 되돌림 |
|---|---|
| 뇌 쪽이 아예 안 되는 경우 | `launcher.py` 540행을 원래대로 + `.env`의 `GOOGLE_API_KEY`에 실제 키 → Gemini Live로 즉시 복귀 |
| 말하는 중에 끊긴다 | AGX `brain/vad.py`의 `fast_confidence=0.99` |
| 맞장구("응")가 거슬린다 | 로봇에서 `local_live.connect(..., backchannel=False)` |
| 응답이 느려도 끊기지 않는 쪽을 원한다 | 위 `fast_confidence=0.99`와 같음 |

로봇 쪽 변경이 한 줄이라, **Gemini로 되돌리는 것도 한 줄**이다. 그게 이 설계의 목적이다.

---

## 6. 아직 없는 것 (기대하지 말 것)

- **감정 담긴 음성 출력** — Piper 한국어 음성은 하나뿐이고 감정 제어가 없다(§8.3)
- **Tailscale 원격** — Stage 6, 미착수. 지금은 같은 LAN만
- **500ms 이하 지연** — 약 2.0초다. 남은 지렛대는 q4 양자화(§13.6)
- **긴 세션 검증** — 대화 기록은 최근 20교환까지만 유지된다(§13.1)
