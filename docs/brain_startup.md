# 뇌 기동 절차 (AGX Orin)

재부팅했거나 뭔가 죽었을 때, **지금 돌고 있는 상태를 그대로** 되살리는 방법.

---

## 한 줄

```bash
bash /home/herobot/moti_brain/scripts/start_brain.sh
```

이미 떠 있는 것은 건드리지 않는다. **몇 번을 돌려도 안전하므로 상태가 의심스러우면
그냥 다시 돌리면 된다** —
다 살아 있으면 몇 초 만에 끝나고, 죽은 것만 골라 올린다.

등록된 사용자가 있으면 이름을 넘긴다 (§4 참고):

```bash
bash scripts/start_brain.sh 조형민
```

---

## ⏱ 얼마나 걸리는가

| 상황 | 시간 |
|---|---|
| 전부 살아 있음 | **몇 초** |
| 뇌 서버만 죽음 | **약 10초** |
| **vLLM까지 죽음 / 재부팅 직후** | **약 29분** |

**29분의 대부분은 vLLM의 "model loading" 구간이고 원인이 규명되지 않았다**(스펙 Q17 —
가중치 읽기는 3.8초인데 그 단계가 28분을 쓴다). 로그가 그동안 한 줄도 안 찍혀서 멈춘 것처럼
보이지만 정상이다. `docker stats moti_vllm`으로 살아있는지 확인할 수 있다.

**그래서 vLLM은 함부로 끄지 않는다.** 이건 원래 상시 가동 서버로 설계된 것이다(스펙 §1).
뇌 서버 코드만 고쳤다면 vLLM은 그대로 두고 뇌 서버만 다시 띄우면 5초다 — 이 둘을 혼동해서
29분을 여러 번 날린 적이 있다.

---

## 스크립트가 하는 일 (직접 하려면)

### 1. 전원 모드

```bash
nvpmodel -q                  # "NV Power Mode: MAXN" 이어야 한다
sudo nvpmodel -m 0           # 아니면 이걸로 바꾸고 재부팅
```

MAXN은 저장되는 설정이라 재부팅해도 유지된다.

`sudo jetson_clocks`는 **유지되지 않고 재부팅마다 풀린다.** 스크립트는 이걸 자동으로 걸지
않는다 — 2026-09-21의 측정값(평균 2.53초)이 `jetson_clocks` 없이 나온 값이라, 지금 상태를
그대로 재현하는 게 목적이기 때문이다. **벤치마크(EXP-8, EXP-10)를 돌릴 때만** 직전에
수동으로 걸고, 그 사실을 결과와 함께 적는다.

### 2. vLLM

```bash
bash /home/herobot/moti_brain/scripts/run_vllm.sh
# 준비 확인 (약 29분 뒤)
curl -sf http://localhost:8000/v1/models
```

실행 인자는 그 스크립트 안에 있고 각각 이유가 주석으로 붙어 있다. 요약하면:

| 인자 | 왜 |
|---|---|
| `--max-model-len 32768` | 실제 페르소나가 18,344토큰이라 16384로는 요청 하나도 못 받는다 |
| `--gpu-memory-utilization 0.40` | 젯슨은 CPU/GPU 메모리가 통합이라 기본 0.9면 시스템이 굶는다. TTS도 같은 보드에 있다 |
| `--mm-processor-cache-gb 0` | 켜두면 `Expected a cached item for mm_hash=...`로 엔진이 죽는다 (스펙 §13.7) |
| `--enable-auto-tool-choice` 외 | gemma4 파서 |

컴파일 캐시(`.vllm_compile_cache/`, 123MB)가 마운트돼 있어서 콜드 컴파일 85초를 다시 내지
않는다. **플래그를 바꾸면 캐시 키가 갈라져 그 85초만 다시 낸다** (29분과는 별개다).

### 3. 뇌 서버

```bash
cd /home/herobot/moti_brain
PYTHONPATH= nohup .venv_tts/bin/python brain/server.py >> logs/brain.log 2>&1 &
```

🔴 **`PYTHONPATH=` 를 반드시 붙인다.** ROS2/HARU setup이 전역 `PYTHONPATH`에 HARU의 venv를
넣어두어서, 빼면 젯슨 torch와 충돌하는 PyPI 패키지가 보이고 `undefined symbol`로 죽는다.
이 프로젝트의 파이썬·pip은 **항상** 이 접두사를 붙여 실행한다.

죽일 때는 `pkill`을 쓰지 않는다 — 패턴이 자기 셸 명령줄에도 걸려서 셸째로 죽는다(exit 144,
두 번 당했다):

```bash
kill "$(pgrep -f 'venv_tts/bin/python [b]rain/server.py')"
```

### 4. 페르소나 예열

```bash
PYTHONPATH= .venv_tts/bin/python scripts/prewarm.py            # 미지 사용자
PYTHONPATH= .venv_tts/bin/python scripts/prewarm.py 조형민       # 등록 사용자도 함께
```

**안 하면 로봇 앞의 첫 사람이 약 17.9초를 기다린다.** 18,344토큰 페르소나의 프리필 값이고,
`launcher.py`는 연결하자마자 인사 턴을 밀어넣으므로 그 침묵이 사람 앞에서 그대로 흐른다.
예열은 아무도 없을 때 그 값을 미리 치르는 것이다. vLLM의 prefix 캐시는 GPU에 있고
**프로세스가 사는 동안 유지**되므로 한 번이면 된다.

무엇이 데워지는가:
- **미지 사용자**(`name=None`) — 얼굴 인식 실패/미등록 경로. 바이트 단위로 재현 가능한
  고정 문자열이라 확실하게 적중한다
- **이름을 넘긴 사용자** — 다만 그 사람에게 기억된 사실(`facts_summary`)이 있으면
  빗나간다. 그 값은 로봇 디스크의 프로필에서 오고 AGX에서는 재현할 수 없다

이름과 facts가 프롬프트의 **1.17% / 3.54% 지점**에 들어가서, 다르면 18,400토큰 중 200~650개만
공유된다. 즉 **사람이 바뀌면 사실상 전부 다시 계산한다.** KV 예산은 페르소나 3.7개분뿐이라
너무 많이 데우면 LRU로 서로 밀어낸다.

---

## 확인

```bash
curl -sf http://localhost:8000/v1/models     # vLLM
ss -ltn | grep 8765                          # 뇌 서버
tail -f /home/herobot/moti_brain/logs/brain.log
```

진짜로 대화가 되는지까지 보려면 (약 1분):

```bash
cd /home/herobot/moti_brain
PYTHONPATH= .venv_tts/bin/python client/test_local_live.py
```

`launcher.py`의 수신 루프를 구조 그대로 재현하고 툴 왕복까지 확인한다.
전체 검사(8종)는 `docs/` 대신 각 테스트를 직접 돌린다 — `brain/test_*.py`, `client/test_*.py`.

---

## 로그에서 읽을 것

| 줄 | 뜻 |
|---|---|
| `robot connected: (...)` | 핸드셰이크 성공 |
| `barge-in: turn cancelled` | 끼어들기. **사용자가 조용한데 반복되면 에코다 → AEC 문제** |
| `backchannel (1440ms of silence)` | 맞장구 발화 |
| `speculation adopted (N ready, ...)` | 투기적 생성이 쓰였다 |
| `client silent 6.5s — closing turn anyway` | 스톨 워치독. 오디오 스트림이 끊겼다 |
| `turn failed` + 트레이스백 | 뇌 쪽 예외. 이게 보이면 트레이스백 전체가 필요하다 |

---

## 잘 안 될 때

| 증상 | 확인할 것 |
|---|---|
| vLLM이 29분 넘게 안 뜬다 | `docker logs moti_vllm | tail -30`. CPU와 블록 I/O가 움직이면 아직 사는 중이다 |
| 컨테이너가 바로 죽는다 | 메모리일 가능성이 높다. `sudo sysctl -w vm.drop_caches=3` 후 재시도 (스펙 §5.3) |
| `ModuleNotFoundError` | `PYTHONPATH=` 를 빠뜨렸다 |
| `undefined symbol` (torch 관련) | 같은 원인. PyPI torch가 젯슨 torch를 가린 것 |
| 8765는 열렸는데 응답이 없다 | vLLM이 죽었을 수 있다. 뇌 서버는 vLLM 없이도 떠 있는다 |
| 첫 인사가 계속 느리다 | 예열이 빗나간 것. 얼굴 인식이 이름을 찾아냈다면 그 이름으로 예열해야 한다 |
| 셸이 exit 144로 죽었다 | `pkill -f` 를 썼다. 위 §3의 `pgrep` + `kill` 방식으로 |

컨테이너를 지워야 한다면 **컴파일 캐시부터 꺼낸다** (마운트를 빠뜨린 채 띄웠던 경우):

```bash
docker cp moti_vllm:/root/.cache/vllm/torch_compile_cache/. /home/herobot/moti_brain/.vllm_compile_cache/
```

---

## 로봇 쪽

로봇을 붙이는 절차는 별도 문서다 — `docs/robot_integration.md`.
로봇이 접속할 주소는 기동 스크립트가 마지막에 출력한다 (유선 `192.168.0.5`, WiFi
`192.168.0.247`).
