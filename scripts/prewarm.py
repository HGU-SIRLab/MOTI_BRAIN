"""페르소나를 미리 프리필해 둔다 — 아무도 기다리지 않을 때 18K 값을 치른다.

왜 필요한가: 한 번도 프리필되지 않은 페르소나로 처음 붙는 세션은 첫 소리까지 약 17.9초가
걸린다(스펙 §13.9). `launcher.py`는 연결하자마자 인사 턴을 밀어넣으므로, 사람이 로봇 앞에
선 채로 그만큼 기다리게 된다. vLLM의 prefix 캐시는 GPU에 있고 **프로세스가 사는 동안
유지되므로**, 여기서 한 번 데워두면 로봇이 같은 문자열을 보낼 때 그대로 적중한다.

무엇을 데우는가: 얼굴 인식이 실패하거나 미등록인 경우 `launcher.py:850`이
`name=None, facts_summary=None`을 넘긴다. 이건 **바이트 단위로 재현 가능한 고정 문자열**이라
확실하게 데울 수 있다. 등록된 사용자의 페르소나는 이름이 프롬프트 1.17% 지점에 들어가
캐시가 거의 공유되지 않으므로, 이름을 알면 인자로 넘겨서 같이 데운다.

사용:
    PYTHONPATH= .venv_tts/bin/python scripts/prewarm.py            # 미지 사용자만
    PYTHONPATH= .venv_tts/bin/python scripts/prewarm.py 조형민 김철수  # 지정한 이름도 함께

주의: KV 예산이 페르소나 3.7개분뿐이다(67,712토큰 ÷ 18,344). 너무 많이 데우면 LRU로
서로 밀어낸다.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/herobot/MOTI-HRI")
try:
    from core.utils import build_persona_system_instruction as build
except ImportError:
    sys.exit("MOTI-HRI를 찾을 수 없다 — /home/herobot/MOTI-HRI 가 있는지 확인할 것")

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"


def warm(system: str, label: str) -> float:
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "."}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    dt = time.perf_counter() - t0
    print(f"  {label:<26} {dt:6.2f}s  ({len(system):,}자)")
    return dt


def main() -> None:
    names = sys.argv[1:]
    print("페르소나 예열 — 여기서 내는 시간만큼 로봇 앞의 사람이 안 기다린다")
    warm(build(name=None, facts_summary=None), "미지 사용자")
    for n in names:
        # facts_summary는 로봇 디스크의 프로필에서 오므로 여기서 정확히 재현할 수 없다.
        # 그 사람에게 기억된 사실이 있으면 이 예열은 빗나가고 첫 턴에 17.9초를 낸다.
        warm(build(name=n, facts_summary=None), f"{n} (facts 없음 가정)")
    dt = warm(build(name=None, facts_summary=None), "미지 사용자 (적중 확인)")
    if dt > 2.0:
        print("\n⚠️  2회차가 2초를 넘었다 — prefix 캐시가 안 듣고 있다. "
              "vLLM 로그와 --gpu-memory-utilization 을 확인할 것.")
    else:
        print("\n예열 완료. 첫 인사가 바로 나온다.")


if __name__ == "__main__":
    main()
