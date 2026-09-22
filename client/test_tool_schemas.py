"""툴 선언이 모델에게 '무엇을 넣어야 하는지'까지 전달되는가.

2026-09-22 실물 세션: 대화 턴마다 `set_emotion({})`이 와서 전부 실패했고, 로컬 뇌 모드에서
로봇 표정이 한 번도 안 바뀌었다. 원인은 `tool_schemas()`가 docstring을 첫 빈 줄에서 잘라
보낸 것 — 이 로봇의 툴은 전부 유효값을 `Args:` 절에 적어두는데, 그게 첫 빈 줄 *뒤*에 있다.
모델은 툴이 있다는 것만 듣고 무엇을 넣을지는 못 들었다.

google-genai는 콜러블을 통째로 넘겨서 모델이 docstring을 다 본다. shim이 그 면을 흉내내는
이상 여기서도 같은 정보가 가야 한다(§11.5).

서버도 vLLM도 필요 없다. Run: PYTHONPATH= .venv_tts/bin/python client/test_tool_schemas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.local_live import tool_schemas  # noqa: E402


# 로봇의 실제 docstring 모양 그대로 (core/{emotion,memory,motion}_tools.py).
def set_emotion(emotion: str) -> str:
    """Set the robot's facial expression to match your current emotional tone.

    Call this once per response, right before or as you start speaking,
    so the robot's face matches what you're about to say.

    Args:
        emotion: one of "neutral", "happy", "excited", "tender", "scared", "angry", "sad", "surprised", "listening", "thinking", "scanning".
    """


def remember_fact(field: str, value: str, confidence: str) -> str:
    """Remember something about the user.

    Args:
        field: what kind of information this is (e.g. "name", "grade", "major", "mbti", "gender", "rc"), or any free-form label if it doesn't fit those.
        value: the value learned, as plain text.
        confidence: "certain" if they stated it directly, "inferred" if you are guessing from context.
    """


def express_gesture(joint: str, intensity: float, speed: str = "normal",
                    repeat: int = 1) -> str:
    """Play a small, tunable body movement on one joint.

    The robot's real signature (`core/motion_tools.py:80`). It used to be written here as
    `play_manual_motion(joint, …)`, a name that does not exist as a tool at all — that
    was an internal helper taking a gesture *name*. The wrong name started in the spec,
    reached this test and the stand-ins, and was sent to the robot as "verified" before
    anyone checked it against the source (§12.6).

    Args:
        joint: one of "right_arm", "left_arm", "shoulder".
        intensity: how big the movement is, from 0.0 (barely noticeable) to 1.0 (full range).
        speed: "slow", "normal", or "fast".
        repeat: how many times to repeat the movement, 1-3.
    """


def no_args() -> str:
    """Just do the thing."""


def main() -> None:
    by_name = {s["function"]["name"]: s["function"]
               for s in tool_schemas([set_emotion, remember_fact,
                                      express_gesture, no_args])}
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # 1) 이게 실물에서 깨졌던 것. 유효값이 모델에게 도달해야 한다.
    emo = by_name["set_emotion"]["parameters"]["properties"]["emotion"]
    check(emo.get("enum") and len(emo["enum"]) == 11,
          f"set_emotion.emotion enum이 11개가 아니다: {emo.get('enum')}")
    check("sad" in (emo.get("enum") or []), "enum에 'sad'가 없다")
    check("emotion:" not in emo.get("description", ""),
          "파라미터 설명에 이름이 중복해 들어갔다")

    # 2) 호출 시점 안내는 두 번째 문단에 있다. 첫 문단만 자르면 사라진다.
    desc = by_name["set_emotion"]["description"]
    check("once per response" in desc,
          f"함수 설명이 첫 문단에서 잘렸다: {desc!r}")
    check("Args:" not in desc, "Args 절이 함수 설명에 중복해 들어갔다")

    # 3) ★ 열거처럼 보이지만 열거가 아닌 것. 여기에 enum을 붙이면 툴이 망가진다 —
    #    docstring이 "or any free-form label"이라고 명시한다.
    field = by_name["remember_fact"]["parameters"]["properties"]["field"]
    check("enum" not in field,
          f"remember_fact.field에 enum이 붙었다 — 자유 입력인데 제한된다: {field.get('enum')}")
    check(field.get("description"), "field에 설명이 없다")

    # 4) "one of"가 있는 것만 enum. 없으면 설명으로만 전달한다.
    #    관절 목록 자체는 로봇이 조정 중이므로(head_nod은 쓰지 않는 모터라 제외 예정)
    #    개수를 단언하지 않는다 — 남의 docstring 변경에 이 테스트가 깨지면 안 된다.
    props = by_name["express_gesture"]["parameters"]["properties"]
    check("right_arm" in (props["joint"].get("enum") or []),
          f"joint enum이 비었거나 틀렸다: {props['joint'].get('enum')}")
    check("enum" not in props["speed"],
          "speed는 'one of'가 없어 enum을 붙이지 않는 게 맞다")
    check(props["intensity"]["type"] == "number", "intensity 타입이 number가 아니다")
    check(props["repeat"]["type"] == "integer", "repeat 타입이 integer가 아니다")
    check("repeat" not in by_name["express_gesture"]["parameters"]["required"],
          "기본값이 있는 인자가 required에 들어갔다")

    # 5) Args 절이 없는 툴도 깨지지 않아야 한다.
    check(by_name["no_args"]["parameters"]["properties"] == {},
          "인자 없는 툴에 인자가 생겼다")

    for name, f in by_name.items():
        n = len(f["parameters"]["properties"])
        enums = sum(1 for v in f["parameters"]["properties"].values() if "enum" in v)
        print(f"  {name:<20} 인자 {n}개, enum {enums}개")

    print()
    if failures:
        print("실패:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("툴 스키마 정상 — 유효값이 모델까지 도달하고, 자유 입력은 제한되지 않는다")


if __name__ == "__main__":
    main()
