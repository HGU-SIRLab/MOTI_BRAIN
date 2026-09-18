# EXP-13 test recordings — manifest

Audio lives in `testdata/` and is **not committed** (public repo, and voice is
cloneable from seconds of audio — see the §15 discussion). The transcripts below
are committed so the pairing survives, and so EXP-4 has reference text for CER later.

Recorded by the project's own developer, whose voice matches the robot's actual user
population (Handong students, per the profile fields in `MOTI-HRI`).

| File | Purpose | Tests |
|---|---|---|
| `a1_tired.wav` | 일상 발화 — 피로 | criterion 1: comprehension |
| `a2_happy.wav` | 일상 발화 — 성취/기쁨 | criterion 1 |
| `a3_anxious.wav` | 일상 발화 — 불안 | criterion 1 |
| `b1_long.wav` | 45~60s 연속 발화 | 30s ceiling + multi-clip split (§12.3) |
| `c1_neutral.wav` | "나 진짜 괜찮아…" 담담하게 | tone channel vs Whisper |
| `c2_suppressed.wav` | 같은 문장, 울컥해서 참으며 | tone channel vs Whisper |
| `c3_bright.wav` | 같은 문장, 밝고 경쾌하게 | tone channel vs Whisper |

## Transcripts

**a1_tired** — 아, 며칠 내내 밤을 새웠더니 진짜 너무 피곤해. 눈이 막 감기네. 너도 피곤하지?

**a2_happy** — 드디어! 며칠 동안 고생했던 프로젝트가 방금 성공적으로 끝났어. 기분 최고야!

**a3_anxious** — 내일 중요한 발표인데 아직 준비를 다 못해서 너무 불안해. 중간에 실수하면 어떡하지?

**b1_long** — 솔직히 요즘 생각이 진짜 많아졌어. 처음에는 그냥 내가 구상한 아이디어를 직접 구현해
보고 싶어서 시작한 건데, 막상 하다 보니까 하드웨어 이슈부터 소프트웨어 통합까지 해결해야 할
문제들이 꼬리에 꼬리를 무는 거야. 특히 원인을 알 수 없는 에러가 계속 뜰 때는 진짜 다 엎어버리고
싶기도 하고... 남들은 척척 잘 해내는 것 같은데 나만 제자리걸음인 것 같아서 가끔 되게 답답해.
그래도 어제 테스트에서 조금 진전이 있는 걸 보니까, 내가 완전 틀린 방향으로 가고 있는 건 아닌 것
같아서 그나마 숨통이 트이더라. 근데 또 당장 다음 주까지 결과물 정리하려면 이번 주말도 꼼짝없이
반납해야겠지. 아무튼 요즘 기분이 좀 롤러코스터 같은데, 이렇게라도 털어놓으니까 속은 좀 시원하네.

**c1_neutral / c2_suppressed / c3_bright** — 나 진짜 괜찮아. 너무 신경 안 써도 돼.

The c-set must be **word-for-word identical**; only prosody differs. That is the whole
point — a Korean-tuned Whisper would emit the same text for all three, so if the model's
three responses differ, the tone channel is doing real work and §7's separate SER model
is redundant.

## Recording conditions that matter

- Speak it, don't read it. Read-aloud prosody differs from spontaneous speech, and
  Whisper degrades more on spontaneous speech — testing on read speech would flatter
  both paths and hide the difference we are trying to measure.
- Same speaker, mic, distance and room for all seven files, or the comparison is confounded.
- `b1_long` must run continuously. A brief pause is fine; 3+ seconds of silence is not,
  because in production VAD would have cut the turn there and the ceiling test would not
  be exercised.
- The c-set differences should be clearly distinguishable but natural — theatrical
  exaggeration would prove nothing about real conversational tone.
