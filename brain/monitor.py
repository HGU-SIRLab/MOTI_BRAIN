"""뇌 내부를 실시간으로 들여다보는 창 — jtop의 파이프라인 판.

로봇이 앞에서 대화하는 동안 뇌 안에서 무슨 일이 일어나는지 보려면 지금은 로그를 tail 하는
수밖에 없는데, 로그에는 VAD 확률도, 턴별 타이밍도, prefix 캐시 적중률도 없다. 여기서
그것들을 한 화면에 모은다.

**설계 원칙 하나: 로봇 경로에 절대 영향을 주지 않는다.** `publish()`는 보는 사람이 없으면
아무것도 하지 않고, 있어도 큐에 넣고 바로 돌아온다. 느린 뷰어나 끊긴 뷰어가 대화를
지연시키면 안 된다 — 모니터는 관찰자이지 참여자가 아니다.

마이크가 필요 없어서 `web/test_client.html`과 달리 보안 컨텍스트 제약이 없다. 그래서 AGX에서
바로 서빙하고 노트북 브라우저로 http://<AGX>:8766/ 을 열면 된다.

Run: 뇌 서버가 자동으로 띄운다 (brain/server.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
import urllib.request
from collections import deque
from pathlib import Path

import websockets
from websockets.http11 import Response

log = logging.getLogger("brain.monitor")

PORT = 8766
PAGE = Path(__file__).resolve().parent.parent / "web" / "monitor.html"
# 뷰어가 세션 중간에 붙어도 직전 맥락은 보이게. 대화 몇 턴이면 충분하다.
RECENT = deque(maxlen=300)
_clients: set = set()
_t0 = time.monotonic()


def publish(_kind: str, /, **data) -> None:
    """이벤트 하나를 뷰어들에게. 보는 사람이 없으면 즉시 반환한다.

    **무슨 일이 있어도 예외를 밖으로 내보내지 않는다.** 이 파일 맨 위에 "로봇 경로에 절대
    영향을 주지 않는다"고 써놓고, 바로 그 변경에서 호출부가 `kind=`를 키워드로 넘겨
    TypeError를 냈고 턴 하나가 통째로 죽었다(2026-09-22). 관찰자가 관찰 대상을 죽이면
    없느니만 못하다.

    그래서 첫 인자를 **위치 전용(`/`)**으로 둔다. 함수 본문의 try로는 못 막는다 — 이름
    충돌은 본문에 들어오기도 전에 호출 바인딩에서 터지기 때문이다. 이제 `kind=`든
    `data=`든 무엇을 키워드로 넘겨도 그냥 필드가 되고, 본문의 try가 나머지를 받는다.
    """
    try:
        ev = {**data, "t": round(time.monotonic() - _t0, 3), "kind": _kind}
        RECENT.append(ev)
        if not _clients:
            return
        payload = json.dumps(ev, ensure_ascii=False, default=repr)
        for q in list(_clients):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass      # 느린 뷰어는 버린다. 대화가 우선이다.
    except Exception:                                  # noqa: BLE001 — 위 docstring
        log.exception("monitor.publish(%r) 실패 — 대화는 계속한다", _kind)


# ---- 시스템/서빙 샘플러 ------------------------------------------------------
_TEGRA = {
    "ram": re.compile(r"RAM (\d+)/(\d+)MB"),
    "gpu": re.compile(r"GR3D_FREQ (\d+)%"),
    "temp": re.compile(r"tj@([\d.]+)C"),
    "gpu_mw": re.compile(r"VDD_GPU_SOC (\d+)mW"),
    "cpu_mw": re.compile(r"VDD_CPU_CV (\d+)mW"),
}


async def _tegrastats() -> None:
    """jtop이 보여주는 것들. sudo 없이 돈다."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "tegrastats", "--interval", "2000",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    except FileNotFoundError:
        log.info("tegrastats 없음 — 시스템 패널은 비워둔다")
        return
    try:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace")
            out = {}
            if m := _TEGRA["ram"].search(line):
                out["ram_used"], out["ram_total"] = int(m[1]), int(m[2])
            if m := _TEGRA["gpu"].search(line):
                out["gpu"] = int(m[1])
            if m := _TEGRA["temp"].search(line):
                out["temp"] = float(m[1])
            watts = 0
            for k in ("gpu_mw", "cpu_mw"):
                if m := _TEGRA[k].search(line):
                    watts += int(m[1])
            if watts:
                out["watts"] = round(watts / 1000, 1)
            if out:
                publish("sys", **out)
    finally:
        proc.terminate()


def _scrape_vllm() -> dict:
    """prefix 캐시 적중률이 여기서 나온다 — 예열이 듣고 있는지 보는 유일한 방법."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=3) as r:
            body = r.read().decode()
    except Exception:                                  # noqa: BLE001 — 없으면 없는 대로
        return {}
    def grab(pat: str) -> float | None:
        m = re.search(pat + r'[^}]*}\s+([0-9.e+]+)', body)
        return float(m[1]) if m else None
    hit = grab(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"')
    miss = grab(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"')
    out = {}
    if hit is not None and miss is not None and (hit + miss):
        out["cache_hit"] = round(100 * hit / (hit + miss), 1)
        out["prefill_tokens"] = int(hit + miss)
    for name, pat in (("running", r"vllm:num_requests_running"),
                      ("waiting", r"vllm:num_requests_waiting"),
                      ("gen_tokens", r"vllm:generation_tokens_total")):
        v = grab(pat)
        if v is not None:
            out[name] = int(v)
    return out


async def _vllm_loop() -> None:
    loop = asyncio.get_running_loop()
    while True:
        stats = await loop.run_in_executor(None, _scrape_vllm)
        if stats:
            publish("vllm", **stats)
        await asyncio.sleep(2.0)


# ---- 전송 --------------------------------------------------------------------
async def _handler(ws) -> None:
    q: asyncio.Queue = asyncio.Queue(maxsize=400)
    _clients.add(q)
    log.info("monitor attached (%d watching)", len(_clients))
    try:
        await ws.send(json.dumps({"kind": "backlog", "events": list(RECENT)},
                                 ensure_ascii=False))
        while True:
            await ws.send(await q.get())
    except websockets.ConnectionClosed:
        pass
    finally:
        _clients.discard(q)
        log.info("monitor detached (%d watching)", len(_clients))


def _static(connection, request):
    """웹소켓이 아닌 요청이면 페이지를 돌려준다 — 별도 파일 서버가 필요 없게."""
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    try:
        body = PAGE.read_bytes()
    except OSError:
        return Response(404, "Not Found", websockets.Headers(), b"monitor.html missing")
    return Response(200, "OK", websockets.Headers({
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": str(len(body)),
    }), body)


async def start(port: int = PORT) -> None:
    """뇌 서버가 부를 것. 실패해도 대화는 계속되어야 하므로 예외를 삼킨다."""
    try:
        await websockets.serve(_handler, "0.0.0.0", port, process_request=_static)
    except OSError as exc:
        log.warning("monitor 포트 %d 못 염 (%s) — 대화에는 영향 없음", port, exc)
        return
    asyncio.create_task(_tegrastats())
    asyncio.create_task(_vllm_loop())
    log.info("monitor on http://0.0.0.0:%d", port)
