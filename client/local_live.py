"""Drop-in replacement for `client.aio.live.connect()` against the local brain.

Goal (§11.5, §20 rule 17): the robot's diff is the `connect()` call and nothing else.
So this deliberately mimics the Gemini Live session surface that `launcher.py` already
uses — the same four methods, and `receive()` as an async generator that ends at each
turn boundary, because launcher re-enters it inside a `while` loop.

It duck-types the `google.genai.types` objects launcher constructs (`Blob`, `Content`,
`FunctionResponse`) rather than importing the SDK: the brain has no reason to depend on
Google's client library, and reading `.data` / `.parts[0].text` / `.id,.name,.response`
is all that is needed.

Usage on the robot — replace

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:

with

    async with local_live.connect(BRAIN_URI, config=config) as session:

`config` may stay a `types.LiveConnectConfig`; only `system_instruction` and `tools`
are read from it.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import typing
import uuid
from types import SimpleNamespace

import websockets

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

# One id per robot process. launcher.py's connection_manager() calls connect() afresh on
# every reconnect, so the id cannot live on the connection object; and it must NOT be
# derived from the persona, because the persona changes mid-session the moment
# remember_fact learns the user's name — which is exactly when losing history would hurt.
SESSION_ID = uuid.uuid4().hex


def tool_schemas(tools) -> list[dict]:
    """Derive OpenAI-style declarations from plain callables.

    `launcher.py` builds `tools` as a list of functions and lets google-genai infer the
    declarations from signatures (§11.5). The brain needs the same information, so we
    infer it here and send it at handshake.
    """
    out = []
    for fn in tools or []:
        if not callable(fn):
            continue
        props, required = {}, []
        try:
            sig = inspect.signature(fn)
            hints = typing.get_type_hints(fn)
        except (TypeError, ValueError, NameError):
            sig, hints = None, {}
        for pname, param in (sig.parameters.items() if sig else []):
            if pname in ("self", "cls"):
                continue
            hint = hints.get(pname, str)
            origin = typing.get_origin(hint)
            if origin is typing.Literal:
                props[pname] = {"type": "string",
                                "enum": [str(a) for a in typing.get_args(hint)]}
            else:
                props[pname] = {"type": _JSON_TYPES.get(hint, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        out.append({"type": "function", "function": {
            "name": getattr(fn, "__name__", "tool"),
            "description": (inspect.getdoc(fn) or "").split("\n\n")[0][:300],
            "parameters": {"type": "object", "properties": props,
                           "required": required}}})
    return out


def _empty_message() -> SimpleNamespace:
    """A message shaped like Gemini's, with everything launcher.py reads present.

    `go_away` and `session_resumption_update` stay None — they exist because Gemini
    expires sessions. The brain does not, but launcher's reconnect loop is still
    load-bearing over Tailscale (§11.5), so the fields must not vanish.
    """
    return SimpleNamespace(
        data=None, tool_call=None, go_away=None, session_resumption_update=None,
        server_content=SimpleNamespace(
            interrupted=False, turn_complete=False,
            input_transcription=None, output_transcription=None))


class Session:
    def __init__(self, ws):
        self._ws = ws
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._pending_rate: int | None = None
        self.resumed = False            # set from the brain's `ready` (§11.5)
        self.turns_carried = 0
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for frame in self._ws:
                if isinstance(frame, bytes):
                    msg = _empty_message()
                    msg.data = frame
                    await self._inbox.put(msg)
                    continue
                payload = json.loads(frame)
                kind = payload.get("t")
                if kind == "audio":
                    self._pending_rate = payload.get("rate")
                    continue
                if kind == "ready":
                    self.resumed = bool(payload.get("resumed"))
                    self.turns_carried = payload.get("turns", 0)
                    continue
                msg = _empty_message()
                if kind == "transcript":
                    field = ("input_transcription" if payload["role"] == "user"
                             else "output_transcription")
                    setattr(msg.server_content, field,
                            SimpleNamespace(text=payload["text"]))
                elif kind == "interrupted":
                    msg.server_content.interrupted = True
                elif kind == "turn_complete":
                    msg.server_content.turn_complete = True
                elif kind == "tool_call":
                    msg.tool_call = SimpleNamespace(function_calls=[
                        SimpleNamespace(id=c.get("id", ""), name=c["name"],
                                        args=c.get("args") or {})
                        for c in payload["calls"]])
                elif kind == "error":
                    continue
                await self._inbox.put(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._inbox.put(None)          # unblock receive() on disconnect

    @property
    def output_rate(self) -> int | None:
        """Sample rate of the most recent audio frame. Piper is 22,050Hz while the
        robot's player was built for Gemini's 24kHz (§8.3) — it must resample or
        reconfigure rather than assume."""
        return self._pending_rate

    async def send_realtime_input(self, audio) -> None:
        await self._ws.send(audio.data)

    async def send_client_content(self, turns=None, turn_complete=True) -> None:
        text = ""
        for part in getattr(turns, "parts", None) or []:
            text += getattr(part, "text", "") or ""
        await self._ws.send(json.dumps({"t": "text", "text": text},
                                       ensure_ascii=False))

    async def send_tool_response(self, function_responses=None) -> None:
        results = [{"id": getattr(r, "id", ""), "name": getattr(r, "name", ""),
                    "result": (getattr(r, "response", None) or {}).get("result")}
                   for r in function_responses or []]
        await self._ws.send(json.dumps({"t": "tool_result", "results": results},
                                       ensure_ascii=False))

    async def receive(self):
        """Yields until the turn ends — Gemini's semantics, which launcher relies on."""
        while True:
            msg = await self._inbox.get()
            if msg is None:
                return
            yield msg
            if msg.server_content and (msg.server_content.turn_complete
                                       or msg.server_content.interrupted):
                return

    async def close(self) -> None:
        self._reader.cancel()


class _Connect:
    def __init__(self, uri: str, config):
        self._uri, self._config = uri, config
        self._ws = None
        self._session: Session | None = None

    async def __aenter__(self) -> Session:
        self._ws = await websockets.connect(self._uri, max_size=None)
        system = getattr(self._config, "system_instruction", None) or ""
        tools = getattr(self._config, "tools", None) or []
        await self._ws.send(json.dumps(
            {"t": "hello", "system": system, "tools": tool_schemas(tools),
             "session_id": SESSION_ID},
            ensure_ascii=False))
        self._session = Session(self._ws)
        return self._session

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()
        if self._ws:
            await self._ws.close()


def connect(uri: str, config=None, **_ignored) -> _Connect:
    """Signature-compatible with `client.aio.live.connect(model=..., config=...)`;
    `model` is accepted and ignored, since the brain decides what it serves."""
    return _Connect(uri, config)
