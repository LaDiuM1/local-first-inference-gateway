"""응답 분류·스트림 커밋 전 검증·스트림 자원 정리 단위 테스트.

httpx·respx 없이 가짜 스트림으로 Chat Completions 성공 판정과 SSE 첫 유효 이벤트 확보, 확정된
스트림의 성공·중간 장애·취소 정리 계약을 결정적으로 검증한다. HTTP 경계 통합은 test_fallback이 맡는다.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from gateway.circuit_breaker import CircuitBreaker, LocalAttempt, LocalDecision
from gateway.relay import _stream_committed
from gateway.relay_common import (
    MAX_PRECOMMIT_INSPECTION_BYTES,
    ManagedStreamingResponse,
    StreamCleanup,
    StreamPrefix,
    StreamTruncatedError,
    is_valid_chat_completion_body,
    iter_committed_stream,
    secure_success_stream,
)

VALID_EVENT = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
DONE_EVENT = b"data: [DONE]\n\n"
ATTEMPT = LocalAttempt(LocalDecision.attempt, 0)


async def _achunks(
    chunks: list[bytes], error: Exception | None = None
) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


class FakeStreamResponse:
    """secure_success_stream용 가짜 응답 — 디코딩된 청크와 닫힘 여부를 제어한다."""

    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False
        self.status_code = 200

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return _achunks(self._chunks, self._error)

    async def aclose(self) -> None:
        self.closed = True


class FakeResponse:
    """iter_committed_stream·_stream_committed용 가짜 응답 — 닫힘만 기록한다."""

    def __init__(self) -> None:
        self.closed = False
        self.status_code = 200

    async def aclose(self) -> None:
        self.closed = True


class FakeBreaker:
    """회로 차단기 호출만 순서대로 기록한다 — 세대 전이 자체는 test_circuit_breaker가 검증한다."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record_success(self, attempt: LocalAttempt) -> None:
        self.calls.append("success")

    def record_failure(self, attempt: LocalAttempt) -> None:
        self.calls.append("failure")

    def release(self, attempt: LocalAttempt) -> None:
        self.calls.append("release")


async def _collect_prefix(prefix: StreamPrefix) -> bytes:
    initial = b"".join(prefix.initial_chunks)
    rest = b"".join([chunk async for chunk in prefix.remaining])
    return initial + rest


def _half_open_probe() -> tuple[CircuitBreaker, LocalAttempt]:
    now = [0.0]
    breaker = CircuitBreaker(1, 10.0, lambda: now[0])
    breaker.record_failure(breaker.begin())
    now[0] = 10.0
    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe
    return breaker, probe


# --- 버퍼링 2xx 본문 분류 ---


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"choices": []}', True),
        (b'{"choices": [{"delta": {}}], "extra": 1}', True),
        (b"{}", False),
        (b'{"id": "x"}', False),
        (b'{"choices": "nope"}', False),
        (b'{"choices": {"a": 1}}', False),
        (b"", False),
        (b"not json", False),
        (b"[1, 2]", False),
        (b'"a string"', False),
        (b"\xff\xfe", False),
        (b'{"choices": [], "extra": NaN}', False),
        (b'{"choices": [], "extra": Infinity}', False),
        (b'{"choices": [], "extra": -Infinity}', False),
        (b'{"choices": [], "extra": 1e400}', False),
    ],
    ids=[
        "empty-choices-list",
        "choices-with-unknown-fields",
        "empty-object",
        "missing-choices",
        "non-list-choices",
        "dict-choices",
        "empty-body",
        "not-json",
        "json-array",
        "json-string",
        "invalid-utf8",
        "nan-constant",
        "infinity-constant",
        "negative-infinity-constant",
        "overflow-number",
    ],
)
def test_is_valid_chat_completion_body(body: bytes, expected: bool) -> None:
    assert is_valid_chat_completion_body(body) is expected


# --- 스트림 커밋 전 검증 ---


@pytest.mark.anyio
async def test_secure_stream_valid_single_event_commits_without_closing() -> None:
    response = FakeStreamResponse([VALID_EVENT, DONE_EVENT])

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert not response.closed
    assert prefix.first_event_end == len(VALID_EVENT)
    assert await _collect_prefix(prefix) == VALID_EVENT + DONE_EVENT


@pytest.mark.anyio
async def test_secure_stream_valid_split_across_chunks_preserves_bytes() -> None:
    chunks = [b'data: {"choi', b'ces":[]}\n\n']
    response = FakeStreamResponse(chunks)

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert await _collect_prefix(prefix) == b"".join(chunks)


@pytest.mark.anyio
async def test_secure_stream_accepts_comment_and_blank_prefix() -> None:
    response = FakeStreamResponse([b": keep-alive\n", b"\n", VALID_EVENT])

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert not response.closed


@pytest.mark.parametrize(
    "chunks",
    [
        [b"event: message\n", VALID_EVENT],
        [b"id: 42\nretry: 500\n", VALID_EVENT],
        [b'event: message\ndata: {"choices":[]}\n\n'],
        [b'data: {"choices":[]}\r\n\r\n'],
        [b'data: {"choices":[]}\r\rdata: [DONE]\r\r'],
        [b'data: {"choices":[]}\r\r'],
        [b'data: {"choi', b'ces":[]}\r\n', b"\r\n"],
        [b'data: {"choices":[]}\r', b"\n\r", b"\n"],
    ],
    ids=[
        "event-field-then-data",
        "id-and-retry-fields",
        "event-field-same-event",
        "crlf-line-endings",
        "cr-only-line-endings",
        "cr-only-final-byte",
        "crlf-split-across-chunks",
        "crlf-delimiters-split-at-cr",
    ],
)
@pytest.mark.anyio
async def test_secure_stream_accepts_auxiliary_fields_and_line_endings(
    chunks: list[bytes],
) -> None:
    # 표준 SSE 보조 필드(event·id·retry)와 CRLF·CR 줄 끝을 유효 스트림으로 받아 커밋한다 —
    # 이들을 무효로 오판하면 정상 로컬 스트림이 실패로 기록돼 뜻하지 않게 OpenAI로 우회한다.
    response = FakeStreamResponse(chunks)

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert not response.closed
    assert await _collect_prefix(prefix) == b"".join(chunks)


@pytest.mark.parametrize("number", [b"NaN", b"1e400"], ids=["constant", "overflow"])
@pytest.mark.anyio
async def test_secure_stream_rejects_non_finite_json_number(number: bytes) -> None:
    response = FakeStreamResponse([b'data: {"choices":[],"x":' + number + b"}\n\n"])

    prefix = await secure_success_stream(response)

    assert prefix is None
    assert response.closed


@pytest.mark.parametrize(
    "chunks",
    [
        [b"<html>error</html>\n"],
        [DONE_EVENT],
        [b'data: {"id":"x"}\n\n'],
        [b"data: not-json\n\n"],
        [b'data: {"choices"'],
        [b'data: {"choices":[]}\n'],
        [],
    ],
    ids=[
        "html",
        "done-only",
        "no-choices",
        "malformed-json",
        "incomplete",
        "data-line-without-event-terminator",
        "empty",
    ],
)
@pytest.mark.anyio
async def test_secure_stream_invalid_returns_none_and_closes(
    chunks: list[bytes],
) -> None:
    response = FakeStreamResponse(chunks)

    prefix = await secure_success_stream(response)

    assert prefix is None
    assert response.closed


@pytest.mark.anyio
async def test_secure_stream_event_terminator_in_later_chunk_commits() -> None:
    # data 라인과 종료 빈 줄이 서로 다른 청크로 와도 이벤트가 완결되면 커밋하고 바이트를 보존한다.
    chunks = [b'data: {"choices":[]}\n', b"\ndata: [DONE]\n\n"]
    response = FakeStreamResponse(chunks)

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert not response.closed
    assert await _collect_prefix(prefix) == b"".join(chunks)


@pytest.mark.anyio
async def test_secure_stream_oversized_valid_event_chunk_is_rejected() -> None:
    # 유효 이벤트를 담았더라도 버퍼가 검사 상한을 넘으면 커밋하지 않는다 — 과도한 버퍼링 차단.
    padding = b"x" * (MAX_PRECOMMIT_INSPECTION_BYTES + 10)
    oversized = b'data: {"choices":[],"pad":"' + padding + b'"}\n\n'
    response = FakeStreamResponse([oversized])

    prefix = await secure_success_stream(response)

    assert prefix is None
    assert response.closed


@pytest.mark.anyio
async def test_secure_stream_large_chunk_commits_at_small_first_event_boundary() -> (
    None
):
    later_event = b'data: {"choices":[{"delta":{"content":"later"}}]}\n\n'
    chunk = VALID_EVENT + later_event * (MAX_PRECOMMIT_INSPECTION_BYTES // 8)
    response = FakeStreamResponse([chunk])

    prefix = await secure_success_stream(response)

    assert prefix is not None
    assert prefix.first_event_end == len(VALID_EVENT)
    # 검사 bytearray에는 상한까지만 복사하고, 이미 받은 큰 transport 청크 자체는 참조로 보존한다.
    assert prefix.initial_chunks == (chunk,)
    assert prefix.initial_chunks[0] is chunk
    assert await _collect_prefix(prefix) == chunk


@pytest.mark.anyio
async def test_secure_stream_first_read_error_returns_none_and_closes() -> None:
    response = FakeStreamResponse([], error=httpx.ReadError("drop"))

    prefix = await secure_success_stream(response)

    assert prefix is None
    assert response.closed


@pytest.mark.anyio
async def test_secure_stream_exceeds_inspection_bound_returns_none() -> None:
    # data 이벤트 없이 빈 줄만 상한을 넘도록 흘리면 무효로 끊는다.
    response = FakeStreamResponse([b"\n" * (MAX_PRECOMMIT_INSPECTION_BYTES + 10)])

    prefix = await secure_success_stream(response)

    assert prefix is None
    assert response.closed


# --- 확정 스트림 자원 정리 ---


FIRST_EVENT = b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
NEXT_EVENT = b'data: {"choices":[{"index":0,"delta":{"content":"b"}}]}\n\n'


@pytest.mark.anyio
async def test_iter_committed_stream_relays_and_closes_on_success() -> None:
    response = FakeResponse()
    prefix = StreamPrefix(
        response, (FIRST_EVENT,), 1, _achunks([NEXT_EVENT, DONE_EVENT])
    )
    cleanup = StreamCleanup(response)

    out = [chunk async for chunk in iter_committed_stream(prefix, cleanup)]

    assert out == [FIRST_EVENT, NEXT_EVENT, DONE_EVENT]
    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_accepts_done_split_across_chunks() -> None:
    response = FakeResponse()
    prefix = StreamPrefix(
        response, (FIRST_EVENT,), 1, _achunks([b"data: [DO", b"NE]\n", b"\n"])
    )
    cleanup = StreamCleanup(response)

    out = [chunk async for chunk in iter_committed_stream(prefix, cleanup)]

    assert b"".join(out) == FIRST_EVENT + DONE_EVENT
    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_accumulates_lineless_chunks_without_rescan() -> (
    None
):
    # 개행 없는 청크는 누적만 한다 — 상한 안에서는 오류 없이 통과하고 [DONE]으로 끝난다.
    response = FakeResponse()
    lineless = [b"data: " + b"x" * 1024] * 32 + [b"\n\n", DONE_EVENT]
    prefix = StreamPrefix(response, (VALID_EVENT,), 1, _achunks(lineless))
    cleanup = StreamCleanup(response)

    out = [chunk async for chunk in iter_committed_stream(prefix, cleanup)]

    assert b"".join(out).endswith(DONE_EVENT)
    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_bounds_empty_data_line_flood() -> None:
    # 값이 빈 `data:` 라인만 무한히 이어지는 이벤트도 상한에 걸려야 한다 —
    # 바이트 길이만 세면 목록 축적과 재합산이 상한을 우회한다.
    response = FakeResponse()
    flood = [b"data:\n" * 8192 for _ in range(9)]
    prefix = StreamPrefix(response, (VALID_EVENT,), 1, _achunks(flood))
    cleanup = StreamCleanup(response)
    generator = iter_committed_stream(prefix, cleanup)

    with pytest.raises(StreamTruncatedError):
        async for _ in generator:
            pass

    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_bounds_unterminated_line_growth() -> None:
    # 종결자 없는 라인이 검사 상한을 넘으면 무한 보관 대신 잘린 스트림으로 끝낸다.
    response = FakeResponse()
    oversized = [b"x" * 8192 for _ in range(9)]
    prefix = StreamPrefix(response, (VALID_EVENT,), 1, _achunks(oversized))
    cleanup = StreamCleanup(response)
    generator = iter_committed_stream(prefix, cleanup)

    with pytest.raises(StreamTruncatedError):
        async for _ in generator:
            pass

    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_raises_on_eof_without_done() -> None:
    response = FakeResponse()
    prefix = StreamPrefix(response, (FIRST_EVENT,), 1, _achunks([NEXT_EVENT]))
    cleanup = StreamCleanup(response)
    generator = iter_committed_stream(prefix, cleanup)

    assert await generator.__anext__() == FIRST_EVENT
    assert await generator.__anext__() == NEXT_EVENT
    # [DONE] 없는 정상 EOF는 잘린 응답이다 — 정상 종료로 위장하지 않는다.
    with pytest.raises(StreamTruncatedError):
        await generator.__anext__()

    assert response.closed


@pytest.mark.anyio
async def test_iter_committed_stream_closes_on_cancellation() -> None:
    response = FakeResponse()
    prefix = StreamPrefix(
        response, (FIRST_EVENT,), 1, _achunks([NEXT_EVENT, DONE_EVENT])
    )
    cleanup = StreamCleanup(response)
    generator = iter_committed_stream(prefix, cleanup)

    assert await generator.__anext__() == FIRST_EVENT
    await generator.aclose()

    assert response.closed


@pytest.mark.anyio
async def test_stream_committed_records_success_and_closes() -> None:
    response = FakeResponse()
    breaker = FakeBreaker()
    prefix = StreamPrefix(response, (FIRST_EVENT,), 1, _achunks([DONE_EVENT]))
    cleanup = StreamCleanup(response, lambda: breaker.release(ATTEMPT))

    out = [
        chunk async for chunk in _stream_committed(prefix, breaker, ATTEMPT, cleanup)
    ]

    assert out == [FIRST_EVENT, DONE_EVENT]
    assert breaker.calls == ["success"]
    assert response.closed


@pytest.mark.anyio
async def test_stream_committed_records_failure_on_eof_without_done() -> None:
    response = FakeResponse()
    breaker = FakeBreaker()
    prefix = StreamPrefix(response, (FIRST_EVENT,), 1, _achunks([NEXT_EVENT]))
    cleanup = StreamCleanup(response, lambda: breaker.release(ATTEMPT))
    generator = _stream_committed(prefix, breaker, ATTEMPT, cleanup)

    assert await generator.__anext__() == FIRST_EVENT
    assert await generator.__anext__() == NEXT_EVENT
    with pytest.raises(StreamTruncatedError):
        await generator.__anext__()

    # [DONE] 없는 EOF는 로컬 실패다 — 회로에 실패를 기록하고 예외로 끝낸다.
    assert breaker.calls == ["failure"]
    assert response.closed


@pytest.mark.anyio
async def test_stream_committed_records_failure_on_mid_stream_error() -> None:
    response = FakeResponse()
    breaker = FakeBreaker()
    prefix = StreamPrefix(
        response, (b"a",), 1, _achunks([b"b"], error=httpx.ReadError("drop"))
    )
    cleanup = StreamCleanup(response, lambda: breaker.release(ATTEMPT))
    generator = _stream_committed(prefix, breaker, ATTEMPT, cleanup)

    assert await generator.__anext__() == b"a"
    assert await generator.__anext__() == b"b"
    with pytest.raises(httpx.ReadError, match="drop"):
        await generator.__anext__()

    # 첫 로컬 바이트를 보낸 뒤의 장애는 실패로 기록하고 예외 종료한다. 정상 EOF로 위장하지 않는다.
    assert breaker.calls == ["failure"]
    assert response.closed


@pytest.mark.anyio
async def test_stream_committed_releases_on_cancellation() -> None:
    response = FakeResponse()
    breaker = FakeBreaker()
    prefix = StreamPrefix(response, (b"a",), 1, _achunks([b"b", b"c"]))
    cleanup = StreamCleanup(response, lambda: breaker.release(ATTEMPT))
    generator = _stream_committed(prefix, breaker, ATTEMPT, cleanup)

    assert await generator.__anext__() == b"a"
    await generator.aclose()

    # 성공·실패로 확정하지 못하고 끝난 취소 경로 — probe 슬롯만 되돌리고 자원을 닫는다.
    assert breaker.calls == ["release"]
    assert response.closed


@pytest.mark.anyio
async def test_half_open_committed_success_closes_breaker() -> None:
    response = FakeResponse()
    breaker, probe = _half_open_probe()
    prefix = StreamPrefix(response, (FIRST_EVENT,), 1, _achunks([DONE_EVENT]))
    cleanup = StreamCleanup(response, lambda: breaker.release(probe))

    assert [
        chunk async for chunk in _stream_committed(prefix, breaker, probe, cleanup)
    ] == [
        FIRST_EVENT,
        DONE_EVENT,
    ]

    assert breaker.begin().decision is LocalDecision.attempt
    assert response.closed


@pytest.mark.anyio
async def test_half_open_committed_failure_reopens_breaker() -> None:
    response = FakeResponse()
    breaker, probe = _half_open_probe()
    prefix = StreamPrefix(
        response, (b"a",), 1, _achunks([], error=httpx.ReadError("drop"))
    )
    cleanup = StreamCleanup(response, lambda: breaker.release(probe))
    generator = _stream_committed(prefix, breaker, probe, cleanup)

    assert await generator.__anext__() == b"a"
    with pytest.raises(httpx.ReadError, match="drop"):
        await generator.__anext__()

    assert breaker.begin().decision is LocalDecision.skip
    assert response.closed


@pytest.mark.anyio
async def test_half_open_committed_cancellation_releases_probe() -> None:
    response = FakeResponse()
    breaker, probe = _half_open_probe()
    prefix = StreamPrefix(response, (b"a",), 1, _achunks([b"b"]))
    cleanup = StreamCleanup(response, lambda: breaker.release(probe))
    generator = _stream_committed(prefix, breaker, probe, cleanup)

    assert await generator.__anext__() == b"a"
    await generator.aclose()

    assert breaker.begin().decision is LocalDecision.probe
    assert response.closed


@pytest.mark.anyio
async def test_asgi_send_failure_closes_upstream_and_releases_half_open_probe() -> None:
    upstream = FakeResponse()
    breaker, probe = _half_open_probe()
    prefix = StreamPrefix(upstream, (b"a",), 1, _achunks([b"b"]))
    cleanup = StreamCleanup(upstream, lambda: breaker.release(probe))
    stream = ManagedStreamingResponse(
        _stream_committed(prefix, breaker, probe, cleanup), cleanup, 200, {}
    )
    sent: list[Message] = []

    async def receive() -> Message:
        raise AssertionError("ASGI 2.4 streaming must not poll receive")

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }

    with pytest.raises(ClientDisconnect):
        await stream(scope, receive, send)

    assert sent[-1]["body"] == b"a"
    assert upstream.closed
    assert breaker.begin().decision is LocalDecision.probe
