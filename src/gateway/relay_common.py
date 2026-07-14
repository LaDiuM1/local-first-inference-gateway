"""로컬 중계와 OpenAI 폴백 중계가 공유하는 상수·헬퍼와 응답 분류.

두 중계 경로가 같은 hop-by-hop 헤더 제외 규칙과 폴백 대상 판정, 사고 수준 필드 이름을 쓰고,
로컬·OpenAI 어느 쪽이든 2xx 응답이 유효한 Chat Completions 응답인지 같은 기준으로 판정한다.
여기에 모아 순환 import 없이 양쪽에서 참조한다.
"""

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import anyio
import httpx
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from gateway.validation import load_standard_json

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# hop-by-hop 헤더(RFC 9110 §7.6.1)는 구간 전용이라 중계하지 않는다.
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# 버퍼링 경로는 디코딩된 본문을 반환하므로 길이·인코딩을 재계산에 맡기고,
# 스트리밍 경로는 HTTP 전송 인코딩을 명시적으로 푼 뒤 SSE 바이트를 흘리므로 길이·인코딩 헤더를
# 재계산에 맡긴다.
EXCLUDED_BUFFERED_HEADERS = HOP_BY_HOP_HEADERS | {"content-length", "content-encoding"}
EXCLUDED_STREAMING_HEADERS = HOP_BY_HOP_HEADERS | {
    "content-length",
    "content-encoding",
}

REASONING_EFFORT_FIELD = "reasoning_effort"
# 로컬 일반 모드 — reasoning_effort 생략 시 게이트웨이가 넣는 기본값.
NORMAL_MODE_EFFORT = "none"
# OpenAI 폴백의 저지연 기본값 — 로컬 일반 모드 none을 이 값으로 바꿔 보낸다.
OPENAI_MINIMAL_EFFORT = "minimal"

# SSE data 이벤트가 유효한 완료 이벤트인지 못 가른 채 버퍼가 이만큼 커지면 무효로 본다.
# 한 SSE chat 이벤트는 델타 하나라 수백 바이트면 충분하므로, 이 상한은 data 이벤트를 끝내 내지 않고
# data 아닌 필드·주석·빈 줄만 흘리거나 비대한 비 SSE 본문을 보내는 스트림을 끊는 안전장치다.
MAX_PRECOMMIT_INSPECTION_BYTES = 65536

_SSE_DATA_FIELD = b"data"
_SSE_COMMENT_PREFIX = b":"
_SSE_DONE_MARKER = b"[DONE]"
_SSE_CR = ord("\r")
_SSE_LF = ord("\n")


def relayed_headers(response: httpx.Response, excluded: set[str]) -> dict[str, str]:
    excluded_names = {name.lower() for name in excluded}
    for connection_value in response.headers.get_list("connection"):
        excluded_names.update(
            option.strip().lower()
            for option in connection_value.split(",")
            if option.strip()
        )
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in excluded_names
    }


async def read_and_close_response(response: httpx.Response) -> bytes:
    """스트리밍 응답 본문을 버퍼링하고 성공·실패 어느 경로에서도 연결을 닫는다."""
    try:
        return await response.aread()
    finally:
        await response.aclose()


def is_fallback_status(status_code: int) -> bool:
    """로컬 응답 상태가 폴백 대상인지 판정한다.

    연결·시간 초과는 예외로 따로 처리하고, 여기서는 응답이 온 경우만 본다. 429·5xx는 로컬이
    유효한 추론 응답을 못 준 상태이고, 404는 설정된 모델을 찾지 못한 경우다. 그 외 4xx는 로컬이
    판단한 클라이언트 오류이므로 폴백하지 않고 그대로 전달한다.
    """
    return status_code == 404 or status_code == 429 or status_code >= 500


def is_success_status(status_code: int) -> bool:
    return 200 <= status_code < 300


def is_valid_chat_completion_body(body: bytes) -> bool:
    """버퍼링된 2xx 본문이 유효한 Chat Completions 응답인지 본다.

    빈 본문·잘린 JSON·JSON이 아닌 오류 페이지·choices 없는 객체는 유효한 추론 응답이 아니다.
    Chat Completions 성공 응답은 choices 리스트를 담은 JSON 객체이므로 그 최소 계약만 확인하고,
    바이트 자체는 검증만 할 뿐 재직렬화하지 않는다.
    """
    if not body:
        return False
    return _is_valid_chat_completion_json(body)


def _is_valid_chat_completion_json(raw: bytes) -> bool:
    try:
        parsed = load_standard_json(raw)
    except ValueError:
        return False
    return _has_choices_list(parsed)


def _has_choices_list(parsed: object) -> bool:
    return isinstance(parsed, dict) and isinstance(parsed.get("choices"), list)


class _PrefixVerdict(StrEnum):
    """스트림 첫 유효 data 이벤트 확보 전, 지금까지 버퍼로 내릴 수 있는 판정."""

    valid = "valid"  # 첫 유효 Chat Completions data 이벤트를 확인 — 커밋 가능.
    invalid = "invalid"  # 비 SSE·깨진 data·유효 이벤트 전 [DONE] — 폴백/오류 대상.
    need_more = "need_more"  # 아직 완결된 판정 라인이 없다 — 다음 청크를 기다린다.


@dataclass(frozen=True)
class _PrefixClassification:
    verdict: _PrefixVerdict
    first_event_end: int | None = None


def _classify_sse_prefix(buffer: bytes | bytearray) -> _PrefixClassification:
    """버퍼의 완결된 SSE 라인만 훑어 첫 완결 data 이벤트를 판정한다.

    SSE 라인 구분자는 CRLF·CR·LF 모두이고 data·event·id·retry 등 여러 필드가 올 수 있다. 판정에 쓰는
    건 data 필드뿐이라 그 외 필드는 규격대로 무시하고, 주석(`:` 시작)과 빈 줄도 건너뛴다. 이벤트는 빈
    줄로 끝나므로 data 라인 하나만 보고 커밋하지 않고 이벤트 종료(빈 줄)까지 본다 — data 라인 뒤 종료
    전에 스트림이 끊기면 클라이언트가 그 이벤트를 디스패치하지 못하기 때문이다. 완결된 첫 이벤트의 data가
    유효한 chat completion JSON이면 유효, [DONE]이거나 깨진 JSON이면 무효로 판정하고, 아직 완결된
    이벤트가 없으면 다음 청크를 기다린다.
    """
    start = 0
    event_data: list[bytes] = []
    event_open = False
    while True:
        completed = _next_sse_line(buffer, start)
        if completed is None:
            return _PrefixClassification(_PrefixVerdict.need_more)
        line, start = completed
        if not line:
            if not event_open:
                continue  # 첫 이벤트 전의 빈 줄 — 건너뛴다.
            # 빈 줄 = 이벤트 종료. 종료 오프셋을 함께 돌려줘 검사 상한을 transport 청크 크기가
            # 아니라 실제 pre-commit prefix에 적용하고, 같은 청크의 뒤쪽 바이트는 그대로 중계한다.
            return _PrefixClassification(_classify_first_event(event_data), start)
        if line.startswith(_SSE_COMMENT_PREFIX):
            continue  # 주석/keep-alive — 이벤트를 시작하지 않고 무시한다.
        field, _, value = line.partition(b":")
        if field == _SSE_DATA_FIELD:
            event_open = True
            event_data.append(value.lstrip(b" "))
        # data 외 필드(event·id·retry 및 미지의 필드)는 SSE 규격상 판정에 쓰지 않아 무시한다.


def _next_sse_line(
    buffer: bytes | bytearray, start: int
) -> tuple[bytes | bytearray, int] | None:
    """start부터 완결된 SSE 한 줄과 다음 줄 시작 위치를 돌려준다 — 없으면 None.

    구분자는 CRLF·CR·LF 모두다. 끝의 CR도 CR-only 줄 끝으로 즉시 인정한다. 다음 청크가 LF로
    시작하면 전체 버퍼를 다시 훑을 때 같은 두 바이트를 CRLF 하나로 보므로, 청크 경계에서도 빈 줄을
    잘못 만들어내지 않으면서 `data: ...\r\r`의 마지막 CR은 즉시 이벤트 종료로 판정할 수 있다.
    """
    index = start
    length = len(buffer)
    while index < length:
        byte = buffer[index]
        if byte == _SSE_LF:
            return buffer[start:index], index + 1
        if byte == _SSE_CR:
            if index + 1 < length and buffer[index + 1] == _SSE_LF:
                return buffer[start:index], index + 2  # CRLF
            return buffer[start:index], index + 1  # 단독 CR
        index += 1
    return None


def _classify_first_event(event_data: list[bytes]) -> _PrefixVerdict:
    """빈 줄로 완결된 첫 이벤트의 data 필드를 판정한다 — 여러 data 라인은 개행으로 이어 붙인다."""
    payload = b"\n".join(event_data)
    if payload.strip() == _SSE_DONE_MARKER:
        return _PrefixVerdict.invalid
    if _is_valid_chat_completion_json(payload):
        return _PrefixVerdict.valid
    return _PrefixVerdict.invalid


@dataclass(frozen=True)
class StreamPrefix:
    """첫 유효 Chat Completions data 이벤트까지 확보한 스트림.

    initial_chunks는 검사하는 동안 이미 받은 디코딩된 SSE 바이트 청크다. 첫 이벤트가 끝난 위치는
    first_event_end로 명시하지만, 같은 transport 청크 뒤쪽의 후속 이벤트도 자르지 않고 청크 참조로
    보존한다. 이를 remaining과 이어 흘리면 손실·중복 없이 application 바이트 순서를 지킨다.
    """

    response: httpx.Response
    initial_chunks: tuple[bytes, ...]
    first_event_end: int
    remaining: AsyncIterator[bytes]


class StreamCleanup:
    """스트림 응답·선택적 미확정 상태를 어느 종료 경로에서도 한 번만 정리한다."""

    def __init__(
        self,
        response: httpx.Response,
        on_abandon: Callable[[], None] | None = None,
    ) -> None:
        self._response = response
        self._on_abandon = on_abandon
        self._resolved = False
        self._abandon_handled = False
        self._closed = False

    def resolve(self) -> None:
        self._resolved = True

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._response.aclose()
            self._closed = True
        finally:
            if (
                not self._resolved
                and not self._abandon_handled
                and self._on_abandon is not None
            ):
                self._abandon_handled = True
                self._on_abandon()


class ManagedStreamingResponse(StreamingResponse):
    """downstream send 실패까지 포함해 body iterator와 upstream 정리를 응답 수준에서 소유한다."""

    def __init__(
        self,
        content: AsyncIterator[bytes],
        cleanup: StreamCleanup,
        status_code: int,
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers)
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                close_iterator = getattr(self.body_iterator, "aclose", None)
                try:
                    if close_iterator is not None:
                        await close_iterator()
                finally:
                    await self._cleanup.close()


async def secure_success_stream(response: httpx.Response) -> StreamPrefix | None:
    """2xx 스트림을 첫 유효 data 이벤트까지 검사한다 — 커밋 전 경계.

    전송 청크가 SSE 라인·이벤트를 임의로 쪼갤 수 있으므로 완결된 data 이벤트가 나올 때까지 원문
    바이트를 모은다. 유효하면 소비한 바이트와 남은 iterator를 담아 돌려주고(응답은 열린 채로 둔다),
    스트림이 끝나거나 검사 상한을 넘거나 깨진/비 SSE 내용이면 응답을 닫고 None을 돌려준다.
    """
    remaining = response.aiter_bytes()
    inspection = bytearray()
    initial_chunks: list[bytes] = []
    while True:
        try:
            chunk = await anext(remaining)
        except (StopAsyncIteration, httpx.RequestError):
            await response.aclose()
            return None
        except BaseException:
            await response.aclose()
            raise
        if not chunk:
            continue

        available = MAX_PRECOMMIT_INSPECTION_BYTES - len(inspection)
        inspected_length = min(len(chunk), available)
        inspection.extend(memoryview(chunk)[:inspected_length])
        initial_chunks.append(chunk)

        classification = _classify_sse_prefix(inspection)
        if classification.verdict is _PrefixVerdict.valid:
            first_event_end = classification.first_event_end
            if first_event_end is None:
                raise RuntimeError("valid SSE classification has no event boundary")
            return StreamPrefix(
                response,
                tuple(initial_chunks),
                first_event_end,
                remaining,
            )
        if classification.verdict is _PrefixVerdict.invalid:
            await response.aclose()
            return None
        # 첫 이벤트가 상한 안에서 끝나지 않았다. 현재 transport 청크의 나머지는 복사하지 않는다.
        if inspected_length < len(chunk) or inspected_length == available:
            await response.aclose()
            return None


async def iter_committed_stream(
    prefix: StreamPrefix, cleanup: StreamCleanup
) -> AsyncIterator[bytes]:
    """확정된 스트림을 원문 바이트 그대로 흘리고, 성공·실패·취소 어느 경로로 끝나도 응답을 닫는다."""
    try:
        for chunk in prefix.initial_chunks:
            yield chunk
        async for chunk in prefix.remaining:
            yield chunk
    finally:
        await cleanup.close()
