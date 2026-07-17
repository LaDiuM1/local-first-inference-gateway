"""요청 관측 — 게이트웨이 경계에서 요청별 처리 경과를 구조화 로그(JSON Lines)로 남긴다.

한 요청은 종료 시점에 한 줄(event=request)로 기록한다: 도착 시각과 요청 식별자, 인증된 호출
주체, 공개 별칭, 업스트림 시작·응답 시작·완료의 상대 시간(ms), 최종 provider와 로컬 실패 사유,
상태 코드·전송 바이트·완결 여부. 회로 차단기 상태 전환은 별도 이벤트(event=circuit)로 남긴다.
프롬프트·이미지·응답 본문과 API 키 원문은 어떤 경우에도 기록하지 않는다.

기록은 요청 경로에서 큐에 넣기만 하고 전용 스레드(QueueListener)가 회전 파일에 쓴다 — 파일
IO가 이벤트 루프를 막지 않는다. 요청 문맥은 contextvar로 전달되므로 중계 경로는 시그니처
변경 없이 관측 지점만 표시하며, 관측 문맥이 없는 호출(단위 테스트 등)에서는 조용히 무시된다.
"""

import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import SimpleQueue
from time import monotonic, time
from uuid import uuid4

from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.errors import internal_server_error_response

LOG_FILE_NAME = "requests.jsonl"
# 한 줄이 수백 바이트라 이 상한으로도 수십만 요청이 남는다 — 사후 진단·벤치마크 근거로 충분하다.
MAX_LOG_FILE_BYTES = 16 * 1024 * 1024
LOG_BACKUP_COUNT = 4

REQUEST_ID_HEADER = b"x-request-id"


class RequestObservation:
    """한 요청의 관측 문맥 — 미들웨어가 만들고 처리 경로가 지점을 표시한다."""

    def __init__(self, request_id: str, started_monotonic: float) -> None:
        self.request_id = request_id
        self._started_monotonic = started_monotonic
        self._fields: dict[str, object] = {}

    def set_value(self, field: str, value: object) -> None:
        self._fields[field] = value

    def set_once(self, field: str, value: object) -> None:
        self._fields.setdefault(field, value)

    def mark_once(self, field: str) -> None:
        elapsed_ms = (monotonic() - self._started_monotonic) * 1000
        self._fields.setdefault(field, round(elapsed_ms, 2))

    def build_record(
        self, *, status: int | None, bytes_out: int, completed: bool, duration_ms: float
    ) -> dict[str, object]:
        """필드가 비어도 키를 고정해 로그 후처리가 일정한 스키마를 보게 한다."""
        fields = self._fields
        return {
            "time": _utc_now(),
            "request_id": self.request_id,
            "started_at": fields.get("started_at"),
            "client": fields.get("client"),
            "key_id": fields.get("key_id"),
            "method": fields.get("method"),
            "path": fields.get("path"),
            "alias": fields.get("alias"),
            "stream": fields.get("stream"),
            "status": status,
            "provider": fields.get("provider"),
            "local_failure_reason": fields.get("local_failure_reason"),
            "upstream_start_ms": fields.get("upstream_start_ms"),
            "response_start_ms": fields.get("response_start_ms"),
            "duration_ms": duration_ms,
            "bytes_out": bytes_out,
            "completed": completed,
        }


_current_observation: ContextVar[RequestObservation | None] = ContextVar(
    "request_observation", default=None
)


def observe_alias(alias: str) -> None:
    """클라이언트가 요청한 공개 별칭 — 먼저 기록된 값이 남는다(내부 정규화 별칭이 덮지 않는다)."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.set_once("alias", alias)


def observe_stream(streaming: bool) -> None:
    observation = _current_observation.get()
    if observation is None:
        return
    observation.set_value("stream", streaming)


def observe_client(client: str, key_id: str) -> None:
    """인증된 호출 주체 — 서비스 이름과 키 식별자만 남기고 키 원문은 받지 않는다."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.set_value("client", client)
    observation.set_value("key_id", key_id)


def observe_provider(provider: str) -> None:
    """최종 응답을 만든 provider(local·openai) — 합성 오류만 낸 요청에는 남지 않는다."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.set_value("provider", provider)


def observe_local_failure(reason: str) -> None:
    """로컬 응답을 쓰지 못한 첫 사유 — 폴백 여부와 무관하게 실패 분류 지점에서 기록한다."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.set_once("local_failure_reason", reason)


def observe_upstream_start() -> None:
    """첫 업스트림 호출 직전 — 도착부터 이 지점까지가 게이트웨이 구간(수신·검증)이다."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.mark_once("upstream_start_ms")


def observe_response_start() -> None:
    """유효한 응답을 확보한 시점 — buffered는 본문 확보, 스트리밍은 첫 유효 이벤트 확보다."""
    observation = _current_observation.get()
    if observation is None:
        return
    observation.mark_once("response_start_ms")


class RequestLogWriter:
    """요청·회로 이벤트를 JSON Lines 회전 파일에 전용 스레드로 기록한다.

    로그 디렉터리를 만들거나 파일을 열지 못하면 그대로 예외를 낸다 — 관측이 조용히 꺼진 채
    운영되는 상태를 만들지 않고 기동 시점에 실패시킨다.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._handler = RotatingFileHandler(
            directory / LOG_FILE_NAME,
            maxBytes=MAX_LOG_FILE_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        record_queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
        self._queue_handler = QueueHandler(record_queue)
        self._listener = QueueListener(record_queue, self._handler)
        self._listener.start()

    def log_request(self, record: dict[str, object]) -> None:
        self._emit({"event": "request", **record})

    def log_circuit_transition(self, alias: str, state: str) -> None:
        self._emit(
            {"event": "circuit", "time": _utc_now(), "alias": alias, "state": state}
        )

    def close(self) -> None:
        """큐에 남은 기록을 모두 파일에 쓴 뒤 스레드와 파일을 닫는다."""
        self._listener.stop()
        self._handler.close()

    def _emit(self, record: dict[str, object]) -> None:
        line = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        self._queue_handler.emit(logging.makeLogRecord({"msg": line}))


class ObservabilityMiddleware:
    """모든 HTTP 요청을 관측 문맥으로 감싸고 종료 시 한 줄을 기록한다.

    가장 바깥 미들웨어로 등록되어 도착 시각(request_started_at)의 유일한 기준점을 만들고,
    모든 응답에 x-request-id 헤더를 추가한다. 스트리밍은 전송이 끝나야 기록되므로 중간에
    끊긴 요청은 completed=false로 남는다.

    처리되지 않은 예외의 500은 더 바깥의 오류 처리기가 이 미들웨어를 거치지 않고 보내므로,
    응답 시작 전 예외면 여기서 규격 500을 직접 보내 x-request-id·상태 기록 계약을 지키고
    예외는 다시 던져 서버 로그를 남긴다.
    """

    def __init__(
        self, app: ASGIApp, writer_provider: Callable[[], RequestLogWriter]
    ) -> None:
        self._app = app
        self._writer_provider = writer_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        started_monotonic = monotonic()
        scope.setdefault("state", {})["request_started_at"] = started_monotonic
        observation = RequestObservation(uuid4().hex, started_monotonic)
        observation.set_value("started_at", round(time(), 3))
        observation.set_value("method", scope["method"])
        observation.set_value("path", scope["path"])

        status: int | None = None
        bytes_out = 0
        completed = False
        response_started = False

        async def observed_send(message: Message) -> None:
            nonlocal status, bytes_out, completed, response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = list(message.get("headers", []))
                headers.append(
                    (REQUEST_ID_HEADER, observation.request_id.encode("ascii"))
                )
                message = {**message, "headers": headers}
            await send(message)
            # 전송이 성공한 메시지만 집계한다 — 실패한 마지막 청크가 완결·바이트로 남지 않는다.
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                bytes_out += len(message.get("body", b""))
                if not message.get("more_body", False):
                    completed = True

        token = _current_observation.set(observation)
        try:
            await self._app(scope, receive, observed_send)
        except ClientDisconnect:
            # 본문 수신 중 클라이언트가 연결을 끊었다 — 게이트웨이 결함이 아니므로 500을 합성하지
            # 않고 조용히 끝낸다. 기록은 finally가 completed=false로 남긴다.
            pass
        except Exception:
            if not response_started:
                await internal_server_error_response()(scope, receive, observed_send)
            raise
        finally:
            _current_observation.reset(token)
            duration_ms = round((monotonic() - started_monotonic) * 1000, 2)
            self._writer_provider().log_request(
                observation.build_record(
                    status=status,
                    bytes_out=bytes_out,
                    completed=completed,
                    duration_ms=duration_ms,
                )
            )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
