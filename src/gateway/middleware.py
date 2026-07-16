"""공개 HTTP 경계의 인증, 요청 본문 크기, 캐시 정책."""

from collections.abc import Callable
from time import monotonic

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.api_keys import ApiKeyStore, ApiKeyStoreError

MAX_REQUEST_BODY_MIB = 32
MAX_REQUEST_BODY_BYTES = MAX_REQUEST_BODY_MIB * 1024 * 1024
API_CACHE_CONTROL = b"no-store"
DOCS_CACHE_CONTROL = b"public, max-age=300"

ApiKeyStoreProvider = Callable[[], ApiKeyStore]
Clock = Callable[[], float]


class AuthenticationMiddleware:
    """보호 경로의 Bearer 키를 본문 수신 전에 검증한다."""

    def __init__(
        self,
        app: ASGIApp,
        store_provider: ApiKeyStoreProvider,
        clock: Clock = monotonic,
    ) -> None:
        self._app = app
        self._store_provider = store_provider
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _requires_authentication(scope["path"]):
            await self._app(scope, receive, send)
            return

        scope.setdefault("state", {})["request_started_at"] = self._clock()
        api_key = _bearer_token(scope)
        if api_key is None:
            await _unauthorized_response()(scope, receive, send)
            return
        try:
            identity = self._store_provider().authenticate(api_key)
        except ApiKeyStoreError:
            await _authentication_unavailable_response()(scope, receive, send)
            return
        if identity is None:
            await _unauthorized_response()(scope, receive, send)
            return
        scope["state"]["api_key_identity"] = identity
        await self._app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Content-Length와 실제 ASGI 수신 바이트를 모두 공개 상한으로 제한한다."""

    def __init__(
        self, app: ASGIApp, maximum_bytes: int = MAX_REQUEST_BODY_BYTES
    ) -> None:
        self._app = app
        self._maximum_bytes = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self._app(scope, receive, send)
            return

        try:
            content_length = _content_length(scope)
        except ValueError:
            await _invalid_content_length_response()(scope, receive, send)
            return
        if content_length is not None and content_length > self._maximum_bytes:
            await _request_too_large_response()(scope, receive, send)
            return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._maximum_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self._app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _request_too_large_response()(scope, receive, send)


class CacheControlMiddleware:
    """민감 응답은 저장 금지하고 공개 연동 문서만 5분 캐시한다."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        cache_control = _cache_control_for(scope["path"])
        if cache_control is None:
            await self._app(scope, receive, send)
            return

        async def cache_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"cache-control"
                ]
                headers.append((b"cache-control", cache_control))
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, cache_send)


class _RequestBodyTooLarge(Exception):
    pass


def _requires_authentication(path: str) -> bool:
    return path == "/health" or path.startswith("/v1/")


def _bearer_token(scope: Scope) -> str | None:
    values = [
        value.decode("latin-1")
        for name, value in scope.get("headers", [])
        if name.lower() == b"authorization"
    ]
    if len(values) != 1:
        return None
    parts = values[0].split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _content_length(scope: Scope) -> int | None:
    values = [
        value.decode("ascii")
        for name, value in scope.get("headers", [])
        if name.lower() == b"content-length"
    ]
    if not values:
        return None
    parsed = {int(value) for value in values}
    if len(parsed) != 1:
        raise ValueError("conflicting Content-Length values")
    length = parsed.pop()
    if length < 0:
        raise ValueError("negative Content-Length")
    return length


def _cache_control_for(path: str) -> bytes | None:
    if path == "/docs":
        return DOCS_CACHE_CONTROL
    if path == "/health" or path.startswith("/v1/"):
        return API_CACHE_CONTROL
    return None


def _unauthorized_response() -> JSONResponse:
    return _error_response(
        401,
        "Invalid or missing API key",
        "authentication_error",
        "invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _authentication_unavailable_response() -> JSONResponse:
    return _error_response(
        503,
        "API key authentication is temporarily unavailable",
        "server_error",
        "authentication_unavailable",
    )


def _invalid_content_length_response() -> JSONResponse:
    return _error_response(
        400,
        "Content-Length must be one non-negative integer",
        "invalid_request_error",
        "invalid_content_length",
    )


def _request_too_large_response() -> JSONResponse:
    return _error_response(
        413,
        f"Request body exceeds the {MAX_REQUEST_BODY_MIB} MiB limit",
        "invalid_request_error",
        "request_too_large",
    )


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )
