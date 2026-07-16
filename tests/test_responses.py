"""검색 모듈이 사용하는 OpenAI Responses 이미지 계약 테스트."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from gateway.config import settings
from gateway.main import create_app

pytestmark = pytest.mark.anyio

LOCAL_URL = f"{settings.ollama_base_url}/v1/chat/completions"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"
OPENAI_VISION_MODEL = "gpt-5.4-nano"
LOCAL_MODEL = "gemma4:12b-it-qat"
TEST_FALLBACK_KEY = "sk-test-responses-fallback-key"
IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"


class GatewayCredentials(Protocol):
    store_path: Path
    api_key: str

    @property
    def authorization_headers(self) -> dict[str, str]: ...


@pytest.fixture
async def responses_client(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(TEST_FALLBACK_KEY))
    app = create_app(api_key_store_path=gateway_credentials.store_path)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers=gateway_credentials.authorization_headers,
        ) as client,
    ):
        yield client


def _request() -> dict:
    return {
        "model": OPENAI_VISION_MODEL,
        "instructions": "이미지에서 보이는 사실만 설명한다.",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "검색용 특징을 설명해 줘."},
                    {"type": "input_image", "image_url": IMAGE_DATA_URL},
                ],
            }
        ],
    }


def _chat_response(text: str, model: str = LOCAL_MODEL) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_750_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


@respx.mock
async def test_responses_converts_search_image_request_and_response(
    responses_client: httpx.AsyncClient,
) -> None:
    local = respx.post(LOCAL_URL).respond(200, json=_chat_response("빨간색 상품 사진"))

    response = await responses_client.post("/v1/responses", json=_request())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == OPENAI_VISION_MODEL
    assert body["output"][0]["content"] == [
        {"type": "output_text", "text": "빨간색 상품 사진", "annotations": []}
    ]
    assert body["usage"]["input_tokens"] == 12
    assert body["usage"]["output_tokens"] == 7

    upstream = json.loads(local.calls.last.request.content)
    assert upstream["model"] == LOCAL_MODEL
    assert upstream["stream"] is False
    assert upstream["reasoning_effort"] == "none"
    assert upstream["messages"] == [
        {"role": "system", "content": "이미지에서 보이는 사실만 설명한다."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "검색용 특징을 설명해 줘."},
                {"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}},
            ],
        },
    ]


@respx.mock
async def test_responses_keeps_response_shape_after_openai_fallback(
    responses_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local unavailable"))
    fallback = respx.post(OPENAI_URL).respond(
        200, json=_chat_response("폴백 이미지 설명", "gpt-5-mini")
    )

    response = await responses_client.post("/v1/responses", json=_request())

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "폴백 이미지 설명"
    fallback_request = json.loads(fallback.calls.last.request.content)
    assert fallback_request["model"] == "gpt-5-mini"
    assert fallback_request["reasoning_effort"] == "minimal"
    assert fallback.calls.last.request.headers["authorization"] == (
        f"Bearer {TEST_FALLBACK_KEY}"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request.pop("model"),
        lambda request: request.update({"model": "unknown-model"}),
        lambda request: request.update({"model": "chat"}),
        lambda request: request.update({"model": "vision"}),
        lambda request: request.update({"input": []}),
        lambda request: request.update({"input": [{"role": "tool", "content": "x"}]}),
        lambda request: request["input"][0]["content"][0].update(
            {"type": "input_audio"}
        ),
        lambda request: request["input"][0]["content"][1].pop("image_url"),
        lambda request: request.update({"stream": True}),
        lambda request: request.update({"max_output_tokens": 500}),
        lambda request: request.update({"temperature": 0.2}),
        lambda request: request["input"][0]["content"][1].update(
            {"detail": "original"}
        ),
    ],
    ids=[
        "missing-model",
        "unknown-model",
        "chat-alias-not-allowed-here",
        "vision-alias-not-allowed-here",
        "empty-input",
        "unsupported-role",
        "unsupported-content",
        "missing-image-url",
        "streaming",
        "unsupported-field-max-output-tokens",
        "unsupported-field-temperature",
        "unsupported-image-detail",
    ],
)
@respx.mock
async def test_invalid_responses_request_is_400_without_upstream(
    responses_client: httpx.AsyncClient, mutate
) -> None:
    request = _request()
    mutate(request)
    local = respx.post(LOCAL_URL)
    fallback = respx.post(OPENAI_URL)

    response = await responses_client.post("/v1/responses", json=request)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not local.called
    assert not fallback.called


@respx.mock
async def test_supported_image_detail_is_forwarded(
    responses_client: httpx.AsyncClient,
) -> None:
    local = respx.post(LOCAL_URL).respond(200, json=_chat_response("세부 분석"))
    request = _request()
    request["input"][0]["content"][1]["detail"] = "low"

    response = await responses_client.post("/v1/responses", json=request)

    assert response.status_code == 200
    upstream = json.loads(local.calls.last.request.content)
    assert upstream["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": IMAGE_DATA_URL, "detail": "low"},
    }


@respx.mock
async def test_responses_alias_is_rejected_on_chat_completions_endpoint(
    responses_client: httpx.AsyncClient,
) -> None:
    """gpt-5.4-nano는 /v1/responses 전용이다 — chat 엔드포인트에서는 업스트림 없이 400이다."""
    local = respx.post(LOCAL_URL)
    fallback = respx.post(OPENAI_URL)

    response = await responses_client.post(
        "/v1/chat/completions",
        json={
            "model": OPENAI_VISION_MODEL,
            "messages": [{"role": "user", "content": "안녕"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not local.called
    assert not fallback.called


# --- 무효 로컬 2xx의 폴백과 회로 차단기 기록 ---

INVALID_CHAT_BODIES = [
    b"not json",
    b'{"id":"x"}',
    b'{"choices":[]}',
    b'{"choices":[{"message":{"role":"assistant","content":null}}]}',
    b'{"choices":[{"message":{"role":"assistant","content":""}}]}',
    b'{"choices":[{"message":{"content":"txt"},"finish_reason":"tool_calls"}]}',
    b'{"choices":[{"message":{"content":"txt"}}]}',
]
INVALID_CHAT_BODY_IDS = [
    "broken-json",
    "missing-choices",
    "empty-choices",
    "null-content",
    "empty-content",
    "unsupported-finish-reason",
    "missing-finish-reason",
]


@pytest.mark.parametrize("invalid_body", INVALID_CHAT_BODIES, ids=INVALID_CHAT_BODY_IDS)
@respx.mock
async def test_responses_falls_back_when_local_2xx_is_not_convertible(
    responses_client: httpx.AsyncClient, invalid_body: bytes
) -> None:
    """로컬이 2xx여도 변환 가능한 출력이 없으면 로컬 실패다 — OpenAI로 폴백해 응답을 만든다."""
    local = respx.post(LOCAL_URL).respond(
        200, content=invalid_body, headers={"content-type": "application/json"}
    )
    fallback = respx.post(OPENAI_URL).respond(
        200, json=_chat_response("폴백 분석", "gpt-5-mini")
    )

    response = await responses_client.post("/v1/responses", json=_request())

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "폴백 분석"
    assert local.call_count == 1
    assert fallback.call_count == 1


@respx.mock
async def test_responses_returns_502_when_fallback_is_also_invalid(
    responses_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).respond(200, json={"choices": []})
    respx.post(OPENAI_URL).respond(200, json={"choices": []})

    response = await responses_client.post("/v1/responses", json=_request())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"


@respx.mock
async def test_invalid_local_responses_open_the_shared_vision_circuit(
    responses_client: httpx.AsyncClient,
) -> None:
    """무효 로컬 2xx는 회로 차단기 실패로 기록된다 — 임계값 이후 로컬을 건너뛰고 바로 폴백한다.

    Responses는 내부적으로 vision으로 정규화되므로 vision 회로를 공유한다.
    """
    threshold = settings.circuit_breaker_failure_threshold
    local = respx.post(LOCAL_URL).respond(200, json={"choices": []})
    respx.post(OPENAI_URL).respond(200, json=_chat_response("폴백", "gpt-5-mini"))

    for _ in range(threshold):
        await responses_client.post("/v1/responses", json=_request())
    assert local.call_count == threshold

    # 회로가 열려 로컬을 건너뛴다 — Responses 요청도, 같은 회로의 vision chat 요청도.
    await responses_client.post("/v1/responses", json=_request())
    assert local.call_count == threshold

    vision_chat = await responses_client.post(
        "/v1/chat/completions",
        json={"model": "vision", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert vision_chat.status_code == 200
    assert local.call_count == threshold


# --- finish_reason 변환 ---


def _chat_response_with_finish(text: str, finish_reason: str) -> dict:
    chat_response = _chat_response(text)
    chat_response["choices"][0]["finish_reason"] = finish_reason
    return chat_response


@pytest.mark.parametrize(
    ("finish_reason", "expected_reason"),
    [("length", "max_output_tokens"), ("content_filter", "content_filter")],
)
@respx.mock
async def test_truncated_chat_response_becomes_incomplete_not_completed(
    responses_client: httpx.AsyncClient, finish_reason: str, expected_reason: str
) -> None:
    respx.post(LOCAL_URL).respond(
        200, json=_chat_response_with_finish("잘린 분석", finish_reason)
    )

    response = await responses_client.post("/v1/responses", json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": expected_reason}
    assert body["completed_at"] is None
    assert body["output"][0]["status"] == "incomplete"
    assert body["output"][0]["content"][0]["text"] == "잘린 분석"


@respx.mock
async def test_normal_stop_keeps_completed_status(
    responses_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).respond(200, json=_chat_response("완결 분석"))

    body = (await responses_client.post("/v1/responses", json=_request())).json()

    assert body["status"] == "completed"
    assert body["incomplete_details"] is None
    assert body["completed_at"] == body["created_at"]
    assert body["output"][0]["status"] == "completed"


# --- usage 상세값 보존 ---


@respx.mock
async def test_responses_usage_preserves_detail_and_total_tokens(
    responses_client: httpx.AsyncClient,
) -> None:
    chat_response = _chat_response("분석")
    chat_response["usage"] = {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    respx.post(LOCAL_URL).respond(200, json=chat_response)

    usage = (await responses_client.post("/v1/responses", json=_request())).json()[
        "usage"
    ]

    assert usage == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 25,
    }


@respx.mock
async def test_responses_usage_defaults_missing_details_to_zero(
    responses_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).respond(200, json=_chat_response("분석"))

    usage = (await responses_client.post("/v1/responses", json=_request())).json()[
        "usage"
    ]

    assert usage == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 19,
    }
