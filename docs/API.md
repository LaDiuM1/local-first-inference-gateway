# openAt Inference API 연동 가이드

이 문서는 특정 언어나 클라이언트 라이브러리에 종속되지 않는 공개 HTTP 계약을 설명한다. 모든 예시의 호스트와 API 키는 실제 발급값으로 바꿔 사용해야 한다.

## Base URL과 인증

OpenAI 호환 Base URL은 다음 형식이다.

```text
https://<public-host>/v1
```

`/v1/*` 요청과 `/health` 요청에는 서비스별로 발급된 API 키를 Bearer 토큰으로 보낸다.

```http
Authorization: Bearer <API_KEY>
```

키가 없거나 형식이 잘못됐거나 폐기된 경우 요청은 추론 provider에 전달되지 않고 `401 Unauthorized`로 끝난다. API 키는 서비스 간 공유하지 않는다.

## 지원 엔드포인트와 모델 별칭

- `POST /v1/chat/completions`: 대화형 생성과 이미지 분석
- `POST /v1/embeddings`: 문자열 또는 문자열 배열 임베딩
- `GET /health`: 인증된 운영 상태 확인
- `GET /docs`: 인증 없이 읽을 수 있는 이 문서

클라이언트는 실제 provider 모델명이 아니라 다음 별칭을 사용한다.

- `chat`: 일반 대화형 생성
- `vision`: 이미지가 포함된 대화형 생성
- `embed`: 임베딩 생성

임베딩 벡터는 항상 **1024차원**이다. 임베딩 모델과 차원은 기존 색인과 같은 벡터 공간을 유지하기 위한 계약이므로 클라이언트가 임의로 바꾸지 않는다.

## Chat Completions

### 일반 응답

```http
POST /v1/chat/completions HTTP/1.1
Host: <public-host>
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "model": "chat",
  "messages": [
    {"role": "user", "content": "배송 지연 안내 문구를 작성해 줘."}
  ]
}
```

성공 응답은 OpenAI Chat Completions 형식의 JSON 객체다.

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "model": "provider-model",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "요청하신 안내 문구입니다."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}
}
```

### SSE 스트리밍

긴 생성은 Cloudflare와 클라이언트의 응답 대기 제한을 피하고 첫 결과를 빨리 받기 위해 `stream: true` 사용을 권장한다.

```http
POST /v1/chat/completions HTTP/1.1
Host: <public-host>
Authorization: Bearer <API_KEY>
Content-Type: application/json
Accept: text/event-stream

{
  "model": "chat",
  "stream": true,
  "messages": [
    {"role": "user", "content": "상품 설명을 세 문단으로 작성해 줘."}
  ]
}
```

응답은 `data:` 이벤트가 이어지는 Server-Sent Events 스트림이며 마지막 이벤트는 `[DONE]`이다.

```text
data: {"choices":[{"index":0,"delta":{"content":"첫"}}]}

data: {"choices":[{"index":0,"delta":{"content":" 문장"}}]}

data: [DONE]

```

첫 유효 이벤트가 전달된 뒤에는 provider가 바뀌지 않는다. 이후 전송 장애가 발생하면 다른 provider의 응답을 이어 붙이지 않고 스트림이 오류로 종료될 수 있으므로 클라이언트는 불완전한 응답을 정상 완료로 처리하지 않아야 한다.

### 사고 수준 (reasoning_effort)

`/v1/chat/completions`는 OpenAI 표준 필드 `reasoning_effort`로 사고 수준을 받는다. 필드를 생략하면 게이트웨이가 저지연 일반 모드(`none`)로 처리하고, 값을 지정하면 그대로 provider에 전달한다. 값은 비어 있지 않은 문자열이어야 하며, 그 외 타입이나 빈 문자열은 업스트림 호출 없이 `400`으로 거절된다.

```json
{
  "model": "chat",
  "reasoning_effort": "high",
  "messages": [
    {"role": "user", "content": "이 주문 데이터의 이상 패턴을 단계적으로 분석해 줘."}
  ]
}
```

사고 모드(`high`)는 정확도가 중요한 복합 요청에 유용하지만 첫 응답까지 수십 초 이상 걸릴 수 있으므로, 지연이 중요한 경로는 필드를 생략해 기본 모드로 호출한다. 로컬 장애로 OpenAI 폴백이 동작할 때 일반 모드(`none`)는 폴백 모델이 지원하는 저지연 값(`minimal`)으로 변환되고, 클라이언트가 지정한 다른 값은 그대로 전달된다. 생략(기본 모드)과 `low`·`medium`·`high`는 로컬·폴백 어느 provider에서도 유효하다 — 그 밖의 값은 게이트웨이 검증은 통과하지만 처리하는 provider가 지원하지 않으면 provider의 `400`이 그대로 전달될 수 있으므로 사용하지 않는 것이 안전하다.

## 이미지 분석

이미지는 OpenAI Chat Completions의 다중 콘텐츠 형식으로 전달하고 `vision` 별칭을 사용한다. 아래 예시의 base64 문자열은 실제 이미지 데이터로 바꾼다.

```json
{
  "model": "vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "이 이미지의 핵심 내용을 설명해 줘."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,<BASE64_IMAGE>"}
        }
      ]
    }
  ]
}
```

## Embeddings

### 문자열과 문자열 배열

단건 입력은 문자열로 보낸다.

```json
{
  "model": "embed",
  "input": "임베딩할 문장",
  "encoding_format": "float"
}
```

배치 입력은 비어 있지 않은 문자열 배열로 보낸다. 응답 `data`의 `index`는 입력 순서를 보존한다.

```json
{
  "model": "embed",
  "input": ["첫 번째 문장", "두 번째 문장"]
}
```

기본 `encoding_format`은 `float`이다. 성공 응답의 각 `embedding`은 1024개 숫자로 구성된다. 아래 응답은 표현 형식만 보여 주기 위해 벡터 값을 축약한 예시다.

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.034]}
  ],
  "model": "provider-model",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

`encoding_format: "base64"`를 지정하면 각 벡터는 1024개의 float32 리틀엔디언 바이트를 base64로 인코딩한 문자열로 반환된다.

```json
{
  "model": "embed",
  "input": "임베딩할 문장",
  "encoding_format": "base64"
}
```

## 요청 크기 제한

`/v1/*` 요청의 전체 HTTP 본문은 최대 **20MiB(20 × 1024 × 1024 바이트)** 다. 이 제한은 이미지 원본 크기가 아니라 base64 인코딩과 JSON 구조를 모두 포함하여 실제 전송되는 최종 본문 크기에 적용된다. `Content-Length`가 없더라도 실제 수신 바이트를 기준으로 검사하며, 초과 요청은 JSON 파싱과 추론 호출 전에 `413 Payload Too Large`로 거절된다.

## 응답 시작 기한

기한은 요청 도착부터 계산한다. 로컬 추론은 90초 안에 응답을 시작해야 하며, `chat`·`vision` 폴백을 포함한 전체 기한은 115초다. 일반 응답은 유효한 전체 본문, 스트리밍 응답은 첫 유효 SSE 이벤트를 확보한 시점을 응답 시작으로 본다. 전체 기한을 넘기면 `504 Gateway Timeout`을 반환한다.

## 오류 형식과 상태 코드

게이트웨이가 직접 만드는 오류는 다음 OpenAI 형식의 본문을 사용한다.

```json
{
  "error": {
    "message": "오류 설명",
    "type": "invalid_request_error",
    "param": null,
    "code": "오류 코드"
  }
}
```

주요 상태 코드는 다음과 같다.

- `400 Bad Request`: JSON 또는 지원 필드와 모델 별칭이 잘못됨
- `401 Unauthorized`: API 키가 없거나 잘못됐거나 폐기됨
- `413 Payload Too Large`: 전체 요청 본문이 20MiB를 초과함
- `502 Bad Gateway`: 로컬 또는 폴백 provider가 유효한 응답을 제공하지 못함
- `503 Service Unavailable`: 인증 저장소 등 게이트웨이 운영 상태를 사용할 수 없음
- `504 Gateway Timeout`: 전체 응답 시작 기한 안에 유효한 결과를 확보하지 못함

일부 provider 오류는 상태와 OpenAI 오류 본문이 그대로 전달될 수 있다. 클라이언트는 특정 메시지 문자열이 아니라 HTTP 상태 코드를 1차 기준으로 처리해야 하며, `error.code`는 값이 없을 수(`null`) 있으므로 보조 신호로만 사용한다.

## 캐시 정책

`/v1/*` 추론 응답과 `/health` 응답은 `Cache-Control: no-store`로 반환된다. `/docs`만 공개 문서로서 5분 동안 캐시될 수 있다.
