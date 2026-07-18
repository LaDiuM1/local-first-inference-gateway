# local-first-inference-gateway

팀 프로젝트 openAt의 MSA 클라이언트에 연결하기 위해 홈 GPU(GTX 1080 Ti 11GB) 한 대로 구축한 OpenAI 호환 추론 게이트웨이다. 로컬 Ollama를 기본 경로로 쓰고 장애 시 OpenAI로 우회해 서비스 연속성을 지키며, 단건 스트리밍 첫 토큰 p95 2.74초와 질의당 한계 전기비 0.028원을 실측으로 확인했다. 게이트웨이 쪽 연동 계약은 완료됐고, 검색 모듈의 실제 E2E 전환은 담당자 일정에 맞춰 대기 중이다.

```text
[openAt MSA 클라이언트 (사용 기술 독립)]
      │  /v1/chat/completions · /v1/responses · /v1/embeddings (OpenAI 호환, model = 공개 별칭)
      ▼
[Cloudflare Named Tunnel] ─ [추론 게이트웨이 (FastAPI)]
      서비스별 API 키 인증 · 별칭 라우팅 · SSE 스트리밍 중계 · 요청 관측(JSON Lines)
      ├─ 1차: Ollama — chat·vision → gemma4:12b-it-qat(GPU) / embed → snowflake-arctic-embed2(CPU 전용 인스턴스)
      └─ 폴백: 로컬 장애 시 chat·vision → OpenAI gpt-5-mini (embed는 폴백하지 않음)
```

## 왜 직접 만들었나

파이널 프로젝트에 RAG·이미지 분석 같은 LLM 기능이 들어가면서 지원받는 API 토큰 한도가 팀 공용의 제약이 됐다. 보유한 GPU로 이 요구를 감당할 수 있다고 판단해 추론 경로 자체를 해결할 문제로 정의하고 직접 구축했다. 계약을 OpenAI 규격으로 통일한 것은 업계 표준이면서 클라이언트 변경을 최소화하는 선택이기 때문이다. 각 판단의 배경과 트레이드오프는 [결정 이력](docs/DECISIONS.md)에 시간순으로 남겼다.

## 클라이언트에서 보이는 모습

OpenAI SDK의 호출 구조를 그대로 두고 base_url·게이트웨이 키·모델 별칭 설정만 전환한다.

```python
from openai import OpenAI

client = OpenAI(base_url="https://<public-host>/v1", api_key="<게이트웨이 발급 키>")
reply = client.chat.completions.create(
    model="chat",  # 실제 모델 선택은 서버 routing.yaml의 책임이다
    messages=[{"role": "user", "content": "주문 지연 안내 문구를 만들어 줘"}],
)
```

기본 별칭은 chat·vision·embed 세 가지고, 이미 OpenAI 규격으로 구현된 검색 모듈을 위해 `gpt-5.4-nano`(/v1/responses)와 `text-embedding-3-small`(1536차원) 호환 별칭도 받는다. 전체 계약과 오류 규격은 [연동 가이드](docs/API.md)에 있다.

## 실측 결과

| 항목 (측정 조건) | 로컬 | 비교 |
|---|---|---|
| 첫 토큰 p95 — 텍스트 단건 스트리밍 32건 | 2.74초 | 합격선 5초 |
| 생성 속도 — 같은 32건의 첫 토큰 이후 usage 토큰·시간 합산 | 30.8 tok/s | — |
| 생성 품질 — 함정·형식 9문항 블라인드 심판(gpt-5.6-sol) | 8.89 | gpt-5-mini 7.67 · gpt-5.4-nano 7.11 |
| 검색 품질 — 자체 질의 24건 Recall@1 | 22/24 | text-embedding-3-small 21/24 |
| 질의당 비용 — 한계 전기비 vs 동일 조건 API 환산 | 0.028원 | 0.141~0.236원 |

동시 요청은 8건까지 오류와 VRAM 초과 없이 처리되지만, 로컬 추론이 직렬이라 대기 시간이 급증한다(동시 8건 첫 토큰 p95 40.4초). 공개 객관 셋 KMMLU 48문항은 로컬 31/48로 비교 모델(각 32/48)과 동급이다. 측정 조건과 재현 경로는 [벤치마크 결과](docs/BENCHMARK_RESULTS.md), 해석과 운영 과제는 [벤치마크 피드백](docs/BENCHMARK_FEEDBACK.md)에 있다.

## 설계에서 지킨 것

- 모델 선택은 서버의 책임이다. 클라이언트는 별칭만 알고, 호환되는 로컬 모델의 매핑은 routing.yaml에서 바꿔 재배포한다.
- 폴백에는 경계가 있다. 별칭별 회로 차단기로 자동 복구하고, 임베딩은 벡터 공간 정합성 때문에 폴백하지 않으며, 스트리밍은 첫 유효 이벤트 이후 provider를 섞지 않는다.
- 모든 요청은 관측된다. JSON Lines 요청 기록과 x-request-id 헤더로 클라이언트 문의를 서버 기록과 대조할 수 있다.
- 운영 코드는 격리해 실행한다. SYSTEM 예약 작업은 관리자 전용 배포 사본만 실행하고, 로그인 없는 재부팅 복구를 실환경에서 확인했다.
- 단계마다 게이트가 있다. 저장소 안에서 완결되는 단계는 당시의 전체 자동화 테스트와 단계별 검증 스크립트(fake 결정적 검증 + 실 Ollama 실측)를 통과해야 다음으로 갔고(현재 자동화 테스트 554건), 외부 팀 일정에 걸린 검증(검색 모듈 E2E)만 예외로 대기한다.

## 문서

- [ROADMAP](docs/ROADMAP.md) — 9단계 계획과 단계별 실측·검증 기록
- [DECISIONS](docs/DECISIONS.md) — 시간순 결정 이력(맥락·결정·트레이드오프)
- [API](docs/API.md) — 클라이언트 기술 독립 연동 가이드(공개 /docs로 서빙되는 원본)
- [OPERATIONS](docs/OPERATIONS.md) — 배포 사본·SYSTEM 실행 경계·키 관리·감시 운영 절차
- [BENCHMARK_RESULTS](docs/BENCHMARK_RESULTS.md) — 공식 측정 결과와 재현 경로
- [BENCHMARK_FEEDBACK](docs/BENCHMARK_FEEDBACK.md) — 결과 해석과 후속 운영 과제

## 개발 검증

```powershell
uv sync
uv run pytest                              # 자동화 테스트 554건 — 업스트림은 목
uv run python scripts/verify_stage5.py     # fake 업스트림 결정적 검증의 대표 예 — 단계별 스크립트 제공
```

실제 스택 실행은 Windows와 Python 3.12에 GPU·CPU용 Ollama 인스턴스 두 개와 라우팅이 요구하는 두 모델, 서비스 API 키 발급이 전제이므로 [운영 가이드](docs/OPERATIONS.md)의 설치 절차를 따른다. OpenAI 폴백은 저장소 `.env`의 `OPENAI_API_KEY`가 있을 때만 동작하고, 없으면 로컬 전용으로 동작하며 폴백이 필요한 요청은 502로 끝난다.
