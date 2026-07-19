<div align="center">

<img src="docs/assets/readme-banner.svg" width="880"
     alt="local-first-inference-gateway — Python 3.12 · FastAPI · Ollama · OpenAI 호환 /v1">

**로컬 Ollama를 우선하고 장애 시 OpenAI로 우회하는, 홈 GPU 한 대로 운영하는 OpenAI 호환 추론 게이트웨이**

</div>

팀 프로젝트 openAt의 MSA 클라이언트들이 호출하는 추론 게이트웨이다. 홈 GPU(GTX 1080 Ti 11GB) 한 대에서 로컬 Ollama를 기본 경로로 쓰고, 로컬 장애 시에만 OpenAI로 우회해 서비스 연속성을 지킨다. 클라이언트가 보는 계약은 지원 범위 내 OpenAI 호환 규격이라, base_url·키·모델 별칭 설정 전환으로 연동한다. 단건 스트리밍 첫 토큰 p95 2.74초와 질의당 추가 전기요금 0.028원을 실측으로 확인했다. 게이트웨이 쪽 연동 계약과 벤치마크는 완료됐고, 검색 모듈의 실제 E2E 전환은 담당자 일정에 맞춰 대기 중이다.

<div align="center">
<img src="docs/assets/architecture.svg" width="880"
     alt="openAt MSA 클라이언트 → Cloudflare Named Tunnel → 추론 게이트웨이(FastAPI) → 1차 경로 Ollama, 로컬 장애 시 OpenAI gpt-5-mini">
</div>

## 실측 결과

성능·품질·비용을 같은 조건에서 직접 측정해, 사전에 정한 합격선 네 항목을 모두 통과했다.

<div align="center">
<img src="docs/assets/metrics-strip.svg" width="880"
     alt="첫 토큰 p95 2.74초 · 생성 30.8 tok/s · 블라인드 심판 품질 8.89 · 질의당 추가 전기요금 0.028원">
</div>

| 항목 | 로컬 | 비교 |
|---|---:|---|
| 첫 토큰 p95 <sub>텍스트 단건 스트리밍 32건</sub> | 2.74초 | 합격선 5초 |
| 생성 속도 <sub>같은 32건의 usage 실측 토큰 기준</sub> | 30.8 tok/s | — |
| 생성 품질 <sub>정답이 없는 9문항(근거 함정·형식)의 블라인드 심판(gpt-5.6-sol) 점수</sub> | 8.89 | gpt-5-mini 7.67 · gpt-5.4-nano 7.11 |
| 검색 품질 <sub>자체 질의 24건 Recall@1</sub> | 22/24 | text-embedding-3-small 21/24 |
| 질의당 추가 전기요금 <sub>실측 전력 기반, 동일 조건 API 환산 대비</sub> | 0.028원 | 0.141~0.236원 |

<sub>0.028원은 이미 가동 중인 장비에서 요청 한 건을 더 처리할 때 드는 전기요금 실측값이다(환율 1,500원/USD·전기 210원/kWh). 장비 구입비와 유휴 전력은 포함하지 않으므로 총소유비용이 같은 비율로 낮아졌다는 뜻은 아니다.</sub>

공개 객관 셋 KMMLU 48문항은 로컬 31/48로 비교 모델(각 32/48)과 동급이고, 자체 고난도 18문항은 로컬 16/18로 두 비교 모델(14/18·12/18)을 앞섰다. 사고 모드(`reasoning_effort: high`)는 고난도 17/18로 정확도가 더 높지만 평균 응답이 80초라, 정확도가 지연보다 중요한 요청에만 쓸 근거로 남겼다. 동시 요청은 8건까지 오류와 VRAM 초과 없이 처리되지만 로컬 추론이 직렬이라 대기 시간이 급증한다(동시 8건 첫 토큰 p95 40.4초). 이에 동시성이 높아지는 시점의 라우팅 정책 조정을 후속 과제로 남겼다. 측정 조건과 재현 경로는 [벤치마크 결과](docs/BENCHMARK_RESULTS.md), 해석과 운영 과제는 [벤치마크 피드백](docs/BENCHMARK_FEEDBACK.md)에 있다.

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

기본 별칭은 chat·vision·embed 세 가지다. 이미 OpenAI 규격으로 구현된 검색 모듈을 위해 `gpt-5.4-nano`(/v1/responses)와 `text-embedding-3-small`(1536차원 zero-padding, 코사인 유사도 보존) 호환 별칭도 받는다. `/v1/responses`의 지원 범위는 비스트리밍 이미지 분석이고, 호환 별칭의 임베딩 벡터는 OpenAI와 차원만 같고 벡터 공간이 다르므로 provider를 전환할 때는 전체 색인을 다시 만드는 것이 전제다. 전체 계약과 오류 규격은 [연동 가이드](docs/API.md)에 있다.

## 왜 직접 만들었나

파이널 프로젝트에 RAG·이미지 분석 같은 LLM 기능이 들어가면서 지원받는 API 토큰 한도가 팀 공용의 제약이 됐다. 나는 이것을 해결해야 할 문제로 정의했고, 사전 리서치에서 보유한 GPU로도 프로젝트 범위의 품질을 감당할 가능성을 확인해 추론 경로 자체를 직접 구축하기로 결정했다. 계약을 OpenAI 규격으로 통일한 것은 업계 표준이면서 클라이언트 변경을 최소화하는 선택이기 때문이다. 각 판단의 배경과 트레이드오프는 [결정 이력](docs/DECISIONS.md)에 시간순으로 남겼다.

## 설계에서 지킨 것

| 원칙 | 구현 |
|---|---|
| 모델 선택은 서버의 책임이다 | 클라이언트는 별칭만 알고, 실제 모델 매핑은 routing.yaml이 담당한다. 모델을 바꿔도 클라이언트는 수정하지 않는다. |
| 잘못된 요청은 업스트림에 닿지 않는다 | 인증 실패·미등록 별칭·규격 위반 요청은 추론 호출 전에 OpenAI 규격 오류로 끝난다. |
| 폴백에는 경계가 있다 | 별칭별 회로 차단기로 자동 복구하고, 임베딩은 벡터 공간 정합성 때문에 폴백하지 않으며, 스트리밍은 첫 유효 이벤트 이후 provider를 섞지 않는다. |
| 모든 요청은 관측된다 | 모든 요청을 JSON Lines 한 줄로 남기고, 응답의 x-request-id 헤더로 클라이언트 문의와 서버 기록을 대조한다. |
| 단계마다 게이트가 있다 | 각 단계는 전체 자동화 테스트(현재 554건)와 단계별 검증 스크립트(fake 결정적 검증 + 실 Ollama 실측)를 통과해야 다음으로 갔다. 외부 팀 일정에 걸린 검색 모듈 E2E만 예외로 대기한다. |

## 운영

게이트웨이·두 Ollama·감시 프로세스는 SYSTEM 예약 작업으로 실행하되, 비관리자가 수정할 수 있는 작업 트리가 아니라 관리자 전용 배포 사본만 실행한다. 재부팅 후 로그인 없이 4.91분 안에 전 기능이 복구되고, 게이트웨이 프로세스를 강제 종료해도 5.4초 만에 복구되는 것을 실환경에서 확인했다. 응답 시작 기한(로컬 90초·전체 115초), 요청 본문 32MiB 상한, 서비스별 API 키(해시만 저장, 개별 폐기)를 운영 계약으로 둔다. 모니터링은 상주 수집 스택을 단일 홈 서버에 과설계라 판단해 제외하고, 관측 로그 후처리와 실시간 상태를 묶은 온디맨드 경량 모니터로 대신한다. 절차 전체는 [운영 가이드](docs/OPERATIONS.md)에 있다.

## 문서

| 문서 | 내용 |
|---|---|
| [ROADMAP](docs/ROADMAP.md) | 9단계 계획과 단계별 실측·검증 기록 |
| [DECISIONS](docs/DECISIONS.md) | 시간순 결정 이력(맥락·결정·트레이드오프) |
| [API](docs/API.md) | 클라이언트 기술 독립 연동 가이드(공개 /docs로 서빙되는 원본) |
| [OPERATIONS](docs/OPERATIONS.md) | 배포 사본·SYSTEM 실행 경계·키 관리·감시 운영 절차 |
| [BENCHMARK_RESULTS](docs/BENCHMARK_RESULTS.md) | 공식 측정 결과와 재현 경로 |
| [BENCHMARK_FEEDBACK](docs/BENCHMARK_FEEDBACK.md) | 결과 해석과 후속 운영 과제 |

## 개발 검증

```powershell
uv sync
uv run pytest                              # 자동화 테스트 554건 — 업스트림은 목
uv run python scripts/verify_stage5.py     # fake 업스트림 결정적 검증의 대표 예 — 단계별 스크립트 제공
```

실제 스택 실행은 Windows와 Python 3.12에 GPU·CPU용 Ollama 인스턴스 두 개, 라우팅이 요구하는 두 모델, 서비스 API 키 발급이 전제이므로 [운영 가이드](docs/OPERATIONS.md)의 설치 절차를 따른다. OpenAI 폴백은 저장소 `.env`의 `OPENAI_API_KEY`가 있을 때만 동작하고, 없으면 로컬 전용으로 동작하며 폴백이 필요한 요청은 502로 끝난다.
