"""운영 모니터 웹 앱 — loopback 전용 온디맨드 대시보드와 상태 JSON.

게이트웨이와 별도 프로세스로 실행되어 게이트웨이 장애 중에도 스택 상태를 볼 수 있다.
프로브 대상은 watchdog과 같은 운영 고정값이며 설정으로 열지 않는다 — 검증 하네스만
앱 팩터리 인자로 임시 로그 경로와 fake 대상을 주입한다.
"""

import asyncio
import json
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from gateway.monitor.metrics import LogReader, LogSnapshot, summarize_requests
from gateway.monitor.probes import (
    probe_disk,
    probe_gateway,
    probe_gpu,
    probe_ollama,
    probe_scheduled_tasks,
    probe_service,
)
from gateway.paths import STATE_DIRECTORY
from gateway.watchdog import (
    CHAT_HEALTH_URL,
    CHAT_TASK_NAME,
    CLOUDFLARED_SERVICE_NAME,
    EMBEDDING_HEALTH_URL,
    EMBEDDING_TASK_NAME,
    GATEWAY_HEALTH_URL,
    GATEWAY_TASK_NAME,
    TASK_PATH,
)

# watchdog이 감시하는 것과 같은 운영 인스턴스 — 주소를 중복 정의하지 않고 파생한다.
CHAT_OLLAMA_BASE_URL = CHAT_HEALTH_URL.removesuffix("/api/version")
EMBEDDING_OLLAMA_BASE_URL = EMBEDDING_HEALTH_URL.removesuffix("/api/version")
# Watchdog 작업명은 자기 자신을 관리하지 않는 watchdog 코드에 없다 — 등록 스크립트의 이름.
STACK_TASK_NAMES = (
    GATEWAY_TASK_NAME,
    CHAT_TASK_NAME,
    EMBEDDING_TASK_NAME,
    "Watchdog",
)
SUMMARY_WINDOWS_SECONDS = (3600, 86400)


@dataclass(frozen=True)
class MonitorSettings:
    log_directory: Path = STATE_DIRECTORY / "logs"
    gateway_health_url: str = GATEWAY_HEALTH_URL
    chat_base_url: str = CHAT_OLLAMA_BASE_URL
    embedding_base_url: str = EMBEDDING_OLLAMA_BASE_URL
    task_path: str = TASK_PATH
    task_names: tuple[str, ...] = STACK_TASK_NAMES
    cloudflared_service: str = CLOUDFLARED_SERVICE_NAME
    # 예약 작업·서비스·GPU·디스크 프로브 — 결정적 검증에서는 끈다.
    include_system_probes: bool = True
    probe_timeout_seconds: float = 3.0
    summary_windows_seconds: tuple[int, ...] = SUMMARY_WINDOWS_SECONDS


async def _guarded_probe(name: str, probe: Awaitable[dict]) -> dict:
    """프로브 예외를 구성요소 실패로 격리한다 — 한 프로브가 상태 조회를 죽이지 않는다.

    업스트림이 무한대 수나 UTF-8로 인코딩할 수 없는 문자열을 돌려주면 상태 JSON
    직렬화 단계에서 응답 전체가 500이 된다 — 직렬화 가능성도 여기서 검증해 해당
    구성요소만 실패로 바꾼다.
    """
    try:
        result = await probe
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return result
    except Exception as error:
        # 예외 메시지 자체에 인코딩 불가 문자가 실려 올 수 있다 — 실패 응답도 안전하게 만든다.
        message = f"{type(error).__name__}: {error}"
        return {
            "name": name,
            "ok": False,
            "detail": message.encode("utf-8", "backslashreplace").decode("utf-8"),
            "data": {},
        }


async def _collect_status(settings: MonitorSettings, log_reader: LogReader) -> dict:
    # loopback 프로브가 환경 프록시를 타지 않도록 격리한다 — watchdog과 같은 이유.
    async with httpx.AsyncClient(
        timeout=settings.probe_timeout_seconds, trust_env=False
    ) as client:
        components = list(
            await asyncio.gather(
                _guarded_probe(
                    "gateway", probe_gateway(client, settings.gateway_health_url)
                ),
                _guarded_probe(
                    "chat_ollama",
                    probe_ollama(client, "chat_ollama", settings.chat_base_url),
                ),
                _guarded_probe(
                    "embedding_ollama",
                    probe_ollama(
                        client, "embedding_ollama", settings.embedding_base_url
                    ),
                ),
            )
        )
    if settings.include_system_probes:
        system_probes = await asyncio.gather(
            _guarded_probe(
                "scheduled_tasks",
                asyncio.to_thread(
                    probe_scheduled_tasks, settings.task_path, settings.task_names
                ),
            ),
            _guarded_probe(
                f"service:{settings.cloudflared_service}",
                asyncio.to_thread(probe_service, settings.cloudflared_service),
            ),
            _guarded_probe("gpu", asyncio.to_thread(probe_gpu)),
            _guarded_probe(
                "disk", asyncio.to_thread(probe_disk, settings.log_directory)
            ),
        )
        components.extend(system_probes)

    def read_and_summarize() -> tuple[LogSnapshot, list[dict]]:
        # 로그 읽기뿐 아니라 창 요약도 워커 스레드에서 한다 — 수만 건 집계가
        # 이벤트 루프를 막지 않는다.
        snapshot = log_reader.read()
        now_epoch = time.time()
        summaries = [
            summarize_requests(
                snapshot.records, now_epoch=now_epoch, window_seconds=window
            )
            for window in settings.summary_windows_seconds
        ]
        return snapshot, summaries

    snapshot, summaries = await asyncio.to_thread(read_and_summarize)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "components": components,
        "requests": summaries,
        "log": {
            "directory": str(settings.log_directory),
            "readable": snapshot.readable,
            "records": len(snapshot.records),
            "invalid_lines": snapshot.invalid_lines,
            "files_skipped": snapshot.files_skipped,
        },
    }


def create_monitor_app(settings: MonitorSettings | None = None) -> FastAPI:
    monitor_settings = settings if settings is not None else MonitorSettings()
    log_reader = LogReader(monitor_settings.log_directory)
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(await _collect_status(monitor_settings, log_reader))

    @app.get("/")
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_PAGE)

    return app


_DASHBOARD_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>local-first-inference-gateway 운영 모니터</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; margin: 2rem; background: #14161a; color: #e6e6e6; }
  h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 1.6rem; }
  table { border-collapse: collapse; margin-top: .5rem; }
  th, td { border: 1px solid #3a3f47; padding: .35rem .7rem; font-size: .85rem; text-align: left; }
  th { background: #1f232a; }
  .ok { color: #6fce7f; } .bad { color: #e8705f; }
  #meta { color: #9aa1ab; font-size: .8rem; margin-top: .4rem; }
</style>
</head>
<body>
<h1>local-first-inference-gateway 운영 모니터</h1>
<div id="meta">불러오는 중…</div>
<h2>구성요소</h2>
<table><thead><tr><th>이름</th><th>상태</th><th>세부</th></tr></thead><tbody id="components"></tbody></table>
<h2>요청 지표</h2>
<table><thead><tr><th>창</th><th>추론 요청</th><th>운영 오류</th><th>클라이언트 4xx</th><th>스트리밍</th><th>provider</th><th>응답 시작 p50/p95(ms)</th><th>전체 p50/p95(ms)</th><th>최대 동시</th><th>비추론 제외</th></tr></thead><tbody id="windows"></tbody></table>
<h2>별칭별 (24시간)</h2>
<table><thead><tr><th>별칭</th><th>요청</th><th>운영 오류</th><th>전체 p95(ms)</th></tr></thead><tbody id="aliases"></tbody></table>
<h2>회로 이벤트 (24시간)</h2>
<table><thead><tr><th>시각</th><th>별칭</th><th>상태</th></tr></thead><tbody id="circuits"></tbody></table>
<script>
function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
}
function latency(summary) {
  if (!summary || summary.p50 === undefined) return '-';
  return summary.p50 + ' / ' + summary.p95;
}
function windowLabel(seconds) {
  return seconds === 3600 ? '1시간' : seconds === 86400 ? '24시간' : seconds + 's';
}
async function refresh() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    let logSummary = status.log.readable
      ? '로그 ' + status.log.records + '건 (손상 줄 ' + status.log.invalid_lines + ')'
      : '⚠ 로그 접근 불가 — 관리자 셸이 필요하거나 경로가 없음';
    if (status.log.files_skipped > 0) {
      logSummary += ' · 회전 경합 스킵 ' + status.log.files_skipped + '개';
    }
    document.getElementById('meta').textContent =
      status.generated_at + ' · ' + logSummary + ' · ' + status.log.directory;
    const components = document.getElementById('components');
    components.replaceChildren();
    for (const component of status.components) {
      const row = document.createElement('tr');
      cell(row, component.name);
      cell(row, component.ok ? '정상' : '이상', component.ok ? 'ok' : 'bad');
      cell(row, component.detail);
      components.appendChild(row);
    }
    const windows = document.getElementById('windows');
    windows.replaceChildren();
    for (const summary of status.requests) {
      const row = document.createElement('tr');
      cell(row, windowLabel(summary.window_seconds));
      cell(row, String(summary.count));
      cell(row, summary.operational_errors + ' (' + (summary.error_rate * 100).toFixed(1) + '%)',
           summary.operational_errors ? 'bad' : 'ok');
      cell(row, String(summary.client_errors));
      cell(row, String(summary.streaming));
      cell(row, Object.entries(summary.providers).map(([k, v]) => k + ' ' + v).join(', ') || '-');
      cell(row, latency(summary.latency_ms.response_start));
      cell(row, latency(summary.latency_ms.total));
      cell(row, String(summary.max_concurrency));
      cell(row, String(summary.non_inference_count));
      windows.appendChild(row);
    }
    const daySummary = status.requests[status.requests.length - 1];
    const aliases = document.getElementById('aliases');
    aliases.replaceChildren();
    for (const [alias, entry] of Object.entries(daySummary.aliases)) {
      const row = document.createElement('tr');
      cell(row, alias);
      cell(row, String(entry.count));
      cell(row, String(entry.operational_errors), entry.operational_errors ? 'bad' : 'ok');
      cell(row, entry.total_p95_ms === undefined ? '-' : String(entry.total_p95_ms));
      aliases.appendChild(row);
    }
    const circuits = document.getElementById('circuits');
    circuits.replaceChildren();
    for (const event of daySummary.circuit_events) {
      const row = document.createElement('tr');
      cell(row, event.time);
      cell(row, event.alias);
      cell(row, event.state, event.state === 'closed' ? 'ok' : 'bad');
      circuits.appendChild(row);
    }
  } catch (error) {
    document.getElementById('meta').textContent = '상태 조회 실패: ' + error;
  }
}
async function pollLoop() {
  // 직전 조회가 끝난 뒤에만 다음 조회를 예약한다 — 프로브 지연 시 요청이 겹치지 않는다.
  await refresh();
  setTimeout(pollLoop, 5000);
}
pollLoop();
</script>
</body>
</html>
"""
