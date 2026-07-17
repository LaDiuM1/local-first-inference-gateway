"""운영 모니터 실행 진입점 — loopback 전용 온디맨드 실행.

실행: .venv\\Scripts\\python.exe -m gateway.monitor [--port 29100] [--log-dir <경로>]
운영 관측 로그는 관리자만 읽을 수 있으므로 운영 지표를 보려면 관리자 셸에서 실행한다.
"""

import argparse
from pathlib import Path

import uvicorn

from gateway.monitor.app import MonitorSettings, create_monitor_app

MONITOR_HOST = "127.0.0.1"
DEFAULT_PORT = 29100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="local-first-inference-gateway 운영 모니터"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="관측 로그 디렉터리 — 생략 시 운영 상태 디렉터리의 logs",
    )
    arguments = parser.parse_args()
    if arguments.log_dir is None:
        settings = MonitorSettings()
    else:
        settings = MonitorSettings(log_directory=arguments.log_dir)
    print(f"운영 모니터: http://{MONITOR_HOST}:{arguments.port}/")
    uvicorn.run(
        create_monitor_app(settings),
        host=MONITOR_HOST,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
