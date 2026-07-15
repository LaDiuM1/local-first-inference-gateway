"""운영 상태와 배포 자산의 고정 경로."""

import ctypes
import os
from pathlib import Path
from uuid import UUID


class OperationalPathError(Exception):
    """운영 상태 파일의 고정 위치를 확인할 수 없다."""


# FOLDERID_ProgramData — 모든 사용자의 공용 애플리케이션 데이터 폴더.
_PROGRAM_DATA_FOLDER_ID = UUID("62ab5d82-fdc1-4dc3-a9dd-070d1d495d97")


class _KnownFolderId(ctypes.Structure):
    _fields_ = [("value", ctypes.c_byte * 16)]


def _program_data_directory() -> Path:
    if os.name != "nt":
        raise OperationalPathError("Windows is required")
    folder_id = _KnownFolderId.from_buffer_copy(_PROGRAM_DATA_FOLDER_ID.bytes_le)
    buffer = ctypes.c_wchar_p()
    result = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(folder_id), 0, None, ctypes.byref(buffer)
    )
    if result != 0:
        raise OperationalPathError("Windows could not resolve the common data folder")
    try:
        return Path(buffer.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(buffer)


def _deployment_root() -> Path:
    # src/gateway/paths.py 기준 — 지금 실행 중인 이 코드가 담긴 배포본의 루트.
    return Path(__file__).resolve().parents[2]


STATE_DIRECTORY = _program_data_directory() / "local-first-inference-gateway"
API_KEY_STORE_PATH = STATE_DIRECTORY / "api-keys.json"
WATCHDOG_KEY_PATH = STATE_DIRECTORY / "watchdog-key.dpapi"
PROTECTED_STATE_PATHS = frozenset({API_KEY_STORE_PATH, WATCHDOG_KEY_PATH})
DEFAULT_WATCHDOG_LOG_PATH = STATE_DIRECTORY / "watchdog.log"
TASK_CONTROL_SCRIPT = _deployment_root() / "scripts" / "task_control.ps1"
PUBLIC_DOCS_PATH = _deployment_root() / "docs" / "API.md"
