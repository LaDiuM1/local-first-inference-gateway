"""원자적 파일 교체와 보호된 운영 상태 파일의 소유권 유지."""

import ctypes
import os
import tempfile
from pathlib import Path

from gateway.paths import PROTECTED_STATE_PATHS

_ADMINISTRATORS_SID = "S-1-5-32-544"
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x1


def atomic_write(path: Path, content: bytes) -> None:
    """운영 상태 파일은 임시 파일의 소유권을 보호한 뒤 원자적으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if _requires_administrative_owner(path):
            _take_administrative_ownership(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _requires_administrative_owner(path: Path) -> bool:
    # 별칭(C:\Users\All Users 등)을 거쳐 온 경로도 같은 자리로 본다 — 보호를 건너뛰면 안 된다.
    return path.resolve() in PROTECTED_STATE_PATHS


def _take_administrative_ownership(path: Path) -> None:
    sid = _administrators_sid()
    try:
        error_code = ctypes.windll.advapi32.SetNamedSecurityInfoW(
            ctypes.c_wchar_p(str(path)),
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION,
            sid,
            None,
            None,
            None,
        )
    finally:
        ctypes.windll.kernel32.LocalFree(sid)
    if error_code != 0:
        raise ctypes.WinError(error_code)


def _administrators_sid() -> ctypes.c_void_p:
    sid = ctypes.c_void_p()
    converted = ctypes.windll.advapi32.ConvertStringSidToSidW(
        ctypes.c_wchar_p(_ADMINISTRATORS_SID), ctypes.byref(sid)
    )
    if not converted:
        raise ctypes.WinError()
    return sid
