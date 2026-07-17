"""라우팅 설정 로드와 별칭 해석 단위 테스트 — 구성 오류는 기동 시 명확히 실패해야 한다."""

from pathlib import Path

import pytest

from gateway.errors import InvalidRequestError
from gateway.routing import (
    EndpointKind,
    RoutingConfigError,
    load_routing_table,
)

VALID_CONFIG = """
routes:
  - alias: chat
    endpoint: chat
    model: gemma4:12b-it-qat
  - alias: vision
    endpoint: chat
    model: gemma4:12b-it-qat
  - alias: embed
    endpoint: embeddings
    model: snowflake-arctic-embed2
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "routing.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_resolves_alias_to_configured_model(tmp_path: Path) -> None:
    table = load_routing_table(_write_config(tmp_path, VALID_CONFIG))

    assert table.resolve(EndpointKind.chat, "chat") == "gemma4:12b-it-qat"
    assert table.resolve(EndpointKind.chat, "vision") == "gemma4:12b-it-qat"
    assert table.resolve(EndpointKind.embeddings, "embed") == "snowflake-arctic-embed2"


def test_config_change_swaps_target_model(tmp_path: Path) -> None:
    swapped = VALID_CONFIG.replace("gemma4:12b-it-qat", "some-other-model")
    table = load_routing_table(_write_config(tmp_path, swapped))

    assert table.resolve(EndpointKind.chat, "chat") == "some-other-model"


def test_alias_bound_to_its_endpoint(tmp_path: Path) -> None:
    table = load_routing_table(_write_config(tmp_path, VALID_CONFIG))

    with pytest.raises(InvalidRequestError):
        table.resolve(EndpointKind.embeddings, "chat")
    with pytest.raises(InvalidRequestError):
        table.resolve(EndpointKind.chat, "embed")


def test_unknown_alias_raises_invalid_request(tmp_path: Path) -> None:
    table = load_routing_table(_write_config(tmp_path, VALID_CONFIG))

    with pytest.raises(InvalidRequestError):
        table.resolve(EndpointKind.chat, "no-such-alias")


def test_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_table(tmp_path / "absent.yaml")


def test_broken_yaml_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_table(_write_config(tmp_path, "routes: [unclosed"))


def test_duplicate_alias_fails_clearly(tmp_path: Path) -> None:
    duplicated = (
        VALID_CONFIG
        + """
  - alias: chat
    endpoint: chat
    model: another-model
"""
    )
    with pytest.raises(RoutingConfigError):
        load_routing_table(_write_config(tmp_path, duplicated))


@pytest.mark.parametrize(
    "content",
    [
        "model: gemma4",
        "routes: []",
        "routes:\n  - alias: chat\n    endpoint: chat",
        "routes:\n  - alias: chat\n    endpoint: audio\n    model: gemma4",
        "routes:\n  - endpoint: chat\n    model: gemma4",
    ],
    ids=[
        "no-routes",
        "empty-routes",
        "missing-model",
        "invalid-endpoint",
        "missing-alias",
    ],
)
def test_incomplete_or_invalid_routing_fails_clearly(
    tmp_path: Path, content: str
) -> None:
    with pytest.raises(RoutingConfigError):
        load_routing_table(_write_config(tmp_path, content))


REQUIRED_ROUTE_BLOCKS = {
    "chat": "  - alias: chat\n    endpoint: chat\n    model: gemma4:12b-it-qat\n",
    "vision": "  - alias: vision\n    endpoint: chat\n    model: gemma4:12b-it-qat\n",
    "embed": "  - alias: embed\n    endpoint: embeddings\n    model: snowflake-arctic-embed2\n",
}


def test_blank_model_string_fails_clearly(tmp_path: Path) -> None:
    # 공백뿐인 모델명은 빈 값과 같다 — 기동 시점에 실패시킨다.
    blanked = VALID_CONFIG.replace("gemma4:12b-it-qat", '"   "', 1)

    with pytest.raises(RoutingConfigError):
        load_routing_table(_write_config(tmp_path, blanked))


@pytest.mark.parametrize("dropped_alias", ["chat", "vision", "embed"])
def test_missing_required_route_fails_clearly(
    tmp_path: Path, dropped_alias: str
) -> None:
    kept = [
        block
        for alias, block in REQUIRED_ROUTE_BLOCKS.items()
        if alias != dropped_alias
    ]
    partial = "routes:\n" + "".join(kept)

    with pytest.raises(RoutingConfigError):
        load_routing_table(_write_config(tmp_path, partial))
