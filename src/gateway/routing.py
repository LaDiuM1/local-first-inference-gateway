"""용도별 별칭 → 실제 모델 라우팅 테이블.

클라이언트는 `chat`·`vision`·`embed` 같은 용도별 별칭만 호출하고, 별칭이 어느 실제 모델로
가는지는 이 테이블이 결정한다. 매핑은 코드가 아니라 YAML 설정에서 로드하므로, 모델 교체는
설정 파일 수정만으로 이뤄진다. 설정에 문제가 있으면 런타임으로 미루지 않고 기동 시점에 실패한다.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from gateway.errors import InvalidRequestError


class EndpointKind(StrEnum):
    """별칭이 서비스하는 OpenAI 엔드포인트. 엔드포인트와 맞지 않는 별칭은 거절 대상이다."""

    chat = "chat"
    embeddings = "embeddings"


@dataclass(frozen=True)
class Route:
    alias: str
    endpoint: EndpointKind
    model: str


# 클라이언트가 의존하는 외부 계약 별칭 — 하나라도 빠지면 기동 시점에 실패시킨다.
REQUIRED_ROUTES: frozenset[tuple[EndpointKind, str]] = frozenset(
    {
        (EndpointKind.chat, "chat"),
        (EndpointKind.chat, "vision"),
        (EndpointKind.embeddings, "embed"),
    }
)


class RoutingConfigError(Exception):
    """라우팅 설정 파일이 없거나 형식·내용이 잘못됐다 — 게이트웨이가 기동하지 못한다."""


class RoutingTable:
    """(엔드포인트, 별칭) → 실제 모델 조회. 등록되지 않은 별칭은 400으로 거절한다."""

    def __init__(self, routes: list[Route]) -> None:
        self._models = {(route.endpoint, route.alias): route.model for route in routes}

    def resolve(self, endpoint: EndpointKind, alias: str) -> str:
        model = self._models.get((endpoint, alias))
        if model is None:
            raise InvalidRequestError(f"unknown model alias '{alias}'")
        return model


def load_routing_table(path: Path) -> RoutingTable:
    if not path.exists():
        raise RoutingConfigError(f"라우팅 설정 파일이 없다: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RoutingConfigError(
            f"라우팅 설정 YAML을 파싱할 수 없다: {error}"
        ) from error
    return _build_table(document)


def _build_table(document: object) -> RoutingTable:
    if not isinstance(document, dict) or "routes" not in document:
        raise RoutingConfigError("라우팅 설정 최상위에 routes 목록이 있어야 한다")
    entries = document["routes"]
    if not isinstance(entries, list) or not entries:
        raise RoutingConfigError("routes는 비어 있지 않은 목록이어야 한다")

    routes: list[Route] = []
    seen_aliases: set[str] = set()
    for index, entry in enumerate(entries):
        route = _parse_route(entry, index)
        if route.alias in seen_aliases:
            raise RoutingConfigError(f"중복된 별칭이 있다: '{route.alias}'")
        seen_aliases.add(route.alias)
        routes.append(route)

    configured = {(route.endpoint, route.alias) for route in routes}
    missing = REQUIRED_ROUTES - configured
    if missing:
        described = ", ".join(
            f"{endpoint.value}:{alias}" for endpoint, alias in sorted(missing)
        )
        raise RoutingConfigError(f"필수 라우팅이 빠져 있다: {described}")
    return RoutingTable(routes)


def _parse_route(entry: object, index: int) -> Route:
    if not isinstance(entry, dict):
        raise RoutingConfigError(f"routes[{index}]는 매핑(dict)이어야 한다")

    alias = _require_config_str(entry, "alias", index)
    model = _require_config_str(entry, "model", index)
    endpoint_value = _require_config_str(entry, "endpoint", index)
    try:
        endpoint = EndpointKind(endpoint_value)
    except ValueError:
        allowed = ", ".join(kind.value for kind in EndpointKind)
        raise RoutingConfigError(
            f"routes[{index}]의 endpoint '{endpoint_value}'가 올바르지 않다 (허용: {allowed})"
        ) from None
    return Route(alias=alias, endpoint=endpoint, model=model)


def _require_config_str(entry: dict, field: str, index: int) -> str:
    # 공백뿐인 문자열은 빈 값과 같다 — 런타임으로 미루지 않고 기동 시점에 실패시킨다.
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RoutingConfigError(
            f"routes[{index}]에 비어 있지 않은 {field} 문자열이 필요하다"
        )
    return value
