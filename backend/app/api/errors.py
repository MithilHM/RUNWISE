"""One error shape for every endpoint.

PRD 9 requires structured errors, and the frontend needs a single contract rather
than two: the analysis routers returned ``{"detail": [{"code", "message", "field"}]}``
while the graph routers returned FastAPI's ``{"detail": "<string>"}``, and a
validation failure returned a third shape entirely. A client cannot render a
message it has to guess the shape of.

Everything here produces::

    {"detail": [{"code": "experiment_not_found",
                 "message": "No experiment with id 'exp_404' exists.",
                 "field": null}]}

``app/main.py`` installs matching handlers for HTTPException, request validation
and database failures, so a code path that forgets to raise through this module
still returns the same envelope.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status


def detail(code: str, message: str, field: str | None = None) -> list[dict[str, Any]]:
    """Build the canonical detail list."""
    return [{"code": code, "message": message, "field": field}]


def http_error(
    status_code: int,
    code: str,
    message: str,
    field: str | None = None,
) -> HTTPException:
    """An HTTPException carrying the canonical detail list."""
    return HTTPException(status_code=status_code, detail=detail(code, message, field))


def not_found(entity: str, entity_id: str) -> HTTPException:
    return http_error(
        status.HTTP_404_NOT_FOUND,
        f"{entity}_not_found",
        f"No {entity} with id {entity_id!r} exists.",
    )


def bad_request(code: str, message: str, field_name: str | None = None) -> HTTPException:
    return http_error(status.HTTP_400_BAD_REQUEST, code, message, field_name)


def graph_unavailable(message: str, code: str = "graph_service_unavailable") -> HTTPException:
    """503 for a dependency that is not wired or not reachable."""
    return http_error(status.HTTP_503_SERVICE_UNAVAILABLE, code, message)


def graph_unreachable(message: str, code: str = "graph_unavailable") -> HTTPException:
    """503 specifically reporting unreachable graph database."""
    return http_error(status.HTTP_503_SERVICE_UNAVAILABLE, code, message)


def unprocessable(code: str, message: str, field_name: str | None = None) -> HTTPException:
    """422 for semantically invalid requests (e.g. pr_auc on multiclass)."""
    return http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, code, message, field_name)


