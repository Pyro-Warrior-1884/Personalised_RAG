"""Exception handling for the API.

The default Django REST Framework handler is replaced so that errors are
returned in a predictable shape and never leak stack traces or internal
paths to the browser.
"""

from __future__ import annotations

import logging

from rest_framework.views import exception_handler

logger = logging.getLogger(__name__)


def api_exception_handler(exc, context):
    """Return a consistent JSON error payload."""
    response = exception_handler(exc, context)
    if response is None:
        # An unexpected error: log it server side, tell the client nothing
        # about the internals.
        logger.exception("Unhandled API error", exc_info=exc)
        from rest_framework.response import Response
        from rest_framework import status

        return Response(
            {
                "error": "server_error",
                "detail": "Something went wrong while processing the request.",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    detail = response.data
    message = detail
    if isinstance(detail, dict):
        message = detail.get("detail") or detail
    if isinstance(message, (list, tuple)) and message:
        message = message[0]
    if isinstance(detail, dict) and not detail.get("detail") and len(detail) == 1:
        # Field errors: {"content": ["..."]} -> ["..."]
        only_value = next(iter(detail.values()))
        message = only_value[0] if isinstance(only_value, (list, tuple)) and only_value else only_value

    response.data = {
        "error": getattr(exc, "default_code", "error"),
        "detail": str(message),
    }
    return response
