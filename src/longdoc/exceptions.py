"""Exception hierarchy for longdoc.

Every error raised by this library derives from :class:`LongDocError`, so callers can
catch one base class. Errors from the Anthropic API itself (rate limits, auth, network)
are *not* wrapped: they propagate as the ``anthropic`` SDK's typed exceptions, which
already carry status codes and request IDs.
"""

from __future__ import annotations


class LongDocError(Exception):
    """Base class for all longdoc errors."""


class ConfigurationError(LongDocError, ValueError):
    """A configuration value is invalid."""


class DocumentLoadError(LongDocError):
    """A document could not be read or parsed."""


class UnsupportedFormatError(DocumentLoadError):
    """The file type is not supported, or its optional dependency is not installed."""


class EmptyDocumentError(DocumentLoadError):
    """The document contains no extractable text."""


class RefusalError(LongDocError):
    """The model declined to answer (``stop_reason == "refusal"``).

    Attributes:
        category: The refusal category reported by the API, if any.
        explanation: The API's explanation, if any.
        request_id: The Anthropic request ID, for support tickets.
    """

    def __init__(
        self,
        category: str | None = None,
        explanation: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.category = category
        self.explanation = explanation
        self.request_id = request_id
        detail = f" ({category})" if category else ""
        super().__init__(f"The model declined to answer{detail}. {explanation or ''}".strip())
