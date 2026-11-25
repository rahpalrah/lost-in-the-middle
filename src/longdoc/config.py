"""Configuration for :class:`longdoc.DocumentQA`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

from .exceptions import ConfigurationError

DEFAULT_MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]
CacheTTL = Literal["5m", "1h"]
Fallbacks = Optional[Union[Literal["default"], Sequence[str]]]  # noqa: UP007, UP045

_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})


class Strategy(str, Enum):
    """How document content is sent to the model.

    ``RETRIEVAL``
        Parent-child retrieval: search small child chunks, send only the parent
        sections they belong to. Smallest prompts; best for one-off questions and for
        documents bigger than you want to pay to send in full.

    ``CACHED_DOCUMENT``
        Send the whole (sectioned) document as a cached prompt prefix. The first call
        pays a cache write; later calls within the TTL read it at a small fraction of
        the input price. Best for many questions in a short window over one document.
    """

    RETRIEVAL = "retrieval"
    CACHED_DOCUMENT = "cached_document"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingConfig:
    """Sizes for parent-child chunking, in characters (~4 characters per token).

    Attributes:
        parent_max_chars: Upper bound on a parent section. Headings also start a new
            parent, so most parents are smaller.
        child_max_chars: Upper bound on a searchable child passage.
        child_overlap_chars: Overlap between neighbouring children, so a fact on a
            boundary is fully inside at least one child.
    """

    parent_max_chars: int = 6_000
    child_max_chars: int = 800
    child_overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.child_max_chars <= 0 or self.parent_max_chars <= 0:
            raise ConfigurationError("Chunk sizes must be positive.")
        if self.child_max_chars > self.parent_max_chars:
            raise ConfigurationError("child_max_chars must not exceed parent_max_chars.")
        if not 0 <= self.child_overlap_chars < self.child_max_chars:
            raise ConfigurationError("child_overlap_chars must be in [0, child_max_chars).")


@dataclass(frozen=True, slots=True, kw_only=True)
class QAConfig:
    """Settings for question answering.

    Attributes:
        model: Claude model ID.
        effort: Thinking depth and token spend (``output_config.effort``). Claude Opus
            5.5 defaults to ``"medium"`` server-side; we set it explicitly.
        max_tokens: Ceiling on the answer length. Requests stream, so large values are
            safe.
        strategy: Default :class:`Strategy` for :meth:`DocumentQA.ask`.
        top_k_children: How many child passages to retrieve per question.
        max_parents: How many parent sections at most to send per question.
        max_context_chars: Character budget for the sections sent per question.
        cache_ttl: Prompt-cache lifetime for ``CACHED_DOCUMENT``: ``"5m"`` (cheaper
            write) or ``"1h"`` (survives longer gaps between questions).
        fallbacks: Server-side refusal fallback. ``"default"`` lets the API choose a
            fallback model by refusal category; a sequence of model IDs names them;
            ``None`` disables fallbacks (use this for models or platforms that do not
            support them).
        system_prompt: Override the built-in instructions. Keep it constant: any change
            invalidates the prompt cache.
        context_rot: Drop weak and near-duplicate hits and put the strongest sections at the
            start and end of the context (see :mod:`longdoc.context`). Off by default; not
            measured to improve answers (no effect on the built-in engine).
        recursive: Split a multi-part question into sub-questions, search each, and merge the
            hits (see :mod:`longdoc.recursive`). Off by default; not measured to help.
        min_relevance: With ``context_rot``, a section is kept only if its score is at least this
            share of the best section's score.
    """

    model: str = DEFAULT_MODEL
    effort: Effort = "medium"
    max_tokens: int = 16_000
    strategy: Strategy = Strategy.RETRIEVAL
    top_k_children: int = 12
    max_parents: int = 4
    max_context_chars: int = 40_000
    cache_ttl: CacheTTL = "5m"
    fallbacks: Fallbacks = "default"
    system_prompt: str | None = None
    context_rot: bool = False
    recursive: bool = False
    min_relevance: float = 0.25

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigurationError("model must be a non-empty model ID.")
        if self.effort not in _EFFORTS:
            raise ConfigurationError(f"effort must be one of {sorted(_EFFORTS)}.")
        if self.max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive.")
        if self.top_k_children <= 0 or self.max_parents <= 0 or self.max_context_chars <= 0:
            raise ConfigurationError("Retrieval limits must be positive.")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ConfigurationError("min_relevance must be between 0 and 1.")
        if self.cache_ttl not in ("5m", "1h"):
            raise ConfigurationError('cache_ttl must be "5m" or "1h".')
        if isinstance(self.fallbacks, str) and self.fallbacks != "default":
            raise ConfigurationError('fallbacks must be "default", a list of model IDs, or None.')
