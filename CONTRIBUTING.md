# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest --cov
```

CI runs the same checks on Python 3.10 to 3.13. Coverage must stay at 85% or above.

## Guidelines

- **Tests make no network calls.** Use the `FakeClient` in `tests/conftest.py`.
- **Keep prompts deterministic.** Prompt caching is a byte-exact prefix match: never put
  timestamps, random IDs or unordered collections in anything that is sent to the model
  before the question. `test_cached_prefix_is_byte_identical_across_questions` guards this.
- **Public API changes** need docstrings (Google style), a README update if user-facing,
  and a `CHANGELOG.md` entry under *Unreleased*.
- **Don't wrap Anthropic SDK exceptions.** Let them propagate; they carry status codes
  and request IDs.
- Follow [Semantic Versioning](https://semver.org/). The version lives in
  `src/longdoc/_version.py`.
