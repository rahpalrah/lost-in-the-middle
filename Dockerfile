# syntax=docker/dockerfile:1
# Standalone longdoc HTTP API.
#
#   docker build -t longdoc .
#   docker run -p 8000:8000 \
#     -e ANTHROPIC_API_KEY=... -e LONGDOC_API_KEY=choose-a-token \
#     -v longdoc-data:/data longdoc

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LONGDOC_STORE_DIR=/data
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --from=build /dist/*.whl /tmp/
# Default: stdlib server (dependency-free). For multi-process: `--framework fastapi --workers N`
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[pdf]" && rm /tmp/*.whl
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["longdoc", "serve", "--host", "0.0.0.0", "--port", "8000"]
