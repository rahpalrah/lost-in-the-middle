# Deployment Guide

## Standalone HTTP Server

The built-in server (`longdoc serve`) is a single-process, multi-threaded Python HTTP server
using only the standard library. It is suitable for small to medium deployments.

```bash
export ANTHROPIC_API_KEY=...
export LONGDOC_API_KEY=choose-a-secure-token
longdoc serve --host 0.0.0.0 --port 8000 --store /var/lib/longdoc-data
```

**Concurrency limits:**
- `--max-concurrency` (default 32): slots semaphore; excess requests get `503 server_busy`.
- Per-client rate limit: `LONGDOC_RATE_LIMIT_PER_MIN` (default none).
- Request size: `LONGDOC_MAX_UPLOAD_BYTES` (default 100 MB).

**Persistence:**
- `--store DIR`: JSON file store for documents (optional; in-memory otherwise).
- Each document is one JSON file; suitable for 100s to 1000s of documents.

## Multi-Process Deployments

For high throughput, run several processes behind a load balancer:

### Option 1: Multiple Built-In Servers (Simplest)

```bash
for i in {1..4}; do
  port=$((8000 + i))
  LONGDOC_STORE_DIR=/var/lib/longdoc-data \
  longdoc serve --host 127.0.0.1 --port $port &
done
# nginx or similar load balances across ports 8001–8004
```

**Limitations:**
- All processes must share the store directory (NFS, shared volume, etc.).
- Max throughput ~300 req/s per process (CPU-bound on typical hardware).

### Option 2: FastAPI with Uvicorn Workers

```bash
pip install "longdoc-qa[fastapi]"
LONGDOC_STORE_DIR=/var/lib/longdoc-data \
longdoc serve --framework fastapi --workers 4
# or directly:
uvicorn longdoc.server.asgi:app --workers 4 --host 0.0.0.0 --port 8000
```

**Advantages:**
- Native async/await with many workers.
- Can scale further with async database backends.

## Docker

### Single Process (Default)

```bash
docker build -t longdoc .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=... \
  -e LONGDOC_API_KEY=choose-a-token \
  -v longdoc-data:/data \
  longdoc
```

The default Dockerfile uses the built-in server (no FastAPI extra needed).

### Multi-Process Cluster (Kubernetes, Docker Compose)

```yaml
# docker-compose.yml
version: "3.8"
services:
  api-1:
    image: longdoc:latest
    ports:
      - "8001:8000"
    environment:
      LONGDOC_API_KEY: token
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LONGDOC_STORE_DIR: /data
    volumes:
      - longdoc-data:/data
  
  api-2:
    image: longdoc:latest
    ports:
      - "8002:8000"
    environment:
      LONGDOC_API_KEY: token
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LONGDOC_STORE_DIR: /data
    volumes:
      - longdoc-data:/data
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api-1
      - api-2

volumes:
  longdoc-data:
```

## Platform Support

### Linux (Primary)

Fully tested and recommended. Uses stdlib `ThreadingHTTPServer` and standard Python
file/JSON operations.

### macOS

✓ Tested: CLI, library, HTTP server, PDF reading, all features work.

Minor considerations:
- File descriptors: `ulimit -n 4096` or higher for many concurrent requests.
- Keychain: if used for credential storage, make sure the process has permission.

### Windows

✓ Works with WSL 2 (Linux subsystem). Native Windows support is untested but likely works because:
- `ThreadingHTTPServer` is cross-platform.
- PDF reader uses only standard library functions available on Windows.
- JSON file store works on Windows (note: file locking may differ).

**Untested on native Windows (outside WSL); if issues arise, WSL 2 is the supported path.**

## TLS / HTTPS

The server listens on `http://` only. For production, use a reverse proxy:

```nginx
upstream longdoc {
  server 127.0.0.1:8000;
  server 127.0.0.1:8001;
  server 127.0.0.1:8002;
}

server {
  listen 443 ssl http2;
  server_name api.example.com;
  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;

  location / {
    proxy_pass http://longdoc;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-Proto https;
  }
}
```

**Trust proxy headers:** set `LONGDOC_TRUST_PROXY=true` so rate limits are based on
`X-Forwarded-For` (the client's real IP), not the proxy's IP.

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.9.0"}
```

The Docker image includes `HEALTHCHECK` using this endpoint.

### Metrics

The server does not export Prometheus metrics yet. To measure throughput and latency:
- Access logs: set `LONGDOC_LOG_LEVEL=DEBUG` to see HTTP requests in stderr.
- Load test: `python benchmarks/load_test.py --url http://localhost:8000 --seconds 30`.

### Error Handling

Errors are logged to stderr and returned as JSON with an HTTP status code and error code:

```json
{
  "error": "rate_limited",
  "message": "rate limit exceeded",
  "details": {"retry_after_seconds": 1}
}
```

Error codes: `not_found`, `rate_limited`, `server_busy`, `bad_request`, `upstream_error`, etc.

## Performance Tuning

### Concurrency Limit

For a document store on NFS or S3, increase `--max-concurrency` if you have spare CPU:

```bash
longdoc serve --max-concurrency 64
```

For a CPU-bound workload (e.g., the built-in BM25 retrieval), no benefit beyond ~4× the
number of CPU cores.

### Rate Limiting

Per-client sliding-window rate limit (default none, i.e., unlimited):

```bash
export LONGDOC_RATE_LIMIT_PER_MIN=1000  # 1000 requests per minute per IP
longdoc serve
```

### Chunk Sizes

The parent-child chunking happens at load time and is per-document. Defaults are good for
most contracts; adjust only if you observe poor retrieval:

```python
from longdoc import DocumentQA, ChunkingConfig, QAConfig

qa = DocumentQA.from_file(
    "contract.pdf",
    chunking=ChunkingConfig(parent_max_chars=8000, child_max_chars=1000),
    config=QAConfig(context_rot=True, recursive=True),
)
```

## Scaling

| Deployment | Throughput | Latency (p95) | Setup | Notes |
|---|---|---|---|---|
| Single built-in server | 300 req/s | 50 ms | Simple | Good for internal tools |
| 4 × built-in behind LB | 1200 req/s | 50 ms | Moderate | NFS/shared store needed |
| FastAPI + 8 workers | 2000 req/s | 100 ms | Easy | Scales further with async DB |
| Kubernetes cluster | 10k+ req/s | 100–200 ms | Complex | Multiple replicas + autoscaling |

(Numbers are for the built-in extractive engine on typical 4-core hardware. With Claude or
OpenAI, the model's latency dominates, so server throughput is less of a bottleneck.)
