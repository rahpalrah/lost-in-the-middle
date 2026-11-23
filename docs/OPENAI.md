# Using OpenAI instead of Claude

longdoc can answer with an OpenAI chat model (or any OpenAI-compatible server). Install the
extra and give it **your own** API key, using one of the safe methods below.

```bash
pip install "longdoc-qa[openai]"
```

> **Read this first: your key is a password.** Anyone who has it can spend your money. This
> repository is **public**, so a key committed here is exposed to the whole internet within
> minutes. Never paste a key into a chat, an issue, a pull request, a screenshot or a log.
> If one leaks, revoke it at once in your OpenAI account and create a new one.

## 1. Create a key with limits

1. Create a key at <https://platform.openai.com/api-keys>. Prefer a **project key** used only for
   this tool.
2. In your OpenAI account, set a **monthly spending limit** for that project. This is your real
   safety net: no setting in longdoc can stop a leaked key from being used elsewhere.

## 2. Give longdoc the key (pick ONE)

### A. A `.env` file on your own machine (easiest)

`.env.example` in the repository root is the template file for this.

```bash
cp .env.example .env
chmod 600 .env            # only you can read it (Linux and macOS)
# open .env in an editor and fill in OPENAI_API_KEY= and LONGDOC_MODEL=
```

`.env` is already listed in `.gitignore`, so git will not add it. Use it explicitly:

```bash
longdoc --env-file .env ask contract.pdf "What is the notice period?" --provider openai --model "$MODEL"
LONGDOC_ENV_FILE=.env longdoc serve --store ./data      # HTTP service (needs LONGDOC_PROVIDER=openai in the file)
```

```python
from longdoc.env import load_env_file

load_env_file(".env")  # real environment variables win; values are never printed
```

Nothing is loaded automatically: you must name the file.

### B. An environment variable

```bash
export OPENAI_API_KEY=...        # your shell; add to your secret manager for servers
docker run -e OPENAI_API_KEY -e LONGDOC_PROVIDER=openai -e LONGDOC_MODEL=... -p 8000:8000 longdoc
```

On servers, use your platform's secret store (Kubernetes Secret, AWS Secrets Manager, and so on),
not a plain file in the image. Never put the key in a `Dockerfile` (`ENV` or `ARG` bake it into
the image).

### C. A cloud coding session (for example Claude Code on the web)

You cannot upload a key file into such a session safely. Instead add `OPENAI_API_KEY` as an
**environment variable** in the cloud environment's settings (environment menu, then Edit), and
allow the network host `api.openai.com` there. Environment variables apply to **new** sessions
only, and anyone who can use that environment can read them.

### D. Continuous integration

Add it as a **repository secret** (GitHub: Settings, Secrets and variables, Actions) and read it
in the workflow as `${{ secrets.OPENAI_API_KEY }}`. Never write the key in the workflow file.

## 3. Choose a model

longdoc does not pick one for you, because available models depend on your account. Use any chat
model you can access:

```python
from longdoc import DocumentQA, QAConfig
from longdoc.openai_client import OpenAIClient

qa = DocumentQA.from_file(
    "contract.pdf",
    client=OpenAIClient(),  # reads OPENAI_API_KEY
    config=QAConfig(model="<your chat model>", fallbacks=None),
)
print(qa.ask("What is the termination notice period?").text)
```

CLI: `--provider openai --model <name>`. Server: `LONGDOC_PROVIDER=openai LONGDOC_MODEL=<name>`.
Leaving the default model (a Claude name) gives a clear error instead of a confusing API failure.

## 4. Check your setup without spending anything

```bash
python -c "from longdoc.openai_client import OpenAIClient; OpenAIClient(); print('key found')"
```

This only checks that a key is present; it does not call OpenAI. A first real question on a
small file costs a fraction of a cent.

## 5. Control cost

- **Provider-side limit** on the key's project (section 1): the only hard guarantee.
- `LONGDOC_RATE_LIMIT_PER_MIN` caps questions per client when you run the HTTP service.
- `QAConfig(max_tokens=...)` caps each answer. Reasoning models count their hidden thinking
  tokens against this cap, so raise it if answers come back empty or truncated.
- The default `Strategy.RETRIEVAL` sends about 2,000 tokens per question; whole-document
  strategies send the entire document each time (OpenAI discounts repeated prompt prefixes
  automatically, and the saving shows in `answer.usage.cache_read_input_tokens`).
- The benchmark needs two explicit steps before it spends anything, and enforces a cap:

  ```bash
  LONGDOC_ALLOW_LIVE_API=1 python benchmarks/contract_review/run.py --openai "$MODEL" \
      --env-file .env --price-in <usd per 1M> --price-out <usd per 1M> \
      --strategies retrieval --max-cost 1
  ```

## 6. Compatible servers

Any server that speaks the OpenAI chat-completions protocol works:

```python
OpenAIClient(base_url="http://localhost:11434/v1", token_param="max_tokens")  # no key needed
```

`token_param="max_tokens"` is what most compatible servers expect; OpenAI's own current models
use the default, `max_completion_tokens`. Azure OpenAI uses a different client class and is not
covered.

## 7. What is and is not verified

- **Verified:** the adapter's request building, response and usage mapping (including cached
  tokens), refusals, truncation, async use, error mapping, key handling and the benchmark's
  safeguards, with fake clients in the test suite (no network, no real key).
- **Not verified:** any call to the real OpenAI service. No OpenAI key was used in development.
  Run a small real check yourself before relying on it, and report problems.
- Claude-only features (effort levels, refusal fallbacks, explicit cache markers) are ignored.

## If a key leaks

1. Revoke it in your OpenAI account immediately, then create a new one.
2. Check the usage dashboard for activity you do not recognise.
3. If it was committed, deleting the file is not enough: it stays in git history. Rotate first;
   then ask for help rewriting history if you want it removed.
4. Turn on GitHub **secret scanning and push protection** for the repository (Settings, Code
   security), which blocks pushes that contain recognised keys.
