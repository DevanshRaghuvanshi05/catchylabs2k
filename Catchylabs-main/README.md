# CatchyLabs — Market Research API

Automated multi-stage market research pipeline powered by **Gemini API**, **SerpAPI**, **RQ + Upstash Redis**, and **Supabase PostgreSQL**.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values. **Never commit `.env` to version control.**

| Variable | Required | Description |
|---|---|---|
| `CATCHYLAB_API_KEY` | ✅ | Secret key for `X-API-Key` header auth (use `openssl rand -hex 32`) |
| `DATABASE_URL` | ✅ | PostgreSQL connection string (Supabase `postgresql://…`) |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `SERPAPI_API_KEY` | ✅ | SerpAPI key for competitor web search |
| `REDIS_URL` | ✅ | Upstash Redis connection string (`rediss://…`) |
| `UPSTASH_REDIS_REST_URL` | ✅ | Upstash REST URL for admin queries |
| `UPSTASH_REDIS_REST_TOKEN` | ✅ | Upstash REST token |
| `SUPABASE_URL` | ☑️ | Supabase project URL used to validate user access tokens for provider routes |
| `SUPABASE_ANON_KEY` | ☑️ | Supabase public anon key used only server-to-server for `/auth/v1/user` validation |
| `PROVIDER_ENCRYPTION_KEY` | ☑️ | URL-safe base64 encoding of 32 random bytes for AES-GCM provider credential encryption |
| `GOOGLE_OAUTH_CLIENT_ID` | ☑️ | Google Cloud OAuth client ID for Gemini Developer API access |
| `GOOGLE_OAUTH_CLIENT_SECRET` | ☑️ | Server-only Google OAuth client secret |
| `GOOGLE_OAUTH_REDIRECT_URI` | ☑️ | Exact HTTPS callback registered with the Google OAuth client |
| `FRONTEND_URL` | ☑️ | Frontend origin used after the Gemini OAuth callback |
| `VOICE_LOCAL_ONLY` | ☑️ | Keep voice-originated AI replies on the local model (default `1`; no provider fallback) |
| `LOCAL_VOICE_LLM_MODEL` | ☑️ | Model name already installed in the local Ollama runtime (default `qwen3:4b`) |
| `LOCAL_STT_BINARY` | ☑️ | Local `whisper-cli` executable path (default `whisper-cli`) |
| `LOCAL_STT_FFMPEG` | ☑️ | Local `ffmpeg` executable path (default `ffmpeg`) |
| `LOCAL_STT_MODEL_PATH` | ☑️ | Absolute path to a manually downloaded multilingual whisper.cpp model |
| `ALLOWED_ORIGINS` | ☑️ | Explicit comma-separated CORS origins (empty = cross-origin access disabled) |
| `MAX_RUNS_PER_DAY` | ☑️ | Daily POST /research cap (default `5`) |
| `MAX_STAGE_RUNS_PER_DAY` | ☑️ | Daily stage-job cap (default `20`) |
| `MAX_SERPAPI_PER_DAY` | ☑️ | Daily SerpAPI call cap (default `15`) |
| `PORT` | ☑️ | uvicorn listen port (default `8000`) |
| `TEST_DATABASE_URL` | 🧪 | Separate test DB — **never the same as DATABASE_URL** |

> ⚠️ **Never put real credentials in `config.py` fallback defaults.** All secrets must come from environment variables or `.env`.

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your real values

# 3. Start the API server + worker
bash start.sh
```

The API server starts on `http://localhost:${PORT:-8000}`.

### Self-hosted voice setup

The voice routes use only local processes: `whisper.cpp`/`whisper-cli` for speech-to-text,
Ollama at the fixed loopback address `127.0.0.1:11434` for voice chat, and browser/OS
speech-synthesis voices with `localService=true` for read-aloud. They do not call Gemini,
OpenAI, Anthropic, or any speech API. `VOICE_LOCAL_ONLY=1` is the default; dictated messages
then use only the configured Ollama model, while typed chat continues to use its explicitly
selected provider. The Research-mode transcript still enters the existing research pipeline
only when the user confirms the research brief; that pipeline may use its configured Gemini
and SerpAPI integrations, but receives text, never the audio recording. There is no cloud
fallback for AI Chat voice turns. Setting `VOICE_LOCAL_ONLY=0` is an explicit
opt-in to sending dictated **text** through the selected chat provider; the audio is still
transcribed locally and is never sent to that provider.

1. Install the local prerequisites yourself. On macOS, install CMake and ffmpeg, then build
   the MIT-licensed whisper.cpp project with its Apple Silicon Metal support:

   ```sh
   brew install cmake ffmpeg
   git clone https://github.com/ggml-org/whisper.cpp.git
   cd whisper.cpp
   cmake -B build
   cmake --build build --config Release -j
   ```

2. **Only after reviewing the model download and storage requirement**, explicitly download
   a multilingual model. The multilingual `base` model is about 142 MiB on disk and about
   388 MB runtime memory according to whisper.cpp; `small` is about 466 MiB / 852 MB and
   `medium` about 1.5 GiB / 2.1 GB. Do not use `.en` models for Hindi.

   ```sh
   ./models/download-ggml-model.sh base
   ```

3. Configure the backend with absolute paths and start Ollama. Install the LLM model only
   after reviewing its separate storage requirements:

   ```sh
   export LOCAL_STT_BINARY="$PWD/build/bin/whisper-cli"
   export LOCAL_STT_MODEL_PATH="$PWD/models/ggml-base.bin"
   export LOCAL_STT_FFMPEG="$(command -v ffmpeg)"
   export VOICE_LOCAL_ONLY=1
   ollama pull qwen3:4b
   ollama serve
   ```

   The application does not execute either model-download command. On the inspected
   development machine (Apple M5, 16 GB unified memory), `ollama` and `ffmpeg` are present,
   but no Ollama models or whisper.cpp executable/model are installed. The multilingual
   `base` speech model and 4B Qwen model are reasonable first candidates; inference speed,
   thermals, and Hindi/Hinglish quality have not been benchmarked here. `small`/`medium`
   Whisper models and larger LLMs use more memory and may be slower. Costs still include
   hardware, electricity, disk, hosting, and maintenance.

4. Set `LOCAL_VOICE_LLM_MODEL` to the exact installed Ollama tag. The UI checks the local
   Ollama model list and local transcription configuration without loading or downloading a
   model. Install a local Hindi system voice in macOS/browser settings for Hindi read-aloud;
   remote browser voices are rejected. No raw recording is retained: audio is temporarily
   written to a short-lived backend temp directory for ffmpeg/whisper.cpp and removed after
   transcription. Audio is sent from the browser to the configured CatchyLab backend, so use
   only a backend you control and secure its transport with HTTPS outside local development.

The browser voice language options are Auto, Hindi, Hinglish, and English. Auto sends an
automatic Whisper language hint; Hindi and English pin the recognition language. Hinglish
uses automatic recognition because code-switching is not reliably supported by Whisper.
Transcripts are editable and are never sent until the user presses the existing Send button.
No live partial transcript or hands-free loop is provided. `GET /voice/status`,
`POST /voice/transcribe`, and `POST /voice/chat` use the existing shared `X-API-Key` dependency.
Do not expose that legacy shared-key API publicly without a user-scoped production gateway.

#### Evaluated local components and licensing

- **Speech recognition:** whisper.cpp software is MIT-licensed. The converted model repository
  declares MIT metadata, and the original Whisper Base model card declares Apache-2.0; retain
  the applicable notices and review both the exact artifact and upstream terms before
  commercial distribution. It supports Apple Silicon/Metal and CPU-only systems. Model disk/RAM estimates are
  above; the small/base multilingual models support Hindi but Hinglish accuracy is not
  guaranteed. The service requires a locally installed executable, ffmpeg, and model file.
- **Local language model:** Ollama is a local runtime; the selected model has its own license.
  The suggested Qwen3-4B model card is Apache-2.0 and lists multilingual capabilities. Its
  generated language/cultural quality still requires evaluation; a model is not marketing
  research evidence. Ollama must already have the exact tag installed. The application never
  pulls it automatically.
- **Speech synthesis:** browser/OS system voices are used only when their `localService`
  property is true. This avoids remote speech services but their availability, language
  coverage, and redistribution/commercial terms are controlled by the operating system and
  installed voice vendor, not CatchyLab. No voice is synthesized if a suitable local voice is
  unavailable.
- **Not selected:** browser `SpeechRecognition` implementations may use a remote service,
  so the app does not invoke them. Faster-Whisper/PyTorch was not chosen for this Apple Silicon
  backend because it adds a larger Python runtime/dependency stack; whisper.cpp is the smaller
  native option already suited to the inspected hardware.

Official references: [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[Whisper license](https://github.com/openai/whisper/blob/main/LICENSE),
[Qwen3-4B model card and license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Ollama model library](https://ollama.com/library/qwen3),
and [local speech voice indicator](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisVoice/localService).

### Health Check

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

---

## Render Deployment

| Setting | Value |
|---|---|
| **Build command** | `pip install -r requirements.txt` |
| **Start command** | `bash start.sh` |
| **Health check path** | `/health` |

Set all environment variables from the table above in the Render dashboard (Environment → Secret Files or individual env vars). **Do not hard-code any values.**

---

## API Endpoints

Legacy research endpoints except `GET /health` require the shared `X-API-Key`. The `/providers/*`
routes instead require a Supabase `Authorization: Bearer <user access token>` and never accept the
shared key as a user identity.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — no auth |
| `GET` | `/providers/connections` | List the signed-in user's provider connections (no credential material) |
| `GET` | `/voice/status` | Check local-only speech, Ollama model, and browser-TTS availability |
| `POST` | `/voice/transcribe` | Transcribe raw audio locally with whisper.cpp (20 MB maximum) |
| `POST` | `/voice/chat` | Generate a dictated-message reply only through the configured loopback Ollama model |
| `POST` | `/providers/{gemini\|claude\|openai}/api-key` | Verify and encrypt that user's API key and discover available models |
| `GET` | `/providers/{provider}/models` | Refresh the connected account's available models |
| `PUT` | `/providers/{provider}/model` | Make a real small request to verify and select exactly one model |
| `POST` | `/providers/{provider}/generate` | Execute a prompt using only the explicitly selected model |
| `DELETE` | `/providers/{provider}/connection` | Remove this user's credential; attempt Gemini OAuth revocation |
| `POST` | `/providers/gemini/oauth/start` | Start Google OAuth authorization-code flow with state and PKCE |
| `GET` | `/providers/gemini/oauth/callback` | Validate one-time state, exchange code server-side, discover models |
| `GET` | `/runs` | List all research runs with counts |
| `POST` | `/research` | Enqueue new research job |
| `GET` | `/research/{run_id}` | Fetch run + evidence |
| `GET` | `/research/{run_id}/full` | Full run view (all pipeline outputs) |
| `GET` | `/research/{run_id}/cost` | Cost breakdown |
| `GET` | `/research/{run_id}/logs` | API call logs |
| `POST` | `/research/{run_id}/consumer-intelligence` | Enqueue Agent 2 |
| `GET` | `/research/{run_id}/consumer-intelligence` | Fetch insights |
| `POST` | `/research/{run_id}/competitor-intelligence` | Enqueue Agent 3 |
| `GET` | `/research/{run_id}/competitor-intelligence` | Fetch competitors |
| `POST` | `/research/{run_id}/creative-gaps` | Enqueue Agent 4 |
| `GET` | `/research/{run_id}/creative-gaps` | Fetch creative gaps |
| `POST` | `/research/{run_id}/opportunities` | Enqueue opportunity scoring |
| `GET` | `/research/{run_id}/opportunities` | Fetch ranked opportunities |
| `POST` | `/research/{run_id}/content` | Enqueue content generation |
| `GET` | `/research/{run_id}/content` | Fetch content pieces |
| `PATCH` | `/content/{piece_id}/status` | Set approved / rejected |
| `PATCH` | `/content/{piece_id}/claims` | Resolve a verified claim |
| `GET` | `/quota-status` | Free-tier quota usage |

---

## Running Tests

Tests use `TEST_DATABASE_URL` exclusively — never `DATABASE_URL`.

```bash
# Set in .env.test or environment:
# TEST_DATABASE_URL=postgresql://...

pytest -v tests/
```

All tests mock Gemini and SerpAPI calls. No live external API calls are made during the test suite.

---

## Schema Migrations

See [`MIGRATION_RUNBOOK.md`](MIGRATION_RUNBOOK.md) for the full step-by-step guide.

```bash
# Dry run (no DB connection)
python migrate_live.py

# Inspect live schema (read-only SELECT)
python migrate_live.py --inspect

# Apply (requires fresh backup + typed project ID)
python migrate_live.py --apply
```

---

## Security Notes

- `X-API-Key` is compared with `secrets.compare_digest` (constant-time, no timing leaks).
- CORS is disabled by default; configure exact `ALLOWED_ORIGINS` values for browser clients.
- Row Level Security (RLS) is enabled on all tables with no public Supabase policies. The API
  connects directly using `DATABASE_URL`, whose database role bypasses RLS.
- `.env`, `.env.test`, `backups/`, `outputs/`, `research_report_*.json`, and `*.dump` files are all `.gitignore`d.
- Content pieces can never be marked `approved` while `claims_to_verify` is non-empty — unverified claims block approval.

### User-owned AI provider connections

The provider endpoints validate Supabase access tokens by calling the configured Supabase Auth
`/auth/v1/user` endpoint. Provider connection rows are keyed by the verified Supabase user UUID and
provider; all reads, updates, and deletes filter by that server-verified owner. API keys and OAuth
tokens are encrypted with AES-GCM before database persistence and are never returned by API
responses. The encryption key is server-only and must be backed up securely; changing it without a
migration makes existing provider credentials unreadable.

API-key connection verifies provider authentication by listing models. A connection remains
`authorized` until the user explicitly chooses a listed model; choosing it performs a small real
provider generation request and marks it `ready` only on success. Generation uses only that
selected model and does not fall back. The user's active provider/model choice is persisted with
their connection and restored after sign-in; selecting a different model deactivates the previous
choice for that user. Claude.ai and ChatGPT sign-ins/subscriptions are not API credentials. Claude
and OpenAI use their documented API-key connection; Gemini additionally offers Google OAuth for the
Gemini Developer API when all OAuth configuration below is present.

For Gemini OAuth, configure an OAuth client in Google Cloud, enable the Generative Language API,
set the exact registered redirect to `https://<api-host>/providers/gemini/oauth/callback`, complete
the consent-screen/test-user and production verification requirements, and configure
`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, and
`FRONTEND_URL`. The integration requests the documented
`https://www.googleapis.com/auth/generative-language.retriever` scope, uses state and PKCE, and
keeps authorization codes out of application access logs. Each user supplies their Google Cloud
quota/billing project ID; requests include `x-goog-user-project` rather than silently charging a
shared CatchyLab project. The authorizing account needs Service Usage Consumer access to that
project. Test the exact project, API enablement,
consent, and production approval in Google Cloud before launch. The OAuth flow was not live-tested
in this workspace.

`init_db()` creates the provider connection, OAuth-state, and generation-usage tables, upgrades
existing OAuth-state tables with the quota-project column, and enables RLS with no public Supabase
policies. Provider access still goes through the backend owner checks; the direct `DATABASE_URL`
role bypasses RLS. The separate legacy research API remains shared-key authenticated, and its
research records are **not** user-scoped. Do not treat this partial provider authentication as
complete multi-tenant authorization for the whole application.

### Deployment boundary

The legacy research API currently authenticates callers with one shared `CATCHYLAB_API_KEY`. It
does not authenticate individual users or scope research runs and generated content to user
identities. Anyone holding the shared key can access the same research records. Treat those routes
as a single-operator/internal prototype; do not expose them as a public multi-tenant service until
end-user authentication and per-record authorization are implemented and tested. The new
`/providers/*` routes use Supabase bearer-token validation and per-user ownership checks. The Vite
proxy described in the frontend README is development-only, not a production security boundary.

Research retrieval currently uses Google web results through SerpAPI and Gemini for extraction
and analysis. Direct Reddit, YouTube, X, Instagram, Facebook, and Canva integrations are not
implemented by this backend.

## Marketing intelligence foundation

The research workflow adds a provider-independent, versioned CatchyLab playbook corpus under
`intelligence/knowledge_base/v1/`. It currently covers marketing fundamentals, consumer behavior,
research methods and quality, brand positioning, creative strategy, analytics/experimentation, and
competitor analysis. Retrieval is a small lexical-overlap implementation; documents are original
operational summaries, not empirical evidence or a substitute for source review. The corpus is
versioned so it can later move to metadata-filtered embeddings without coupling it to a model.
Version `1.1.0` includes English source metadata and Devanagari/Romanized-Hindi retrieval keywords;
the content itself remains English and is not represented as a translation or reviewed
multilingual corpus.
Version `1.1.0` includes English source metadata and Devanagari/Romanized Hindi retrieval keywords;
the content itself remains English and is explicitly not represented as a translation or reviewed
multilingual corpus.
Relevant guidance is selected for Gemini's consumer-intelligence, competitive-intelligence,
creative-gap, opportunity-scoring, and content-generation prompts. Prompt context labels these
documents as internal frameworks, not external evidence, and output grounding continues to rely
on the existing evidence and entity validators.

Each research run now receives a structured research plan, evidence provenance registry, and
explicit quality-review boundaries. The context is persisted in the `research_contexts` table,
which is created by the existing `init_db` startup process and included in the no-public-policy
RLS table set. `GET /research/{run_id}/full` returns it. Reviews intentionally label public search
excerpts as secondary evidence, avoid implying representative sampling or causality, and mark
source quality/independence for human review.
The full-run response also returns available source publication dates and the source record's
first-indexed timestamp. Because sources are deduplicated across runs, that timestamp is not
necessarily the time the source was retrieved for the current run. Claims remain explicitly
unverified until their excerpt-to-claim support and original source are reviewed; model confidence
values are not calibrated.
Semantic recall is constrained to exact case-insensitive brand and market matches to avoid
reusing evidence from unrelated research runs. This does not provide user/tenant isolation for
research records; only the separate provider-connection routes below are user-scoped.

Research and creative output supports `auto`, `english`, `hindi`, and `hinglish` language
preferences. Run preferences are persisted with the research brief; AI chat uses the selected
preference for the generation request. Automatic language/script detection is a lightweight
script-and-lexicon heuristic, not a validated classifier, and must not be used to infer identity,
region, or audience attributes. Search prompts request relevant Hindi or Romanized-Hindi queries
for Hindi/Hinglish briefs, while source snippets and titles remain in their original form. The
stored language coverage is only a heuristic label for actual query strings, not a retrieval
quality score or evidence of source availability. Extracted claims are model summaries, never
declared verified translations.

`intelligence.dataset` defines provenance, rights, reviewer, confidence, split, and exact-duplicate
leakage guards for future supervised examples. The current unit tests use synthetic, unreviewed
fixtures only. No reviewed multilingual training dataset, licensed campaign corpus, trained model,
or fine-tuning run exists. Public brand and ad libraries are not ingested; source-specific
permissions, retention, licensing, and access configuration must be established before collecting
campaign examples.

Research and creative output supports `auto`, `english`, `hindi`, and `hinglish` language
preferences. Run preferences are persisted with the research brief; AI chat uses the selected
preference for the generation request. Automatic language/script detection is a lightweight
script-and-lexicon heuristic, not a validated classifier, and must not be used to infer identity,
region, or audience attributes. Search prompts request relevant Hindi or Romanized-Hindi queries
for Hindi/Hinglish briefs, while source snippets and titles remain in their original form. The
stored language coverage is only a heuristic label for actual query strings, not a retrieval
quality score or evidence of source availability. Extracted claims are model summaries, never
declared verified translations.

`intelligence.dataset` defines provenance, rights, reviewer, confidence, split, and leakage guards
for future supervised examples. The current unit tests use synthetic, unreviewed fixtures only.
No reviewed multilingual training dataset, licensed campaign corpus, trained model, or fine-tuning
run exists. Public brand and ad libraries are not ingested; source-specific permissions, retention,
licensing, and access configuration must be established before collecting campaign examples.

The research pipeline remains configured with its existing shared Gemini credential and does not
add primary interviews/surveys or Claude/OpenAI research adapters. The separate provider chat
endpoint below supports user-owned credentials and selected models. The legacy shared API key and
current database role do not provide tenant isolation for research, so those routes remain an
internal/single-operator prototype.

## Provider authentication support and setup boundary

Supabase email/password sign-in and server-side Supabase token validation are implemented for the
provider routes. Per-user credentials are encrypted with AES-GCM; available models are fetched
from the user's provider account. A model becomes ready only after a short real generation request
to that exact model succeeds. Chat calls use only the selected provider/model, and the UI does not
silently fall back. The separate `GEMINI_API_KEY` remains a server-side shared credential for the
research pipeline and is not used as a user's provider connection.

| Provider | Documented API authentication | Account/OAuth boundary |
|---|---|---|
| Google Gemini Developer API | API key or Google OAuth | OAuth is documented for the Gemini API. Setup requires a Google Cloud project, Generative Language API enabled, OAuth consent configuration and test users as applicable, OAuth client credentials, an exact server redirect URI, and server-side authorization-code/token handling. Google's Python OAuth example uses `generative-language.retriever`; the gcloud example also requests `cloud-platform`. Prefer the narrower supported scope where possible and confirm current scope and verification requirements before production approval. |
| Anthropic Claude | Claude Console API key; workload identity federation | Workload identity federation exchanges a workload's OIDC identity for a short-lived API token; it is not end-user OAuth for a browser SaaS account. App Attest is for registered native iOS/macOS apps, not this web app. Claude.ai sign-in/subscription is not Claude API access. |
| OpenAI | OpenAI Platform API key | ChatGPT account sign-in/subscription is not an API credential. No ChatGPT session, cookie, or subscription token is supported or used by CatchyLab. |

Official setup references:

- [Gemini API OAuth guide](https://ai.google.dev/gemini-api/docs/oauth) and [Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key)
- [Claude API authentication](https://platform.claude.com/docs/en/manage-claude/authentication) and [Claude Workload Identity Federation](https://platform.claude.com/docs/en/manage-claude/workload-identity-federation)
- [OpenAI API authentication](https://platform.openai.com/docs/api-reference/authentication) and [API key safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

Configure `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `PROVIDER_ENCRYPTION_KEY` on the backend. The
frontend needs `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` (the public anon key only). To enable
Gemini OAuth, set `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`,
`GOOGLE_OAUTH_REDIRECT_URI`, and `FRONTEND_URL` server-side. Register the exact HTTPS callback
`https://<api-host>/providers/gemini/oauth/callback`, enable the Generative Language API, and
complete Google consent/test-user and production verification requirements. Each user must supply
their own quota/billing project ID and authorize the account to use that project. The integration
sends it in `x-goog-user-project`, uses state, PKCE, the documented
`generative-language.retriever` scope, server-side token exchange, and refresh-token persistence.
OAuth setup and live provider credentials/API/model access have not been verified in this workspace.
The direct database role bypasses RLS, so provider isolation currently
relies on server-validated Supabase user identity and owner filters; no public RLS policies are
added. The legacy research routes remain outside that ownership boundary.