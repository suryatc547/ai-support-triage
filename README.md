# Support Ticket System with AI & Hybrid RAG

An enterprise-ready, AI-driven support ticket management system that automatically ingests emails via IMAP, screens them with deterministic security guardrails, categorizes inquiries, and assigns them to the most relevant support engineers using a hybrid RAG (Retrieval-Augmented Generation) pipeline with BM25 search and a multi-provider LLM chain.

---

## 🛠️ Architecture & Tech Stack

- **Backend API**: FastAPI (Python 3.10+, tested on 3.14)
- **Database**: SQLite with SQLAlchemy ORM (WAL journal, enforced foreign keys, busy timeout)
- **Email Ingestion**: IMAP via the standard library (`imaplib`)
- **Security Guardrails**: deterministic validation (domain blocklists, typosquat detection, attachment screening) that runs *before* any LLM call, with an optional LLM assist for ambiguous cases
- **AI & RAG Engine**: LangChain + multi-provider LLM chain (Gemini primary → OpenRouter → Groq → local [ModelM](./modelm/README.md)) + BM25 keyword retrieval (`rank_bm25`)
- **Frontend UI**: React 19 + TypeScript + Vite (`/web`)
- **Code Quality**:
  - Backend: [Ruff](https://docs.astral.sh/ruff/) (lint + format)
  - Frontend: [ESLint](https://eslint.org/) with `@typescript-eslint`, TypeScript strict build

---

## 📁 Project Structure

```text
support/
├── api/
│   ├── pyproject.toml              # Ruff configuration
│   ├── requirements.txt            # Backend dependencies
│   ├── support.db                  # SQLite database (git-ignored)
│   ├── src/
│   │   ├── configs/
│   │   │   ├── config.py           # App + IMAP configuration (env-driven)
│   │   │   └── security_config.py  # Guardrail blocklists & thresholds
│   │   ├── controllers/
│   │   │   └── ticket_controller.py  # API route handlers
│   │   ├── knowledge_base/         # Per-team routing policy docs (BM25 index)
│   │   ├── models/
│   │   │   ├── database.py         # SQLAlchemy engine + idempotent migrations
│   │   │   └── models.py           # User / Ticket ORM models
│   │   ├── scripts/
│   │   │   └── seed_data.py        # Seed the support staff table
│   │   ├── services/
│   │   │   ├── email_service.py    # SMTP forwarding of assigned tickets
│   │   │   ├── imap_service.py     # IMAP polling & email ingestion
│   │   │   ├── mail_analyzer.py    # Pre-LLM support/marketing/auto-reply gate
│   │   │   ├── rag.py              # Hybrid BM25 + LLM ticket routing
│   │   │   ├── security_analyzer.py # Optional LLM-assisted phishing check
│   │   │   └── validation_service.py # Deterministic guardrail engine
│   │   ├── logging_config.py       # Trace-ID-aware logging setup
│   │   ├── main.py                 # FastAPI application entrypoint
│   │   ├── middleware.py           # TraceIdMiddleware + APIKeyMiddleware
│   │   └── schemas.py              # Pydantic response schemas (API contract)
│   └── tests/                      # Pytest suite (73 tests)
├── modelm/                         # Custom local ticket classifier (ModelM)
│   ├── README.md
│   ├── model.py                    # NumPy TF-IDF + softmax classifier
│   ├── train.py                    # Training pipeline (dataset + KB + staff)
│   ├── langchain_model.py          # LangChain BaseChatModel adapter
│   ├── dataset.py                  # Domain training tickets (5 departments)
│   ├── weights.json / metrics.json # Trained artifacts (intentionally tracked)
│   └── tests/                      # Model tests (12 tests)
├── web/
│   ├── src/
│   │   ├── components/             # Reusable UI components (one folder each)
│   │   │   ├── Loader/
│   │   │   ├── SecurityBadge/
│   │   │   ├── SecurityFilter/
│   │   │   ├── ThemeToggle/
│   │   │   ├── TicketCard/
│   │   │   ├── TicketDetail/
│   │   │   ├── TicketList/
│   │   │   └── ViewToggle/
│   │   ├── App.tsx                 # Main dashboard UI
│   │   ├── main.tsx                # React entrypoint
│   │   └── types.ts                # Shared TypeScript types
│   ├── eslint.config.js            # Flat ESLint configuration
│   ├── package.json
│   └── vite.config.ts              # Vite + dev proxy to the API
├── AGENTS.md                       # Agent instructions & standards
├── .github/workflows/ci.yml        # CI: lint + tests + build (GitHub Actions)
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Node.js 18+ and `pnpm`

---

### 2. Backend Setup (`api/`)

1. **Navigate to the API folder:**
   ```powershell
   cd api
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Seed the support staff table** (run once):
   ```powershell
   python -m src.scripts.seed_data
   ```

5. **Environment Configuration:**
   Configure the variables below (see `.env.example`). Secrets belong in `.env`, never in code:
   - `IMAP_EMAIL` / `IMAP_PASSWORD`: mail account used for ingestion.
   - `IMAP_HOST` (defaults to `imap.gmail.com`) and `IMAP_FOLDER` (defaults to `inbox`).
   - `GOOGLE_API_KEY` / `GEMINI_MODEL`: primary Gemini provider.
   - `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`: first fallback provider.
   - `GROQ_API_KEY` / `GROQ_MODEL`: second fallback provider.
   - `API_KEY` (or comma-separated `API_KEYS`): when set, the API requires a matching `X-API-Key` header on every `/api/*` request (see [API Authentication](#api-authentication)).
   - `EMAIL_FORWARDING`: set to `true` to forward every assigned ticket to the staff member's email over SMTP (off by default; `SMTP_HOST`/`SMTP_PORT`/`SMTP_USER`/`SMTP_PASSWORD` optional — defaults to `smtp.gmail.com:587` and reuses IMAP credentials).
   - Optional `SECURITY_*` overrides for guardrail blocklists/thresholds (see `security_config.py`).

6. **Run the backend server:**
   ```powershell
   uvicorn src.main:app --reload --port 8000
   ```
   - API Server: `http://localhost:8000`
   - Interactive Docs (Swagger): `http://localhost:8000/docs`

---

### 3. Frontend Setup (`web/`)

1. **Navigate to the Web folder:**
   ```powershell
   cd web
   ```

2. **Install pnpm dependencies:**
   ```powershell
   pnpm install
   ```

3. **Start the development server:**
   ```powershell
   pnpm dev
   ```
   - Web App: `http://localhost:5173` (Vite proxies `/api` to `http://localhost:8000`)

---

## 🔍 Linting & Code Quality

### Backend (Ruff)
```powershell
cd api
python -m ruff check .                 # lint
python -m ruff check --fix .           # auto-fix lint
python -m ruff format .                # format
```

### Backend (Tests)
```powershell
cd api
python -m pytest -q
```
The suite covers the BM25 fallback router and context retrieval, the LLM-output sanitizer, the deterministic guardrails, the security config loader, the provider injection, and the trace-ID / API-auth middleware — all fully offline (no API keys / no DB).

### Frontend (ESLint + TypeScript)
```powershell
cd web
pnpm lint
pnpm build        # tsc -b && vite build
```

### CI (GitHub Actions)
`.github/workflows/ci.yml` runs the full backend lint/format/test suite and the
frontend lint/typecheck/build pipeline on every push and pull request.

---

## 🛡️ Inbound Email Security Pipeline

Every message is screened by deterministic, offline guardrails **before** any analyzer or LLM runs:

1. **Sender validation**: disposable/throwaway domains and known spam domains are hard-blocked; lookalike (typosquat) domains close to the organisation's trusted domains are flagged.
2. **Attachment screening**: executable/script extensions (`.exe`, `.ps1`, `.js`, …) are hard-blocked; double-extension, macro-Office, archive, and oversized attachments are flagged.
3. **Verdicts**: `pass` (processed normally), `flag` (processed but surfaced in the UI), or `quarantine` (a `quarantined` ticket is recorded, never auto-assigned).
4. **Hybrid LLM assist** *(optional, best-effort)*: for a `flagged` mail that is not quarantined, the LLM chain may escalate it to `quarantine` if it judges it a phishing/social-engineering attempt. It can never downgrade a hard block.

The dashboard shows security badges (Quarantined / Flagged with suspicion score), a **Security Findings** panel in the ticket detail modal, and a **security filter** (`All / Quarantined / Flagged`) on the toolbar.

---

## 🤖 RAG Routing Pipeline

1. **Email Fetching**: `imap_service.py` connects via IMAP, deduplicates on the RFC `Message-ID`, and reads new/unseen messages.
2. **Guardrails**: `validation_service.py` screens each message (see above) before anything else runs.
3. **Pre-LLM Analyzer**: `mail_analyzer.py` cheaply skips clearly non-support mail (newsletters, auto-replies, marketing) to avoid wasting LLM calls.
4. **Context Retrieval**: `rag.py` builds BM25 contexts from support staff expertise plus per-team routing policy docs in `knowledge_base/`.
5. **LLM Routing**: A provider chain (Gemini → OpenRouter → Groq → local [ModelM](./modelm/README.md), each with 429 retry) produces structured JSON with category + `assignee_id`. The local model is enabled by default and guarantees a verdict even during total cloud outages; set `USE_LOCAL_MODEL_FIRST=true` to evaluate tickets locally first, or `USE_LOCAL_MODEL=false` to disable it. The response is sanitized before touching the DB (`_clean_llm_result`): non-string categories are normalized, and `assignee_id` is kept only when it is an integer that refers to a real support staff member. If every provider fails at runtime, the ticket is left **unassigned** for manual review rather than guessed into a team.
6. **Persistence & Presentation**: The ticket is stored in SQLite and immediately visible on the React dashboard. When `EMAIL_FORWARDING=true`, assigned tickets are also forwarded to the staff member's email via SMTP (`email_service.py`).

---

## 🪵 Logging & Observability

All application logs share a single format with a **trace ID prefix** so a full
request can be followed from start to finish:

```text
[req-a1b2c3d4e5f6] INFO  2026-09-07 12:01:08 app.request START GET /api/tickets?page=1&limit=5
[req-a1b2c3d4e5f6] INFO  2026-09-07 12:01:08 src.services.rag No LLM provider configured. Falling back to BM25 routing.
[req-a1b2c3d4e5f6] INFO  2026-09-07 12:01:08 app.request END GET /api/tickets?page=1&limit=5 -> 200 in 16.4ms
```

- `logging_config.py` owns the central format; a `RequestIdFilter` injects the
  active trace ID into every record logged through the root handlers.
- `TraceIdMiddleware` (`middleware.py`) assigns one trace ID per HTTP request —
  reusing a client-supplied `X-Request-ID` when present (sanitized), otherwise
  generating `req-<12 hex>` — logs a `START`/`END` line pair with status and
  duration, and echoes the ID back via the `X-Request-ID` response header.
- IMAP sync runs get their own `sync-*` ID (background tasks do not inherit the
  HTTP request context), keeping the whole ingestion pipeline traceable.
- Records emitted outside any request/run use a `[-]` placeholder so the prefix
  column is always present.

---

## 🔐 API Authentication

When `API_KEY` (or a comma-separated `API_KEYS`) is set in the environment,
every `/api/*` request must send an `X-API-Key` header matching one of the
configured keys. `APIKeyMiddleware` compares it in constant time
(`secrets.compare_digest`) and otherwise answers `401` without reaching the
application.

- Leaving `API_KEY` empty disables authentication entirely (local/dev convenience).
- `/docs`, `/openapi.json`, `/redoc`, and `/health` remain open for exploration.
- The web client sends the key automatically when `VITE_API_KEY` is set in
  `web/.env` (Vite exposes only `VITE_*` variables to the browser).