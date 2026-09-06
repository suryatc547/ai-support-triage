# Support Ticket System with AI & Hybrid RAG

An enterprise-ready, AI-driven support ticket management system that automatically ingests emails via IMAP, screens them with deterministic security guardrails, categorizes inquiries, and assigns them to the most relevant support engineers using a hybrid RAG (Retrieval-Augmented Generation) pipeline with BM25 search and a multi-provider LLM chain.

---

## 🛠️ Architecture & Tech Stack

- **Backend API**: FastAPI (Python 3.10+, tested on 3.14)
- **Database**: SQLite with SQLAlchemy ORM
- **Email Ingestion**: IMAP via the standard library (`imaplib`)
- **Security Guardrails**: deterministic validation (domain blocklists, typosquat detection, attachment screening) that runs *before* any LLM call, with an optional LLM assist for ambiguous cases
- **AI & RAG Engine**: LangChain + multi-provider LLM chain (Gemini primary → OpenRouter → Groq) + BM25 keyword retrieval (`rank_bm25`)
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
│   │   │   ├── imap_service.py     # IMAP polling & email ingestion
│   │   │   ├── mail_analyzer.py    # Pre-LLM support/marketing/auto-reply gate
│   │   │   ├── rag.py              # Hybrid BM25 + LLM ticket routing
│   │   │   ├── security_analyzer.py # Optional LLM-assisted phishing check
│   │   │   └── validation_service.py # Deterministic guardrail engine
│   │   ├── main.py                 # FastAPI application entrypoint
│   │   └── schemas.py              # Pydantic response schemas (API contract)
│   └── tests/                      # Pytest suite (35 tests)
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
The suite covers the BM25 fallback router and context retrieval, the LLM-output sanitizer, the deterministic guardrails, and the security config loader — all fully offline (no API keys / no DB).

### Frontend (ESLint + TypeScript)
```powershell
cd web
pnpm lint
pnpm build        # tsc -b && vite build
```

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
5. **LLM Routing**: A provider chain (Gemini → OpenRouter → Groq, each with 429 retry) produces structured JSON with category + `assignee_id`. The response is sanitized before touching the DB (`_clean_llm_result`): non-string categories are normalized, and `assignee_id` is kept only when it is an integer that refers to a real support staff member. If every provider fails at runtime, the ticket is left **unassigned** for manual review rather than guessed into a team.
6. **Persistence & Presentation**: The ticket is stored in SQLite and immediately visible on the React dashboard.