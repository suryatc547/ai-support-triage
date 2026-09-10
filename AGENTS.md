---
trigger: always_on
description: Privacy rules and project context for the support ticket system.
---

# Project Context and Instructions

- **Goal**: We are creating a support ticket system.
- **Email Ingestion**: We listen to an email inbox using IMAP.
- **Ticket Assignment**: Any received email related to a support ask/question needs to be assigned to a particular person for that support area.
- **Database**: We use SQLite. The SQLAlchemy engine enforces foreign keys, WAL journal mode, and a busy timeout via a `connect` event listener in `models/database.py`.
- **API Auth**: When `API_KEY`/`API_KEYS` is configured, `APIKeyMiddleware` requires `X-API-Key` on every `/api/*` request (constant-time compare). Keep `TraceIdMiddleware` outermost and the docs/`/health` exemptions intact.
- **CI**: `.github/workflows/ci.yml` gates commits — backend ruff (lint + format) and pytest, frontend eslint and `tsc -b && vite build`.
- **AI & RAG Architecture**: We need to architect a RAG (Retrieval-Augmented Generation) system. We will provide the mail inbox content and RAG context to the AI to analyze the type of support. Based on that analysis, the ticket will be assigned to the appropriate person.
- **Multi-Provider LLM Chain**: Routing uses an ordered provider chain (Gemini primary -> OpenRouter -> Groq -> local `modelm` ModelM), each with 429 retry via `invoke_with_retry`. The local model (`modelm/langchain_model.py`) is on by default (`USE_LOCAL_MODEL`, flip to `USE_LOCAL_MODEL_FIRST=true` to prefer it); it must stay last in the chain unless explicitly configured otherwise. BM25 retrieval/fallback is local (`rank_bm25`); `langchain-community` must not be added back — it is sunset. Assignee matching is order-independent (exact dept/name match first, then word-boundary tokens) — never reintroduce bare-substring aliases like `"it"`.
- **Email Forwarding**: When `EMAIL_FORWARDING=true`, `email_service.py` forwards each assigned ticket to the staff member's email over SMTP (`smtp_config`, IMAP creds as fallback). Forwarding is non-fatal (failures log a warning, counted in the sync summary) and gated — never send mail unless the env flag is set.
- **LLM Output Sanitization**: LLM JSON responses are validated before persistence (`_clean_llm_result`): category is normalized to a lowercase string, and `assignee_id` is kept only when it is an int referring to a real user. When every provider fails at runtime, tickets are left **unassigned** (`unclassified`) for manual review.
- **Security Guardrails**: Every inbound email runs through deterministic, offline validation (`validation_service.py`) BEFORE any analyzer or LLM call. Verdicts: `pass` / `flag` / `quarantine`. Quarantined mail is persisted with `status="quarantined"`, flagged, and never auto-assigned. The optional LLM assist (`security_analyzer.py`) can only escalate `flag` -> `quarantine`, never weaken a block.
- **Frontend**: We will use React under the `/web` folder.
- **Quality Standards**: We must follow enterprise-level code quality standards and write comprehensive unit tests.
- **Logging & Trace IDs**: All application logs flow through `logging_config.setup_logging()` (single `[<trace-id>]`-prefixed format). Every HTTP request gets a trace ID via `TraceIdMiddleware` (`X-Request-ID` header when provided/sanitized, else generated `req-*`), echoed in the response header; IMAP sync runs use a `sync-*` ID because background tasks do not inherit the HTTP context. Keep loggers module-scoped (`logging.getLogger(__name__)`); never `print()`.

# Privacy Rules

- Do NOT read, output, or modify `.env` files.
- Do NOT read or output environment variables containing secrets or credentials.

# File Naming and Structure Standards

- **Backend**: Use pythonic `snake_case` for folder and file names (e.g., `services/imap_service.py`). Group files logically (e.g., `models`, `services`, `controllers`).
- **Frontend**: Components must be placed in `src/components/<ComponentName>/<ComponentName>.<extension>` (e.g., `src/components/TicketCard/TicketCard.tsx`). Use `PascalCase` for component names and their containing folders. Use `pnpm` as the package manager for the `/web` workspace.
