---
trigger: always_on
description: Privacy rules and project context for the support ticket system.
---

# Project Context and Instructions

- **Goal**: We are creating a support ticket system.
- **Email Ingestion**: We listen to an email inbox using IMAP.
- **Ticket Assignment**: Any received email related to a support ask/question needs to be assigned to a particular person for that support area.
- **Database**: We use SQLite.
- **AI & RAG Architecture**: We need to architect a RAG (Retrieval-Augmented Generation) system. We will provide the mail inbox content and RAG context to the AI to analyze the type of support. Based on that analysis, the ticket will be assigned to the appropriate person.
- **Multi-Provider LLM Chain**: Routing uses an ordered provider chain (Gemini primary -> OpenRouter -> Groq fallbacks), each with 429 retry via `invoke_with_retry`. BM25 retrieval/fallback is local (`rank_bm25`); `langchain-community` must not be added back — it is sunset.
- **LLM Output Sanitization**: LLM JSON responses are validated before persistence (`_clean_llm_result`): category is normalized to a lowercase string, and `assignee_id` is kept only when it is an int referring to a real user. When every provider fails at runtime, tickets are left **unassigned** (`unclassified`) for manual review.
- **Security Guardrails**: Every inbound email runs through deterministic, offline validation (`validation_service.py`) BEFORE any analyzer or LLM call. Verdicts: `pass` / `flag` / `quarantine`. Quarantined mail is persisted with `status="quarantined"`, flagged, and never auto-assigned. The optional LLM assist (`security_analyzer.py`) can only escalate `flag` -> `quarantine`, never weaken a block.
- **Frontend**: We will use React under the `/web` folder.
- **Quality Standards**: We must follow enterprise-level code quality standards and write comprehensive unit tests.

# Privacy Rules

- Do NOT read, output, or modify `.env` files.
- Do NOT read or output environment variables containing secrets or credentials.

# File Naming and Structure Standards

- **Backend**: Use pythonic `snake_case` for folder and file names (e.g., `services/imap_service.py`). Group files logically (e.g., `models`, `services`, `controllers`).
- **Frontend**: Components must be placed in `src/components/<ComponentName>/<ComponentName>.<extension>` (e.g., `src/components/TicketCard/TicketCard.tsx`). Use `PascalCase` for component names and their containing folders. Use `pnpm` as the package manager for the `/web` workspace.
