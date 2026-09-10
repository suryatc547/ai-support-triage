# Support Hub Web Client

Modern React + TypeScript + Vite frontend dashboard for the Support Ticket System.

---

## 💻 Tech Stack & Features

- **React 19** & **TypeScript**
- **Vite 8** with Fast HMR and API proxying to backend (`/api` -> `http://localhost:8000`)
- **Component Architecture**: Reusable UI components in `src/components/<ComponentName>/`
- **Linting**: ESLint with `@typescript-eslint` and React Hooks rules
- **Theme**: A `ThemeToggle` for Default / Light / Dark modes (persisted to `localStorage`)
- **Ticket views**: card grid and list view toggle
- **Security UI**: `SecurityBadge` (Quarantined / Flagged with suspicion score), a `SecurityFilter` (All / Quarantined / Flagged) on the toolbar, and a Security Findings panel in the ticket detail modal
- **Resilience**: inline/block `Loader` components and clear error states (an alert is shown when the API call fails instead of stale data)
- **Live sync**: "Sync Emails" button triggers an IMAP sync against the backend

---

## 🚀 Running Locally

```powershell
# 1. Install dependencies (pnpm is the package manager)
pnpm install

# 2. Run dev server
pnpm dev

# 3. Linting
pnpm lint

# 4. Production Build (tsc -b && vite build)
pnpm build
```

---

## 🔐 API Key

If the backend has API authentication enabled (`API_KEY` in its env), set
`VITE_API_KEY` in `web/.env` to the same value and the app will attach an
`X-API-Key` header to every request automatically. Without it the app talks to
the backend anonymously (fine when the backend runs without an API key).