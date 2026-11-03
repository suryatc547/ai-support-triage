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

If the backend has API authentication enabled (`API_KEY` in its env), the app
still works with `pnpm dev`: the Vite dev-server proxy reads the repo-root
`.env` (or the shell environment) and injects the `X-API-Key` header
server-side, so the key never enters the browser bundle.

For a production build, serve `dist/` behind a reverse proxy / API gateway that
adds the same `X-API-Key` header for `/api/*` requests. There is deliberately no
`VITE_API_KEY`: any value compiled into the client is public.