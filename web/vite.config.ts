import path from 'node:path'

import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  // Resolve the API key server-side only. `loadEnv` reads the repo-root `.env`
  // (where the backend keeps its config), and the value is used solely as a
  // dev-proxy header — it is never assigned to a `VITE_*` variable, so it can
  // never be compiled into the client bundle.
  const env = loadEnv(mode, path.resolve(process.cwd(), '..'), '')
  const rawKeys = env.API_KEYS ?? process.env.API_KEYS ?? ''
  const apiKey = env.API_KEY ?? process.env.API_KEY ?? rawKeys.split(',')[0]?.trim() ?? ''

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          headers: apiKey ? { 'X-API-Key': apiKey } : {},
        },
      },
    },
  }
})
