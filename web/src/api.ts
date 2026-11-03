// The browser never holds the API key. In development, the Vite dev-server
// proxy injects `X-API-Key` server-side (see vite.config.ts); in production the
// deployment gateway / reverse proxy must do the same. Anything compiled into
// the client bundle is public, so a `VITE_*` key is never used here.
export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const hasJsonBody = typeof init.body === 'string';
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(hasJsonBody ? { 'Content-Type': 'application/json' } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    throw new Error(`Request failed (status ${response.status})`);
  }
  return (await response.json()) as T;
}
