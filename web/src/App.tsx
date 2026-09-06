import { useState, useEffect } from 'react'
import './index.css'
import { TicketCard } from './components/TicketCard/TicketCard'
import { TicketList } from './components/TicketList/TicketList'
import { ViewToggle, type ViewMode } from './components/ViewToggle/ViewToggle'
import { ThemeToggle } from './components/ThemeToggle/ThemeToggle'
import { TicketDetail } from './components/TicketDetail/TicketDetail'
import { Loader } from './components/Loader/Loader'
import { SecurityFilter, type SecurityFilterValue } from './components/SecurityFilter/SecurityFilter'
import type { Ticket, TicketPage } from './types'

const DEFAULT_LIMIT = 10;
const MIN_LOADER_MS = 250;
// How long to wait after triggering a background sync before re-fetching, and
// how long to keep the Sync button in its "syncing" state.
const SYNC_REFRESH_MS = 3000;

function App() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>('cards');
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);

  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [security, setSecurity] = useState<SecurityFilterValue>('all');
  const [page, setPage] = useState(1);
  const limit = DEFAULT_LIMIT;
  const [total, setTotal] = useState(0);
  const [pages, setPages] = useState(1);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      setLoading(true);
      const startedAt = Date.now();
      try {
        const params = new URLSearchParams({
          page: String(page),
          limit: String(limit),
        });
        if (search) {
          params.set('search', search);
        }
        if (security && security !== 'all') {
          params.set('security', security);
        }
        const response = await fetch(`/api/tickets?${params.toString()}`);
        if (!response.ok) {
          throw new Error(`Failed to load tickets (status ${response.status})`);
        }
        const data: TicketPage = await response.json();
        if (isMounted) {
          setTickets(data.items);
          setTotal(data.total);
          setPages(data.pages);
          setError(null);
        }
      } catch (e) {
        console.error('Error fetching tickets', e);
        if (isMounted) {
          setError(e instanceof Error ? e.message : 'Failed to load tickets');
        }
      } finally {
        // Keep the loader visible for at least the minimum duration so it is
        // perceptible even when the (local) API responds almost instantly.
        const elapsed = Date.now() - startedAt;
        const remaining = Math.max(0, MIN_LOADER_MS - elapsed);
        setTimeout(() => {
          if (isMounted) {
            setLoading(false);
          }
        }, remaining);
      }
    };

    load();
    return () => {
      isMounted = false;
    };
  }, [page, limit, search, security, reloadKey]);

  const handleSearch = () => {
    setPage(1);
    setSearch(searchInput);
  };

  const handleSecurityChange = (value: SecurityFilterValue) => {
    setSecurity(value);
    setPage(1);
  };

  const triggerSync = async () => {
    setSyncing(true);
    try {
      const response = await fetch('/api/sync', { method: 'POST' });
      if (!response.ok) {
        console.error('Sync failed with status', response.status);
        return;
      }
      // Re-fetch after the background sync has had time to process.
      setTimeout(() => setReloadKey((k) => k + 1), SYNC_REFRESH_MS);
    } catch (e) {
      console.error('Error syncing', e);
    } finally {
      setTimeout(() => setSyncing(false), SYNC_REFRESH_MS);
    }
  };

  const from = total === 0 ? 0 : (page - 1) * limit + 1;
  const to = Math.min(page * limit, total);

  return (
    <div className="dashboard">
      <header className="header">
        <div className="header-left">
          <h1>Support Hub AI</h1>
        </div>
        <div className="header-actions">
          <ThemeToggle />
          <button className="sync-btn" onClick={triggerSync} disabled={syncing}>
            {syncing ? (
              <Loader size="sm" inline label="Syncing..." />
            ) : (
              'Sync Emails'
            )}
          </button>
        </div>
      </header>

      <div className="toolbar">
        <div className="search-box">
          <input
            type="text"
            className="search-input"
            placeholder="Search tickets..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button className="search-btn" onClick={handleSearch}>Search</button>
        </div>
        <SecurityFilter value={security} onChange={handleSecurityChange} />
        <ViewToggle view={view} onChange={setView} />
      </div>

      {loading ? (
        <Loader label="Loading tickets..." />
      ) : error ? (
        <div className="empty error" role="alert">
          {error}
        </div>
      ) : (
        <>
          {tickets.length === 0 ? (
            <p className="empty">No tickets found. Try syncing emails!</p>
          ) : view === 'cards' ? (
            <div className="tickets-grid">
              {tickets.map(ticket => (
                <TicketCard key={ticket.id} ticket={ticket} onClick={() => setSelectedTicket(ticket)} />
              ))}
            </div>
          ) : (
            <TicketList tickets={tickets} onSelect={setSelectedTicket} />
          )}

          <div className="pagination">
            <span className="pagination-info">
              Showing {from}-{to} of {total}
            </span>
            <div className="pagination-controls">
              <button
                className="page-btn"
                onClick={() => setPage((p) => p - 1)}
                disabled={page <= 1}
              >
                Prev
              </button>
              <span className="page-indicator">
                Page {page} of {pages}
              </span>
              <button
                className="page-btn"
                onClick={() => setPage((p) => p + 1)}
                disabled={page >= pages}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}

      {selectedTicket && (
        <TicketDetail ticket={selectedTicket} onClose={() => setSelectedTicket(null)} />
      )}
    </div>
  )
}

export default App
