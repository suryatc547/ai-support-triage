import { useEffect, useState } from 'react';
import './TicketDetail.css';
import { apiFetch } from '../../api';
import type { Ticket, User } from '../../types';
import { SecurityBadge } from '../SecurityBadge/SecurityBadge';

interface TicketDetailProps {
  ticket: Ticket;
  onClose: () => void;
  onUpdated: (ticket: Ticket) => void;
}

export function TicketDetail({ ticket, onClose, onUpdated }: TicketDetailProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedAssignee, setSelectedAssignee] = useState<number | ''>('');
  const [acting, setActing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Load the staff directory only when the ticket needs triage.
  useEffect(() => {
    let isMounted = true;
    if (ticket.assignee) {
      return;
    }
    apiFetch<User[]>('/api/users')
      .then((data) => {
        if (isMounted) {
          setUsers(data);
        }
      })
      .catch(() => {
        if (isMounted) {
          setActionError('Failed to load staff list');
        }
      });
    return () => {
      isMounted = false;
    };
  }, [ticket.assignee]);

  const handleReanalyze = async () => {
    setActing(true);
    setActionError(null);
    try {
      const updated = await apiFetch<Ticket>(`/api/tickets/${ticket.id}/reanalyze`, {
        method: 'POST',
      });
      onUpdated(updated);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Re-analysis failed');
    } finally {
      setActing(false);
    }
  };

  const handleAssign = async () => {
    if (selectedAssignee === '') {
      return;
    }
    setActing(true);
    setActionError(null);
    try {
      const updated = await apiFetch<Ticket>(`/api/tickets/${ticket.id}/assign`, {
        method: 'POST',
        body: JSON.stringify({ assignee_id: selectedAssignee }),
      });
      onUpdated(updated);
      setSelectedAssignee('');
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Assignment failed');
    } finally {
      setActing(false);
    }
  };

  const date = new Date(ticket.created_at);
  const dateStr = date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  const timeStr = date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="ticket-type">{ticket.ticket_type}</span>
            <SecurityBadge ticket={ticket} />
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <h2 className="modal-subject">{ticket.subject}</h2>

        <div className="modal-meta">
          <div className="meta-item">
            <span className="meta-label">From</span>
            <span className="meta-value">{ticket.sender_email}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Received</span>
            <span className="meta-value">{dateStr} at {timeStr}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Status</span>
            <span className="meta-value">{ticket.status}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Assigned To</span>
            <span className="meta-value">
              {ticket.assignee ? (
                <>
                  {ticket.assignee.name}
                  <span className="meta-email"> ({ticket.assignee.email})</span>
                </>
              ) : (
                <span className="unassigned-text">Unassigned</span>
              )}
            </span>
          </div>
          {ticket.suspicion_score > 0 && (
            <div className="meta-item">
              <span className="meta-label">Suspicion Score</span>
              <span className="meta-value">{ticket.suspicion_score}/100</span>
            </div>
          )}
        </div>

        {!ticket.assignee && (
          <div className="modal-section">
            <span className="meta-label">Triage Actions</span>
            <div className="modal-actions">
              <button
                className="modal-btn"
                onClick={handleReanalyze}
                disabled={acting}
              >
                {acting ? 'Analyzing...' : 'Re-analyze with AI'}
              </button>
              <div className="assign-control">
                <select
                  className="assign-select"
                  value={selectedAssignee}
                  onChange={(e) =>
                    setSelectedAssignee(e.target.value === '' ? '' : Number(e.target.value))
                  }
                  disabled={acting}
                  aria-label="Assign to"
                >
                  <option value="">Assign to...</option>
                  {users.map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.name}
                      {u.department ? ` — ${u.department}` : ''}
                    </option>
                  ))}
                </select>
                <button
                  className="modal-btn modal-btn-primary"
                  onClick={handleAssign}
                  disabled={acting || selectedAssignee === ''}
                >
                  Assign
                </button>
              </div>
            </div>
            {actionError && (
              <div className="action-error" role="alert">
                {actionError}
              </div>
            )}
          </div>
        )}

        {ticket.validation_findings && ticket.validation_findings.length > 0 && (
          <div className="modal-section">
            <span className="meta-label">Security Findings</span>
            <ul className="findings-list">
              {ticket.validation_findings.map((f, i) => (
                <li key={`${f.type}-${i}`} className={`finding finding-${f.severity}`}>
                  <span className="finding-severity">{f.severity}</span>
                  <span className="finding-message">{f.message}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="modal-section">
          <span className="meta-label">Message</span>
          <div className="modal-body">{ticket.body}</div>
        </div>
      </div>
    </div>
  );
}
