import './SecurityBadge.css';
import type { Ticket } from '../../types';

interface SecurityBadgeProps {
  ticket: Ticket;
  showScore?: boolean;
}

export function SecurityBadge({ ticket, showScore = true }: SecurityBadgeProps) {
  const quarantined = ticket.status === 'quarantined' || ticket.security_flag === 'quarantined';
  const flagged = ticket.security_flag === 'flagged';

  if (!quarantined && !flagged) {
    return null;
  }

  const label = quarantined ? 'Quarantined' : 'Flagged';
  const cls = quarantined ? 'security-badge-quarantined' : 'security-badge-flagged';

  return (
    <span className={`security-badge ${cls}`} title="Security flagged for review">
      <span className="security-badge-dot" aria-hidden="true" />
      {label}
      {showScore && ticket.suspicion_score > 0 && (
        <span className="security-badge-score">{ticket.suspicion_score}</span>
      )}
    </span>
  );
}
