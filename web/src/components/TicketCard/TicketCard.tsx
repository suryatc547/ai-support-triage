import './TicketCard.css';
import type { Ticket } from '../../types';
import { SecurityBadge } from '../SecurityBadge/SecurityBadge';

interface TicketCardProps {
  ticket: Ticket;
  onClick: () => void;
}

export function TicketCard({ ticket, onClick }: TicketCardProps) {
  const quarantined = ticket.status === 'quarantined' || ticket.security_flag === 'quarantined';
  const cardClass = `ticket-card ${quarantined ? 'ticket-card-quarantined' : ''}`.trim();

  return (
    <div className={cardClass} onClick={onClick} role="button" tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}>
      <div className="ticket-header">
        <span className="ticket-type">{ticket.ticket_type}</span>
        <div className="ticket-header-right">
          <SecurityBadge ticket={ticket} />
          <span className="ticket-date">{new Date(ticket.created_at).toLocaleDateString()}</span>
        </div>
      </div>
      <h3 className="ticket-subject">{ticket.subject}</h3>
      <p className="ticket-body">{ticket.body}</p>
      <div className="ticket-footer">
        <span className="ticket-sender">{ticket.sender_email}</span>
        <div className="assignee">
          {ticket.assignee ? (
            <>
              <div className="assignee-avatar">
                {ticket.assignee.name.charAt(0).toUpperCase()}
              </div>
              <div className="assignee-info">
                <span>{ticket.assignee.name}</span>
                <span className="assignee-email">{ticket.assignee.email}</span>
              </div>
            </>
          ) : (
            <>
              <div className="assignee-avatar unassigned">?</div>
              <span className="unassigned-text">Unassigned</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
