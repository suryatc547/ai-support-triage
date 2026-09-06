import './TicketList.css';
import type { Ticket } from '../../types';
import { SecurityBadge } from '../SecurityBadge/SecurityBadge';

interface TicketListProps {
  tickets: Ticket[];
  onSelect: (ticket: Ticket) => void;
}

export function TicketList({ tickets, onSelect }: TicketListProps) {
  return (
    <div className="ticket-list">
      <div className="ticket-list-head">
        <span className="col-ticket">Ticket</span>
        <span className="col-type">Type</span>
        <span className="col-sender">From</span>
        <span className="col-assignee">Assigned To</span>
        <span className="col-date">Date</span>
      </div>
      {tickets.length === 0 ? (
        <div className="ticket-list-empty">No tickets to display.</div>
      ) : (
        tickets.map((ticket) => {
          const quarantined =
            ticket.status === 'quarantined' || ticket.security_flag === 'quarantined';
          return (
            <div
              className={`ticket-list-row ${quarantined ? 'ticket-list-row-quarantined' : ''}`.trim()}
              key={ticket.id}
              onClick={() => onSelect(ticket)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(ticket);
                }
              }}
            >
              <div className="col-ticket">
                <div className="row-subject">{ticket.subject}</div>
                <div className="row-body">{ticket.body}</div>
              </div>
              <span className="col-type">
                <span className="row-type">{ticket.ticket_type}</span>
                <SecurityBadge ticket={ticket} showScore={false} />
              </span>
              <span className="col-sender row-sender">{ticket.sender_email}</span>
              <span className="col-assignee">
                {ticket.assignee ? (
                  <span className="row-assignee">
                    <span className="row-assignee-name">
                      <span className="row-avatar">
                        {ticket.assignee.name.charAt(0).toUpperCase()}
                      </span>
                      {ticket.assignee.name}
                    </span>
                    <span className="row-assignee-email">{ticket.assignee.email}</span>
                  </span>
                ) : (
                  <span className="row-assignee unassigned">Unassigned</span>
                )}
              </span>
              <span className="col-date row-date">
                {new Date(ticket.created_at).toLocaleDateString()}
              </span>
            </div>
          );
        })
      )}
    </div>
  );
}
