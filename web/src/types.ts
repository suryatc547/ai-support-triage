export interface Assignee {
  id: number;
  name: string;
  email: string;
}

export interface ValidationFinding {
  type: string;
  severity: string;
  message: string;
}

export interface Ticket {
  id: number;
  subject: string;
  body: string;
  sender_email: string;
  ticket_type: string;
  status: string;
  created_at: string;
  assignee: Assignee | null;
  security_flag: string | null;
  suspicion_score: number;
  validation_findings: ValidationFinding[];
}

export interface TicketPage {
  items: Ticket[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}
