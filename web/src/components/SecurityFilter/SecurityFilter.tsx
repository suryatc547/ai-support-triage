import './SecurityFilter.css';

export type SecurityFilterValue = 'all' | 'quarantined' | 'flagged';

interface SecurityFilterProps {
  value: SecurityFilterValue;
  onChange: (value: SecurityFilterValue) => void;
}

const OPTIONS: { value: SecurityFilterValue; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'quarantined', label: 'Quarantined' },
  { value: 'flagged', label: 'Flagged' },
];

export function SecurityFilter({ value, onChange }: SecurityFilterProps) {
  return (
    <div className="security-filter" role="group" aria-label="Filter by security status">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          className={`security-filter-btn ${value === opt.value ? 'active' : ''}`}
          onClick={() => onChange(opt.value)}
          type="button"
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
