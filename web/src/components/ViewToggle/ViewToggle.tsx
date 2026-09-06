import './ViewToggle.css';

export type ViewMode = 'cards' | 'list';

interface ViewToggleProps {
  view: ViewMode;
  onChange: (view: ViewMode) => void;
}

export function ViewToggle({ view, onChange }: ViewToggleProps) {
  return (
    <div className="view-toggle" role="group" aria-label="View mode">
      <button
        className={`view-toggle-btn ${view === 'cards' ? 'active' : ''}`}
        onClick={() => onChange('cards')}
        type="button"
      >
        Cards
      </button>
      <button
        className={`view-toggle-btn ${view === 'list' ? 'active' : ''}`}
        onClick={() => onChange('list')}
        type="button"
      >
        List
      </button>
    </div>
  );
}
