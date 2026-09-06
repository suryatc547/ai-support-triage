import './Loader.css';

interface LoaderProps {
  size?: 'sm' | 'md' | 'lg';
  inline?: boolean;
  label?: string;
}

export function Loader({ size = 'md', inline = false, label }: LoaderProps) {
  return (
    <div
      className={`loader ${inline ? 'loader-inline' : 'loader-block'}`}
      role="status"
      aria-live="polite"
    >
      <span className={`loader-spinner loader-spinner-${size}`} aria-hidden="true" />
      {label && <span className="loader-label">{label}</span>}
    </div>
  );
}
