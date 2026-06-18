import clsx from "clsx";

/**
 * FilterPanel — CB7's filter row for the EDA Explorer.
 *
 * Layout (DESIGN.md "Layout & Spacing" — 12-col grid + 24px
 * gutter): a horizontal row of label + select / input combos.
 * The selected value drives the parent's useSurface query
 * (filter changes re-render the right-hand surface).
 */
export interface FilterOption<T extends string> {
  value: T;
  label: string;
}

export interface FilterPanelProps {
  series: FilterOption<string>[];
  selectedSeries: string;
  onSeriesChange: (value: string) => void;
  granularity: FilterOption<string>[];
  selectedGranularity: string;
  onGranularityChange: (value: string) => void;
  /** Date range is read-only for CB7 (surface state shape is per-run, not per-day). */
  dateRangeLabel?: string;
}

export function FilterPanel({
  series,
  selectedSeries,
  onSeriesChange,
  granularity,
  selectedGranularity,
  onGranularityChange,
  dateRangeLabel,
}: FilterPanelProps): JSX.Element {
  return (
    <div className="grid grid-cols-1 gap-stack-md rounded-md border border-border-slate bg-surface-container-lowest p-stack-md shadow-card md:grid-cols-3">
      <FilterField label="Series">
        <select
          value={selectedSeries}
          onChange={(e) => onSeriesChange(e.target.value)}
          className={clsx(
            "w-full rounded-md border border-border-slate bg-surface-container-lowest px-stack-md py-stack-sm",
            "text-body-md text-text-main focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/15",
          )}
        >
          {series.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </FilterField>
      <FilterField label="Granularity">
        <select
          value={selectedGranularity}
          onChange={(e) => onGranularityChange(e.target.value)}
          className={clsx(
            "w-full rounded-md border border-border-slate bg-surface-container-lowest px-stack-md py-stack-sm",
            "text-body-md text-text-main focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/15",
          )}
        >
          {granularity.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </FilterField>
      <FilterField label="Date range">
        <p className="rounded-md border border-border-slate bg-surface-container-low px-stack-md py-stack-sm text-body-md text-text-muted">
          {dateRangeLabel ?? "All available weeks"}
        </p>
      </FilterField>
    </div>
  );
}

function FilterField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <label className="flex flex-col gap-stack-sm">
      <span className="font-mono text-label-caps uppercase text-text-muted">
        {label}
      </span>
      {children}
    </label>
  );
}
