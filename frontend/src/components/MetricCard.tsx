import clsx from "clsx";
import type { ReactNode } from "react";

/**
 * MetricCard — centerpiece of the cockpit (DESIGN.md "Metric Cards"
 * section, lines 168-169).
 *
 * Layout: white card, 1px slate border, 24px padding, optional 2px
 * electric-cyan top accent for "active" / primary metrics. Data
 * values render in JetBrains Mono.
 */
export interface MetricCardProps {
  /** Small label above the value (uppercase label-caps). */
  label: string;
  /** Big data value (rendered in data-mono). */
  value: ReactNode;
  /** Optional caption below the value (e.g. "+12% vs last run"). */
  caption?: string;
  /** When true, the 2px Electric Cyan top accent appears. */
  active?: boolean;
  /** Optional status chip slot in the top-right corner. */
  status?: ReactNode;
}

export function MetricCard({
  label,
  value,
  caption,
  active = false,
  status,
}: MetricCardProps): JSX.Element {
  return (
    <article
      className={clsx(
        "rounded-lg border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card",
        active && "shadow-metric-active",
      )}
    >
      <header className="flex items-start justify-between gap-stack-sm">
        <p className="font-mono text-label-caps uppercase text-text-muted">
          {label}
        </p>
        {status}
      </header>
      <p className="mt-stack-sm font-mono text-data-mono text-text-main">
        {value}
      </p>
      {caption ? (
        <p className="mt-stack-sm text-body-sm text-text-muted">{caption}</p>
      ) : null}
    </article>
  );
}
