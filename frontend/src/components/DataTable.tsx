import clsx from "clsx";
import type { ReactNode } from "react";

/**
 * DataTable — flat tabular layout per DESIGN.md "Data Tables" section.
 *
 * "Tables should use a strictly flat design with horizontal-only
 * borders in #e2e8f0. Row hover states use a #f8fafc background
 * color to maintain context during horizontal scanning."
 *
 * Cells render in Inter; numeric columns (when `numeric: true`)
 * switch to JetBrains Mono via font-mono + text-data-mono.
 */
export interface DataTableColumn<T> {
  /** Header label (label-caps uppercase rendered). */
  header: string;
  /** Cell accessor. */
  accessor: (row: T) => ReactNode;
  /** Optional row key for stable sorting. */
  sortKey?: (row: T) => string | number;
  /** Render in mono (right-aligned) — useful for IDs + numbers. */
  numeric?: boolean;
  /** Tailwind width class (e.g. "w-32"). */
  width?: string;
}

export interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  /** Stable key extractor. */
  rowKey: (row: T) => string;
  /** Empty-state message. */
  emptyMessage?: string;
  /** Optional caption row at the bottom (e.g. "Showing N rows"). */
  caption?: ReactNode;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  emptyMessage = "No rows.",
  caption,
}: DataTableProps<T>): JSX.Element {
  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg text-body-md text-text-muted shadow-card">
        {emptyMessage}
      </div>
    );
  }
  return (
    <div className="overflow-x-auto rounded-md border border-border-slate bg-surface-container-lowest shadow-card">
      <table className="w-full text-body-sm">
        <thead>
          <tr className="border-b border-border-slate bg-surface-container-low text-left">
            {columns.map((col, idx) => (
              <th
                key={idx}
                className={clsx(
                  "px-stack-md py-stack-sm font-mono text-label-caps uppercase text-text-muted",
                  col.numeric && "text-right",
                  col.width,
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              className="border-b border-border-slate last:border-b-0 hover:bg-surface transition-colors"
            >
              {columns.map((col, idx) => (
                <td
                  key={idx}
                  className={clsx(
                    "px-stack-md py-stack-md text-text-main",
                    col.numeric && "font-mono text-data-mono text-right",
                    col.width,
                  )}
                >
                  {col.accessor(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {caption ? (
        <div className="border-t border-border-slate bg-surface-container-low px-stack-md py-stack-sm text-body-sm text-text-muted">
          {caption}
        </div>
      ) : null}
    </div>
  );
}
