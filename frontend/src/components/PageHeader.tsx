import type { ReactNode } from "react";

/**
 * PageHeader — top of every surface page.
 *
 * Layout: label-caps category line + headline-lg title + optional
 * body-md subtitle + optional right-aligned action slot.
 *
 * DESIGN.md ref: line 167-177 ("Metric Cards" + "Buttons" sections
 * imply this is the title block; the heading type scale lives at
 * lines 67-71).
 */
export interface PageHeaderProps {
  /** Label above the title (e.g. "MISSION CONTROL"). */
  eyebrow?: string;
  /** Big page title. */
  title: string;
  /** Optional subtitle / description line. */
  subtitle?: string;
  /** Optional right-aligned action (e.g. a "Refresh" button). */
  actions?: ReactNode;
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: PageHeaderProps): JSX.Element {
  return (
    <header className="flex flex-wrap items-end justify-between gap-stack-md border-b border-border-slate bg-surface-container-lowest px-stack-lg py-stack-md">
      <div className="min-w-0">
        {eyebrow ? (
          <p className="font-mono text-label-caps uppercase text-text-muted">
            {eyebrow}
          </p>
        ) : null}
        <h1 className="mt-stack-sm text-headline-lg text-text-main">{title}</h1>
        {subtitle ? (
          <p className="mt-stack-sm max-w-2xl text-body-md text-text-muted">
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-stack-sm">{actions}</div> : null}
    </header>
  );
}
