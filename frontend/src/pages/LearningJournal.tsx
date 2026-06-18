import { useState } from "react";
import clsx from "clsx";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { StatusChip } from "@/components/StatusChip";
import { useSurface } from "@/api/hooks";

/**
 * LearningJournal — CB11 bespoke surface.
 *
 * Reads useSurface("learning_journal", runId). Renders the six
 * Phase 1 workspace markdown artifacts (LEARNINGS, DECISIONS,
 * ASSUMPTIONS, RUNBOOK, MODEL_REGISTRY, PROMOTION_DECISIONS) as
 * a tabbed viewer + active/retired card KPIs.
 */
export interface LearningJournalProps {
  runId: string;
}

interface LearningJournalState {
  active_cards?: number;
  retired_cards?: number;
  LEARNINGS?: string | null;
  DECISIONS?: string | null;
  ASSUMPTIONS?: string | null;
  RUNBOOK?: string | null;
  MODEL_REGISTRY?: string | null;
  PROMOTION_DECISIONS?: string | null;
}

const ARTIFACTS = [
  { key: "LEARNINGS", label: "Learnings" },
  { key: "DECISIONS", label: "Decisions" },
  { key: "ASSUMPTIONS", label: "Assumptions" },
  { key: "RUNBOOK", label: "Runbook" },
  { key: "MODEL_REGISTRY", label: "Model registry" },
  { key: "PROMOTION_DECISIONS", label: "Promotions" },
] as const;

type ArtifactKey = (typeof ARTIFACTS)[number]["key"];

export function LearningJournal({ runId }: LearningJournalProps): JSX.Element {
  const surface = useSurface("learning_journal", runId);
  const state = (surface.data?.state ?? {}) as LearningJournalState;
  const [active, setActive] = useState<ArtifactKey>("LEARNINGS");

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Learning Journal"
        title={`Run ${runId}`}
        subtitle="Phase 1 workspace markdown. Six artifacts the planner reviews when the run lands."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Active cards"
          value={(state.active_cards ?? 0).toString()}
          caption="Learnings currently in use"
          active
          status={<StatusChip label="ACTIVE" tone="success" />}
        />
        <MetricCard
          label="Retired cards"
          value={(state.retired_cards ?? 0).toString()}
          caption="Lessons that no longer apply"
        />
        <MetricCard
          label="Artifacts"
          value={`${ARTIFACTS.filter((a) => Boolean(state[a.key as ArtifactKey])).length} / ${ARTIFACTS.length}`}
          caption="Workspace markdown present"
        />
      </section>

      <section className="rounded-md border border-border-slate bg-surface-container-lowest shadow-card">
        <nav className="flex flex-wrap gap-stack-sm border-b border-border-slate bg-surface-container-low p-stack-sm">
          {ARTIFACTS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setActive(key)}
              className={clsx(
                "rounded-md px-stack-md py-stack-sm text-body-sm font-medium transition-colors",
                active === key
                  ? "bg-primary text-on-primary"
                  : "text-text-muted hover:bg-surface-container hover:text-text-main",
              )}
            >
              {label}
            </button>
          ))}
        </nav>
        <article className="max-h-96 overflow-auto p-stack-lg">
          {state[active] == null ? (
            <p className="text-body-md italic text-text-muted">
              No {ARTIFACTS.find((a) => a.key === active)?.label} artifact for
              this run.
            </p>
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-data-mono text-text-main">
              {state[active]}
            </pre>
          )}
        </article>
      </section>
    </div>
  );
}
