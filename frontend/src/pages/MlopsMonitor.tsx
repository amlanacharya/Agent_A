import { useState } from "react";
import clsx from "clsx";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { StatusChip } from "@/components/StatusChip";
import { useSurface } from "@/api/hooks";

/**
 * MlopsMonitor — CB12 bespoke surface.
 *
 * Reads useSurface("mlops_monitor", runId). Renders the four
 * Phase 7 markdown artifacts as a tabbed view (Monitoring,
 * Drift, Override analysis, Model health) + 4 KPI cards
 * (one per artifact presence + total artifacts present).
 */
export interface MlopsMonitorProps {
  runId: string;
}

interface MlopsMonitorState {
  "MONITORING_REPORT.md"?: string | null;
  "DRIFT_REPORT.md"?: string | null;
  "OVERRIDE_ANALYSIS.md"?: string | null;
  "MODEL_HEALTH.md"?: string | null;
  // Index signature lets the page look up by artifact key in
  // a loop without TS noUncheckedIndexedAccess warnings on
  // each lookup (the union above is not directly assignable
  // from a string).
  [key: string]: string | null | undefined;
}

const ARTIFACTS = [
  {
    key: "MONITORING_REPORT.md",
    label: "Monitoring",
    tone: "info" as const,
  },
  {
    key: "DRIFT_REPORT.md",
    label: "Drift",
    tone: "warning" as const,
  },
  {
    key: "OVERRIDE_ANALYSIS.md",
    label: "Overrides",
    tone: "critical" as const,
  },
  {
    key: "MODEL_HEALTH.md",
    label: "Health",
    tone: "success" as const,
  },
];

type ArtifactKey = (typeof ARTIFACTS)[number]["key"];

export function MlopsMonitor({ runId }: MlopsMonitorProps): JSX.Element {
  const surface = useSurface("mlops_monitor", runId);
  const state = (surface.data?.state ?? {}) as MlopsMonitorState;
  const [active, setActive] = useState<ArtifactKey>("MONITORING_REPORT.md");

  const presentCount = ARTIFACTS.filter((a) => Boolean(state[a.key])).length;

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="MLOps Monitor"
        title={`Run ${runId}`}
        subtitle="Phase 7 monitoring artifacts. Read by the planner on every refresh; written by the run's monitoring tick."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-2 xl:grid-cols-4">
        {ARTIFACTS.map(({ key, label, tone }) => (
          <MetricCard
            key={key}
            label={label}
            value={state[key] ? "Present" : "Missing"}
            caption={state[key] ? "Last refresh attached" : "Run has not produced this artifact yet"}
            status={
              <StatusChip
                label={state[key] ? "OK" : "PENDING"}
                tone={state[key] ? "success" : tone}
              />
            }
          />
        ))}
      </section>

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Artifacts present"
          value={`${presentCount} / ${ARTIFACTS.length}`}
          caption="Phase 7 monitoring coverage"
          active
        />
        <MetricCard
          label="Run"
          value={runId}
          caption="Active run"
        />
        <MetricCard
          label="Refresh"
          value="On focus"
          caption="TanStack Query refetchOnWindowFocus"
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
        <article className="max-h-[36rem] overflow-auto p-stack-lg">
          {state[active] == null ? (
            <p className="text-body-md italic text-text-muted">
              No {ARTIFACTS.find((a) => a.key === active)?.label} artifact
              for this run. The next monitoring tick will produce it.
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
