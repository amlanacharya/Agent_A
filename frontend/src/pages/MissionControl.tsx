import { MetricCard } from "@/components/MetricCard";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { PageHeader } from "@/components/PageHeader";
import { useSurface } from "@/api/hooks";

/**
 * MissionControl — the live-state cockpit view (CB5).
 *
 * Reads the mission_control surface and renders three zones:
 *
 * 1. KPI grid (4 cards): current step, active agent, confidence,
 *    blockers count.
 * 2. Approval banner: surfaces the `approval_needed` flag + the
 *    `verifier_gate` so the planner sees it without scrolling.
 * 3. Activity feed: the tool_result + code_escalation_status +
 *    blockers as a vertical timeline (recent events at the top).
 *
 * The state shape comes from ``CockpitState.to_public_dict()``
 * (see backend/forecasting/cockpit_state.py). CB5 does not extend
 * the contract — only renders it.
 */
export interface MissionControlProps {
  runId: string;
}

interface MissionControlState {
  run_id?: string;
  current_step?: string;
  active_agent?: string;
  tool_result?: string | null;
  code_escalation_status?: string | null;
  code_attempt?: number | null;
  verifier_gate?: string | null;
  approval_needed?: boolean;
  confidence?: string;
  blockers?: string[];
}

export function MissionControl({ runId }: MissionControlProps): JSX.Element {
  const surface = useSurface("mission_control", runId);
  const state = (surface.data?.state ?? {}) as MissionControlState;

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Mission Control"
        title={`Run ${runId}`}
        subtitle="Live platform state. The cockpit reads the same CockpitState the agents see."
      />

      {surface.isError ? (
        <ErrorBanner message={String(surface.error)} />
      ) : null}

      <KpiGrid state={state} />

      {state.approval_needed ? (
        <ApprovalBanner
          gate={state.verifier_gate ?? "unknown gate"}
          attempt={state.code_attempt ?? null}
        />
      ) : null}

      <ActivityFeed state={state} />
    </div>
  );
}

function KpiGrid({ state }: { state: MissionControlState }): JSX.Element {
  const confidenceTone: StatusChipTone =
    state.confidence === "high"
      ? "success"
      : state.confidence === "low"
        ? "critical"
        : "warning";
  return (
    <section className="grid grid-cols-1 gap-stack-md md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        label="Current step"
        value={state.current_step ?? "—"}
        caption="What the platform is doing right now"
        active
      />
      <MetricCard
        label="Active agent"
        value={state.active_agent ?? "—"}
        caption="Top of the agent stack"
      />
      <MetricCard
        label="Confidence"
        value={state.confidence?.toUpperCase() ?? "—"}
        caption="Harness signal"
        status={
          <StatusChip
            label={state.confidence ?? "unknown"}
            tone={confidenceTone}
          />
        }
      />
      <MetricCard
        label="Blockers"
        value={(state.blockers?.length ?? 0).toString()}
        caption={
          (state.blockers?.length ?? 0) === 0
            ? "No blockers"
            : "Open issues — see activity feed"
        }
      />
    </section>
  );
}

function ApprovalBanner({
  gate,
  attempt,
}: {
  gate: string;
  attempt: number | null;
}): JSX.Element {
  return (
    <aside className="flex flex-wrap items-center justify-between gap-stack-md rounded-md border border-warning-amber/30 bg-warning-amber/10 p-stack-md">
      <div>
        <p className="font-mono text-label-caps uppercase text-warning-amber">
          Approval Needed
        </p>
        <p className="mt-stack-sm text-body-md text-text-main">
          The harness paused at <span className="font-mono">{gate}</span>
          {attempt !== null ? (
            <>
              {" "}
              (attempt {attempt} / 3)
            </>
          ) : null}
          . Acknowledge in the Replenishment Board to resume.
        </p>
      </div>
      <StatusChip label="Awaiting" tone="warning" icon="⏸" />
    </aside>
  );
}

function ActivityFeed({ state }: { state: MissionControlState }): JSX.Element {
  const events: Array<{ ts: string; label: string; tone: StatusChipTone; body: string }> = [];
  if (state.tool_result) {
    events.push({
      ts: "now",
      label: "Tool result",
      tone: "info",
      body: state.tool_result,
    });
  }
  if (state.code_escalation_status) {
    events.push({
      ts: "recent",
      label: "Code escalation",
      tone: state.code_escalation_status.includes("fail") ? "critical" : "info",
      body: state.code_escalation_status,
    });
  }
  for (const blocker of state.blockers ?? []) {
    events.push({
      ts: "open",
      label: "Blocker",
      tone: "critical",
      body: blocker,
    });
  }
  return (
    <section className="rounded-lg border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card">
      <header className="flex items-center justify-between">
        <h2 className="text-headline-md text-text-main">Recent activity</h2>
        <StatusChip label={`${events.length} event(s)`} tone="neutral" />
      </header>
      {events.length === 0 ? (
        <p className="mt-stack-md text-body-md text-text-muted">
          No events yet for this run. Start a run from the FastAPI
          surface to populate the cockpit.
        </p>
      ) : (
        <ol className="mt-stack-md flex flex-col gap-stack-md">
          {events.map((event, idx) => (
            <li
              key={idx}
              className="flex items-start gap-stack-md border-l-2 border-border-slate pl-stack-md"
            >
              <StatusChip label={event.ts} tone={event.tone} />
              <div>
                <p className="text-body-md font-semibold text-text-main">
                  {event.label}
                </p>
                <p className="mt-stack-sm text-body-sm text-text-muted">
                  {event.body}
                </p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function ErrorBanner({ message }: { message: string }): JSX.Element {
  return (
    <aside className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
      Failed to load mission control state: {message}
    </aside>
  );
}
