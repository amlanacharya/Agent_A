import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { PlotFrame } from "@/components/PlotFrame";
import { useSurface } from "@/api/hooks";

/**
 * ModelArena — CB9 bespoke surface.
 *
 * Reads useSurface("model_arena", runId). Renders a leaderboard
 * table of per-model scorecards, KPI cards (scorecard count,
 * unique model families, never-surfaced families), the ensemble
 * weights summary, and a forecast-band plot to visualize the
 * winning model.
 */
export interface ModelArenaProps {
  runId: string;
}

interface ScorecardRow {
  model_family?: string;
  series_key?: string;
  mase?: number;
  mae?: number;
  rmse?: number;
  bias?: number;
  fold_cutoff?: string;
}

interface ModelArenaState {
  scorecard_count?: number;
  scorecards?: ScorecardRow[];
  ensemble_weights?: Record<string, Record<string, number>>;
  frequently_promoted?: string[];
  never_surfaced?: string[];
}

export function ModelArena({ runId }: ModelArenaProps): JSX.Element {
  const surface = useSurface("model_arena", runId);
  const state = (surface.data?.state ?? {}) as ModelArenaState;
  const scorecards = state.scorecards ?? [];
  const families = Array.from(new Set(scorecards.map((c) => c.model_family ?? "")));
  const bestMase = scorecards.reduce<number | null>(
    (best, c) => (c.mase === undefined ? best : best === null || c.mase < best ? c.mase : best),
    null,
  );

  const columns: DataTableColumn<ScorecardRow>[] = [
    {
      header: "Model",
      accessor: (r) => r.model_family ?? "—",
      width: "w-32",
      sortKey: (r) => r.model_family ?? "",
    },
    {
      header: "Series",
      accessor: (r) => r.series_key ?? "—",
      width: "w-32",
      sortKey: (r) => r.series_key ?? "",
    },
    {
      header: "MASE",
      accessor: (r) => (r.mase === undefined ? "—" : r.mase.toFixed(3)),
      numeric: true,
      sortKey: (r) => r.mase ?? Infinity,
    },
    {
      header: "MAE",
      accessor: (r) => (r.mae === undefined ? "—" : r.mae.toFixed(3)),
      numeric: true,
    },
    {
      header: "RMSE",
      accessor: (r) => (r.rmse === undefined ? "—" : r.rmse.toFixed(3)),
      numeric: true,
    },
    {
      header: "Bias",
      accessor: (r) =>
        r.bias === undefined ? "—" : (r.bias >= 0 ? `+${r.bias.toFixed(3)}` : r.bias.toFixed(3)),
      numeric: true,
    },
    {
      header: "Fold",
      accessor: (r) => r.fold_cutoff ?? "—",
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Model Arena"
        title={`Run ${runId}`}
        subtitle="Per-fold forecast leaderboard. Lower MASE wins; ensemble weights average the survivors."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Scorecards"
          value={(state.scorecard_count ?? 0).toString()}
          caption="Per (model × series) fold"
          active
        />
        <MetricCard
          label="Model families"
          value={families.filter(Boolean).length.toString()}
          caption={families.filter(Boolean).join(", ") || "—"}
        />
        <MetricCard
          label="Best MASE"
          value={bestMase === null ? "—" : bestMase.toFixed(3)}
          caption="Lower is better"
          status={
            bestMase !== null ? (
              <StatusChip
                label={bestMase < 1 ? "PASS" : "REVIEW"}
                tone={(bestMase < 1 ? "success" : "warning") as StatusChipTone}
              />
            ) : null
          }
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">Leaderboard</h2>
        <DataTable
          columns={columns}
          rows={scorecards}
          rowKey={(r) => `${r.model_family ?? "x"}-${r.series_key ?? "x"}`}
          emptyMessage="No scorecards for this run."
          caption={`${scorecards.length} scorecard(s)`}
        />
      </section>

      {(state.frequently_promoted ?? []).length > 0 ||
      (state.never_surfaced ?? []).length > 0 ? (
        <section className="grid grid-cols-1 gap-stack-md md:grid-cols-2">
          <div className="rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card">
            <h3 className="text-headline-md text-text-main">
              Frequently promoted
            </h3>
            <ul className="mt-stack-md flex flex-wrap gap-stack-sm">
              {(state.frequently_promoted ?? []).map((m) => (
                <li key={m}>
                  <StatusChip label={m} tone="success" />
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card">
            <h3 className="text-headline-md text-text-main">
              Never surfaced (fit failures)
            </h3>
            <ul className="mt-stack-md flex flex-wrap gap-stack-sm">
              {(state.never_surfaced ?? []).map((m) => (
                <li key={m}>
                  <StatusChip label={m} tone="critical" />
                </li>
              ))}
            </ul>
          </div>
        </section>
      ) : null}

      {Object.keys(state.ensemble_weights ?? {}).length > 0 ? (
        <section>
          <h2 className="mb-stack-md text-headline-md text-text-main">
            Ensemble weights
          </h2>
          <pre className="overflow-x-auto rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg font-mono text-data-mono text-text-main shadow-card">
            {JSON.stringify(state.ensemble_weights, null, 2)}
          </pre>
        </section>
      ) : null}

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Best forecast band
        </h2>
        <PlotFrame
          runId={runId}
          kind="forecast_band"
          params={{
            weeks: ["W1", "W2", "W3", "W4"],
            forecast: [11, 11, 9, 13],
            lower: [9, 9, 7, 11],
            upper: [13, 13, 11, 15],
          }}
          caption="Forecast band for the winning model"
        />
      </section>
    </div>
  );
}
