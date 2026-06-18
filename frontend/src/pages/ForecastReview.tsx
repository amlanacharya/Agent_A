import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { PlotFrame } from "@/components/PlotFrame";
import { useSurface } from "@/api/hooks";

/**
 * ForecastReview — CB10 bespoke surface.
 *
 * Reads useSurface("forecast_review", runId). Renders the per-series
 * Foundry results, the overall MASE + target-met KPIs, the
 * backtest plot, and the foundry narrative.
 */
export interface ForecastReviewProps {
  runId: string;
}

interface SeriesResultRow {
  series_key?: string;
  sb_class?: string;
  mase_target?: number;
  best_model?: string;
  target_met?: boolean;
  overall_mase?: number;
}

interface ForecastReviewState {
  overall_mase?: number;
  target_met_fraction?: number;
  series_results?: SeriesResultRow[];
  narrative?: string;
}

export function ForecastReview({ runId }: ForecastReviewProps): JSX.Element {
  const surface = useSurface("forecast_review", runId);
  const state = (surface.data?.state ?? {}) as ForecastReviewState;
  const seriesResults = state.series_results ?? [];
  const targetMet = state.target_met_fraction ?? 0;

  const columns: DataTableColumn<SeriesResultRow>[] = [
    {
      header: "Series",
      accessor: (r) => r.series_key ?? "—",
      width: "w-32",
      sortKey: (r) => r.series_key ?? "",
    },
    {
      header: "Demand class",
      accessor: (r) =>
        r.sb_class ? (
          <StatusChip label={r.sb_class} tone={classTone(r.sb_class)} />
        ) : (
          "—"
        ),
    },
    {
      header: "MASE target",
      accessor: (r) =>
        r.mase_target === undefined ? "—" : r.mase_target.toFixed(2),
      numeric: true,
    },
    {
      header: "MASE actual",
      accessor: (r) =>
        r.overall_mase === undefined ? "—" : r.overall_mase.toFixed(3),
      numeric: true,
    },
    {
      header: "Best model",
      accessor: (r) => r.best_model ?? "—",
      width: "w-32",
    },
    {
      header: "Target met",
      accessor: (r) =>
        r.target_met === undefined ? (
          "—"
        ) : (
          <StatusChip
            label={r.target_met ? "MET" : "MISS"}
            tone={(r.target_met ? "success" : "critical") as StatusChipTone}
          />
        ),
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Forecast Review"
        title={`Run ${runId}`}
        subtitle="Per-series Phase 4 forecast outcome. Overall MASE + target-met fraction drive the Foundry verdict."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Overall MASE"
          value={
            state.overall_mase === undefined ? "—" : state.overall_mase.toFixed(3)
          }
          caption="Weighted across all series"
          active
          status={
            state.overall_mase !== undefined ? (
              <StatusChip
                label={state.overall_mase < 1 ? "PASS" : "REVIEW"}
                tone={
                  (state.overall_mase < 1 ? "success" : "warning") as StatusChipTone
                }
              />
            ) : null
          }
        />
        <MetricCard
          label="Target met"
          value={`${Math.round(targetMet * 100)}%`}
          caption="Series hitting MASE target"
        />
        <MetricCard
          label="Series results"
          value={seriesResults.length.toString()}
          caption="Per-series verdicts"
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Series results
        </h2>
        <DataTable
          columns={columns}
          rows={seriesResults}
          rowKey={(r) => `${r.series_key ?? "x"}`}
          emptyMessage="No series results for this run."
          caption={`${seriesResults.length} series`}
        />
      </section>

      <section className="grid grid-cols-1 gap-stack-md lg:grid-cols-2">
        <PlotFrame
          runId={runId}
          kind="forecast_band"
          params={{
            weeks: ["W1", "W2", "W3", "W4"],
            forecast: [11, 11, 9, 13],
            lower: [9, 9, 7, 11],
            upper: [13, 13, 11, 15],
          }}
          caption="Forecast band (best model)"
        />
        <PlotFrame
          runId={runId}
          kind="backtest"
          params={{
            folds: ["fold_1", "fold_2", "fold_3"],
            actual: [10, 12, 8],
            forecast: [11, 11, 9],
          }}
          caption="Backtest: actual vs forecast per fold"
        />
      </section>

      {state.narrative ? (
        <section className="rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card">
          <h2 className="text-headline-md text-text-main">Foundry narrative</h2>
          <p className="mt-stack-md whitespace-pre-line text-body-md text-text-muted">
            {state.narrative}
          </p>
        </section>
      ) : null}
    </div>
  );
}

function classTone(sbClass: string): StatusChipTone {
  if (sbClass === "SMOOTH") return "success";
  if (sbClass === "INTERMITTENT") return "warning";
  if (sbClass === "ERRATIC" || sbClass === "LUMPY") return "critical";
  return "neutral";
}
