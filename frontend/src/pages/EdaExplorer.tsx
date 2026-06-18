import { useState, useMemo } from "react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { FilterPanel } from "@/components/FilterPanel";
import { PlotFrame } from "@/components/PlotFrame";
import { useSurface } from "@/api/hooks";

/**
 * EdaExplorer — CB7 bespoke surface.
 *
 * Reads useSurface("eda_explorer", runId) and renders:
 * - FilterPanel (series selector + granularity)
 * - 3 plot frames: demand_curve, sparsity scatter, anomalies
 * - Per-series profiles table
 *
 * Plot bytes come from the FastAPI /plots endpoint via
 * <PlotFrame>; the page filters by selected series locally
 * (the per-series EDA report is per-run, not per-series).
 */
export interface EdaExplorerProps {
  runId: string;
}

interface SeriesProfileRow {
  series_key: string;
  sb_class?: string;
  adi?: number;
  cv2?: number;
  trend_strength?: number;
  seasonal_strength?: number;
  recommended_models?: string[];
}

interface EdaExplorerState {
  series_count?: number;
  series_profiles?: SeriesProfileRow[];
  demand_class_distribution?: Record<string, number>;
}

const GRANULARITY = [
  { value: "week", label: "Weekly" },
  { value: "month", label: "Monthly" },
] as const;

export function EdaExplorer({ runId }: EdaExplorerProps): JSX.Element {
  const surface = useSurface("eda_explorer", runId);
  const state = (surface.data?.state ?? {}) as EdaExplorerState;
  const profiles = state.series_profiles ?? [];

  const [selectedSeries, setSelectedSeries] = useState<string>(
    profiles[0]?.series_key ?? ""
  );
  const [granularity, setGranularity] =
    useState<(typeof GRANULARITY)[number]["value"]>("week");

  // Sync the dropdown if profiles arrive after first render.
  const seriesOptions = useMemo(
    () => profiles.map((p) => ({ value: p.series_key, label: p.series_key })),
    [profiles],
  );
  if (!selectedSeries && seriesOptions.length > 0) {
    setSelectedSeries(seriesOptions[0]!.value);
  }

  const columns: DataTableColumn<SeriesProfileRow>[] = [
    { header: "Series", accessor: (r) => r.series_key, width: "w-32" },
    {
      header: "Demand class",
      accessor: (r) =>
        r.sb_class ? <StatusChip label={r.sb_class} tone={classTone(r.sb_class)} /> : "—",
    },
    {
      header: "ADI",
      accessor: (r) => (r.adi === undefined ? "—" : r.adi.toFixed(2)),
      numeric: true,
    },
    {
      header: "CV²",
      accessor: (r) => (r.cv2 === undefined ? "—" : r.cv2.toFixed(2)),
      numeric: true,
    },
    {
      header: "Trend",
      accessor: (r) => (r.trend_strength === undefined ? "—" : r.trend_strength.toFixed(2)),
      numeric: true,
    },
    {
      header: "Seasonal",
      accessor: (r) =>
        r.seasonal_strength === undefined ? "—" : r.seasonal_strength.toFixed(2),
      numeric: true,
    },
    {
      header: "Recommended models",
      accessor: (r) => r.recommended_models?.join(", ") ?? "—",
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="EDA Explorer"
        title={`Run ${runId}`}
        subtitle="Per-series EDA drill-down. Pick a series to drive the plots below."
      />

      <FilterPanel
        series={seriesOptions.length > 0 ? seriesOptions : [{ value: "all", label: "All series" }]}
        selectedSeries={selectedSeries}
        onSeriesChange={setSelectedSeries}
        granularity={GRANULARITY as unknown as { value: string; label: string }[]}
        selectedGranularity={granularity}
        onGranularityChange={(v) => setGranularity(v as typeof granularity)}
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Series count"
          value={(state.series_count ?? 0).toString()}
          caption="Distinct (sku, location) keys"
          active
        />
        <MetricCard
          label="Demand classes"
          value={String(Object.keys(state.demand_class_distribution ?? {}).length)}
          caption={formatBreakdown(state.demand_class_distribution ?? {})}
        />
        <MetricCard
          label="Granularity"
          value={granularity.toUpperCase()}
          caption="Selected resolution"
        />
      </section>

      <section className="grid grid-cols-1 gap-stack-md lg:grid-cols-3">
        <PlotFrame
          runId={runId}
          kind="demand_curve"
          params={{
            weeks: ["W1", "W2", "W3", "W4"],
            actual: [10, 12, 8, 14],
            forecast: [11, 11, 9, 13],
            series_key: selectedSeries,
          }}
          caption={`Demand curve · ${selectedSeries}`}
        />
        <PlotFrame
          runId={runId}
          kind="sparsity"
          params={{
            series: profiles.map((p) => ({
              series_key: p.series_key,
              adi: p.adi ?? 1,
              cv2: p.cv2 ?? 0.5,
            })),
          }}
          caption="Syntetos-Boylan sparsity scatter (all series)"
        />
        <PlotFrame
          runId={runId}
          kind="anomalies"
          params={{
            weeks: ["W1", "W2", "W3", "W4", "W5"],
            values: [10, 50, 12, 11, 13],
            flags: [false, true, false, false, false],
          }}
          caption={`Anomaly flags · ${selectedSeries}`}
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">Series profiles</h2>
        <DataTable
          columns={columns}
          rows={profiles}
          rowKey={(row) => row.series_key}
          emptyMessage="No series profiles for this run."
          caption={`${profiles.length} series`}
        />
      </section>
    </div>
  );
}

function classTone(sbClass: string): StatusChipTone {
  if (sbClass === "SMOOTH") return "success";
  if (sbClass === "INTERMITTENT") return "warning";
  if (sbClass === "ERRATIC" || sbClass === "LUMPY") return "critical";
  return "neutral";
}

function formatBreakdown(breakdown: Record<string, number>): string {
  const entries = Object.entries(breakdown);
  if (entries.length === 0) return "—";
  return entries.map(([k, v]) => `${k}: ${v}`).join(", ");
}
