import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { useSurface } from "@/api/hooks";

/**
 * DataHealth — CB6 bespoke surface.
 *
 * Reads useSurface("data_health", runId) and renders the
 * headline numbers from the Phase 2 EDA report summary:
 * series count, segment count, demand-class breakdown, segment
 * profiles table, narrative.
 */
export interface DataHealthProps {
  runId: string;
}

interface SegmentProfileRow {
  segment_id: string;
  series_count: number;
  median_adi?: number;
  median_cv2?: number;
  demand_class_distribution?: Record<string, number>;
  forecastability_breakdown?: Record<string, number>;
}

interface DataHealthState {
  series_count?: number;
  segment_count?: number;
  segment_profiles?: SegmentProfileRow[];
  demand_class_breakdown?: Record<string, number>;
  narrative?: string;
}

export function DataHealth({ runId }: DataHealthProps): JSX.Element {
  const surface = useSurface("data_health", runId);
  const state = (surface.data?.state ?? {}) as DataHealthState;
  const breakdown = state.demand_class_breakdown ?? {};
  const profiles = state.segment_profiles ?? [];

  const columns: DataTableColumn<SegmentProfileRow>[] = [
    {
      header: "Segment",
      accessor: (row) => row.segment_id,
      sortKey: (row) => row.segment_id,
      width: "w-32",
    },
    {
      header: "Series",
      accessor: (row) => row.series_count,
      sortKey: (row) => row.series_count,
      numeric: true,
      width: "w-24",
    },
    {
      header: "Median ADI",
      accessor: (row) => formatNumber(row.median_adi),
      numeric: true,
      width: "w-32",
    },
    {
      header: "Median CV²",
      accessor: (row) => formatNumber(row.median_cv2),
      numeric: true,
      width: "w-32",
    },
    {
      header: "Demand classes",
      accessor: (row) => formatBreakdown(row.demand_class_distribution ?? {}),
    },
    {
      header: "Forecastability",
      accessor: (row) => formatBreakdown(row.forecastability_breakdown ?? {}),
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Data Health"
        title={`Run ${runId}`}
        subtitle="Phase 2 EDA summary: series count, segments, demand-class mix, narrative."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-3">
        <MetricCard
          label="Series"
          value={(state.series_count ?? 0).toString()}
          caption="Distinct (sku, location) keys"
          active
        />
        <MetricCard
          label="Segments"
          value={(state.segment_count ?? 0).toString()}
          caption="Forecastability clusters"
        />
        <MetricCard
          label="Demand classes"
          value={String(Object.keys(breakdown).length)}
          caption={formatBreakdown(breakdown)}
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Segment profiles
        </h2>
        <DataTable
          columns={columns}
          rows={profiles}
          rowKey={(row) => row.segment_id}
          emptyMessage="No segment profiles for this run."
          caption={`${profiles.length} segment(s)`}
        />
      </section>

      {state.narrative ? (
        <section className="rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg shadow-card">
          <h2 className="text-headline-md text-text-main">Narrative</h2>
          <p className="mt-stack-md whitespace-pre-line text-body-md text-text-muted">
            {state.narrative}
          </p>
        </section>
      ) : null}
    </div>
  );
}

function formatNumber(value: number | undefined): string {
  if (value === undefined) return "—";
  return value.toFixed(2);
}

function formatBreakdown(breakdown: Record<string, number>): string {
  const entries = Object.entries(breakdown);
  if (entries.length === 0) return "—";
  return entries.map(([k, v]) => `${k}: ${v}`).join(", ");
}
