import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { useSurface } from "@/api/hooks";

/**
 * ReplenishmentBoard — CB11 bespoke surface.
 *
 * Reads useSurface("replenishment_board", runId). Renders the
 * per-series ReplenishmentRecommendation rows + the batch
 * rollup (count, total order quantity, per-tier breakdown).
 *
 * NOTE: The acknowledge action is not wired in CB11 — Phase 6
 * exposes POST /approvals/{id}/acknowledge but the prototype
 * HTML files don't exercise that interaction. The frontend
 * surfaces the action as a StatusChip in the table; the actual
 * acknowledgment call is a separate CB (or a Phase 10+ work).
 */
export interface ReplenishmentBoardProps {
  runId: string;
}

interface Recommendation {
  series_key?: string;
  lead_time_days?: number;
  forecast_std?: number;
  lead_time_demand?: number;
  safety_stock?: number;
  reorder_point?: number;
  target_inventory?: number;
  current_inventory?: number;
  open_purchase_orders?: number;
  order_quantity?: number;
  approval_tier?: string;
}

interface ReplenishmentBoardState {
  recommendation_count?: number;
  recommendations?: Recommendation[];
  total_order_quantity?: number;
  approval_tier_breakdown?: Record<string, number>;
}

export function ReplenishmentBoard({ runId }: ReplenishmentBoardProps): JSX.Element {
  const surface = useSurface("replenishment_board", runId);
  const state = (surface.data?.state ?? {}) as ReplenishmentBoardState;
  const recommendations = state.recommendations ?? [];
  const tierBreakdown = state.approval_tier_breakdown ?? {};
  const totalQty = state.total_order_quantity ?? 0;

  const columns: DataTableColumn<Recommendation>[] = [
    {
      header: "Series",
      accessor: (r) => r.series_key ?? "—",
      width: "w-32",
      sortKey: (r) => r.series_key ?? "",
    },
    {
      header: "Lead time (d)",
      accessor: (r) => r.lead_time_days ?? "—",
      numeric: true,
    },
    {
      header: "ROP",
      accessor: (r) =>
        r.reorder_point === undefined ? "—" : r.reorder_point.toFixed(0),
      numeric: true,
    },
    {
      header: "Current",
      accessor: (r) =>
        r.current_inventory === undefined ? "—" : r.current_inventory.toFixed(0),
      numeric: true,
    },
    {
      header: "Order qty",
      accessor: (r) =>
        r.order_quantity === undefined ? "—" : r.order_quantity.toFixed(0),
      numeric: true,
    },
    {
      header: "Tier",
      accessor: (r) =>
        r.approval_tier ? (
          <StatusChip label={r.approval_tier} tone={tierTone(r.approval_tier)} />
        ) : (
          "—"
        ),
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Replenishment Board"
        title={`Run ${runId}`}
        subtitle="Phase 5 batch summary. Per-series recommendations + per-tier rollup."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-4">
        <MetricCard
          label="Recommendations"
          value={(state.recommendation_count ?? 0).toString()}
          caption="Series in this batch"
          active
        />
        <MetricCard
          label="Total order qty"
          value={totalQty.toFixed(0)}
          caption="Sum across the batch"
        />
        <MetricCard
          label="High tier"
          value={(tierBreakdown["high"] ?? 0).toString()}
          caption="Requires fast review"
        />
        <MetricCard
          label="Medium / low"
          value={`${tierBreakdown["medium"] ?? 0} / ${tierBreakdown["low"] ?? 0}`}
          caption="Medium / low tier counts"
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Recommendations
        </h2>
        <DataTable
          columns={columns}
          rows={recommendations}
          rowKey={(r) => `${r.series_key ?? "x"}`}
          emptyMessage="No replenishment recommendations for this run."
          caption={`${recommendations.length} recommendation(s)`}
        />
      </section>
    </div>
  );
}

function tierTone(tier: string): StatusChipTone {
  if (tier === "high") return "critical";
  if (tier === "medium") return "warning";
  if (tier === "low") return "success";
  return "neutral";
}
