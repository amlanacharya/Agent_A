import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import { PlotFrame } from "@/components/PlotFrame";
import { useSurface } from "@/api/hooks";

/**
 * FeatureFactory — CB8 bespoke surface.
 *
 * Reads useSurface("feature_factory", runId). The state is a
 * dict keyed by series_key whose value is the FeatureFlags
 * model_dump (the 8 family booleans). Plus an optional
 * recommended_models_per_series dict.
 */
export interface FeatureFactoryProps {
  runId: string;
}

interface FeatureFlags {
  lag?: boolean;
  rolling?: boolean;
  calendar?: boolean;
  price_promo?: boolean;
  stockout_availability?: boolean;
  hierarchy?: boolean;
  lifecycle?: boolean;
  intermittency?: boolean;
}

interface FeatureFactoryState {
  [seriesKey: string]:
    | FeatureFlags
    | { recommended_models_per_series?: Record<string, string[]> }
    | undefined;
}

const FAMILY_KEYS: Array<{ key: keyof FeatureFlags; label: string }> = [
  { key: "lag", label: "Lag" },
  { key: "rolling", label: "Rolling" },
  { key: "calendar", label: "Calendar" },
  { key: "price_promo", label: "Price / promo" },
  { key: "stockout_availability", label: "Stockout" },
  { key: "hierarchy", label: "Hierarchy" },
  { key: "lifecycle", label: "Lifecycle" },
  { key: "intermittency", label: "Intermittency" },
];

export function FeatureFactory({ runId }: FeatureFactoryProps): JSX.Element {
  const surface = useSurface("feature_factory", runId);
  const state = (surface.data?.state ?? {}) as FeatureFactoryState;
  const recommendedMap =
    (state["recommended_models_per_series"] as
      | Record<string, string[]>
      | undefined) ?? {};

  // Strip out the recommended_models_per_series entry; the rest
  // is FeatureFlags per series.
  const flagsBySeries: Record<string, FeatureFlags> = {};
  for (const [k, v] of Object.entries(state)) {
    if (k === "recommended_models_per_series") continue;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      flagsBySeries[k] = v as FeatureFlags;
    }
  }
  const seriesList = Object.keys(flagsBySeries);
  const enabledCounts = FAMILY_KEYS.map(({ key, label }) => {
    const enabled = seriesList.filter((s) => flagsBySeries[s]?.[key]).length;
    return { key, label, enabled, total: seriesList.length };
  });

  const columns: DataTableColumn<{ series_key: string; enabled_count: number }> = [
    {
      header: "Series",
      accessor: (r) => r.series_key,
      width: "w-32",
      sortKey: (r) => r.series_key,
    },
    {
      header: "Families enabled",
      accessor: (r) => `${r.enabled_count} / ${FAMILY_KEYS.length}`,
      numeric: true,
    },
    {
      header: "Recommended models",
      accessor: (r) => recommendedMap[r.series_key]?.join(", ") ?? "—",
    },
  ];

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Feature Factory"
        title={`Run ${runId}`}
        subtitle="Per-series FeatureFlags: which of the 8 feature families are enabled."
      />

      {surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : null}

      <section className="grid grid-cols-1 gap-stack-md md:grid-cols-4">
        <MetricCard
          label="Series"
          value={seriesList.length.toString()}
          caption="With feature config"
          active
        />
        {enabledCounts.slice(0, 3).map(({ key, label, enabled, total }) => (
          <MetricCard
            key={key}
            label={label}
            value={`${enabled} / ${total}`}
            caption="Series with this family enabled"
          />
        ))}
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Per-series families
        </h2>
        <div className="overflow-x-auto rounded-md border border-border-slate bg-surface-container-lowest shadow-card">
          <table className="w-full text-body-sm">
            <thead>
              <tr className="border-b border-border-slate bg-surface-container-low text-left">
                <th className="px-stack-md py-stack-sm font-mono text-label-caps uppercase text-text-muted">
                  Series
                </th>
                {FAMILY_KEYS.map(({ key, label }) => (
                  <th
                    key={key}
                    className="px-stack-md py-stack-sm text-center font-mono text-label-caps uppercase text-text-muted"
                  >
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {seriesList.map((series) => (
                <tr
                  key={series}
                  className="border-b border-border-slate last:border-b-0 hover:bg-surface transition-colors"
                >
                  <td className="px-stack-md py-stack-md font-mono text-data-mono text-text-main">
                    {series}
                  </td>
                  {FAMILY_KEYS.map(({ key }) => {
                    const on = !!flagsBySeries[series]?.[key];
                    return (
                      <td key={key} className="px-stack-md py-stack-md text-center">
                        <StatusChip
                          label={on ? "ON" : "off"}
                          tone={(on ? "success" : "neutral") as StatusChipTone}
                        />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Recommended models
        </h2>
        <DataTable
          columns={columns}
          rows={seriesList.map((s) => {
            const enabled_count = FAMILY_KEYS.filter((f) => flagsBySeries[s]?.[f.key]).length;
            return { series_key: s, enabled_count };
          })}
          rowKey={(r) => r.series_key}
          emptyMessage="No recommended models for this run."
        />
      </section>

      <section>
        <h2 className="mb-stack-md text-headline-md text-text-main">
          Feature importance
        </h2>
        <PlotFrame
          runId={runId}
          kind="feature_importance"
          params={{
            features: seriesList.flatMap((s) =>
              FAMILY_KEYS.filter((f) => flagsBySeries[s]?.[f.key]).map((f, i) => ({
                name: `${s} · ${f.label}`,
                importance: Math.max(0.05, 1 - i * 0.1),
              })),
            ),
          }}
          caption="Feature importance (heuristic from enabled families)"
        />
      </section>
    </div>
  );
}
