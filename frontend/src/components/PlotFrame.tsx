import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { renderPlot } from "@/api/client";
import type { PlotKind, PlotResponse } from "@/api/client";

/**
 * PlotFrame — single-source wrapper around the FastAPI /plots
 * endpoint. Renders one of the 7 plot kinds as a base64-decoded
 * <img>, with loading + error states. CB7 uses this for the
 * 3 EDA plot kinds; CB10 + CB12 reuse it for the remaining 4.
 *
 * The actual chart library (Recharts / D3 / visx) is intentionally
 * out of scope: the platform's plot engine produces PNG bytes
 * server-side, so the frontend is a pure consumer. CB12 may add
 * a chart-library path for SVG-bearing plot kinds if the PNG
 * approach turns out to be too lossy for forecasting fan charts;
 * for now the bytes-in-bytes-out contract is the right one.
 */
export interface PlotFrameProps {
  runId: string;
  kind: PlotKind;
  /** Engine params (validated server-side; missing params → 400). */
  params?: Record<string, unknown>;
  /** Caption rendered under the image. */
  caption?: string;
  /** Tailwind height class (default h-72). */
  heightClass?: string;
}

export function PlotFrame({
  runId,
  kind,
  params = {},
  caption,
  heightClass = "h-72",
}: PlotFrameProps): JSX.Element {
  const query = usePlot(runId, kind, params);
  return (
    <figure
      className={`flex flex-col gap-stack-sm rounded-md border border-border-slate bg-surface-container-lowest p-stack-md shadow-card ${heightClass}`}
    >
      <PlotFrameContent query={query} kind={kind} />
      {caption ? (
        <figcaption className="text-body-sm text-text-muted">{caption}</figcaption>
      ) : null}
    </figure>
  );
}

function PlotFrameContent({
  query,
  kind,
}: {
  query: UseQueryResult<PlotResponse, Error>;
  kind: PlotKind;
}): JSX.Element {
  if (query.isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center text-body-sm text-text-muted">
        Loading {kind}…
      </div>
    );
  }
  if (query.isError) {
    return (
      <div className="flex flex-1 items-center justify-center rounded-md bg-critical-rose/10 p-stack-md text-body-sm text-critical-rose">
        {String(query.error)}
      </div>
    );
  }
  if (!query.data) {
    return <div className="flex flex-1 items-center justify-center text-body-sm text-text-muted">No plot.</div>;
  }
  // Build the data: URL. PNG and SVG+xml are both supported by
  // the FastAPI plot engine (api/plots.py renders to one or the
  // other depending on the kind).
  const src = `data:${query.data.content_type};base64,${query.data.bytes_b64}`;
  return (
    <img
      src={src}
      alt={kind}
      className="h-full w-full rounded-md object-contain"
      data-plot-kind={kind}
    />
  );
}

/**
 * usePlot — TanStack Query wrapper around renderPlot().
 * Lives in PlotFrame.tsx (not hooks.ts) because it's plot-specific;
 * useSurface stays the canonical surface hook.
 */
function usePlot(
  runId: string,
  kind: PlotKind,
  params: Record<string, unknown>
): UseQueryResult<PlotResponse, Error> {
  return useQuery({
    queryKey: ["plots", runId, kind, params] as const,
    queryFn: () => renderPlot({ run_id: runId, kind, params }),
    enabled: Boolean(runId),
    // Plot bytes are heavy + don't change frequently inside a
    // session; 5min stale time keeps the cockpit snappy.
    staleTime: 5 * 60_000,
    retry: 1,
  });
}
