/**
 * TanStack Query wrappers around the typed API client.
 *
 * Each hook owns its cache key + stale time so the surface pages
 * (CB5+) can call `useSurface(name, runId)` and trust the lifecycle.
 */

import {
  useMutation,
  useQuery,
  type UseMutationResult,
  type UseQueryResult,
} from "@tanstack/react-query";
import {
  advanceRun,
  fetchCockpitState,
  fetchSurface,
  listSurfaces,
  postMessage,
  uploadCsv,
  type AdvanceRequest,
  type AdvanceResponse,
  type CockpitStateResponse,
  type MessageRequest,
  type MessageResponse,
  type PlotResponse,
  type SurfaceName,
  type SurfaceSnapshot,
  type UploadResponse,
} from "./client";

export type SurfacesList = { surfaces: SurfaceName[] };

export const queryKeys = {
  surfaces: ["surfaces"] as const,
  surface: (name: SurfaceName, runId: string) =>
    ["surfaces", name, runId] as const,
  cockpitState: (runId: string) => ["cockpit-state", runId] as const,
};

/** Query the surface registry (10 surfaces registered by the dev launcher). */
export function useSurfaces(): UseQueryResult<SurfacesList, Error> {
  return useQuery({
    queryKey: queryKeys.surfaces,
    queryFn: () => listSurfaces(),
  });
}

/** Query a single surface snapshot for a given run. */
export function useSurface(
  name: SurfaceName,
  runId: string,
): UseQueryResult<SurfaceSnapshot, Error> {
  return useQuery({
    queryKey: queryKeys.surface(name, runId),
    queryFn: () => fetchSurface(name, runId),
    enabled: Boolean(name) && Boolean(runId),
  });
}

/**
 * Live CockpitState poll — every 5s while a run is in flight.
 *
 * The RunConsole's left rail subscribes to this query so the
 * operator sees the current_step / active_agent / blockers
 * update without polling the full surface (which is heavier).
 * ``refetchInterval: 5000`` keeps the wire cost low while still
 * catching a phase advance within ~5s. The query is gated on
 * ``runId`` so a missing URL doesn't fire a useless fetch.
 */
export function useCockpitState(
  runId: string,
  options: { enabled?: boolean; refetchInterval?: number } = {},
): UseQueryResult<CockpitStateResponse, Error> {
  const { enabled = true, refetchInterval = 5000 } = options;
  return useQuery({
    queryKey: queryKeys.cockpitState(runId),
    queryFn: () => fetchCockpitState(runId),
    enabled: Boolean(runId) && enabled,
    refetchInterval,
    // CockpitState is small + the live pulse of the cockpit; a
    // 0s stale time makes the polling actually deliver fresh data
    // on every tick instead of returning the cached previous result.
    staleTime: 0,
  });
}

/**
 * POST /uploads — multipart CSV upload mutation.
 *
 * On success the caller ``navigate``s to
 * ``/runs/${runId}/console`` so the RunConsole page takes over.
 * The mutation exposes ``isPending`` for the spinner,
 * ``error`` for the toast, and ``data`` for the response.
 */
export function useUploadCsv(): UseMutationResult<
  UploadResponse,
  Error,
  { file: File; domain: string }
> {
  return useMutation({
    mutationFn: ({ file, domain }) => uploadCsv(file, domain),
  });
}

/**
 * POST /messages — chat-loop mutation.
 *
 * On success the RunConsole's chat panel appends the new
 * ``reply`` + ``possibilities`` to the message stack. The
 * mutation result also carries the (possibly updated) RunState
 * which the page writes to local state.
 */
export function usePostMessage(): UseMutationResult<
  MessageResponse,
  Error,
  MessageRequest
> {
  return useMutation({
    mutationFn: (request) => postMessage(request),
  });
}

/**
 * POST /runs/{run_id}/advance — driver-button mutation.
 *
 * The Run Console's "Advance to next phase" button calls
 * this. On success the page reads ``advanced_to`` to decide
 * which surface to deep-link to (forecast_review /
 * replenishment_board) when the run lands at report_ready.
 * ``force`` is the operator escape hatch for the chat gate.
 */
export function useAdvanceRun(): UseMutationResult<
  AdvanceResponse,
  Error,
  { runId: string; request?: AdvanceRequest }
> {
  return useMutation({
    mutationFn: ({ runId, request }) => advanceRun(runId, request ?? {}),
  });
}

/** Re-export the PlotResponse type for surface components that render plots. */
export type { PlotResponse };
