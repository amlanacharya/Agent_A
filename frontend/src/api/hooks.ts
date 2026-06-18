/**
 * TanStack Query wrappers around the typed API client.
 *
 * Each hook owns its cache key + stale time so the surface pages
 * (CB5+) can call `useSurface(name, runId)` and trust the lifecycle.
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import {
  fetchSurface,
  listSurfaces,
  type PlotResponse,
  type SurfaceName,
  type SurfaceSnapshot,
} from "./client";

export type SurfacesList = { surfaces: SurfaceName[] };

export const queryKeys = {
  surfaces: ["surfaces"] as const,
  surface: (name: SurfaceName, runId: string) =>
    ["surfaces", name, runId] as const,
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

/** Re-export the PlotResponse type for surface components that render plots. */
export type { PlotResponse };
