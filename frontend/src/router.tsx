import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { GenericSurfacePage } from "@/pages/SurfacePage";
import { MissionControl } from "@/pages/MissionControl";

/**
 * Router — CB4's routes. CB5–CB12 add bespoke pages and slot them
 * into ``children`` below.
 *
 * Path conventions:
 * - ``/`` → redirect to the default surface (Mission Control).
 * - ``/surfaces/:name/:runId`` → the per-surface page. CB5–CB12
 *   replace the generic fallback with bespoke components; the path
 *   shape stays stable so deep links and the top-nav menu keep
 *   working across the phase.
 */
export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/surfaces/mission_control/dev-run" replace /> },
      {
        path: "surfaces/mission_control/:runId",
        element: <MissionControlRoute />,
      },
      { path: "surfaces/:name/:runId", element: <GenericSurfacePage /> },
    ],
  },
]);

/**
 * Tiny wrapper that extracts :runId and hands it to <MissionControl>.
 * Keeping the router declaration lean — the heavy lifting lives in
 * MissionControl itself.
 */
import { useParams } from "react-router-dom";

function MissionControlRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <MissionControl runId={runId} />;
}
