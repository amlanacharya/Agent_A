import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { GenericSurfacePage } from "@/pages/SurfacePage";

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
      { path: "surfaces/:name/:runId", element: <GenericSurfacePage /> },
    ],
  },
]);
