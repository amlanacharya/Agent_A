import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { GenericSurfacePage } from "@/pages/SurfacePage";
import { MissionControl } from "@/pages/MissionControl";
import { DataHealth } from "@/pages/DataHealth";
import { EdaExplorer } from "@/pages/EdaExplorer";
import { FeatureFactory } from "@/pages/FeatureFactory";
import { ModelArena } from "@/pages/ModelArena";
import { ForecastReview } from "@/pages/ForecastReview";
import { ReplenishmentBoard } from "@/pages/ReplenishmentBoard";
import { LearningJournal } from "@/pages/LearningJournal";
import { MlopsMonitor } from "@/pages/MlopsMonitor";
import { RunConsole } from "@/pages/RunConsole";

/**
 * Router — CB4's routes. CB5–CB12 add bespoke pages and slot them
 * into ``children`` below. CB6 (Phase 10) adds the ``/`` (upload)
 * and ``/runs/:runId/console`` (chat + advance) routes for the
 * cockpit driver.
 *
 * Path conventions:
 * - ``/`` → RunConsole (upload form when no :runId).
 * - ``/runs/:runId/console`` → RunConsole (chat + advance).
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
      { index: true, element: <RunConsole /> },
      {
        path: "runs/:runId/console",
        element: <RunConsoleRoute />,
      },
      {
        path: "surfaces/mission_control/:runId",
        element: <MissionControlRoute />,
      },
      {
        path: "surfaces/data_health/:runId",
        element: <DataHealthRoute />,
      },
      {
        path: "surfaces/eda_explorer/:runId",
        element: <EdaExplorerRoute />,
      },
      {
        path: "surfaces/feature_factory/:runId",
        element: <FeatureFactoryRoute />,
      },
      {
        path: "surfaces/model_arena/:runId",
        element: <ModelArenaRoute />,
      },
      {
        path: "surfaces/forecast_review/:runId",
        element: <ForecastReviewRoute />,
      },
      {
        path: "surfaces/replenishment_board/:runId",
        element: <ReplenishmentBoardRoute />,
      },
      {
        path: "surfaces/learning_journal/:runId",
        element: <LearningJournalRoute />,
      },
      {
        path: "surfaces/mlops_monitor/:runId",
        element: <MlopsMonitorRoute />,
      },
      { path: "surfaces/:name/:runId", element: <GenericSurfacePage /> },
    ],
  },
]);

/**
 * Tiny wrappers that extract :runId and hand it to each page.
 * Keeping the router declaration lean — the heavy lifting lives in
 * the page components themselves.
 */
import { useParams } from "react-router-dom";

function RunConsoleRoute(): JSX.Element {
  // The RunConsole reads :runId from the URL itself, so we don't
  // need to pass it as a prop. Routing through this wrapper keeps
  // the route declaration shape consistent with the surface routes.
  return <RunConsole />;
}

function MissionControlRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <MissionControl runId={runId} />;
}

function DataHealthRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <DataHealth runId={runId} />;
}

function EdaExplorerRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <EdaExplorer runId={runId} />;
}

function FeatureFactoryRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <FeatureFactory runId={runId} />;
}

function ModelArenaRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <ModelArena runId={runId} />;
}

function ForecastReviewRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <ForecastReview runId={runId} />;
}

function ReplenishmentBoardRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <ReplenishmentBoard runId={runId} />;
}

function LearningJournalRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <LearningJournal runId={runId} />;
}

function MlopsMonitorRoute(): JSX.Element {
  const { runId = "dev-run" } = useParams<{ runId: string }>();
  return <MlopsMonitor runId={runId} />;
}
