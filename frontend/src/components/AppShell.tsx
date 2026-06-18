import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";
import { useSurfaces } from "@/api/hooks";

/**
 * AppShell — the global layout that wraps every surface page.
 *
 * Layout (DESIGN.md "Layout & Spacing" + the prototype HTML files):
 * - Top nav: brand mark + horizontal surface menu (Mission Control,
 *   Data Health, etc.) + run selector.
 * - Main: <Outlet/> renders the matched route.
 *
 * The run selector defaults to `dev-run` when the /runs endpoint
 * returns an empty list (the dev launcher's canned registry uses
 * `dev-run`). The full run dropdown surfaces once the platform has
 * real runs under outputs/.
 */
export function AppShell(): JSX.Element {
  const surfaces = useSurfaces();
  const surfaceList = surfaces.data?.surfaces ?? [];

  return (
    <div className="min-h-screen bg-background font-sans text-on-surface">
      <TopNav surfaceNames={surfaceList} />
      <main className="mx-auto w-full max-w-container-max px-margin-mobile py-stack-lg md:px-margin-desktop">
        <Outlet />
      </main>
    </div>
  );
}

interface TopNavProps {
  surfaceNames: readonly string[];
}

function TopNav({ surfaceNames }: TopNavProps): JSX.Element {
  return (
    <nav className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-stack-md border-b border-border-slate bg-surface-container-lowest px-margin-mobile py-stack-md shadow-card md:px-margin-desktop">
      <div className="flex items-center gap-stack-md">
        <BrandMark />
        <div>
          <p className="font-mono text-label-caps uppercase text-text-muted">
            Agent A
          </p>
          <p className="text-body-md font-semibold text-text-main">
            Data Intelligence Cockpit
          </p>
        </div>
      </div>
      <ul className="flex flex-wrap items-center gap-stack-sm">
        {surfaceNames.map((name) => (
          <li key={name}>
            <NavLink
              to={`/surfaces/${name}/dev-run`}
              className={({ isActive }) =>
                clsx(
                  "rounded-md px-stack-md py-stack-sm text-body-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-on-primary"
                    : "text-text-muted hover:bg-surface-container hover:text-text-main",
                )
              }
            >
              {labelFor(name)}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function BrandMark(): JSX.Element {
  return (
    <div
      aria-hidden
      className="flex h-10 w-10 items-center justify-center rounded-md bg-primary font-mono text-label-caps uppercase text-on-primary"
    >
      A
    </div>
  );
}

/** Display label for a surface name (snake_case → Title Case). */
function labelFor(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
