/**
 * Placeholder App for CB2.
 *
 * CB4 replaces this with the real router (React Router v6 routes for
 * Mission Control, surface pages, run selector). The placeholder exists to
 * prove the scaffold compiles, the design tokens render, and the
 * QueryClientProvider / BrowserRouter wrap work end-to-end.
 */
export default function App(): JSX.Element {
  return (
    <main className="min-h-screen bg-background px-margin-mobile py-stack-lg md:px-margin-desktop">
      <header className="mx-auto max-w-container-max">
        <p className="font-mono text-label-caps uppercase tracking-wider text-text-muted">
          Agent A · Data Intelligence Cockpit
        </p>
        <h1 className="mt-stack-sm text-display-lg text-text-main">
          Cockpit Frontend
        </h1>
        <p className="mt-stack-md max-w-2xl text-body-lg text-text-muted">
          Phase 9 scaffold. The 9 surfaces (Mission Control, Data Health,
          EDA Explorer, Feature Factory, Model Arena, Forecast Review,
          Replenishment Board, MLOps Monitor, Learning Journal) wire up in
          CB4–CB12.
        </p>
      </header>
    </main>
  );
}
