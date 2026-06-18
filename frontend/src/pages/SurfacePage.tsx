import { useParams } from "react-router-dom";
import { PageHeader } from "@/components/PageHeader";
import { useSurface } from "@/api/hooks";
import type { SurfaceName } from "@/api/client";

/**
 * GenericSurfacePage — fallback route for surfaces that don't have
 * a bespoke implementation yet (CB5–CB12 add bespoke pages). Renders
 * the raw state as JSON inside a card so the route is still useful
 * during development.
 */
export function GenericSurfacePage(): JSX.Element {
  const params = useParams<{ name: string; runId: string }>();
  const name = (params.name ?? "") as SurfaceName;
  const runId = params.runId ?? "";

  const surface = useSurface(name, runId);

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow={name.replace(/_/g, " ").toUpperCase()}
        title={`${name} · ${runId}`}
        subtitle="Generic surface view. CB5–CB12 replace this with bespoke layouts for each surface kind."
      />
      {surface.isLoading ? (
        <p className="text-body-md text-text-muted">Loading surface state…</p>
      ) : surface.isError ? (
        <p className="rounded-md border border-critical-rose/30 bg-critical-rose/10 p-stack-md text-body-md text-critical-rose">
          {String(surface.error)}
        </p>
      ) : surface.data ? (
        <pre className="overflow-x-auto rounded-md border border-border-slate bg-surface-container-lowest p-stack-lg font-mono text-data-mono text-text-main shadow-card">
          {JSON.stringify(surface.data.state, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
