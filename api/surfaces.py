"""Phase 8 CB4: CockpitSurface interface + the first two surfaces.

The cockpit UI consumes ``SurfaceSnapshot``s — one per
``SurfaceName`` — from the FastAPI router. Each surface is a
pure function of (run_id, optional context) returning a
typed snapshot. The ``SurfaceRegistry`` is the typed
dispatch seam: the router calls ``registry.render(surface,
run_id)`` and the registry routes to the right surface.

The two surfaces in this module:

* ``MissionControlSurface`` — reads the platform's live
  state (the 7 fields ``cockpit_state.CockpitState`` already
  exposes) and surfaces them in the mission-control surface.
* ``MlopsMonitorSurface`` — reads the four Phase 7 markdown
  artifacts from ``outputs/{run_id}/`` and surfaces their
  content (or ``None`` if the artifact does not exist yet).

Design:

* **Pure function, no I/O at the interface.** The surface
  ``render`` method takes a ``run_id`` and returns a
  ``SurfaceSnapshot``. The I/O (reading the cockpit state,
  reading the markdown files) happens in the surface's
  provider / constructor — the interface is pure.
* **Provider injection.** The cockpit state and the
  artifacts root are passed in at construction time so the
  surface is unit-testable without monkey-patching the
  filesystem. The FastAPI router (CB8) wires the production
  providers.
* **SurfaceRegistry is the typed dispatch seam.** A future
  external surface (a remote service, a third-party
  dashboard) can register itself behind the same
  ``CockpitSurface`` interface without changing the
  FastAPI router.
* **Duplicate registration is a programming error.** The
  registry raises a typed ``DuplicateSurfaceError`` rather
  than silently overwriting — the surface name is the
  contract and a collision means a misconfigured router.
* **Unknown surface is a 404.** The registry raises a
  typed ``UnknownSurfaceError`` rather than returning an
  empty snapshot — the FastAPI router translates this to
  a 404.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.models import SurfaceName, SurfaceSnapshot

if TYPE_CHECKING:
    import pandas as pd
    from forecasting.contracts import (
        EDAReport,
        FeatureFlags,
        ForecastHarnessReport,
        FoundryReport,
        ReplenishmentRecommendation,
        SegmentMap,
        SeriesDemandProfile,
    )


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


class SurfaceError(Exception):
    """Base class for surface registry errors."""


class UnknownSurfaceError(SurfaceError):
    """The requested surface name is not registered.

    The FastAPI router translates this to a 404. The cockpit
    UI shows a 'this surface is not yet wired' widget.
    """

    def __init__(self, surface: str) -> None:
        super().__init__(f"Unknown cockpit surface: {surface!r}")
        self.surface = surface


class DuplicateSurfaceError(SurfaceError):
    """The same surface name was registered twice.

    A programming error, not a runtime error: the registry
    refuses to silently overwrite an existing surface
    because a collision usually means a misconfigured
    router.
    """

    def __init__(self, surface: str) -> None:
        super().__init__(f"Surface {surface!r} is already registered")
        self.surface = surface


# ---------------------------------------------------------------------------
# CockpitSurface — the interface
# ---------------------------------------------------------------------------


class CockpitSurface(ABC):
    """The interface every cockpit surface implements.

    The contract is small on purpose: one method
    (``render``) that takes a ``run_id`` and returns a
    ``SurfaceSnapshot``. The router calls the interface;
    the in-process surfaces (CB4-CB7) implement it; a
    future external surface can drop in behind the same
    surface.
    """

    surface: SurfaceName  # set by the concrete subclass

    @abstractmethod
    def render(self, run_id: str) -> SurfaceSnapshot:
        """Render the surface for the given run."""


# ---------------------------------------------------------------------------
# MissionControlSurface
# ---------------------------------------------------------------------------


CockpitStateProvider = Callable[[str], Any]
"""Type alias for the cockpit-state provider callable.

The provider takes a ``run_id`` and returns the
``CockpitState`` for that run. The production wiring
(CB8) reads ``run_state.json`` and builds a
``CockpitState``; tests pass a lambda that returns a
fixed state.
"""


class MissionControlSurface(CockpitSurface):
    """The Mission Control surface — the platform's live state.

    Surfaces the 7 live-state fields ``cockpit_state``
    already exposes (current_step, tool_result, code
    escalation status, attempt count, verifier gate,
    approval needed, confidence / blockers) so the
    cockpit UI can show "what is the platform doing right
    now" in one place.
    """

    surface: SurfaceName = "mission_control"

    def __init__(self, *, cockpit_state_provider: CockpitStateProvider) -> None:
        self._cockpit_state = cockpit_state_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        state = self._cockpit_state(run_id)
        return SurfaceSnapshot(
            run_id=run_id,
            surface="mission_control",
            state=state.to_public_dict(),
        )


# ---------------------------------------------------------------------------
# MlopsMonitorSurface
# ---------------------------------------------------------------------------


# The four Phase 7 markdown artifacts the surface reads.
# The set is closed (the plan's artifact checklist); a new
# artifact is a deliberate addition to this tuple and a
# matching update to the writer (CB6) + plan doc.
_MONITORING_ARTIFACTS = (
    "MONITORING_REPORT.md",
    "DRIFT_REPORT.md",
    "OVERRIDE_ANALYSIS.md",
    "MODEL_HEALTH.md",
)


class MlopsMonitorSurface(CockpitSurface):
    """The MLOps Monitor surface — the four Phase 7 markdown artifacts.

    Reads each artifact from ``outputs/{run_id}/`` and
    surfaces its content (or ``None`` if the artifact
    does not exist yet — the surface shows 'no report
    yet' rather than failing). The cockpit renders the
    four artifacts as a tabbed view under the MLOps
    Monitor tab.
    """

    surface: SurfaceName = "mlops_monitor"

    def __init__(self, *, artifacts_root: Path) -> None:
        self._artifacts_root = Path(artifacts_root)

    def render(self, run_id: str) -> SurfaceSnapshot:
        run_dir = self._artifacts_root / run_id
        state: dict[str, str | None] = {}
        for filename in _MONITORING_ARTIFACTS:
            path = run_dir / filename
            if path.exists():
                state[filename] = path.read_text(encoding="utf-8")
            else:
                state[filename] = None
        return SurfaceSnapshot(
            run_id=run_id,
            surface="mlops_monitor",
            state=state,
        )


# ---------------------------------------------------------------------------
# SurfaceRegistry
# ---------------------------------------------------------------------------


# The head preview is capped at 50 rows. A larger table
# would bloat the FastAPI response and slow the cockpit
# render; the cockpit can fetch a larger slice via a
# future ``?offset=`` parameter.
_TABLE_HEAD_LIMIT = 50


EDAReportProvider = Callable[[str], "EDAReport | None"]
"""Type alias for the EDA-report provider callable.

The provider takes a ``run_id`` and returns the
``EDAReport`` for that run, or ``None`` if the run has
not yet reached the EDA phase. The production wiring
(CB8) reads the EDA report from ``outputs/{run_id}/``
or reconstructs it from ``run_state.json``; tests pass
a lambda that returns a fixed report.
"""

CanonicalTableProvider = Callable[[str], "pd.DataFrame | None"]
"""Type alias for the canonical-table provider callable.

The provider takes a ``run_id`` and returns the
post-Preflight canonical demand table for that run,
or ``None`` if the run has not yet reached Preflight.
"""

SegmentMapProvider = Callable[[str], "SegmentMap | None"]
"""Type alias for the segment-map provider callable.

The provider takes a ``run_id`` and returns the
``SegmentMap`` for that run, or ``None`` if not
yet assigned.
"""

FeatureFlagsProvider = Callable[[str], "dict[str, FeatureFlags] | None"]
"""Type alias for the per-series FeatureFlags provider.

Returns a ``{series_key: FeatureFlags}`` dict, or
``None`` if the run has not yet built feature flags.
"""

SeriesProfilesProvider = Callable[[str], "list[SeriesDemandProfile] | None"]
"""Type alias for the per-series EDA profile provider."""

HarnessReportProvider = Callable[[str], "ForecastHarnessReport | None"]
"""Type alias for the Phase 4 forecast harness report provider."""

FoundryReportProvider = Callable[[str], "FoundryReport | None"]
"""Type alias for the Phase 4 Foundry report provider."""

ReplenishmentRecommendationsProvider = Callable[[str], "list[ReplenishmentRecommendation] | None"]
"""Type alias for the Phase 5 replenishment-batch provider.

Returns a list of per-series recommendations, or
``None`` if the run has not yet reached the
replenishment phase.
"""


class DataHealthSurface(CockpitSurface):
    """The Data Health surface — the Phase 2 EDA report summary.

    Surfaces the headline numbers the cockpit needs to
    render the data-health widget: series count, segment
    count, demand-class breakdown, per-segment profiles,
    and the EDA narrative. The full EDA report is also
    available via the per-segment / per-series fields
    so the cockpit can drill down.
    """

    surface: SurfaceName = "data_health"

    def __init__(self, *, eda_report_provider: EDAReportProvider) -> None:
        self._eda_report_provider = eda_report_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        report = self._eda_report_provider(run_id)
        if report is None:
            return SurfaceSnapshot(
                run_id=run_id,
                surface="data_health",
                state={
                    "series_count": 0,
                    "segment_count": 0,
                    "segment_profiles": [],
                    "demand_class_breakdown": {},
                    "narrative": "",
                },
            )
        # The per-segment profiles serialize via model_dump so the
        # response is JSON-safe over HTTP. The demand-class
        # breakdown is the union of per-segment breakdowns.
        demand_class_breakdown: dict[str, int] = {}
        for profile in report.segment_profiles:
            for sb_class, count in profile.demand_class_distribution.items():
                demand_class_breakdown[sb_class] = (
                    demand_class_breakdown.get(sb_class, 0) + count
                )
        return SurfaceSnapshot(
            run_id=run_id,
            surface="data_health",
            state={
                "series_count": len(report.series_profiles),
                "segment_count": len(report.segment_profiles),
                "segment_profiles": [
                    profile.model_dump() for profile in report.segment_profiles
                ],
                "demand_class_breakdown": demand_class_breakdown,
                "narrative": report.narrative,
            },
        )


class CanonicalTableBuilderSurface(CockpitSurface):
    """The Canonical Table Builder surface — the post-Preflight demand table.

    Surfaces the head (first ``_TABLE_HEAD_LIMIT`` rows) +
    the column list + the row count + the per-segment
    series count. The cockpit shows the head as a preview
    table; the segment widget shows the per-segment
    series count.
    """

    surface: SurfaceName = "canonical_table_builder"

    def __init__(
        self,
        *,
        canonical_table_provider: CanonicalTableProvider,
        segment_map_provider: SegmentMapProvider | None = None,
    ) -> None:
        self._canonical_table_provider = canonical_table_provider
        self._segment_map_provider = segment_map_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        df = self._canonical_table_provider(run_id)
        if df is None or df.empty:
            state: dict[str, Any] = {
                "row_count": 0,
                "column_count": 0,
                "columns": [],
                "head": [],
                "segments": [],
            }
            return SurfaceSnapshot(
                run_id=run_id,
                surface="canonical_table_builder",
                state=state,
            )
        head_records = df.head(_TABLE_HEAD_LIMIT).to_dict(orient="records")
        segments_state: list[dict[str, Any]] = []
        if self._segment_map_provider is not None:
            segment_map = self._segment_map_provider(run_id)
            if segment_map is not None:
                segments_state = [
                    {
                        "segment_id": seg.segment_id,
                        "label": seg.label,
                        "series_count": len(seg.series_keys),
                    }
                    for seg in segment_map.segments
                ]
        return SurfaceSnapshot(
            run_id=run_id,
            surface="canonical_table_builder",
            state={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "head": head_records,
                "segments": segments_state,
            },
        )


class EdaExplorerSurface(CockpitSurface):
    """The EDA Explorer surface — per-series EDA drill-down.

    Surfaces the ``series_profiles`` from the Phase 2
    EDA report (per-series ADI / CV² / trend /
    seasonality / demand class) plus the union
    demand-class distribution. The cockpit's EDA
    Explorer lets the planner drill into one series
    at a time.
    """

    surface: SurfaceName = "eda_explorer"

    def __init__(self, *, eda_report_provider: EDAReportProvider) -> None:
        self._eda_report_provider = eda_report_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        report = self._eda_report_provider(run_id)
        if report is None:
            return SurfaceSnapshot(
                run_id=run_id,
                surface="eda_explorer",
                state={
                    "series_count": 0,
                    "series_profiles": [],
                    "demand_class_distribution": {},
                },
            )
        demand_class_distribution: dict[str, int] = {}
        for profile in report.segment_profiles:
            for sb_class, count in profile.demand_class_distribution.items():
                demand_class_distribution[sb_class] = (
                    demand_class_distribution.get(sb_class, 0) + count
                )
        return SurfaceSnapshot(
            run_id=run_id,
            surface="eda_explorer",
            state={
                "series_count": len(report.series_profiles),
                "series_profiles": [
                    profile.model_dump() for profile in report.series_profiles
                ],
                "demand_class_distribution": demand_class_distribution,
            },
        )


class FeatureFactorySurface(CockpitSurface):
    """The Feature Factory surface — per-series FeatureFlags + recommendations.

    Surfaces the per-series ``FeatureFlags`` (which
    feature families are enabled) plus the
    recommended-models list per series. The cockpit's
    Feature Factory widget shows the config + a link
    to the feature-importance chart.
    """

    surface: SurfaceName = "feature_factory"

    def __init__(
        self,
        *,
        feature_flags_provider: FeatureFlagsProvider,
        series_profiles_provider: SeriesProfilesProvider | None = None,
    ) -> None:
        self._feature_flags_provider = feature_flags_provider
        self._series_profiles_provider = series_profiles_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        flags = self._feature_flags_provider(run_id) or {}
        state: dict[str, Any] = {
            series_key: flags[series_key].model_dump()
            for series_key in flags
        }
        if self._series_profiles_provider is not None:
            profiles = self._series_profiles_provider(run_id) or []
            state["recommended_models_per_series"] = {
                profile.series_key: list(profile.recommended_models)
                for profile in profiles
            }
        return SurfaceSnapshot(
            run_id=run_id,
            surface="feature_factory",
            state=state,
        )


class ModelArenaSurface(CockpitSurface):
    """The Model Arena surface — the Phase 4 forecast harness leaderboard.

    Surfaces the per-model scorecards (model_family,
    series_key, mase, bias), the ensemble weights, the
    frequently-promoted list, and the never-surfaced
    list (fit-failed families). The cockpit's Model
    Arena is the per-run leaderboard the planner reads
    when deciding which model to promote.
    """

    surface: SurfaceName = "model_arena"

    def __init__(self, *, harness_report_provider: HarnessReportProvider) -> None:
        self._harness_report_provider = harness_report_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        report = self._harness_report_provider(run_id)
        if report is None or report.ensemble is None:
            return SurfaceSnapshot(
                run_id=run_id,
                surface="model_arena",
                state={
                    "scorecard_count": 0,
                    "scorecards": [],
                    "ensemble_weights": {},
                    "frequently_promoted": [],
                    "never_surfaced": [],
                },
            )
        return SurfaceSnapshot(
            run_id=run_id,
            surface="model_arena",
            state={
                "scorecard_count": len(report.scorecards),
                "scorecards": [
                    card.model_dump() for card in report.scorecards
                ],
                "ensemble_weights": dict(report.ensemble.weights),
                "frequently_promoted": list(
                    report.ensemble.frequently_promoted
                ),
                "never_surfaced": list(report.ensemble.never_surfaced),
            },
        )


class ForecastReviewSurface(CockpitSurface):
    """The Forecast Review surface — the per-series Phase 4 forecast outcome.

    Surfaces the per-series ``SeriesResult`` (best
    model, MASE target, target met, demand class) plus
    the overall MASE / target-met fraction and the
    Foundry narrative. The cockpit's Forecast Review
    is the planner's per-series drill-down.
    """

    surface: SurfaceName = "forecast_review"

    def __init__(self, *, foundry_report_provider: FoundryReportProvider) -> None:
        self._foundry_report_provider = foundry_report_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        report = self._foundry_report_provider(run_id)
        if report is None:
            return SurfaceSnapshot(
                run_id=run_id,
                surface="forecast_review",
                state={
                    "overall_mase": 0.0,
                    "target_met_fraction": 0.0,
                    "series_results": [],
                    "narrative": "",
                },
            )
        return SurfaceSnapshot(
            run_id=run_id,
            surface="forecast_review",
            state={
                "overall_mase": report.overall_mase,
                "target_met_fraction": report.target_met_fraction,
                "series_results": [
                    result.model_dump() for result in report.series_results
                ],
                "narrative": report.narrative,
            },
        )


class ReplenishmentBoardSurface(CockpitSurface):
    """The Replenishment Board surface — the Phase 5 batch summary.

    Surfaces the per-series
    ``ReplenishmentRecommendation`` (lead time, safety
    stock, ROP, target inventory, current inventory,
    open POs, order quantity, approval tier) plus the
    batch rollup (count, total order quantity, per-tier
    breakdown). The cockpit's Replenishment Board is
    the planner's approval widget.
    """

    surface: SurfaceName = "replenishment_board"

    def __init__(self, *, recommendations_provider: ReplenishmentRecommendationsProvider) -> None:
        self._recommendations_provider = recommendations_provider

    def render(self, run_id: str) -> SurfaceSnapshot:
        recommendations = self._recommendations_provider(run_id) or []
        # The per-tier breakdown is a count of how many
        # recommendations fell into each tier. The total
        # order quantity is the sum across the batch.
        approval_tier_breakdown: dict[str, int] = {}
        total_order_quantity = 0.0
        for rec in recommendations:
            tier = str(rec.approval_tier)
            approval_tier_breakdown[tier] = approval_tier_breakdown.get(tier, 0) + 1
            total_order_quantity += float(rec.order_quantity)
        return SurfaceSnapshot(
            run_id=run_id,
            surface="replenishment_board",
            state={
                "recommendation_count": len(recommendations),
                "recommendations": [
                    rec.model_dump() for rec in recommendations
                ],
                "total_order_quantity": total_order_quantity,
                "approval_tier_breakdown": approval_tier_breakdown,
            },
        )


class LearningJournalSurface(CockpitSurface):
    """The Learning Journal surface — the Phase 1 workspace markdown.

    Surfaces the six workspace markdown artifacts the
    cockpit renders (LEARNINGS.md, DECISIONS.md,
    ASSUMPTIONS.md, RUNBOOK.md, MODEL_REGISTRY.md,
    PROMOTION_DECISIONS.md) plus a card-lifecycle summary
    (active / retired card counts parsed from
    LEARNINGS.md).

    Missing artifacts surface as ``None`` so the
    cockpit renders an empty-state widget rather than
    a 404. A workspace directory that does not exist
    yet is also handled (no crash).
    """

    surface: SurfaceName = "learning_journal"

    # The six workspace artifacts. The set is closed
    # (the Phase 1 plan's artifact checklist); a new
    # artifact is a deliberate addition to this tuple
    # and a matching update to the Phase 1 workspace.
    _WORKSPACE_FILES = (
        "LEARNINGS.md",
        "DECISIONS.md",
        "ASSUMPTIONS.md",
        "RUNBOOK.md",
        "MODEL_REGISTRY.md",
        "PROMOTION_DECISIONS.md",
    )

    def __init__(self, *, workspace_root: Path) -> None:
        self._workspace_root = Path(workspace_root)

    def render(self, run_id: str) -> SurfaceSnapshot:
        state: dict[str, object] = {}
        for filename in self._WORKSPACE_FILES:
            path = self._workspace_root / filename
            state[filename] = (
                path.read_text(encoding="utf-8")
                if path.exists()
                else None
            )
        # Parse the card-lifecycle summary from the
        # LEARNINGS.md front matter. The Phase 1
        # workspace writes a ``## Active`` and
        # ``## Retired`` section heading; the surface
        # counts bullet items under each.
        active_cards = 0
        retired_cards = 0
        learnings = state.get("LEARNINGS.md")
        if isinstance(learnings, str):
            active_cards, retired_cards = _parse_card_lifecycle(learnings)
        state["active_cards"] = active_cards
        state["retired_cards"] = retired_cards
        return SurfaceSnapshot(
            run_id=run_id,
            surface="learning_journal",
            state=state,
        )


def _parse_card_lifecycle(learnings_text: str) -> tuple[int, int]:
    """Parse active / retired card counts from LEARNINGS.md text.

    The Phase 1 workspace writes a ``## Active`` and
    ``## Retired`` section; each section's entries are
    bulleted. Returns ``(active_count, retired_count)``.
    """
    active_count = 0
    retired_count = 0
    section: str | None = None
    for line in learnings_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            if heading == "active":
                section = "active"
            elif heading == "retired":
                section = "retired"
            else:
                section = None
            continue
        if section is None:
            continue
        if stripped.startswith("- "):
            if section == "active":
                active_count += 1
            elif section == "retired":
                retired_count += 1
    return active_count, retired_count


class SurfaceRegistry:
    """The typed dispatch seam for cockpit surfaces.

    The router calls ``registry.render(surface, run_id)``;
    the registry routes to the registered surface. A
    future external surface registers itself behind the
    same ``CockpitSurface`` interface. Duplicate
    registration is a programming error (raises
    ``DuplicateSurfaceError``); unknown surface is a 404
    (raises ``UnknownSurfaceError``).
    """

    def __init__(self) -> None:
        self._surfaces: dict[SurfaceName, CockpitSurface] = {}

    def register(self, surface: CockpitSurface) -> None:
        """Register a surface. Duplicate names raise ``DuplicateSurfaceError``."""
        if surface.surface in self._surfaces:
            raise DuplicateSurfaceError(surface.surface)
        self._surfaces[surface.surface] = surface

    def render(self, surface: SurfaceName, run_id: str) -> SurfaceSnapshot:
        """Render the named surface for the given run."""
        impl = self._surfaces.get(surface)
        if impl is None:
            raise UnknownSurfaceError(surface)
        return impl.render(run_id)

    def list_surfaces(self) -> list[SurfaceName]:
        """Return the set of registered surface names (for the UI menu)."""
        return sorted(self._surfaces)


__all__ = (
    "CanonicalTableBuilderSurface",
    "CanonicalTableProvider",
    "CockpitSurface",
    "CockpitStateProvider",
    "DataHealthSurface",
    "DuplicateSurfaceError",
    "EdaExplorerSurface",
    "EDAReportProvider",
    "FeatureFactorySurface",
    "FeatureFlagsProvider",
    "ForecastReviewSurface",
    "FoundryReportProvider",
    "HarnessReportProvider",
    "LearningJournalSurface",
    "MissionControlSurface",
    "MlopsMonitorSurface",
    "ModelArenaSurface",
    "ReplenishmentBoardSurface",
    "ReplenishmentRecommendationsProvider",
    "SegmentMapProvider",
    "SeriesProfilesProvider",
    "SurfaceError",
    "SurfaceRegistry",
    "UnknownSurfaceError",
)
