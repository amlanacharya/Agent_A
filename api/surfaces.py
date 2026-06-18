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
from typing import Any

from api.models import SurfaceName, SurfaceSnapshot


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
    "CockpitSurface",
    "CockpitStateProvider",
    "DuplicateSurfaceError",
    "MissionControlSurface",
    "MlopsMonitorSurface",
    "SurfaceError",
    "SurfaceRegistry",
    "UnknownSurfaceError",
)
