"""Phase 8 CB2: PlotEngine ABC + InProcessPlotEngine default.

The plot engine is the boundary between the FastAPI surface
(the cockpit UI) and the plot-generation code (CB3). The ABC
is the platform's seam: an alternative implementation (a
remote plotting microservice, a Jupyter rendering kernel, a
third-party charting library) can drop in behind the same
interface without changing the FastAPI routers or the UI.

The interface is intentionally narrow — one method
(``render``) plus the typed error surface. The full per-kind
plot generation lives in ``api/plots.py`` (CB3); the engine
in this module is the routing + dispatch seam.

Design:

* **ABC seam, in-process default.** The same pattern as
  Phase 6's ``ApprovalGateway`` and ``Scheduler``, and Phase
  7's ``MonitorSnapshot`` consumers: a small ABC the
  production code talks to, with an in-process default the
  deployment uses today. A future external rendering
  service plugs in behind the same interface.
* **Pure dispatch, no I/O at the seam.** The engine does
  not touch the filesystem or the network. The CB3 plot
  functions are pure; the engine is a typed router.
* **Typed error surface.** ``PlotEngineError`` is the base;
  ``UnknownPlotKindError`` is the typed subclass the cockpit
  surfaces as a "this kind is not implemented" widget. New
  error kinds are deliberate additions.
* **Placeholder PNG so the seam round-trips end-to-end before
  CB3.** The default ``_placeholder_png`` returns a 1x1
  transparent PNG (8-byte signature + 13-byte IHDR + checksum).
  The full per-kind generators replace the placeholder in
  CB3; the engine surface stays unchanged.
"""

from __future__ import annotations

import base64
import struct
import zlib
from abc import ABC, abstractmethod

from api.models import CockpitPlotRequest, PlotKind, PlotResponse


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


class PlotEngineError(Exception):
    """Base class for plot-engine errors."""


class UnknownPlotKindError(PlotEngineError):
    """The requested plot kind is not implemented by this engine.

    The kind is included in the message so the cockpit can
    surface it as a "this plot kind is not yet wired" widget
    without a separate code path.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(f"Unknown plot kind: {kind!r}")
        self.kind = kind


# ---------------------------------------------------------------------------
# PlotEngine ABC
# ---------------------------------------------------------------------------


class PlotEngine(ABC):
    """Interface every plot engine implements.

    The contract is small on purpose: one method (``render``)
    that takes a typed ``CockpitPlotRequest`` and returns a
    typed ``PlotResponse``. The cockpit UI talks to the
    interface; the in-process default (CB3) implements it;
    a future external rendering service can drop in behind
    the same surface.
    """

    @abstractmethod
    def render(self, request: CockpitPlotRequest) -> PlotResponse:
        """Render a plot to PNG (or SVG) bytes and return the typed response.

        Raises :class:`UnknownPlotKindError` if the request's kind
        is not implemented by this engine. The HTTP layer
        translates the typed error to a 4xx response.
        """


# ---------------------------------------------------------------------------
# Placeholder PNG (replaced by per-kind generators in CB3)
# ---------------------------------------------------------------------------


def _placeholder_png(width: int = 1, height: int = 1) -> bytes:
    """Build a tiny valid PNG with the given dimensions.

    The bytes are a real PNG (8-byte signature + IHDR + IDAT +
    IEND chunks), so the cockpit can render the placeholder
    inline before CB3 lands the per-kind generators. The
    engine surface stays unchanged across the CB2 -> CB3
    transition.
    """
    # PNG signature.
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 13 bytes (width, height, bit depth, color type, ...)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT: a single scanline of all-transparent RGBA pixels.
    raw = b"\x00" + (b"\x00\x00\x00\x00" * (width * height))
    idat_data = zlib.compress(raw, 9)
    idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
    idat = (
        struct.pack(">I", len(idat_data))
        + b"IDAT"
        + idat_data
        + struct.pack(">I", idat_crc)
    )
    # IEND.
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# InProcessPlotEngine — default implementation
# ---------------------------------------------------------------------------


class InProcessPlotEngine(PlotEngine):
    """The in-process default implementation.

    Delegates to the per-kind generator functions in
    ``api/plots.py`` (CB3). Today (CB2) the per-kind
    functions are not yet implemented, so the engine
    returns a 1x1 placeholder PNG for every kind. The
    CB3 commit replaces the placeholder with the real
    per-kind generators; the engine surface and the
    HTTP layer do not change.
    """

    def render(self, request: CockpitPlotRequest) -> PlotResponse:
        # The full per-kind dispatch lives in CB3. Today, every
        # kind routes to a placeholder PNG so the seam
        # round-trips end-to-end. The kind validation is
        # the only thing this engine does today; CB3
        # extends it with real per-kind generators.
        if request.kind not in set(PlotKind.__args__):  # type: ignore[attr-defined]
            raise UnknownPlotKindError(str(request.kind))
        png = _placeholder_png()
        return PlotResponse(
            kind=request.kind,  # type: ignore[arg-type]
            content_type="image/png",
            bytes_b64=base64.b64encode(png).decode("ascii"),
            width=1,
            height=1,
        )


__all__ = (
    "InProcessPlotEngine",
    "PlotEngine",
    "PlotEngineError",
    "UnknownPlotKindError",
)
