"""Phase 8 CB3: pure-function plot generation engine.

The seven plot kinds the plan calls for, all rendered as
PNG bytes via a small stdlib renderer (no matplotlib). The
renderer is a pure function of a pixels callback; the
per-kind generators in this module build the callback
from the typed ``CockpitPlotRequest.params`` and return a
typed ``PlotResponse``.

The seven kinds:

* ``demand_curve`` — actual + forecast over time
* ``sparsity`` — ADI / CV² scatter per series
* ``anomalies`` — point flags over time (spikes / drops)
* ``forecast_band`` — fan chart around a point forecast
* ``backtest`` — actual vs forecast across folds
* ``feature_importance`` — bar chart of feature gains
* ``drift_chart`` — per-segment MASE over time

Design:

* **Pure functions.** Every kind generator is a pure
  function of ``(CockpitPlotRequest) -> PlotResponse``.
  Same input -> same bytes. No I/O, no LLM, no globals.
* **Pure-Python PNG via stdlib.** The renderer uses
  ``struct`` + ``zlib`` to emit a real PNG. The seam is
  open for a future matplotlib-backed implementation
  to drop in behind the same ``render_png`` function.
* **Param validation is the engine's job, not the HTTP
  layer's.** The HTTP layer validates the kind (Pydantic
  Literal); the engine validates the per-kind params.
  A missing or malformed param raises a typed
  ``ValueError`` the HTTP layer translates to a 4xx.
* **Default dimensions are 800 x 400.** The cockpit can
  override per-request via ``params["width"]`` /
  ``params["height"]``; the engine clamps to
  ``[MIN_DIM, MAX_DIM]`` so a malicious client cannot
  request a 100k x 100k image.
"""

from __future__ import annotations

import base64
import struct
import zlib
from collections.abc import Callable
from typing import Any

from api.models import CockpitPlotRequest, PlotResponse


# Default plot dimensions (cockpit can override per-request).
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 400
MIN_DIM = 64
MAX_DIM = 4096

# Color palette (RGB triples). The palette is small and
# deterministic so the PNG bytes are stable across runs.
BG_COLOR = (255, 255, 255)
GRID_COLOR = (220, 220, 220)
AXIS_COLOR = (60, 60, 60)
ACTUAL_COLOR = (40, 80, 160)
FORECAST_COLOR = (220, 100, 40)
BAND_COLOR = (220, 100, 40)
ANOMALY_COLOR = (200, 30, 30)
FEATURE_COLOR = (80, 130, 200)
DRIFT_PALETTE = (
    (40, 80, 160),
    (220, 100, 40),
    (40, 160, 80),
    (180, 80, 180),
    (200, 150, 30),
)


# ---------------------------------------------------------------------------
# Pure-Python PNG renderer
# ---------------------------------------------------------------------------


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk (length + type + data + CRC)."""
    length = struct.pack(">I", len(data))
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return length + chunk_type + data + struct.pack(">I", crc)


def render_png(
    *,
    width: int,
    height: int,
    pixels: Callable[[int, int], tuple[int, int, int]],
) -> bytes:
    """Render a width x height PNG whose pixels come from ``pixels(x, y)``.

    The callback returns an ``(r, g, b)`` triple for each
    ``(x, y)`` coordinate. ``(0, 0)`` is the top-left; the
    renderer flips Y internally so the callback sees a
    conventional top-down coordinate system.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"width and height must be positive (got {width} x {height})"
        )
    # Build the IDAT scanlines. Each scanline starts with a
    # filter byte (0 = None) followed by RGB triples.
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        row.append(0)  # filter type
        for x in range(width):
            r, g, b = pixels(x, y)
            # Clamp to 0..255 so a buggy callback cannot
            # produce a malformed PNG.
            row.append(max(0, min(255, r)))
            row.append(max(0, min(255, g)))
            row.append(max(0, min(255, b)))
        rows.append(bytes(row))
    raw = b"".join(rows)
    idat = zlib.compress(raw, 6)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return sig + _png_chunk(b"IHDR", ihdr_data) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


# Sentinel raised when the renderer is asked for an out-of-range
# dimension. The HTTP layer translates this to a 4xx response.
class PNGSignatureError(ValueError):
    """Raised when a PNG signature check fails (defensive)."""


# ---------------------------------------------------------------------------
# Dimension + palette helpers
# ---------------------------------------------------------------------------


def _resolve_dimensions(params: dict[str, Any]) -> tuple[int, int]:
    """Read width / height from params, clamping to ``[MIN_DIM, MAX_DIM]``."""
    width = int(params.get("width", DEFAULT_WIDTH))
    height = int(params.get("height", DEFAULT_HEIGHT))
    width = max(MIN_DIM, min(MAX_DIM, width))
    height = max(MIN_DIM, min(MAX_DIM, height))
    return width, height


def _make_canvas(
    width: int,
    height: int,
    bg: tuple[int, int, int] = BG_COLOR,
) -> list[list[tuple[int, int, int]]]:
    """Build a width x height pixel grid filled with ``bg``."""
    return [[bg for _ in range(width)] for _ in range(height)]


def _draw_hline(
    canvas: list[list[tuple[int, int, int]]],
    x0: int,
    x1: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a horizontal line at row ``y`` from x0..x1 inclusive."""
    height = len(canvas)
    width = len(canvas[0])
    if not (0 <= y < height):
        return
    for x in range(max(0, x0), min(width, x1 + 1)):
        canvas[y][x] = color


def _draw_vline(
    canvas: list[list[tuple[int, int, int]]],
    x: int,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a vertical line at column ``x`` from y0..y1 inclusive."""
    height = len(canvas)
    width = len(canvas[0])
    if not (0 <= x < width):
        return
    for y in range(max(0, y0), min(height, y1 + 1)):
        canvas[y][x] = color


def _draw_axes(
    canvas: list[list[tuple[int, int, int]]],
    margin: int = 40,
) -> None:
    """Draw a light frame + axes around the plot area."""
    width = len(canvas[0])
    height = len(canvas)
    # Top + bottom edges.
    _draw_hline(canvas, 0, width - 1, margin, GRID_COLOR)
    _draw_hline(canvas, 0, width - 1, height - margin - 1, GRID_COLOR)
    # Left + right edges.
    _draw_vline(canvas, margin, 0, height - 1, GRID_COLOR)
    _draw_vline(canvas, width - margin - 1, 0, height - 1, GRID_COLOR)
    # Axes (slightly darker).
    _draw_hline(canvas, margin, width - margin - 1, height - margin - 1, AXIS_COLOR)
    _draw_vline(canvas, margin, margin, height - margin - 1, AXIS_COLOR)


def _plot_line(
    canvas: list[list[tuple[int, int, int]]],
    values: list[float],
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
    margin: int = 40,
) -> None:
    """Plot a polyline of ``values`` into ``canvas``."""
    width = len(canvas[0])
    height = len(canvas)
    if not values or y_max == y_min:
        return
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    n = len(values)
    pts: list[tuple[int, int]] = []
    for i, v in enumerate(values):
        x = margin + (i * plot_w) // max(1, n - 1)
        # Map y so y_max sits at the top of the plot area.
        norm = (v - y_min) / (y_max - y_min)
        y = margin + plot_h - int(norm * plot_h)
        y = max(margin, min(height - margin - 1, y))
        pts.append((x, y))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        _draw_line_segment(canvas, x0, y0, x1, y1, color)


def _draw_line_segment(
    canvas: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a single line segment using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _plot_pixel(canvas, x, y, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _plot_pixel(
    canvas: list[list[tuple[int, int, int]]],
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Plot a single pixel if it's inside the canvas."""
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def _canvas_to_response(
    kind: str,
    canvas: list[list[tuple[int, int, int]]],
) -> PlotResponse:
    """Convert a pixel grid into a typed PlotResponse."""
    height = len(canvas)
    width = len(canvas[0])
    png = render_png(
        width=width,
        height=height,
        pixels=lambda x, y: canvas[y][x],
    )
    return PlotResponse(
        kind=kind,  # type: ignore[arg-type]
        content_type="image/png",
        bytes_b64=base64.b64encode(png).decode("ascii"),
        width=width,
        height=height,
    )


def _require_params(
    params: dict[str, Any],
    *keys: str,
) -> list[Any]:
    """Pull ``keys`` from ``params`` in order; raise ``ValueError`` if any are missing."""
    missing = [k for k in keys if k not in params]
    if missing:
        raise ValueError(f"Missing required params: {missing}")
    return [params[k] for k in keys]


# ---------------------------------------------------------------------------
# render_demand_curve
# ---------------------------------------------------------------------------


def render_demand_curve(request: CockpitPlotRequest) -> PlotResponse:
    """Plot actual + forecast over time.

    Required params: ``weeks`` (list[str]), ``actual`` (list[float]),
    ``forecast`` (list[float]). All three must have the same length.
    """
    weeks, actual, forecast = _require_params(request.params, "weeks", "actual", "forecast")
    if not (len(weeks) == len(actual) == len(forecast)):
        raise ValueError(
            f"weeks / actual / forecast must have the same length "
            f"(got {len(weeks)} / {len(actual)} / {len(forecast)})"
        )
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    all_values = list(actual) + list(forecast)
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    _plot_line(canvas, list(actual), y_min, y_max, ACTUAL_COLOR)
    _plot_line(canvas, list(forecast), y_min, y_max, FORECAST_COLOR)
    return _canvas_to_response("demand_curve", canvas)


# ---------------------------------------------------------------------------
# render_sparsity
# ---------------------------------------------------------------------------


def render_sparsity(request: CockpitPlotRequest) -> PlotResponse:
    """Plot ADI / CV² per series as a scatter.

    Required params: ``series`` (list of
    ``{series_key, adi, cv2}`` dicts). The X axis is ADI,
    the Y axis is CV². Demand class quadrants are drawn
    as light grid lines (the Syntetos-Boylan matrix from
    the EDA toolbox).
    """
    (series,) = _require_params(request.params, "series")
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    if not series:
        return _canvas_to_response("sparsity", canvas)
    # ADI / CV² ranges. ADI 0..5, CV² 0..3 are the standard
    # Syntetos-Boylan quadrants; we clamp outliers to the box.
    x_min, x_max = 0.0, 5.0
    y_min, y_max = 0.0, 3.0
    margin = 40
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    for entry in series:
        adi = float(entry.get("adi", 0.0))
        cv2 = float(entry.get("cv2", 0.0))
        # Clamp to the quadrant box.
        adi = max(x_min, min(x_max, adi))
        cv2 = max(y_min, min(y_max, cv2))
        x = margin + int((adi - x_min) / (x_max - x_min) * plot_w)
        y = margin + plot_h - int((cv2 - y_min) / (y_max - y_min) * plot_h)
        # Plot a 5x5 dot for each series.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                _plot_pixel(canvas, x + dx, y + dy, FORECAST_COLOR)
    return _canvas_to_response("sparsity", canvas)


# ---------------------------------------------------------------------------
# render_anomalies
# ---------------------------------------------------------------------------


def render_anomalies(request: CockpitPlotRequest) -> PlotResponse:
    """Plot values over time with point flags at anomaly indices.

    Required params: ``weeks`` (list[str]), ``values`` (list[float]).
    Optional params: ``flags`` (list[bool], same length as values;
    missing entries are treated as ``False``).
    """
    weeks, values = _require_params(request.params, "weeks", "values")
    flags = list(request.params.get("flags", []))
    if len(weeks) != len(values):
        raise ValueError(
            f"weeks and values must have the same length "
            f"(got {len(weeks)} / {len(values)})"
        )
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    if not values:
        return _canvas_to_response("anomalies", canvas)
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    _plot_line(canvas, list(values), y_min, y_max, ACTUAL_COLOR)
    # Mark anomalies.
    margin = 40
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    n = len(values)
    for i, flag in enumerate(flags):
        if not flag:
            continue
        x = margin + (i * plot_w) // max(1, n - 1)
        norm = (values[i] - y_min) / (y_max - y_min)
        y = margin + plot_h - int(norm * plot_h)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                _plot_pixel(canvas, x + dx, y + dy, ANOMALY_COLOR)
    return _canvas_to_response("anomalies", canvas)


# ---------------------------------------------------------------------------
# render_forecast_band
# ---------------------------------------------------------------------------


def render_forecast_band(request: CockpitPlotRequest) -> PlotResponse:
    """Plot a forecast fan chart.

    Required params: ``weeks`` (list[str]), ``forecast`` (list[float]),
    ``lower`` (list[float]), ``upper`` (list[float]). All four must
    have the same length.
    """
    weeks, forecast, lower, upper = _require_params(
        request.params, "weeks", "forecast", "lower", "upper"
    )
    if not (len(weeks) == len(forecast) == len(lower) == len(upper)):
        raise ValueError(
            "weeks / forecast / lower / upper must all have the same length"
        )
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    all_values = list(lower) + list(upper) + list(forecast)
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    _plot_line(canvas, list(lower), y_min, y_max, BAND_COLOR)
    _plot_line(canvas, list(upper), y_min, y_max, BAND_COLOR)
    _plot_line(canvas, list(forecast), y_min, y_max, FORECAST_COLOR)
    return _canvas_to_response("forecast_band", canvas)


# ---------------------------------------------------------------------------
# render_backtest
# ---------------------------------------------------------------------------


def render_backtest(request: CockpitPlotRequest) -> PlotResponse:
    """Plot actual vs forecast across folds.

    Required params: ``folds`` (list[str]), ``actual`` (list[float]),
    ``forecast`` (list[float]). All three must have the same length.
    """
    folds, actual, forecast = _require_params(
        request.params, "folds", "actual", "forecast"
    )
    if not (len(folds) == len(actual) == len(forecast)):
        raise ValueError(
            "folds / actual / forecast must all have the same length"
        )
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    all_values = list(actual) + list(forecast)
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    _plot_line(canvas, list(actual), y_min, y_max, ACTUAL_COLOR)
    _plot_line(canvas, list(forecast), y_min, y_max, FORECAST_COLOR)
    return _canvas_to_response("backtest", canvas)


# ---------------------------------------------------------------------------
# render_feature_importance
# ---------------------------------------------------------------------------


def render_feature_importance(request: CockpitPlotRequest) -> PlotResponse:
    """Plot a bar chart of feature importances.

    Required params: ``features`` (list of ``{name, importance}`` dicts).
    """
    (features,) = _require_params(request.params, "features")
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    if not features:
        return _canvas_to_response("feature_importance", canvas)
    importances = [float(f.get("importance", 0.0)) for f in features]
    max_importance = max(importances) or 1.0
    n = len(features)
    margin = 40
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = max(2, plot_w // max(1, n))
    for i, importance in enumerate(importances):
        x0 = margin + i * bar_w
        bar_h = int((importance / max_importance) * plot_h)
        for x in range(x0, min(width - margin, x0 + bar_w - 1)):
            for y in range(height - margin - 1 - bar_h, height - margin - 1):
                _plot_pixel(canvas, x, y, FEATURE_COLOR)
    return _canvas_to_response("feature_importance", canvas)


# ---------------------------------------------------------------------------
# render_drift_chart
# ---------------------------------------------------------------------------


def render_drift_chart(request: CockpitPlotRequest) -> PlotResponse:
    """Plot per-segment MASE over time.

    Required params: ``runs`` (list[str]), ``segments`` (dict of
    segment_id -> list[float], one entry per run).
    """
    runs, segments = _require_params(request.params, "runs", "segments")
    width, height = _resolve_dimensions(request.params)
    canvas = _make_canvas(width, height)
    _draw_axes(canvas)
    if not runs or not segments:
        return _canvas_to_response("drift_chart", canvas)
    all_values: list[float] = []
    for values in segments.values():
        all_values.extend(values)
    if not all_values:
        return _canvas_to_response("drift_chart", canvas)
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    for i, (segment_id, values) in enumerate(segments.items()):
        color = DRIFT_PALETTE[i % len(DRIFT_PALETTE)]
        if len(values) != len(runs):
            raise ValueError(
                f"segment {segment_id!r} has {len(values)} values; "
                f"expected {len(runs)} (one per run)"
            )
        _plot_line(canvas, list(values), y_min, y_max, color)
    return _canvas_to_response("drift_chart", canvas)


# ---------------------------------------------------------------------------
# Per-kind dispatch (used by the engine)
# ---------------------------------------------------------------------------


_PER_KIND = {
    "demand_curve": render_demand_curve,
    "sparsity": render_sparsity,
    "anomalies": render_anomalies,
    "forecast_band": render_forecast_band,
    "backtest": render_backtest,
    "feature_importance": render_feature_importance,
    "drift_chart": render_drift_chart,
}


def render_kind(request: CockpitPlotRequest) -> PlotResponse:
    """Dispatch to the per-kind generator. Used by the engine.

    Raises :class:`UnknownPlotKindError` (defined in
    ``api.plot_engine``) if the kind is not implemented.
    """
    from api.plot_engine import UnknownPlotKindError  # local import to avoid cycle

    handler = _PER_KIND.get(request.kind)
    if handler is None:
        raise UnknownPlotKindError(str(request.kind))
    return handler(request)


__all__ = (
    "render_anomalies",
    "render_backtest",
    "render_demand_curve",
    "render_drift_chart",
    "render_feature_importance",
    "render_forecast_band",
    "render_kind",
    "render_png",
    "render_sparsity",
    # Sentinel
    "PNGSignatureError",
)
