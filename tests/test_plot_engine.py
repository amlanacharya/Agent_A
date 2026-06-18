"""Tests for Phase 8 CB3: plot generation engine.

Covers the 7 per-kind plot generators in ``api/plots.py`` plus
the pure-Python PNG renderer they all use. The engine takes a
typed ``CockpitPlotRequest`` and returns a typed ``PlotResponse``
with real (non-placeholder) PNG bytes.

The seven kinds:

* ``demand_curve`` — actual + forecast + band over time
* ``sparsity`` — ADI / CV² per series (per the Phase 2 EDA toolbox)
* ``anomalies`` — point-out flags over time (spikes / drops)
* ``forecast_band`` — fan chart around a point forecast
* ``backtest`` — actual vs forecast across folds
* ``feature_importance`` — bar chart of feature gains
* ``drift_chart`` — per-segment MASE delta over time

Design rules pinned by the tests:

* All 7 generators are pure functions of their typed inputs.
* The PNG renderer is a pure-Python stdlib implementation
  (struct + zlib + no matplotlib). The seam is open for a
  future external renderer (a third-party library, a remote
  plotting service) to drop in behind the same surface.
* The PNG bytes are a real PNG (8-byte signature + IHDR +
  IDAT + IEND chunks) so the cockpit can render them
  inline.
* Each generator returns a ``PlotResponse`` with positive
  width / height and a non-empty ``bytes_b64`` field.
"""

from __future__ import annotations

import base64
import struct
import zlib

import pytest

from api.models import CockpitPlotRequest
from api.plot_engine import InProcessPlotEngine
from api.plots import (
    PNGSignatureError,
    render_png,
    render_demand_curve,
    render_sparsity,
    render_anomalies,
    render_forecast_band,
    render_backtest,
    render_feature_importance,
    render_drift_chart,
)


# ---------------------------------------------------------------------------
# render_png — the underlying stdlib PNG renderer
# ---------------------------------------------------------------------------


def test_render_png_returns_real_png_signature() -> None:
    """The first 8 bytes of the output are the PNG signature."""
    out = render_png(
        width=10,
        height=10,
        pixels=lambda x, y: (255, 255, 255),
    )
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_png_dimensions_match_ihdr() -> None:
    """The IHDR chunk carries the requested width and height."""
    out = render_png(
        width=42,
        height=17,
        pixels=lambda x, y: (0, 0, 0),
    )
    # The IHDR data starts at byte offset 8 (sig) + 4 (length) + 4 (type) = 16.
    # First 4 bytes after the type are width, next 4 are height.
    width = struct.unpack(">I", out[16:20])[0]
    height = struct.unpack(">I", out[20:24])[0]
    assert width == 42
    assert height == 17


def test_render_png_pure_function() -> None:
    """Same inputs -> same bytes (the renderer is deterministic)."""
    fn = lambda x, y: (x * 2, y * 2, (x + y) % 256)  # noqa: E731
    out1 = render_png(width=8, height=8, pixels=fn)
    out2 = render_png(width=8, height=8, pixels=fn)
    assert out1 == out2


def test_render_png_raises_on_invalid_dimensions() -> None:
    """A zero or negative dimension is a contract violation."""
    with pytest.raises(ValueError):
        render_png(width=0, height=10, pixels=lambda x, y: (0, 0, 0))
    with pytest.raises(ValueError):
        render_png(width=10, height=0, pixels=lambda x, y: (0, 0, 0))


def test_render_png_pixels_callback_receives_xy() -> None:
    """The pixels callback is called for every (x, y) coordinate."""
    seen: list[tuple[int, int]] = []
    def fn(x, y):
        seen.append((x, y))
        return (0, 0, 0)
    render_png(width=3, height=2, pixels=fn)
    assert sorted(seen) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


# ---------------------------------------------------------------------------
# render_demand_curve
# ---------------------------------------------------------------------------


def test_render_demand_curve_returns_valid_plot_response() -> None:
    """A demand curve request renders to a valid PlotResponse."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="demand_curve",
        params={
            "weeks": ["2024-W01", "2024-W02", "2024-W03", "2024-W04"],
            "actual": [10.0, 12.0, 9.0, 11.0],
            "forecast": [10.5, 11.5, 9.5, 10.5],
        },
    )
    response = render_demand_curve(request)
    assert response.kind == "demand_curve"
    assert response.content_type == "image/png"
    assert response.bytes_b64
    assert response.width > 0
    assert response.height > 0
    raw = base64.b64decode(response.bytes_b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_demand_curve_missing_params_raises() -> None:
    """A demand curve request without weeks / actual / forecast is invalid."""
    request = CockpitPlotRequest(run_id="r1", kind="demand_curve", params={})
    with pytest.raises(ValueError):
        render_demand_curve(request)


def test_render_demand_curve_mismatched_lengths_raises() -> None:
    """weeks and actual of different lengths is a contract violation."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="demand_curve",
        params={
            "weeks": ["W1", "W2", "W3"],
            "actual": [10.0, 12.0],  # shorter
            "forecast": [10.5, 11.5, 9.5],
        },
    )
    with pytest.raises(ValueError):
        render_demand_curve(request)


# ---------------------------------------------------------------------------
# render_sparsity
# ---------------------------------------------------------------------------


def test_render_sparsity_renders_adi_cv2_per_series() -> None:
    """A sparsity request renders per-series ADI / CV² points."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="sparsity",
        params={
            "series": [
                {"series_key": "SKU_1", "adi": 1.2, "cv2": 0.5},
                {"series_key": "SKU_2", "adi": 4.0, "cv2": 1.5},
            ],
        },
    )
    response = render_sparsity(request)
    assert response.kind == "sparsity"
    assert response.bytes_b64
    assert response.width > 0


def test_render_sparsity_empty_series_returns_placeholder() -> None:
    """An empty series list renders an empty plot (no crash)."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="sparsity",
        params={"series": []},
    )
    response = render_sparsity(request)
    assert response.kind == "sparsity"
    assert response.bytes_b64  # still a valid PNG, just empty


# ---------------------------------------------------------------------------
# render_anomalies
# ---------------------------------------------------------------------------


def test_render_anomalies_renders_point_flags() -> None:
    """An anomalies request renders point flags over the timeline."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="anomalies",
        params={
            "weeks": ["W1", "W2", "W3", "W4", "W5"],
            "values": [10.0, 50.0, 12.0, 8.0, 60.0],
            "flags": [False, True, False, False, True],
        },
    )
    response = render_anomalies(request)
    assert response.kind == "anomalies"
    assert response.bytes_b64


def test_render_anomalies_missing_flags_treats_none_as_false() -> None:
    """A request without flags renders the values without point flags."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="anomalies",
        params={
            "weeks": ["W1", "W2", "W3"],
            "values": [10.0, 12.0, 9.0],
        },
    )
    response = render_anomalies(request)
    assert response.kind == "anomalies"
    assert response.bytes_b64


# ---------------------------------------------------------------------------
# render_forecast_band
# ---------------------------------------------------------------------------


def test_render_forecast_band_renders_fan_chart() -> None:
    """A forecast band request renders a fan chart around the point forecast."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="forecast_band",
        params={
            "weeks": ["W1", "W2", "W3", "W4"],
            "forecast": [10.0, 12.0, 14.0, 13.0],
            "lower": [8.0, 10.0, 11.0, 10.0],
            "upper": [12.0, 14.0, 17.0, 16.0],
        },
    )
    response = render_forecast_band(request)
    assert response.kind == "forecast_band"
    assert response.bytes_b64


# ---------------------------------------------------------------------------
# render_backtest
# ---------------------------------------------------------------------------


def test_render_backtest_renders_actual_vs_forecast() -> None:
    """A backtest request renders actual vs forecast across folds."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="backtest",
        params={
            "folds": ["fold_1", "fold_2", "fold_3"],
            "actual": [10.0, 12.0, 14.0],
            "forecast": [10.5, 11.5, 13.5],
        },
    )
    response = render_backtest(request)
    assert response.kind == "backtest"
    assert response.bytes_b64


# ---------------------------------------------------------------------------
# render_feature_importance
# ---------------------------------------------------------------------------


def test_render_feature_importance_renders_bar_chart() -> None:
    """A feature importance request renders a bar chart of feature gains."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="feature_importance",
        params={
            "features": [
                {"name": "lag_1", "importance": 0.30},
                {"name": "rolling_mean_4", "importance": 0.25},
                {"name": "promo_flag", "importance": 0.15},
            ],
        },
    )
    response = render_feature_importance(request)
    assert response.kind == "feature_importance"
    assert response.bytes_b64


# ---------------------------------------------------------------------------
# render_drift_chart
# ---------------------------------------------------------------------------


def test_render_drift_chart_renders_per_segment_delta() -> None:
    """A drift chart request renders per-segment MASE deltas over time."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="drift_chart",
        params={
            "runs": ["r1", "r2", "r3"],
            "segments": {
                "G1": [0.80, 0.85, 0.95],
                "G2": [1.00, 1.05, 1.10],
            },
        },
    )
    response = render_drift_chart(request)
    assert response.kind == "drift_chart"
    assert response.bytes_b64


# ---------------------------------------------------------------------------
# InProcessPlotEngine — full dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [
    "demand_curve",
    "sparsity",
    "anomalies",
    "forecast_band",
    "backtest",
    "feature_importance",
    "drift_chart",
])
def test_engine_dispatches_to_all_seven_kinds(kind: str) -> None:
    """Every kind routes to a real generator (not the placeholder)."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind=kind,  # type: ignore[arg-type]
        params=_params_for(kind),
    )
    response = InProcessPlotEngine().render(request)
    assert response.kind == kind
    assert response.content_type == "image/png"
    assert response.bytes_b64
    assert response.width > 0
    assert response.height > 0


def _params_for(kind: str) -> dict:
    """Minimal valid params for each plot kind (used by the parametrize)."""
    if kind == "demand_curve":
        return {
            "weeks": ["W1", "W2", "W3"],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.1, 1.9, 3.1],
        }
    if kind == "sparsity":
        return {
            "series": [
                {"series_key": "A", "adi": 1.0, "cv2": 0.5},
            ],
        }
    if kind == "anomalies":
        return {
            "weeks": ["W1", "W2", "W3"],
            "values": [1.0, 2.0, 3.0],
            "flags": [False, True, False],
        }
    if kind == "forecast_band":
        return {
            "weeks": ["W1", "W2", "W3"],
            "forecast": [1.0, 2.0, 3.0],
            "lower": [0.8, 1.8, 2.8],
            "upper": [1.2, 2.2, 3.2],
        }
    if kind == "backtest":
        return {
            "folds": ["f1", "f2"],
            "actual": [1.0, 2.0],
            "forecast": [1.1, 1.9],
        }
    if kind == "feature_importance":
        return {
            "features": [
                {"name": "a", "importance": 0.5},
                {"name": "b", "importance": 0.3},
            ],
        }
    if kind == "drift_chart":
        return {
            "runs": ["r1", "r2"],
            "segments": {"G1": [0.8, 0.9]},
        }
    return {}
