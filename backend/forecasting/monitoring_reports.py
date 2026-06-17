"""Phase 7 CB6: markdown report writers for the four monitoring artifacts.

The platform emits four recurring markdown artifacts under
``outputs/{run_id}/``:

* ``MONITORING_REPORT.md`` — the top-level rollup. Surfaces every
  signal kind the engine observes, in one stable greppable file.
* ``DRIFT_REPORT.md`` — the data and model drift detail. Schema
  changes, missing feeds, distribution shifts, new SKU / location
  keys, and the per-segment MASE delta.
* ``OVERRIDE_ANALYSIS.md`` — planner overrides and approval /
  rejection / deferral patterns pulled from the in-process
  approval audit log.
* ``MODEL_HEALTH.md`` — model health detail. Population-level
  MASE / bias, per-segment degradation, and the interval
  calibration seam (None until the platform grows quantile
  support).

Each report has two functions:

* ``format_<report>(snapshot) -> str`` — a pure function that
  renders the snapshot to stable markdown. Same input -> same
  output. No I/O. The format is a public contract: a future
  reader can grep the file and answer "what was the MASE
  delta for G1 in r2?" without running any code.
* ``write_<report>(snapshot, output_dir) -> None`` — a thin
  I/O wrapper that calls the format function and persists to
  ``output_dir / FILENAME``. The writer creates the directory
  if it does not exist (matches the platform's other write
  paths; see ``run_state.create_run_state``).

Design:

* **Pure formatter, separate I/O function.** Same pattern as
  ``promotion.format_promotion_decision`` /
  ``promotion.write_promotion_decision`` — the format is
  pure, the writer is a thin I/O wrapper. The pure / I/O
  split lets the cockpit render the report to the API
  response without writing to disk, and lets the writer
  persist the same content for the audit trail.
* **Empty sections use a ``(none)`` marker.** A drift report
  with no missing feeds is a heading followed by ``(none)``,
  not a heading followed by nothing — the markdown is
  well-formed and a reader can grep for ``(none)`` to find
  the absence of a signal kind.
* **Stable filenames.** The four filenames match the plan's
  artifact checklist. They are exported as constants so a
  downstream consumer (the cockpit, the report-listing
  endpoint) can import them rather than hard-coding the
  string.
* **The reports are generated artifacts, not the source of
  truth.** The ``.gitignore`` excludes the four filenames;
  the workspace markdown in ``learning_workspace.py`` is
  the durable record. The reports are a snapshot view of
  one monitoring tick; the workspace is the cross-run
  learning.
"""

from __future__ import annotations

from pathlib import Path

from forecasting.contracts import (
    DistributionShift,
    MonitorSnapshot,
    SchemaChange,
    SegmentDegradation,
)


# The four filenames match the plan's artifact checklist. They
# are exported as constants so a downstream consumer (the
# cockpit, a future /reports/{run_id} endpoint) can import them
# rather than hard-coding the string. A typo in a hard-coded
# string would be invisible at test time; an import mismatch
# would fail at the first import.
MONITORING_REPORT_FILENAME = "MONITORING_REPORT.md"
DRIFT_REPORT_FILENAME = "DRIFT_REPORT.md"
OVERRIDE_ANALYSIS_FILENAME = "OVERRIDE_ANALYSIS.md"
MODEL_HEALTH_FILENAME = "MODEL_HEALTH.md"


# The ``(none)`` marker is the empty-section convention. A
# reader can grep for the literal ``(none)`` to find the
# absence of a signal kind — useful for the cockpit's
# "no-drift" widget.
_NONE = "(none)"


# ---------------------------------------------------------------------------
# format_monitoring_report — top-level rollup
# ---------------------------------------------------------------------------


def _format_distribution_shifts(shifts: list[DistributionShift]) -> str:
    """Render the distribution-shift table."""
    if not shifts:
        return _NONE
    lines = ["| column | metric | previous | current | pct_change |", "| --- | --- | ---: | ---: | ---: |"]
    for shift in shifts:
        pct = f"{shift.pct_change:+.2%}"
        lines.append(
            f"| {shift.column} | {shift.metric} | "
            f"{shift.previous:.4f} | {shift.current:.4f} | {pct} |"
        )
    return "\n".join(lines)


def _format_schema_changes(changes: list[SchemaChange]) -> str:
    """Render the schema-change list."""
    if not changes:
        return _NONE
    return "\n".join(
        f"- **{change.kind}** on `{change.column}` — {change.detail}"
        for change in changes
    )


def _format_new_keys(keys) -> str:
    """Render the new-keys list."""
    if not keys.new_skus and not keys.new_locations:
        return _NONE
    lines: list[str] = []
    if keys.new_skus:
        lines.append(
            "- New SKUs: " + ", ".join(f"`{s}`" for s in keys.new_skus)
        )
    if keys.new_locations:
        lines.append(
            "- New locations: "
            + ", ".join(f"`{loc}`" for loc in keys.new_locations)
        )
    return "\n".join(lines)


def _format_segment_degradation(
    segments: list[SegmentDegradation],
) -> str:
    """Render the per-segment degradation table."""
    if not segments:
        return _NONE
    lines = [
        "| segment | mase_previous | mase_current | mase_delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seg in segments:
        lines.append(
            f"| {seg.segment_id} | {seg.mase_previous:.4f} | "
            f"{seg.mase_current:.4f} | {seg.mase_delta:+.4f} |"
        )
    return "\n".join(lines)


def format_monitoring_report(snapshot: MonitorSnapshot) -> str:
    """Render the top-level monitoring rollup to markdown.

    The report includes every signal kind the engine observes,
    in one stable greppable file. The format is the public
    contract — a future reader can grep the file and answer
    questions like "what was the MASE delta for G1 in r2?"
    without running any code.

    Empty sections render as ``(none)`` so the markdown is
    well-formed and a reader can grep for that literal to
    find the absence of a signal kind.
    """
    header = (
        f"# Monitoring Report — run {snapshot.run_id} "
        f"(vs {snapshot.previous_run_id})\n\n"
        f"Generated at: {snapshot.generated_at}\n"
    )
    return (
        header
        + "## Data Drift\n\n"
        + f"### Schema changes\n\n{_format_schema_changes(snapshot.data.schema_changes)}\n\n"
        + f"### Missing feeds\n\n{_format_list(snapshot.data.missing_feeds)}\n\n"
        + f"### Distribution shifts\n\n{_format_distribution_shifts(snapshot.data.distribution_shifts)}\n\n"
        + f"### New SKU / location keys\n\n{_format_new_keys(snapshot.data.new_keys)}\n\n"
        + "## Model Drift\n\n"
        + f"- MASE: previous={snapshot.model.mase_previous:.4f}, "
        + f"current={snapshot.model.mase_current:.4f}, "
        + f"delta={snapshot.model.mase_delta:+.4f}\n"
        + f"- Bias: previous={snapshot.model.bias_previous:+.4f}, "
        + f"current={snapshot.model.bias_current:+.4f}, "
        + f"delta={snapshot.model.bias_delta:+.4f}\n"
        + f"### Per-segment degradation\n\n{_format_segment_degradation(snapshot.model.segment_degradation)}\n\n"
        + "## Business Outcomes\n\n"
        + f"- Expected stockouts: {snapshot.business.expected_stockouts:.4f}\n"
        + f"- Expected overstock: {snapshot.business.expected_overstock:.4f}\n"
        + f"- Service level: {snapshot.business.service_level:.2%}\n\n"
        + "## Approval Patterns\n\n"
        + _format_approval_patterns(snapshot.business.approval_patterns)
        + "\n"
    )


def _format_list(items: list[str]) -> str:
    """Render a list of strings as a bulleted list, or ``(none)`` if empty."""
    if not items:
        return _NONE
    return "\n".join(f"- `{item}`" for item in items)


def _format_approval_patterns(patterns: dict[str, int]) -> str:
    """Render the approval-pattern count map as a small table."""
    if not patterns:
        return _NONE
    lines = ["| decision | count |", "| --- | ---: |"]
    for decision in sorted(patterns):
        lines.append(f"| {decision} | {patterns[decision]} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# format_drift_report — data + model drift detail
# ---------------------------------------------------------------------------


def format_drift_report(snapshot: MonitorSnapshot) -> str:
    """Render the data + model drift detail to markdown.

    The drift report is the per-signal view of the monitoring
    layer: every drift observation the engine made, in one
    file. The monitoring report is the top-level rollup; the
    drift report is the per-signal deep-dive.
    """
    header = (
        f"# Drift Report — run {snapshot.run_id} "
        f"(vs {snapshot.previous_run_id})\n\n"
        f"Generated at: {snapshot.generated_at}\n"
    )
    return (
        header
        + "## Data Drift\n\n"
        + f"### Schema changes\n\n{_format_schema_changes(snapshot.data.schema_changes)}\n\n"
        + f"### Missing feeds\n\n{_format_list(snapshot.data.missing_feeds)}\n\n"
        + f"### Distribution shifts\n\n{_format_distribution_shifts(snapshot.data.distribution_shifts)}\n\n"
        + f"### New SKU / location keys\n\n{_format_new_keys(snapshot.data.new_keys)}\n\n"
        + "## Model Drift\n\n"
        + f"- MASE: previous={snapshot.model.mase_previous:.4f}, "
        + f"current={snapshot.model.mase_current:.4f}, "
        + f"delta={snapshot.model.mase_delta:+.4f}\n"
        + f"- Bias: previous={snapshot.model.bias_previous:+.4f}, "
        + f"current={snapshot.model.bias_current:+.4f}, "
        + f"delta={snapshot.model.bias_delta:+.4f}\n"
        + f"### Per-segment degradation\n\n{_format_segment_degradation(snapshot.model.segment_degradation)}\n"
    )


# ---------------------------------------------------------------------------
# format_override_analysis — planner overrides + approval patterns
# ---------------------------------------------------------------------------


def format_override_analysis(snapshot: MonitorSnapshot) -> str:
    """Render the override analysis to markdown.

    The override report is the planner-facing view: free-form
    descriptions of human decisions that diverged from the
    platform's recommendation, plus the approval / rejection /
    deferral counts pulled from the in-process approval audit
    log.
    """
    header = (
        f"# Override Analysis — run {snapshot.run_id}\n\n"
        f"Generated at: {snapshot.generated_at}\n\n"
        f"vs {snapshot.previous_run_id}\n"
    )
    return (
        header
        + "## Planner Overrides\n\n"
        + _format_list(snapshot.business.planner_overrides)
        + "\n\n"
        + "## Approval Patterns\n\n"
        + _format_approval_patterns(snapshot.business.approval_patterns)
        + "\n"
    )


# ---------------------------------------------------------------------------
# format_model_health — MASE / bias / segment degradation / interval seam
# ---------------------------------------------------------------------------


def format_model_health(snapshot: MonitorSnapshot) -> str:
    """Render the model health detail to markdown.

    The model health report is the data-scientist-facing view:
    population-level MASE / bias, the per-segment degradation
    table, and the interval calibration seam (which stays None
    until the platform grows quantile support).
    """
    header = (
        f"# Model Health — run {snapshot.run_id} "
        f"(vs {snapshot.previous_run_id})\n\n"
        f"Generated at: {snapshot.generated_at}\n"
    )
    interval_line = (
        "- Interval calibration: not yet emitted (the platform does "
        "not yet produce quantile forecasts; the seam will fill in "
        "when quantile support lands)\n"
        if snapshot.model.interval_calibration is None
        else f"- Interval calibration: {snapshot.model.interval_calibration:.2%}\n"
    )
    return (
        header
        + "## Population Metrics\n\n"
        + f"- MASE: previous={snapshot.model.mase_previous:.4f}, "
        + f"current={snapshot.model.mase_current:.4f}, "
        + f"delta={snapshot.model.mase_delta:+.4f}\n"
        + f"- Bias: previous={snapshot.model.bias_previous:+.4f}, "
        + f"current={snapshot.model.bias_current:+.4f}, "
        + f"delta={snapshot.model.bias_delta:+.4f}\n"
        + interval_line
        + "\n## Per-Segment Degradation\n\n"
        + _format_segment_degradation(snapshot.model.segment_degradation)
        + "\n"
    )


# ---------------------------------------------------------------------------
# write_* — I/O wrappers
# ---------------------------------------------------------------------------


def _write(snapshot: MonitorSnapshot, output_dir: Path, filename: str, body: str) -> Path:
    """Persist ``body`` to ``output_dir / filename``.

    Creates ``output_dir`` (and any missing parents) if it does
    not exist. The function is the shared I/O helper for the
    four writers; the format function (the ``body`` argument) is
    the one that knows the report shape.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(body, encoding="utf-8")
    return path


def write_monitoring_report(snapshot: MonitorSnapshot, output_dir: Path) -> Path:
    """Persist the monitoring report to ``output_dir / MONITORING_REPORT.md``."""
    return _write(snapshot, output_dir, MONITORING_REPORT_FILENAME, format_monitoring_report(snapshot))


def write_drift_report(snapshot: MonitorSnapshot, output_dir: Path) -> Path:
    """Persist the drift report to ``output_dir / DRIFT_REPORT.md``."""
    return _write(snapshot, output_dir, DRIFT_REPORT_FILENAME, format_drift_report(snapshot))


def write_override_analysis(snapshot: MonitorSnapshot, output_dir: Path) -> Path:
    """Persist the override analysis to ``output_dir / OVERRIDE_ANALYSIS.md``."""
    return _write(snapshot, output_dir, OVERRIDE_ANALYSIS_FILENAME, format_override_analysis(snapshot))


def write_model_health(snapshot: MonitorSnapshot, output_dir: Path) -> Path:
    """Persist the model health report to ``output_dir / MODEL_HEALTH.md``."""
    return _write(snapshot, output_dir, MODEL_HEALTH_FILENAME, format_model_health(snapshot))


__all__ = (
    # Filename constants (the platform surface, importable by name)
    "MONITORING_REPORT_FILENAME",
    "DRIFT_REPORT_FILENAME",
    "OVERRIDE_ANALYSIS_FILENAME",
    "MODEL_HEALTH_FILENAME",
    # Pure formatters (the platform surface, importable by name)
    "format_monitoring_report",
    "format_drift_report",
    "format_override_analysis",
    "format_model_health",
    # I/O writers (the platform surface, importable by name)
    "write_monitoring_report",
    "write_drift_report",
    "write_override_analysis",
    "write_model_health",
)
