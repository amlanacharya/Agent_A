"""Tests for Phase 8 CB7: Replenishment Board + Learning Journal surfaces.

Two more cockpit surfaces, both reading existing
platform state:

* ``ReplenishmentBoardSurface`` — surfaces the latest
  ``ReplenishmentRecommendation`` per series (lead time
  demand, safety stock, ROP, target inventory, current
  inventory, open POs, order quantity, approval tier).
  The cockpit's Replenishment Board is the per-series
  replenishment drill-down + the batch summary the
  planner approves.

* ``LearningJournalSurface`` — surfaces the workspace
  markdown artifacts (``LEARNINGS.md``, ``DECISIONS.md``,
  ``ASSUMPTIONS.md``, ``RUNBOOK.md``, ``MODEL_REGISTRY.md``,
  ``PROMOTION_DECISIONS.md``) the Phase 1 learning
  workspace writes. The cockpit's Learning Journal
  surfaces the cross-run learnings + the active /
  retired card lifecycle.

Both use the same provider-injection pattern as CB4-CB6.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.models import SurfaceSnapshot
from api.surfaces import (
    LearningJournalSurface,
    ReplenishmentBoardSurface,
)


# ---------------------------------------------------------------------------
# ReplenishmentBoardSurface
# ---------------------------------------------------------------------------


def _replenishment_batch() -> "list[object]":
    from forecasting.replenishment import ReplenishmentRecommendation

    return [
        ReplenishmentRecommendation(
            series_key="SKU_1",
            lead_time_days=7,
            forecast_std=2.0,
            lead_time_demand=70.0,
            safety_stock=15.0,
            reorder_point=85.0,
            target_inventory=100.0,
            current_inventory=20.0,
            open_purchase_orders=0.0,
            order_quantity=80.0,
            approval_tier="medium",
        ),
        ReplenishmentRecommendation(
            series_key="SKU_2",
            lead_time_days=14,
            forecast_std=1.0,
            lead_time_demand=140.0,
            safety_stock=10.0,
            reorder_point=150.0,
            target_inventory=160.0,
            current_inventory=5.0,
            open_purchase_orders=50.0,
            order_quantity=105.0,
            approval_tier="large",
        ),
    ]


def test_replenishment_board_surfaces_per_series_recommendations() -> None:
    """The surface surfaces the per-series recommendations for the cockpit drill-down."""
    surface = ReplenishmentBoardSurface(
        recommendations_provider=lambda rid: _replenishment_batch()
    )
    snapshot = surface.render("r1")
    assert snapshot.surface == "replenishment_board"
    state = snapshot.state
    assert state["recommendation_count"] == 2
    by_key = {r["series_key"]: r for r in state["recommendations"]}
    assert by_key["SKU_1"]["order_quantity"] == pytest.approx(80.0)
    assert by_key["SKU_1"]["approval_tier"] == "medium"
    assert by_key["SKU_2"]["approval_tier"] == "large"


def test_replenishment_board_surfaces_approval_tier_breakdown() -> None:
    """The surface surfaces the approval-tier breakdown for the cockpit's batch widget."""
    surface = ReplenishmentBoardSurface(
        recommendations_provider=lambda rid: _replenishment_batch()
    )
    snapshot = surface.render("r1")
    breakdown = snapshot.state["approval_tier_breakdown"]
    assert breakdown["medium"] == 1
    assert breakdown["large"] == 1


def test_replenishment_board_surfaces_total_order_quantity() -> None:
    """The surface surfaces the total order quantity across the batch."""
    surface = ReplenishmentBoardSurface(
        recommendations_provider=lambda rid: _replenishment_batch()
    )
    snapshot = surface.render("r1")
    assert snapshot.state["total_order_quantity"] == pytest.approx(185.0)


def test_replenishment_board_handles_empty_batch() -> None:
    """A run with no replenishment batch yet surfaces a 'no recommendations' widget."""
    surface = ReplenishmentBoardSurface(recommendations_provider=lambda rid: [])
    snapshot = surface.render("r-empty")
    assert snapshot.state["recommendation_count"] == 0
    assert snapshot.state["recommendations"] == []
    assert snapshot.state["total_order_quantity"] == 0.0


def test_replenishment_board_handles_missing_provider() -> None:
    """A run with no provider wired surfaces a 'no data' placeholder."""
    surface = ReplenishmentBoardSurface(recommendations_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["recommendation_count"] == 0


# ---------------------------------------------------------------------------
# LearningJournalSurface
# ---------------------------------------------------------------------------


def test_learning_journal_surfaces_workspace_markdown(tmp_path: Path) -> None:
    """The surface surfaces the workspace markdown artifacts the cockpit renders."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "LEARNINGS.md").write_text("# Learnings\n")
    (workspace / "DECISIONS.md").write_text("# Decisions\n")
    (workspace / "ASSUMPTIONS.md").write_text("# Assumptions\n")
    surface = LearningJournalSurface(workspace_root=workspace)
    snapshot = surface.render("r1")
    assert snapshot.surface == "learning_journal"
    state = snapshot.state
    assert state["LEARNINGS.md"] == "# Learnings\n"
    assert state["DECISIONS.md"] == "# Decisions\n"
    assert state["ASSUMPTIONS.md"] == "# Assumptions\n"


def test_learning_journal_handles_missing_workspace(tmp_path: Path) -> None:
    """A workspace with no markdown yet surfaces every artifact as None."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    surface = LearningJournalSurface(workspace_root=workspace)
    snapshot = surface.render("r-empty")
    for filename in (
        "LEARNINGS.md",
        "DECISIONS.md",
        "ASSUMPTIONS.md",
        "RUNBOOK.md",
        "MODEL_REGISTRY.md",
        "PROMOTION_DECISIONS.md",
    ):
        assert snapshot.state[filename] is None


def test_learning_journal_handles_missing_workspace_dir(tmp_path: Path) -> None:
    """A workspace that does not exist yet surfaces every artifact as None (no crash)."""
    workspace = tmp_path / "does-not-exist"
    surface = LearningJournalSurface(workspace_root=workspace)
    snapshot = surface.render("r-empty")
    for filename in (
        "LEARNINGS.md",
        "DECISIONS.md",
        "ASSUMPTIONS.md",
        "RUNBOOK.md",
        "MODEL_REGISTRY.md",
        "PROMOTION_DECISIONS.md",
    ):
        assert snapshot.state[filename] is None


def test_learning_journal_surfaces_card_lifecycle_summary(tmp_path: Path) -> None:
    """The surface surfaces a card-lifecycle summary (active / retired counts)
    parsed from the LEARNINGS.md front matter the Phase 1 workspace writes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    # The Phase 1 workspace writes a front-matter block at the
    # top of LEARNINGS.md with `## Active` and `## Retired`
    # sections. The surface counts entries under each.
    (workspace / "LEARNINGS.md").write_text(
        "# LEARNINGS\n\n"
        "## Active\n\n"
        "- [card-1] (active, 5 runs)\n"
        "- [card-2] (active, 3 runs)\n\n"
        "## Retired\n\n"
        "- [card-3] (retired, 2 runs)\n"
    )
    surface = LearningJournalSurface(workspace_root=workspace)
    snapshot = surface.render("r1")
    assert snapshot.state["active_cards"] == 2
    assert snapshot.state["retired_cards"] == 1
