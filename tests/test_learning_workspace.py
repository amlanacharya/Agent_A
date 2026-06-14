import pytest

from forecasting.learning_workspace import (
    LearningEntry,
    LearningPromotionError,
    MemoryLayerError,
    create_run_workspace,
    promote_learning,
    validate_memory_layer,
)
from forecasting.run_state import (
    HaltedRunError,
    Phase,
    RunNotFoundError,
    create_run_state,
    save_run_state,
)


def test_create_run_workspace_writes_required_markdown_artifacts(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)

    assert workspace.run_id == run_id
    assert workspace.path.exists()
    assert set(workspace.artifacts) == {
        "CONTEXT.md",
        "DATA_CONTRACT.md",
        "LEARNINGS.md",
        "ASSUMPTIONS.md",
        "DECISIONS.md",
        "RUNBOOK.md",
        "MODEL_REGISTRY.md",
        "PROMOTION_DECISIONS.md",
    }
    assert workspace.artifacts["LEARNINGS.md"].read_text().startswith("# Learnings")


def test_create_run_workspace_does_not_overwrite_existing_artifacts(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    learnings = workspace.artifacts["LEARNINGS.md"]
    learnings.write_text("# Learnings\n\nExisting approved note.\n")

    create_run_workspace(run_id)

    assert learnings.read_text() == "# Learnings\n\nExisting approved note.\n"


def test_promote_learning_appends_approved_entry_under_tier_heading(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    entry = LearningEntry(
        tier="verifier",
        category="model_selection",
        statement="Aggregate-and-allocate beat direct SKU-location models for sparse series.",
        evidence="Backtest WAPE improved from 0.74 to 0.61 on segment G2.",
        status="approved",
    )

    promote_learning(workspace, entry)

    text = workspace.artifacts["LEARNINGS.md"].read_text()
    assert "## Verifier-Promoted Learnings" in text
    assert "**model_selection**" in text
    assert "Aggregate-and-allocate beat direct SKU-location models" in text
    assert "Backtest WAPE improved" in text


def test_promote_learning_appends_to_existing_tier_section(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    promote_learning(
        workspace,
        LearningEntry(
            tier="human",
            category="business_semantics",
            statement="Demand means shipped units.",
            evidence="Planner confirmed during pack review.",
            status="approved",
        ),
    )
    promote_learning(
        workspace,
        LearningEntry(
            tier="verifier",
            category="model_selection",
            statement="Sparse segment needs aggregate fallback.",
            evidence="Backtest bias improved.",
            status="approved",
        ),
    )
    promote_learning(
        workspace,
        LearningEntry(
            tier="human",
            category="policy",
            statement="High-value items require manual approval.",
            evidence="User confirmed approval threshold.",
            status="approved",
        ),
    )

    text = workspace.artifacts["LEARNINGS.md"].read_text()
    human_section = text.split("## Human-Approved Learnings", 1)[1]
    assert human_section.index("Demand means shipped units") < human_section.index(
        "High-value items require manual approval"
    )
    assert human_section.index("High-value items require manual approval") < human_section.index(
        "## Verifier-Promoted Learnings"
    )


def test_promote_learning_rejects_non_approved_entry(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    entry = LearningEntry(
        tier="human",
        category="business_semantics",
        statement="Quantity means shipped units.",
        evidence="Planner confirmation pending.",
        status="proposed",
    )

    with pytest.raises(LearningPromotionError):
        promote_learning(workspace, entry)


def test_create_run_workspace_requires_existing_run_state(run_id, tmp_outputs):
    with pytest.raises(RunNotFoundError):
        create_run_workspace(run_id)


def test_create_run_workspace_rejects_halted_run(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.halt_reason = "guard budget exceeded"
    state.phase = Phase.HALTED
    save_run_state(state)

    with pytest.raises(HaltedRunError):
        create_run_workspace(run_id)


def test_promote_learning_rejects_halted_run(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    state.halt_reason = "manual halt"
    state.phase = Phase.HALTED
    save_run_state(state)

    with pytest.raises(HaltedRunError):
        promote_learning(
            workspace,
            LearningEntry(
                tier="auto",
                category="schema",
                statement="SKU is item_id.",
                evidence="Header mapping confidence 0.98.",
                status="approved",
            ),
        )


@pytest.mark.parametrize("layer", ["global", "customer", "project"])
def test_validate_memory_layer_accepts_known_layers(layer):
    assert validate_memory_layer(layer) == layer


def test_validate_memory_layer_rejects_unknown_layer():
    with pytest.raises(MemoryLayerError):
        validate_memory_layer("shared_customer")
