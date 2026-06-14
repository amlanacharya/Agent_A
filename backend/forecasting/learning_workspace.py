from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field

from forecasting.run_state import HaltedRunError, Phase, load_run_state, run_dir

MemoryLayer = Literal["global", "customer", "project"]
LearningTier = Literal["auto", "verifier", "human"]
LearningStatus = Literal["proposed", "approved", "rejected"]

REQUIRED_ARTIFACTS: dict[str, str] = {
    "CONTEXT.md": "# Run Context\n",
    "DATA_CONTRACT.md": "# Data Contract\n",
    "LEARNINGS.md": "# Learnings\n",
    "ASSUMPTIONS.md": "# Assumptions\n",
    "DECISIONS.md": "# Decisions\n",
    "RUNBOOK.md": "# Runbook\n",
    "MODEL_REGISTRY.md": "# Model Registry\n",
    "PROMOTION_DECISIONS.md": "# Promotion Decisions\n",
}

TIER_HEADINGS: dict[LearningTier, str] = {
    "auto": "Auto-Promoted Learnings",
    "verifier": "Verifier-Promoted Learnings",
    "human": "Human-Approved Learnings",
}


class MemoryLayerError(ValueError):
    pass


class LearningPromotionError(ValueError):
    pass


class LearningEntry(BaseModel):
    tier: LearningTier
    category: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    status: LearningStatus


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    path: Path
    artifacts: dict[str, Path]


def validate_memory_layer(layer: str) -> MemoryLayer:
    if layer not in set(get_args(MemoryLayer)):
        raise MemoryLayerError(f"Unknown memory layer: {layer}")
    return layer  # type: ignore[return-value]


def create_run_workspace(run_id: str) -> RunWorkspace:
    _assert_mutable_run(run_id)
    workspace_path = run_dir(run_id) / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    for name, initial_content in REQUIRED_ARTIFACTS.items():
        path = workspace_path / name
        if not path.exists():
            path.write_text(initial_content)
        artifacts[name] = path

    return RunWorkspace(run_id=run_id, path=workspace_path, artifacts=artifacts)


def promote_learning(workspace: RunWorkspace, entry: LearningEntry) -> None:
    _assert_mutable_run(workspace.run_id)
    if entry.status != "approved":
        raise LearningPromotionError("Only approved learning entries can be promoted")

    learnings_path = workspace.artifacts["LEARNINGS.md"]
    heading = TIER_HEADINGS[entry.tier]
    text = learnings_path.read_text()
    learning_block = (
        f"- **{entry.category}**: {entry.statement}\n"
        f"  Evidence: {entry.evidence}\n"
    )
    learnings_path.write_text(_append_under_heading(text, heading, learning_block))


def _assert_mutable_run(run_id: str) -> None:
    state = load_run_state(run_id)
    if state.phase == Phase.HALTED:
        raise HaltedRunError(run_id)


def _append_under_heading(text: str, heading: str, block: str) -> str:
    marker = f"## {heading}"
    if marker not in text:
        return text.rstrip() + f"\n\n{marker}\n\n{block}"

    start = text.index(marker)
    next_heading = text.find("\n## ", start + len(marker))
    insertion_point = len(text) if next_heading == -1 else next_heading
    before = text[:insertion_point].rstrip()
    after = text[insertion_point:].lstrip("\n")

    updated = before + "\n\n" + block
    if after:
        updated += "\n" + after
    return updated
