from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from forecasting.run_state import run_dir

Layer = Literal[
    "eda",
    "schema_mapping",
    "canonical_table",
    "feature_engineering",
    "forecasting_model",
    "diagnostics",
]

ProblemKind = Literal[
    "data",
    "business_semantics",
    "platform_limitation",
    "model_limitation",
]

ALLOWED_LAYERS: set[str] = {
    "eda",
    "schema_mapping",
    "canonical_table",
    "feature_engineering",
    "forecasting_model",
    "diagnostics",
}
ALLOWED_PROBLEM_KINDS: set[str] = {
    "data",
    "business_semantics",
    "platform_limitation",
    "model_limitation",
}
MAX_CODE_ATTEMPTS = 3


class EscalationLimitReached(Exception):
    pass


@dataclass(frozen=True)
class FailureReport:
    run_id: str
    layer: Layer
    status: Literal["blocked"]
    blocker: str
    evidence: list[str]
    attempts: int
    failed_reasons: dict[int, str]
    problem_kind: ProblemKind
    recommended_next_action: str


# ---------------------------------------------------------------------------
# Why EscalationTracker is file-backed and survives restarts.
# ---------------------------------------------------------------------------
# A custom model family is a permanent addition to the registry - once
# accepted, the next run will see it. The 3-attempt cap exists to bound
# the LLM-driven trial-and-error *within* a run, but the attempt ledger
# itself must persist across runs so a crashed session's attempts are
# honoured when the human returns. The contrasting design lives in
# ``config_escalation.py`` - a FeatureFlag flip is per-run only, so its
# per-action counter stays in memory. If unified observability across
# both paths is ever needed, the seam is an ``AttemptTracker`` Protocol.
# ---------------------------------------------------------------------------


@dataclass
class EscalationTracker:
    run_id: str
    layer: Layer
    attempts: int = field(default=0, init=False)
    status: str = field(default="toolbox", init=False)
    failed_reasons: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.layer not in ALLOWED_LAYERS:
            raise ValueError(f"invalid escalation layer: {self.layer}")
        self._load()

    def request_code_attempt(self, reason: str) -> int:
        del reason
        if self.attempts >= MAX_CODE_ATTEMPTS:
            raise EscalationLimitReached(
                f"layer {self.layer} reached {MAX_CODE_ATTEMPTS} code-generation attempts"
            )
        self.attempts += 1
        self.status = "code_escalation"
        self._save()
        return self.attempts

    def record_failed_attempt(self, attempt_no: int, reason: str) -> None:
        if attempt_no < 1 or attempt_no > self.attempts:
            raise ValueError(f"attempt_no must reference a requested attempt: {attempt_no}")
        self.failed_reasons[attempt_no] = reason
        self._save()

    def declare_failure_report(
        self,
        *,
        blocker: str,
        evidence: list[str],
        problem_kind: ProblemKind,
        recommended_next_action: str,
    ) -> FailureReport:
        if self.attempts != MAX_CODE_ATTEMPTS or len(self.failed_reasons) != MAX_CODE_ATTEMPTS:
            raise ValueError(
                f"failure report requires exactly three recorded failed attempts; "
                f"attempts={self.attempts}, recorded={len(self.failed_reasons)}"
            )
        if problem_kind not in ALLOWED_PROBLEM_KINDS:
            raise ValueError(f"invalid problem kind: {problem_kind}")

        self.status = "blocked"
        report = FailureReport(
            run_id=self.run_id,
            layer=self.layer,
            status="blocked",
            blocker=blocker,
            evidence=list(evidence),
            attempts=self.attempts,
            failed_reasons=dict(sorted(self.failed_reasons.items())),
            problem_kind=problem_kind,
            recommended_next_action=recommended_next_action,
        )
        self._save()
        return report

    @property
    def path(self) -> Path:
        return run_dir(self.run_id) / "escalations" / f"{self.layer}.json"

    def _load(self) -> None:
        path = self.path
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.attempts = int(data.get("attempts", 0))
        self.status = str(data.get("status", "toolbox"))
        self.failed_reasons = {
            int(attempt_no): str(reason)
            for attempt_no, reason in data.get("failed_reasons", {}).items()
        }

    def _save(self) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "layer": self.layer,
                    "attempts": self.attempts,
                    "status": self.status,
                    "failed_reasons": self.failed_reasons,
                },
                indent=2,
            )
        )
