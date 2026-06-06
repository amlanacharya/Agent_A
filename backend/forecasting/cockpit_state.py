from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from forecasting.run_state import Phase, RunState

Confidence = Literal["low", "medium", "high"]
CodeEscalationStatus = Literal["not_requested", "toolbox", "code_escalation", "blocked"]


class CockpitState(BaseModel):
    run_id: str
    current_step: str
    active_agent: str
    tool_result: str | None = None
    code_escalation_status: CodeEscalationStatus | None = None
    code_attempt: int | None = Field(default=None, ge=1, le=3)
    verifier_gate: str | None = None
    approval_needed: bool = False
    confidence: Confidence = "medium"
    blockers: list[str] = Field(default_factory=list)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "current_step": self.current_step,
            "active_agent": self.active_agent,
            "tool_result": self.tool_result,
            "code_escalation_status": self.code_escalation_status,
            "code_attempt": self.code_attempt,
            "verifier_gate": self.verifier_gate,
            "approval_needed": self.approval_needed,
            "confidence": self.confidence,
            "blockers": list(self.blockers),
        }

    def with_blocker(self, message: str) -> CockpitState:
        return self.model_copy(
            update={
                "blockers": [*self.blockers, message],
                "confidence": "low",
            }
        )

    def mark_approval_needed(self, gate: str) -> CockpitState:
        return self.model_copy(
            update={
                "approval_needed": True,
                "verifier_gate": gate,
            }
        )

    @classmethod
    def from_run_state(
        cls,
        run_state: RunState,
        current_step: str,
        active_agent: str,
    ) -> CockpitState:
        state = cls(
            run_id=run_state.run_id,
            current_step=current_step,
            active_agent=active_agent,
        )
        phase = run_state.phase.value if isinstance(run_state.phase, Phase) else run_state.phase
        if phase == Phase.HALTED.value and run_state.halt_reason:
            return state.with_blocker(f"Run halted: {run_state.halt_reason}")
        return state
