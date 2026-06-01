from __future__ import annotations

import json
from typing import Literal

import anthropic
from pydantic import BaseModel

from forecasting.contracts import IntentPack
from forecasting.run_state import RunState


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent: str | None = None


class LensInput(BaseModel):
    conversation_history: list[ConversationTurn]
    user_message: str
    pipeline_state: RunState


client = anthropic.Anthropic()

_SYSTEM = """
You are Lens, an intent classifier for a demand forecasting assistant.
Classify the user's latest message into exactly one intent type and return a single JSON
object. No prose - JSON only.

Intent types:
- SCOPE_RESPONSE    - answering Meridian's scoping question
- OVERRIDE          - contradicting an agent recommendation backed by data
- ADVANCE_PIPELINE  - approving progression to the next pipeline phase
- WHAT_IF_REQUEST   - requesting a scenario / what-if analysis
- CLARIFICATION     - asking a question
- CORRECTION        - fixing a prior statement (only valid in meridian_scoping)

Weighting rules:
1. pipeline_state.phase and the last assistant message are the strongest signal for
   short ambiguous messages ("ok", "yes", "fine", "sure").
2. Short message after a risk warning -> SCOPE_RESPONSE.
3. Short message after "shall we proceed?" -> ADVANCE_PIPELINE.
4. Only WHAT_IF_REQUEST if the user explicitly describes a scenario change.
5. Set confidence honestly. Unsure between two -> confidence < 0.6.
6. raw_quote: verbatim excerpt (<=20 words) from the user message.

Return schema (JSON, no markdown):
{
  "intent": "<IntentType>",
  "entities": {"skus": [], "segments": [], "dates": [], "metrics": [], "scenario": null},
  "confidence": 0.0,
  "raw_quote": ""
}
""".strip()


def classify_intent(inp: LensInput) -> IntentPack:
    messages = [{"role": turn.role, "content": turn.content} for turn in inp.conversation_history]
    messages.append({"role": "user", "content": inp.user_message})

    state = inp.pipeline_state
    system = (
        f"{_SYSTEM}\n\n"
        f"pipeline_state: phase={state.phase} pack_confirmed={state.pack_confirmed} "
        f"open_risks={state.open_risks} override_count={state.override_count}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.0,
        system=system,
        messages=messages,
    )
    payload = json.loads(response.content[0].text.strip())
    return IntentPack.model_validate(payload)
