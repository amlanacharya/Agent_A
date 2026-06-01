import json
from unittest.mock import MagicMock, patch

from forecasting.agents.lens import classify_intent
from forecasting.contracts import IntentPack
from forecasting.run_state import RunState


def _make_input(msg: str, phase: str = "preflight"):
    from forecasting.agents.lens import ConversationTurn, LensInput

    _ = ConversationTurn
    rs = RunState(
        run_id="r1",
        phase=phase,
        domain="fmcg",
        created_at="2024-01-01T00:00:00+00:00",
    )
    return LensInput(
        conversation_history=[],
        user_message=msg,
        pipeline_state=rs,
    )


def _mock_response(intent: str, confidence: float = 0.9):
    pack = {
        "intent": intent,
        "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
        "confidence": confidence,
        "raw_quote": "test",
    }
    content = MagicMock()
    content.text = json.dumps(pack)
    resp = MagicMock()
    resp.content = [content]
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def test_classify_returns_intent_pack():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        result = classify_intent(_make_input("yes that looks right"))
    assert isinstance(result, IntentPack)
    assert result.intent == "SCOPE_RESPONSE"
    assert result.confidence == 0.9


def test_advance_pipeline_intent():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("ADVANCE_PIPELINE")
        result = classify_intent(_make_input("ok let's proceed to modelling"))
    assert result.intent == "ADVANCE_PIPELINE"


def test_model_is_haiku():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        classify_intent(_make_input("yes"))
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "haiku" in call_kwargs["model"]


def test_temperature_zero():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        classify_intent(_make_input("yes"))
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.0
