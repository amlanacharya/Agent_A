import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from forecasting.contracts import IntentPack
from forecasting.run_state import RunState


def _make_input(msg: str, phase: str = "preflight", history=None):
    from forecasting.agents.lens import LensInput

    rs = RunState(
        run_id="r1",
        phase=phase,
        domain="fmcg",
        created_at="2024-01-01T00:00:00+00:00",
    )
    return LensInput(
        conversation_history=history or [],
        user_message=msg,
        pipeline_state=rs,
    )


def _mock_response(intent: str, confidence: float = 0.9, text=None, block_type="text"):
    payload = {
        "intent": intent,
        "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
        "confidence": confidence,
        "raw_quote": "test",
    }
    content = MagicMock()
    content.type = block_type
    if text is None:
        content.text = json.dumps(payload)
    else:
        content.text = text
    resp = MagicMock()
    resp.content = [content]
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def test_import_does_not_construct_anthropic_client():
    with patch("anthropic.Anthropic") as mock_ctor:
        import forecasting.agents.lens as lens_module

        importlib.reload(lens_module)
    mock_ctor.assert_not_called()


def test_classify_returns_intent_pack():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
    result = classify_intent(_make_input("yes that looks right"), injected_client=injected)
    assert isinstance(result, IntentPack)
    assert result.intent == "SCOPE_RESPONSE"
    assert result.confidence == 0.9


def test_advance_pipeline_intent():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("ADVANCE_PIPELINE")
    result = classify_intent(_make_input("ok let's proceed to modelling"), injected_client=injected)
    assert result.intent == "ADVANCE_PIPELINE"


def test_model_is_haiku():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
    classify_intent(_make_input("yes"), injected_client=injected)
    call_kwargs = injected.messages.create.call_args.kwargs
    assert "haiku" in call_kwargs["model"]


def test_temperature_zero():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
    classify_intent(_make_input("yes"), injected_client=injected)
    call_kwargs = injected.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.0


def test_message_construction_includes_history_and_user_message():
    from forecasting.agents.lens import ConversationTurn, classify_intent

    history = [
        ConversationTurn(role="assistant", content="Need scope details", agent="meridian"),
        ConversationTurn(role="user", content="Horizon 8 weeks"),
    ]
    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
    classify_intent(_make_input("Proceed", history=history), injected_client=injected)
    messages = injected.messages.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "assistant", "content": "Need scope details"},
        {"role": "user", "content": "Horizon 8 weeks"},
        {"role": "user", "content": "Proceed"},
    ]


def test_system_prompt_contains_pipeline_state_fields():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
    classify_intent(_make_input("yes", phase="meridian_scoping"), injected_client=injected)
    system = injected.messages.create.call_args.kwargs["system"]
    assert "phase=meridian_scoping" in system
    assert "pack_confirmed=False" in system
    assert "open_risks=0" in system
    assert "override_count=0" in system


def test_fenced_json_is_accepted():
    from forecasting.agents.lens import classify_intent

    text = """```json
{"intent":"SCOPE_RESPONSE","entities":{"skus":[],"segments":[],"dates":[],"metrics":[]},"confidence":0.7,"raw_quote":"ok"}
```"""
    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE", text=text)
    result = classify_intent(_make_input("ok"), injected_client=injected)
    assert result.intent == "SCOPE_RESPONSE"


def test_trailing_text_rejected():
    from forecasting.agents.lens import LensResponseError, classify_intent

    text = '{"intent":"SCOPE_RESPONSE","entities":{"skus":[],"segments":[],"dates":[],"metrics":[]},"confidence":0.7,"raw_quote":"ok"} trailing'
    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE", text=text)
    with pytest.raises(LensResponseError, match="trailing"):
        classify_intent(_make_input("ok"), injected_client=injected)


def test_empty_content_rejected():
    from forecasting.agents.lens import LensResponseError, classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE", text="   ")
    with pytest.raises(LensResponseError, match="empty"):
        classify_intent(_make_input("ok"), injected_client=injected)


def test_non_text_content_block_rejected():
    from forecasting.agents.lens import LensResponseError, classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE", block_type="tool_use")
    with pytest.raises(LensResponseError, match="not text"):
        classify_intent(_make_input("ok"), injected_client=injected)


def test_invalid_confidence_propagates_validation_error():
    from forecasting.agents.lens import classify_intent

    injected = MagicMock()
    injected.messages.create.return_value = _mock_response("SCOPE_RESPONSE", confidence=1.5)
    with pytest.raises(ValidationError):
        classify_intent(_make_input("ok"), injected_client=injected)


def test_intent_pack_confidence_bounds():
    with pytest.raises(ValidationError):
        IntentPack.model_validate(
            {
                "intent": "SCOPE_RESPONSE",
                "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
                "confidence": 1.5,
                "raw_quote": "x",
            }
        )
    with pytest.raises(ValidationError):
        IntentPack.model_validate(
            {
                "intent": "SCOPE_RESPONSE",
                "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
                "confidence": -0.1,
                "raw_quote": "x",
            }
        )


def test_intent_pack_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        IntentPack.model_validate(
            {
                "intent": "SCOPE_RESPONSE",
                "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
                "confidence": 0.5,
                "raw_quote": "x",
                "extra_top": "boom",
            }
        )
    with pytest.raises(ValidationError):
        IntentPack.model_validate(
            {
                "intent": "SCOPE_RESPONSE",
                "entities": {
                    "skus": [],
                    "segments": [],
                    "dates": [],
                    "metrics": [],
                    "unexpected": "boom",
                },
                "confidence": 0.5,
                "raw_quote": "x",
            }
        )
