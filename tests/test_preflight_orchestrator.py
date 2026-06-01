import json
from pathlib import Path

import pytest

from forecasting.data_store import get_series_keys, replace_run
from forecasting.preflight import PreflightBlockingError, run_preflight

PLAYBOOK = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 4,
}


def _csv(n_weeks: int = 12) -> bytes:
    rows = [f"2024-W{w + 1:02d},{sku},NORTH,{float(w + 1)}" for sku in ["SKU_A", "SKU_B"] for w in range(n_weeks)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def _csv_for_skus(skus: list[str], n_weeks: int = 12) -> bytes:
    rows = [f"2024-W{w + 1:02d},{sku},NORTH,{float(w + 1)}" for sku in skus for w in range(n_weeks)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def test_preflight_populates_data_store_and_quality(run_id, tmp_outputs):
    bundle = run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    assert len(get_series_keys(run_id)) == 2
    assert bundle.data_quality_report.blocking_issues == []
    assert bundle.data_quality_report.series_count == 2


def test_preflight_writes_json_with_required_keys(run_id, tmp_outputs):
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    pf_path = tmp_outputs / run_id / "preflight.json"
    assert pf_path.exists()
    data = json.loads(pf_path.read_text())
    assert set(data.keys()) == {"bundle", "segment_map", "break_candidates", "per_series"}
    assert set(data["per_series"].keys()) == {"adi_cv2", "zero_runs", "spikes", "promo_align", "trend", "seasonality"}


def test_preflight_blocks_all_zero(run_id, tmp_outputs):
    rows = [f"2024-W{w:02d},SKU_A,NORTH,0.0" for w in range(1, 13)]
    csv_bytes = ("week,sku,region,demand\n" + "\n".join(rows)).encode()
    with pytest.raises(PreflightBlockingError) as exc_info:
        run_preflight(run_id, csv_bytes, domain="fmcg", playbook=PLAYBOOK)
    assert any(i.code == "ALL_ZERO_DEMAND" for i in exc_info.value.issues)
    assert "ALL_ZERO_DEMAND" in str(exc_info.value)


def test_preflight_blocks_corrupt_file(run_id, tmp_outputs):
    with pytest.raises(PreflightBlockingError) as exc_info:
        run_preflight(run_id, b"\x00\x01\x02corrupted", domain="fmcg", playbook=PLAYBOOK)
    assert any(i.code == "UNPARSEABLE_FILE" for i in exc_info.value.issues)
    assert "UNPARSEABLE_FILE" in str(exc_info.value)


def test_preflight_blocks_when_below_min_series(run_id, tmp_outputs):
    playbook = {**PLAYBOOK, "min_series": 3}
    with pytest.raises(PreflightBlockingError) as exc_info:
        run_preflight(run_id, _csv(), domain="fmcg", playbook=playbook)
    assert any(i.code == "BELOW_MIN_SERIES" for i in exc_info.value.issues)


def test_preflight_adds_short_history_warning(run_id, tmp_outputs):
    playbook = {**PLAYBOOK, "min_history_periods": 30}
    bundle = run_preflight(run_id, _csv(n_weeks=12), domain="fmcg", playbook=playbook)
    assert any(w.code == "SHORT_HISTORY" for w in bundle.data_quality_report.warnings)


def test_preflight_rerun_replaces_series_for_same_run_id(run_id, tmp_outputs):
    run_preflight(run_id, _csv_for_skus(["SKU_A", "SKU_B"]), domain="fmcg", playbook=PLAYBOOK)
    run_preflight(run_id, _csv_for_skus(["SKU_Z"]), domain="fmcg", playbook=PLAYBOOK)
    assert set(get_series_keys(run_id)) == {"SKU_Z|NORTH"}


def test_preflight_write_failure_keeps_store_untouched(run_id, tmp_outputs, monkeypatch):
    replace_run(run_id, {"EXISTING|NORTH": json_to_df("2024-01-01", 5.0)})

    def _boom(self, target):
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated write failure"):
        run_preflight(run_id, _csv_for_skus(["SKU_NEW"]), domain="fmcg", playbook=PLAYBOOK)

    assert set(get_series_keys(run_id)) == {"EXISTING|NORTH"}


def json_to_df(date: str, demand: float):
    import pandas as pd

    return pd.DataFrame({"date": [date], "demand": [demand]})
