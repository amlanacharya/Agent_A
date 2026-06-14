import io
import uuid
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture()
def run_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rows = []
    for sku in ["SKU_A", "SKU_B"]:
        for week in range(1, 53):
            rows.append({"week": f"2024-W{week:02d}", "sku": sku, "region": "NORTH", "demand": float(week % 10 + 1)})
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_outputs(tmp_path: Path, monkeypatch) -> Path:
    import forecasting.run_state as rs
    monkeypatch.setattr(rs, "OUTPUTS_ROOT", tmp_path)
    return tmp_path
