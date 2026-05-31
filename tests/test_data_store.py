import pandas as pd
import pytest
from forecasting.data_store import (
    SeriesNotFoundError,
    delete_run,
    filename_to_key,
    get_series,
    get_series_keys,
    key_to_filename,
    store_series,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2024-01-01"], "demand": [10.0]})


def test_store_and_retrieve(run_id):
    df = make_df()
    store_series(run_id, "SKU_A|NORTH", df)
    result = get_series(run_id, "SKU_A|NORTH")
    assert list(result.columns) == ["date", "demand"]
    assert result["demand"].iloc[0] == 10.0


def test_get_series_keys(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    store_series(run_id, "SKU_B|NORTH", make_df())
    keys = get_series_keys(run_id)
    assert set(keys) == {"SKU_A|NORTH", "SKU_B|NORTH"}


def test_series_not_found_raises(run_id):
    with pytest.raises(SeriesNotFoundError) as exc_info:
        get_series(run_id, "MISSING")
    assert run_id in str(exc_info.value)
    assert "MISSING" in str(exc_info.value)


def test_delete_run_clears_all(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    delete_run(run_id)
    assert get_series_keys(run_id) == []


def test_delete_run_no_op_on_missing():
    delete_run("nonexistent-run")


def test_overwrite_existing_series(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    new_df = pd.DataFrame({"date": ["2024-02-01"], "demand": [99.0]})
    store_series(run_id, "SKU_A|NORTH", new_df)
    assert get_series(run_id, "SKU_A|NORTH")["demand"].iloc[0] == 99.0


def test_post_store_mutation_does_not_alter_stored_data(run_id):
    df = make_df()
    store_series(run_id, "SKU_A|NORTH", df)
    df.loc[0, "demand"] = 77.0
    assert get_series(run_id, "SKU_A|NORTH")["demand"].iloc[0] == 10.0


def test_post_read_mutation_does_not_alter_stored_data(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    result = get_series(run_id, "SKU_A|NORTH")
    result.loc[0, "demand"] = 88.0
    assert get_series(run_id, "SKU_A|NORTH")["demand"].iloc[0] == 10.0


def test_key_filename_round_trip():
    key = "SKU_A|NORTH"
    assert filename_to_key(key_to_filename(key)) == key


def test_pipe_encodes_to_percent_7c():
    assert key_to_filename("SKU_A|NORTH") == "SKU_A%7CNORTH"


def test_non_collision_examples_produce_different_filenames():
    first = key_to_filename("SKU_A|NORTH")
    second = key_to_filename("SKU|A_NORTH")
    assert first != second


def test_keys_without_pipes_round_trip_unchanged():
    key = "SKUA_NORTH"
    assert key_to_filename(key) == key
    assert filename_to_key(key) == key
