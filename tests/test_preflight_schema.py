import pandas as pd

from forecasting.tools.preflight_schema import (
    _normalise_key,
    _parse_dates,
    build_series_keys,
    detect_frequency_and_grain,
    map_schema,
    profile_uploaded_data,
)

PLAYBOOK = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 4,
}


def make_df(n_weeks: int = 10) -> pd.DataFrame:
    rows = []
    for sku in ["A", "B"]:
        for w in range(n_weeks):
            rows.append(
                {
                    "week": f"2024-W{w + 1:02d}",
                    "sku": sku,
                    "region": "NORTH",
                    "demand": float(w + 1),
                }
            )
    return pd.DataFrame(rows)


def test_profile_no_blocking_issues_and_series_count_provisional_zero():
    rpt = profile_uploaded_data(make_df())
    assert rpt.blocking_issues == []
    assert rpt.series_count == 0


def test_profile_all_zero_demand_blocks():
    df = make_df()
    df["demand"] = 0.0
    rpt = profile_uploaded_data(df)
    assert any(b.code == "ALL_ZERO_DEMAND" for b in rpt.blocking_issues)


def test_profile_missing_demand_col_blocks():
    df = make_df().drop(columns=["demand"])
    rpt = profile_uploaded_data(df)
    assert any(b.code == "MISSING_DEMAND_COLUMN" for b in rpt.blocking_issues)


def test_profile_missing_date_col_blocks():
    df = pd.DataFrame({"sku": ["A", "B"], "region": ["NORTH", "NORTH"], "demand": [1.0, 2.0]})
    rpt = profile_uploaded_data(df)
    assert any(b.code == "MISSING_DATE_COLUMN" for b in rpt.blocking_issues)


def test_map_schema_detects_columns():
    df = make_df()
    schema = map_schema(df, PLAYBOOK)
    assert schema.date_col == "week"
    assert schema.demand_col == "demand"
    assert set(schema.grain_cols) == {"sku", "region"}


def test_detect_frequency_weekly():
    schema = map_schema(make_df(52), PLAYBOOK)
    grain_rpt = detect_frequency_and_grain(make_df(52), schema)
    assert grain_rpt.detected_frequency == "weekly"


def test_build_series_keys_pipe_delimited():
    schema = map_schema(make_df(), PLAYBOOK)
    series_map = build_series_keys(make_df(), schema, PLAYBOOK)
    for key in series_map:
        assert "|" in key


def test_series_keys_uppercased_and_normalized():
    df = make_df()
    df["sku"] = ["a-1"] * len(df)
    schema = map_schema(df, PLAYBOOK)
    series_map = build_series_keys(df, schema, PLAYBOOK)
    for key in series_map:
        assert key == key.upper()
        assert key.split("|")[0] == "A1"


def test_series_df_has_exact_date_and_demand_cols():
    schema = map_schema(make_df(), PLAYBOOK)
    series_map = build_series_keys(make_df(), schema, PLAYBOOK)
    for sub_df in series_map.values():
        assert list(sub_df.columns) == ["date", "demand"]


def test_parse_dates_supports_iso_week_as_monday():
    parsed = _parse_dates(pd.Series(["2024-W01", "2024-W02"]))
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01")
    assert parsed.iloc[1] == pd.Timestamp("2024-01-08")


def test_normalise_key_enforces_allowed_charset():
    key = _normalise_key(("sku 1", "north/west"))
    assert key == "SKU_1|NORTHWEST"

