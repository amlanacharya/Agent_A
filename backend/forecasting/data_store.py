from typing import Dict

import pandas as pd

_store: Dict[str, Dict[str, pd.DataFrame]] = {}


class SeriesNotFoundError(Exception):
    def __init__(self, run_id: str, series_key: str):
        super().__init__(f"Series '{series_key}' not found for run '{run_id}'")
        self.run_id = run_id
        self.series_key = series_key


def store_series(run_id: str, series_key: str, df: pd.DataFrame) -> None:
    _store.setdefault(run_id, {})[series_key] = df


def get_series(run_id: str, series_key: str) -> pd.DataFrame:
    try:
        return _store[run_id][series_key]
    except KeyError:
        raise SeriesNotFoundError(run_id, series_key)


def get_series_keys(run_id: str) -> list[str]:
    return list(_store.get(run_id, {}).keys())


def delete_run(run_id: str) -> None:
    _store.pop(run_id, None)


def key_to_filename(series_key: str) -> str:
    """Reversible key-to-filename mapping for Windows-safe series file names.

    Windows file names cannot contain '|', so this mapping encodes it as '%7C'.
    The reverse mapping in filename_to_key restores the original series key.
    """
    return series_key.replace("|", "%7C")


def filename_to_key(stem: str) -> str:
    return stem.replace("%7C", "|")
