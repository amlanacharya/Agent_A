"""Domain playbook loader for Phase 10 (cockpit driver).

A domain playbook is the dict the Preflight layer consumes to map a
domain's CSV columns (e.g. FMCG's ``week/sku/region/demand``) to the
canonical schema. Today only the FMCG playbook is shipped; adding a
new domain is one new factory function + one entry in
:func:`playbook_for`.

A future phase can swap this in-memory dict for a YAML loader that
reads ``playbooks/*.yaml`` files — the public surface (``playbook_for``)
stays the same.
"""
from __future__ import annotations


FMCG_PLAYBOOK: dict[str, object] = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 12,
}


def fmcg_playbook() -> dict[str, object]:
    """Return the canonical FMCG playbook (a fresh dict per call).

    Returns a copy so callers cannot mutate the module-level constant
    across runs.
    """
    return dict(FMCG_PLAYBOOK)


_REGISTRY: dict[str, callable] = {
    "fmcg": fmcg_playbook,
}


def playbook_for(domain: str) -> dict[str, object]:
    """Return the playbook dict for ``domain``.

    Raises ``ValueError`` for unknown domains so the HTTP layer can
    translate to a 400 with a clear message. The closed set of
    shipped domains is ``fmcg`` today; Phase 10.x may add retail /
    hospitality / etc.
    """
    loader = _REGISTRY.get(domain.lower())
    if loader is None:
        raise ValueError(
            f"unknown domain {domain!r}; shipped domains are {sorted(_REGISTRY)}"
        )
    return loader()


__all__ = ("fmcg_playbook", "playbook_for")
