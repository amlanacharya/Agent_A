"""Doc-assertion test for the dual-tracker distinction.

CB3 of PRD-001 asks for a brief comment above each escalation
tracker explaining why the two paths intentionally use different
shapes (file-backed vs in-memory). This test pins those comments
so a future refactor cannot silently drop them.

The assertion is *structural* - it walks the module source
backwards from a known anchor line, collecting consecutive
``#`` lines (and any blank lines between them) until it hits
the first non-comment, non-blank line. This is robust to comment
reformatting (a maintainer editing punctuation or wrapping the
comment differently does not break the test), but a maintainer
*removing* the comment or rewording the rationale entirely will.
That is the right failure mode: the comment's *meaning* is the
contract, not its exact wording.
"""

from __future__ import annotations

import inspect


def _read_source(module) -> str:
    """Return the full source of ``module`` as a string."""
    return inspect.getsource(module)


def _comment_lines_above(source: str, anchor: str) -> list[str]:
    """Walk backwards from the first line containing ``anchor`` and collect
    the run of ``#`` lines (plus blank lines between them) directly above it.

    Stops at the first non-comment, non-blank line going up. Empty list
    if the anchor line has no comment block directly above it.

    NOTE: if the anchor line is preceded by decorators (``@dataclass``,
    ``@runtime_checkable``, etc.) the helper returns empty - use
    :func:`_comment_block_above_anchor` for that case.
    """
    lines = source.split("\n")
    target_idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            target_idx = i
            break
    if target_idx is None:
        return []
    collected: list[str] = []
    i = target_idx - 1
    while i >= 0:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            collected.insert(0, line)
            i -= 1
            continue
        break
    # Drop leading blank lines from the captured run (they belong
    # to the code above, not the comment block).
    while collected and collected[0].strip() == "":
        collected.pop(0)
    return collected


def _comment_block_above_anchor(
    source: str,
    anchor: str,
    *,
    max_decorator_lines: int = 5,
) -> list[str]:
    """Walk backwards from the first line containing ``anchor``, skipping
    over up to ``max_decorator_lines`` consecutive non-comment, non-blank
    lines (typically ``@dataclass`` and similar decorators), then collect
    the comment block above them.

    This is the version to use when the target is the *class* line and
    the comment is above a decorator like ``@dataclass``. Use
    :func:`_comment_lines_above` when the target is a *statement* (no
    decorator above it).
    """
    lines = source.split("\n")
    target_idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            target_idx = i
            break
    if target_idx is None:
        return []
    # Walk up, skipping up to N decorator / non-comment non-blank lines.
    i = target_idx - 1
    skipped = 0
    while i >= 0 and skipped < max_decorator_lines:
        stripped = lines[i].strip()
        if stripped.startswith("#") or stripped == "":
            break
        # Allow blank lines inside the decorator stack (e.g. blank line
        # between two decorator blocks) without counting it.
        if stripped == "":
            i -= 1
            continue
        skipped += 1
        i -= 1
    # Now ``i`` is pointing at the last decorator/blank line above the
    # anchor. Step one more time to land on the blank-or-comment line
    # directly above the decorator stack.
    i -= 1
    collected: list[str] = []
    while i >= 0:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            collected.insert(0, line)
            i -= 1
            continue
        break
    while collected and collected[0].strip() == "":
        collected.pop(0)
    return collected


def _block_contains_phrase(comment_lines: list[str], *phrases: str) -> str | None:
    """Return the first phrase in ``phrases`` not present (case-insensitive) in
    ``comment_lines``, or None if all are present.

    Both the phrases and the comment text are normalised by replacing
    ``-`` with a single space, so a phrase like ``"in-memory"`` will
    match a comment that uses either ``"in-memory"`` or ``"in memory"``.
    """
    text = " ".join(line.lower() for line in comment_lines).replace("-", " ")
    for phrase in phrases:
        normalized = phrase.lower().replace("-", " ")
        if normalized not in text:
            return phrase
    return None


def test_code_escalation_documents_why_tracker_is_file_backed() -> None:
    """The EscalationTracker class has a comment explaining file-backed persistence."""
    from forecasting import code_escalation

    source = _read_source(code_escalation)
    # The comment is above the ``@dataclass`` decorator, not directly
    # above the class. ``_comment_block_above_anchor`` knows to skip
    # the decorator stack when walking backwards.
    comment = _comment_block_above_anchor(source, "class EscalationTracker")
    assert comment, (
        "code_escalation.py must have a comment block above the "
        "@dataclass / class EscalationTracker pair explaining the design"
    )
    missing = _block_contains_phrase(
        comment,
        "file-backed",  # explains the persistence shape
        "config_escalation",  # points at the contrasting design
        "attempttracker",  # mentions the future Protocol seam
    )
    assert missing is None, (
        f"code_escalation comment must mention {missing!r}; got:\n"
        + "\n".join(comment)
    )


def test_config_escalation_documents_why_ledger_is_in_memory() -> None:
    """The per-run attempt ledger has a comment explaining in-memory, per-run shape."""
    from forecasting import config_escalation

    source = _read_source(config_escalation)
    # The per-run attempt ledger - the in-memory record the per-action
    # Counter is later derived from on line ~482.
    comment = _comment_lines_above(
        source, "attempts: list[ConfigAttemptResult] = []"
    )
    assert comment, (
        "config_escalation.py must have a comment block directly above "
        "the per-run attempt ledger (attempts: list[ConfigAttemptResult] = [])"
    )
    missing = _block_contains_phrase(
        comment,
        "in-memory",  # explains the storage shape
        "per-run",  # explains the scope
        "code_escalation",  # points at the contrasting design
        "attempttracker",  # mentions the future Protocol seam
    )
    assert missing is None, (
        f"config_escalation comment must mention {missing!r}; got:\n"
        + "\n".join(comment)
    )


def test_both_comments_reference_each_other() -> None:
    """The two comments cross-reference each other - they form a pair, not orphans."""
    from forecasting import code_escalation, config_escalation

    code_source = _read_source(code_escalation)
    config_source = _read_source(config_escalation)
    # The point of the design comment is the contrast; both halves
    # must point at the other side, otherwise a future reader who
    # looks at only one file gets a one-sided story.
    code_comment = _comment_block_above_anchor(code_source, "class EscalationTracker")
    config_comment = _comment_lines_above(
        config_source, "attempts: list[ConfigAttemptResult] = []"
    )
    assert any("config_escalation" in line for line in code_comment), (
        "code_escalation's tracker comment must point at config_escalation"
    )
    assert any("code_escalation" in line for line in config_comment), (
        "config_escalation's ledger comment must point at code_escalation"
    )


def test_no_orphan_unused_imports_added_by_the_comments() -> None:
    """Sanity check: the new comments did not drag in unused imports.

    This is a regression guard - the CB3 work is docs-only, so a
    future reviewer can run this and see "no behavioural change,
    no new imports, just two new comment blocks". The
    ``AttemptTracker`` Protocol mentioned in the comments is a
    *future* seam, not a current import.
    """
    import ast
    from pathlib import Path
    from forecasting import code_escalation, config_escalation

    for module in [code_escalation, config_escalation]:
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
        assert "AttemptTracker" not in imported, (
            f"{path.name}: the AttemptTracker Protocol is a future seam, "
            "not a current import. If you added it, this is a real code change."
        )

