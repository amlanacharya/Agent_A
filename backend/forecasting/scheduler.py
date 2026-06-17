"""Phase 6 CB4: LocalScheduler — the in-process stand-in for UiPath Triggers.

The platform schedules recurring work (data refresh, validation,
forecast generation, review, monitoring, drift investigation). In
production those triggers live in UiPath Orchestrator (Time, Queue,
Event triggers). In this repo the default implementation is
``LocalScheduler`` — a pure-Python, in-process tick loop that:

* owns a list of ``ScheduledJobTrigger``s
* evaluates each trigger's ``cron`` expression on every ``tick()``
* calls an injected runner for every trigger that fires
* records each run to ``outputs/{run_id}/scheduler_runs.jsonl``
* refuses to run a trigger concurrently with itself — the previous
  run must be in a terminal state (``succeeded`` / ``failed`` /
  ``skipped``) before the next can fire

Design:

* Pure glue. No domain logic, no LLM calls, no platform imports
  beyond ``contracts.py``. The runner is injected; CB5 wires the
  real platform entry points.
* Deterministic with explicit ``now=``. Tests pass an explicit
  datetime; production calls ``tick()`` with no argument and the
  scheduler reads ``datetime.now(timezone.utc)``.
* Triggers persist to ``state_path`` (JSON); the file is rewritten
  on every mutation. On a process restart the scheduler rebuilds
  the trigger list from disk.
* Every ``ScheduledJobRun`` is appended to a per-run JSONL audit
  log at ``outputs/{run_id}/scheduler_runs.jsonl``. The directory
  layout mirrors the gateway's audit log so a single ``outputs``
  root serves both.

Supported cron forms (no full cron parser — keep it tiny):

* ``"every Nm"`` — every N minutes (``N`` in 1..59)
* ``"every Nh"`` — every N hours (``N`` in 1..23)
* ``"hourly"`` — top of every hour
* ``"daily HH:MM"`` — once per day at UTC HH:MM (24-hour clock)

Any other form raises ``InvalidCronExpressionError`` at
``register()`` time, not at tick time.
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from forecasting.contracts import (
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobTrigger,
)


Runner = Callable[[ScheduledJobTrigger], None]


class SchedulerError(Exception):
    """Base class for scheduler-level errors."""


class InvalidCronExpressionError(SchedulerError):
    """The cron string is not in one of the supported forms."""


class TriggerNotFoundError(SchedulerError):
    """The trigger_id is not registered."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Cron expression parser
# ---------------------------------------------------------------------------

# Allowed cron forms. The compiled regexes are intentionally strict:
# accepting a wider grammar is a common footgun and the only forms
# the platform emits are these four.
_RE_EVERY_M = re.compile(r"^every\s+([1-9]\d*)m$")
_RE_EVERY_H = re.compile(r"^every\s+([1-9]\d*)h$")
_RE_HOURLY = re.compile(r"^hourly$")
_RE_DAILY = re.compile(r"^daily\s+([01]\d|2[0-3]):([0-5]\d)$")

# Upper bound on every-N-hours to keep the tick math simple. The
# platform's schedules are fine-grained (minutes to daily); anything
# coarser belongs in a real cron daemon, not this scheduler.
_MAX_HOURS_INTERVAL = 23
_MAX_MINUTES_INTERVAL = 59


def _parse_cron(expr: str) -> Callable[[datetime], bool]:
    """Return a predicate ``now -> bool`` that fires the trigger.

    Raises ``InvalidCronExpressionError`` if the expression is not
    in one of the supported forms.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise InvalidCronExpressionError(
            f"Cron expression must be a non-empty string; got {expr!r}"
        )
    expr = expr.strip()
    if (m := _RE_EVERY_M.match(expr)) is not None:
        n = int(m.group(1))
        if n > _MAX_MINUTES_INTERVAL:
            raise InvalidCronExpressionError(
                f"'every {n}m' exceeds maximum {_MAX_MINUTES_INTERVAL}m"
            )

        def _fires(now: datetime) -> bool:
            return now.minute % n == 0

        return _fires
    if (m := _RE_EVERY_H.match(expr)) is not None:
        n = int(m.group(1))
        if n > _MAX_HOURS_INTERVAL:
            raise InvalidCronExpressionError(
                f"'every {n}h' exceeds maximum {_MAX_HOURS_INTERVAL}h"
            )

        def _fires(now: datetime) -> bool:
            return now.hour % n == 0 and now.minute == 0

        return _fires
    if _RE_HOURLY.match(expr) is not None:

        def _fires(now: datetime) -> bool:  # type: ignore[no-redef]
            return now.minute == 0

        return _fires
    if (m := _RE_DAILY.match(expr)) is not None:
        hh, mm = int(m.group(1)), int(m.group(2))

        def _fires(now: datetime) -> bool:  # type: ignore[no-redef]
            return now.hour == hh and now.minute == mm

        return _fires
    raise InvalidCronExpressionError(
        f"Unsupported cron expression: {expr!r}. "
        f"Supported forms: 'every Nm', 'every Nh', 'hourly', 'daily HH:MM'."
    )


# Statuses that mean "the previous run is still in progress" — the
# scheduler must not fire a second run on top of one of these.
_ACTIVE_STATUSES = ("running", "awaiting_approval")


# ---------------------------------------------------------------------------
# Scheduler interface
# ---------------------------------------------------------------------------


class Scheduler(ABC):
    """Interface every scheduler implements.

    The UiPath-side implementation translates the same trigger list
    into Orchestrator Triggers (Time, Queue, Event). The in-process
    implementation owns the trigger list and the tick loop.
    """

    @abstractmethod
    def register(self, trigger: ScheduledJobTrigger) -> None: ...

    @abstractmethod
    def unregister(self, trigger_id: str) -> None: ...

    @abstractmethod
    def list_triggers(self) -> list[ScheduledJobTrigger]: ...

    @abstractmethod
    def tick(
        self,
        runner: Runner,
        now: datetime | None = None,
    ) -> list[ScheduledJobRun]:
        """Evaluate every enabled trigger. For each one that fires,
        invoke ``runner(trigger)`` and record a ``ScheduledJobRun``.

        ``now`` is for testing; if None, the scheduler reads
        ``datetime.now(timezone.utc)``.

        Returns the list of runs produced by this tick, ordered by
        (started_at, trigger_id).
        """


# ---------------------------------------------------------------------------
# LocalScheduler
# ---------------------------------------------------------------------------


class LocalScheduler(Scheduler):
    """The in-process default scheduler.

    Persists triggers to ``state_path`` (JSON). The state file is
    rewritten on every register/unregister; on startup the file is
    loaded if present. Runs are appended to
    ``outputs/{run_id}/scheduler_runs.jsonl`` (a per-run audit log
    matching the gateway's convention).
    """

    def __init__(self, state_path: Path) -> None:
        self._state_path = Path(state_path)
        self._triggers: dict[str, ScheduledJobTrigger] = {}
        # Cache the parsed cron predicate per trigger_id so we don't
        # re-parse on every tick.
        self._predicates: dict[str, Callable[[datetime], bool]] = {}
        # Lazy-load on first read so an empty/missing file is fine.
        self._load()

    # ----- state persistence ------------------------------------------------

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        data = json.loads(self._state_path.read_text(encoding="utf-8"))
        triggers = data.get("triggers", [])
        for t in triggers:
            trigger = ScheduledJobTrigger.model_validate(t)
            # Re-parse the cron at load time so a bad trigger fails
            # loud at startup, not silently on the first tick.
            self._predicates[trigger.trigger_id] = _parse_cron(trigger.cron)
            self._triggers[trigger.trigger_id] = trigger

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "triggers": [t.model_dump(mode="json") for t in self._triggers.values()]
        }
        self._state_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    # ----- Scheduler surface -----------------------------------------------

    def register(self, trigger: ScheduledJobTrigger) -> None:
        # Parse first so a bad trigger is rejected before we mutate state.
        predicate = _parse_cron(trigger.cron)
        self._triggers[trigger.trigger_id] = trigger
        self._predicates[trigger.trigger_id] = predicate
        self._save()

    def unregister(self, trigger_id: str) -> None:
        if trigger_id not in self._triggers:
            raise TriggerNotFoundError(f"Unknown trigger_id: {trigger_id}")
        del self._triggers[trigger_id]
        self._predicates.pop(trigger_id, None)
        self._save()

    def list_triggers(self) -> list[ScheduledJobTrigger]:
        # Return in a stable order: by trigger_id.
        return sorted(self._triggers.values(), key=lambda t: t.trigger_id)

    def tick(
        self,
        runner: Runner,
        now: datetime | None = None,
    ) -> list[ScheduledJobRun]:
        when = now if now is not None else datetime.now(timezone.utc)
        runs: list[ScheduledJobRun] = []
        for trigger in self.list_triggers():
            if not trigger.enabled:
                continue
            predicate = self._predicates[trigger.trigger_id]
            if not predicate(when):
                continue
            run = self._fire(trigger, runner, when)
            runs.append(run)
        # Stable order: started_at first, then trigger_id.
        runs.sort(key=lambda r: (r.started_at, r.trigger_id))
        return runs

    # ----- internals --------------------------------------------------------

    def _fire(
        self,
        trigger: ScheduledJobTrigger,
        runner: Runner,
        when: datetime,
    ) -> ScheduledJobRun:
        started_at = _now_iso()
        # Skip-while-active guard.
        if self._has_active_run(trigger.trigger_id):
            run = ScheduledJobRun(
                run_id=_new_id("run"),
                trigger_id=trigger.trigger_id,
                kind=trigger.kind,
                started_at=started_at,
                finished_at=started_at,  # immediate skip; no work done
                status="skipped",
                error="previous run still active",
            )
            self.record_run(run)
            return run
        # Try the runner. Any exception is captured as a failed run
        # so the scheduler can move on to the next trigger.
        try:
            runner(trigger)
        except Exception as exc:  # noqa: BLE001 — boundary: capture and continue
            run = ScheduledJobRun(
                run_id=_new_id("run"),
                trigger_id=trigger.trigger_id,
                kind=trigger.kind,
                started_at=started_at,
                finished_at=_now_iso(),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.record_run(run)
            return run
        run = ScheduledJobRun(
            run_id=_new_id("run"),
            trigger_id=trigger.trigger_id,
            kind=trigger.kind,
            started_at=started_at,
            finished_at=_now_iso(),
            status="succeeded",
        )
        self.record_run(run)
        return run

    def _has_active_run(self, trigger_id: str) -> bool:
        """True if the most recent run for this trigger is in an
        active (non-terminal) state.

        Reads the per-trigger audit log and looks at the last entry.
        Cheap for a single-process POC; the full-chain test (CB5)
        can add an in-memory cache if the log grows.
        """
        trigger = self._triggers.get(trigger_id)
        if trigger is None:
            return False
        log_path = self._run_log_path(trigger)
        if not log_path.exists():
            return False
        last_line = ""
        for line in log_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                last_line = stripped
        if not last_line:
            return False
        last = ScheduledJobRun.model_validate_json(last_line)
        return last.status in _ACTIVE_STATUSES

    def _run_log_path(self, trigger: ScheduledJobTrigger) -> Path:
        """Audit log path for one trigger.

        Keyed on the **trigger's** run_id (the parent run the trigger
        is doing work for), not the per-tick record id. This groups
        all runs belonging to the same parent run under one log file
        and lets the skip check look at the right history.

        Layout: ``{state_path parent}/outputs/{run_id}/scheduler_runs.jsonl``.
        Anchored to the state file's parent (which is typically a
        ``tmp_path`` in tests) so tests don't write into
        ``backend/outputs/`` by accident.
        """
        run_id = trigger.run_id or "default"
        return self._state_path.parent / "outputs" / run_id / "scheduler_runs.jsonl"

    # ----- public helpers (used by tests + CB5) ----------------------------

    def record_run(self, run: ScheduledJobRun) -> None:
        """Append a run to the per-trigger audit log.

        The log path is derived from the **trigger** the run belongs
        to (looked up by ``run.trigger_id``), so all runs of the same
        trigger land in the same file. Public so tests can seed the
        log with a known 'running' / 'awaiting_approval' prior run to
        exercise the skip path, and so the full-chain integration
        test (CB5) can park a run at 'awaiting_approval' when an
        approval request is raised.
        """
        trigger = self._triggers.get(run.trigger_id)
        # If the trigger was unregistered between fire-time and now,
        # fall back to the 'default' log so the audit trail is not
        # silently dropped.
        path = (
            self._run_log_path(trigger)
            if trigger is not None
            else self._state_path.parent / "outputs" / "default" / "scheduler_runs.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(run.model_dump_json())
            f.write("\n")
