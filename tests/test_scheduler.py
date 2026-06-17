"""Tests for Phase 6 CB4: LocalScheduler.

Covers the cron-style tick over the 6 ScheduledJobKinds. The scheduler
is glue: register triggers, tick, the runner is called for the ones
that fire, the resulting ScheduledJobRun is recorded. The runner is
injected so the scheduler stays decoupled from the platform's domain
code (CB5 wires the real platform entry points).

Cron expression forms (only these three are supported):

* ``"every Nm"`` / ``"every Nh"`` — every N minutes/hours
* ``"hourly"`` — top of every hour
* ``"daily HH:MM"`` — once per day at UTC HH:MM

Unsupported forms raise a clear error at register time, not at tick
time, so misconfigured triggers fail loud when configured rather than
silently at the first tick.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from forecasting.contracts import (
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobTrigger,
)
from forecasting.scheduler import (
    InvalidCronExpressionError,
    LocalScheduler,
    Scheduler,
    TriggerNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "scheduler_state.json"


@pytest.fixture()
def scheduler(state_path: Path) -> LocalScheduler:
    return LocalScheduler(state_path=state_path)


def make_trigger(
    trigger_id: str = "tr-1",
    cron: str = "hourly",
    kind: str = "data_refresh",
    enabled: bool = True,
) -> ScheduledJobTrigger:
    return ScheduledJobTrigger(
        trigger_id=trigger_id,
        kind=kind,  # type: ignore[arg-type]
        cron=cron,
        enabled=enabled,
        created_at="2026-06-17T00:00:00Z",
        created_by="tester",
    )


# ---------------------------------------------------------------------------
# ABC + interface
# ---------------------------------------------------------------------------


def test_local_scheduler_is_a_scheduler() -> None:
    """The in-process implementation satisfies the abstract interface."""
    assert issubclass(LocalScheduler, Scheduler)


def test_abstract_scheduler_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Scheduler()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# register / unregister / list_triggers
# ---------------------------------------------------------------------------


def test_register_persists_trigger(scheduler: LocalScheduler) -> None:
    trigger = make_trigger()
    scheduler.register(trigger)
    listed = scheduler.list_triggers()
    assert len(listed) == 1
    assert listed[0].trigger_id == "tr-1"
    assert listed[0].cron == "hourly"


def test_register_duplicate_id_replaces(
    scheduler: LocalScheduler,
) -> None:
    """Registering the same id twice replaces the prior trigger. The
    safer behaviour is to allow replacement so the user can edit a
    trigger's cron without removing and re-adding (which would race
    with a tick)."""
    scheduler.register(make_trigger(cron="hourly"))
    scheduler.register(make_trigger(cron="every 10m"))
    assert len(scheduler.list_triggers()) == 1
    assert scheduler.list_triggers()[0].cron == "every 10m"


def test_register_rejects_invalid_cron(scheduler: LocalScheduler) -> None:
    """Invalid forms fail at register time, not at tick time."""
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron="* * * * *"))  # full cron, not supported
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron="every 5s"))  # seconds not supported
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron="daily 25:00"))  # bad hour
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron="every 0m"))  # zero interval
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron=""))  # empty


def test_register_does_not_persist_invalid_trigger(
    scheduler: LocalScheduler,
) -> None:
    """A rejected register call leaves state untouched."""
    with pytest.raises(InvalidCronExpressionError):
        scheduler.register(make_trigger(cron="* * * * *"))
    assert scheduler.list_triggers() == []


def test_unregister_removes_trigger(scheduler: LocalScheduler) -> None:
    scheduler.register(make_trigger())
    scheduler.unregister("tr-1")
    assert scheduler.list_triggers() == []


def test_unregister_unknown_raises(scheduler: LocalScheduler) -> None:
    with pytest.raises(TriggerNotFoundError):
        scheduler.unregister("tr-missing")


def test_persistence_across_instance_replacement(state_path: Path) -> None:
    """Triggers survive a process restart (round-trip via JSON)."""
    s1 = LocalScheduler(state_path=state_path)
    s1.register(make_trigger(trigger_id="tr-1", cron="every 5m"))
    s1.register(make_trigger(trigger_id="tr-2", cron="hourly"))

    s2 = LocalScheduler(state_path=state_path)
    listed = s2.list_triggers()
    assert {t.trigger_id for t in listed} == {"tr-1", "tr-2"}
    # Order is not guaranteed (dict iteration); check both are present.


# ---------------------------------------------------------------------------
# tick: cron evaluation (deterministic with explicit `now=`)
# ---------------------------------------------------------------------------


def test_tick_every_5m_fires_on_5m_boundary(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="every 5m"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    assert len(runs) == 1
    assert runs[0].trigger_id == "tr-1"
    assert runs[0].status == "succeeded"


def test_tick_every_5m_does_not_fire_off_boundary(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="every 5m"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 3, tzinfo=timezone.utc)
    )
    assert runs == []


def test_tick_hourly_fires_at_top_of_hour(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="hourly"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    )
    assert len(runs) == 1


def test_tick_hourly_does_not_fire_at_minute_30(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="hourly"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 30, tzinfo=timezone.utc)
    )
    assert runs == []


def test_tick_daily_02_00_fires_at_02_00(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="daily 02:00"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 2, 0, tzinfo=timezone.utc)
    )
    assert len(runs) == 1


def test_tick_daily_02_00_does_not_fire_at_03_00(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="daily 02:00"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 3, 0, tzinfo=timezone.utc)
    )
    assert runs == []


def test_tick_disabled_trigger_never_fires(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(enabled=False, cron="every 5m"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    assert runs == []


def test_tick_two_triggers_evaluates_both(
    scheduler: LocalScheduler,
) -> None:
    """Two triggers, two independent fire schedules. The scheduler
    evaluates every trigger on every tick — neither one prevents the
    other from firing."""
    scheduler.register(make_trigger(trigger_id="tr-1", cron="every 7m"))
    scheduler.register(make_trigger(trigger_id="tr-2", cron="every 11m"))
    # At 12:05: neither fires (5 % 7 != 0, 5 % 11 != 0).
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    assert runs == []
    # At 12:07: tr-1 fires (7 % 7 == 0), tr-2 does not (7 % 11 != 0).
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 7, tzinfo=timezone.utc)
    )
    assert len(runs) == 1
    assert runs[0].trigger_id == "tr-1"
    # At 12:11: tr-2 fires (11 % 11 == 0), tr-1 does not (11 % 7 != 0).
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 11, tzinfo=timezone.utc)
    )
    assert len(runs) == 1
    assert runs[0].trigger_id == "tr-2"


def test_tick_returns_runs_ordered_by_started_at(
    scheduler: LocalScheduler,
) -> None:
    """When two triggers fire in the same tick, the returned list is
    ordered by started_at (and tied-start triggers by trigger_id, for
    determinism)."""
    scheduler.register(make_trigger(trigger_id="tr-b", cron="every 5m"))
    scheduler.register(make_trigger(trigger_id="tr-a", cron="every 5m"))
    runs = scheduler.tick(
        runner=lambda t: None, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    assert len(runs) == 2
    # Same started_at; fall back to trigger_id ordering.
    assert runs[0].trigger_id == "tr-a"
    assert runs[1].trigger_id == "tr-b"


# ---------------------------------------------------------------------------
# tick: runner integration
# ---------------------------------------------------------------------------


def test_tick_runner_is_called_once_per_fire(
    scheduler: LocalScheduler,
) -> None:
    scheduler.register(make_trigger(cron="every 5m"))
    calls: list[ScheduledJobTrigger] = []
    scheduler.tick(
        runner=calls.append,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(calls) == 1
    assert calls[0].trigger_id == "tr-1"


def test_tick_runner_can_return_status(
    scheduler: LocalScheduler,
) -> None:
    """The runner receives the trigger; the resulting run is recorded
    with status='succeeded' and a finished_at timestamp."""
    scheduler.register(make_trigger(cron="every 5m"))
    runs = scheduler.tick(
        runner=lambda t: None,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "succeeded"
    assert run.started_at != ""
    assert run.finished_at is not None
    assert run.error is None


def test_tick_runner_exception_records_failed_status(
    scheduler: LocalScheduler,
) -> None:
    """A runner that raises is caught: the run is recorded as 'failed'
    with the exception message in the error field, and the scheduler
    continues processing the other triggers (one runner's failure
    must not block the next)."""
    scheduler.register(make_trigger(trigger_id="tr-bad", cron="every 5m"))
    scheduler.register(make_trigger(trigger_id="tr-good", cron="every 5m"))

    def selective_runner(t: ScheduledJobTrigger) -> None:
        if t.trigger_id == "tr-bad":
            raise RuntimeError("simulated platform failure")

    runs = scheduler.tick(
        runner=selective_runner,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(runs) == 2
    by_id = {r.trigger_id: r for r in runs}
    assert by_id["tr-bad"].status == "failed"
    assert by_id["tr-bad"].error is not None
    assert "simulated platform failure" in by_id["tr-bad"].error
    assert by_id["tr-good"].status == "succeeded"


# ---------------------------------------------------------------------------
# tick: skip when previous run is still active
# ---------------------------------------------------------------------------


def test_tick_skips_when_previous_run_still_running(
    scheduler: LocalScheduler,
) -> None:
    """A trigger whose previous run is still active is skipped with a
    new run record status='skipped' and an explanatory error. The
    scheduler must never run a trigger concurrently with itself."""
    scheduler.register(make_trigger(cron="every 5m"))
    # First tick: runner records a "long-running" by raising a sentinel
    # that the runner-aware version of the test would hold; for this
    # test the simpler approach is to inject a "previous run still
    # active" state by writing a ScheduledJobRun with status='running'
    # to the run history. Easiest path: tick once with a runner that
    # claims awaiting_approval status (not 'succeeded' or 'failed'),
    # then tick again. The first tick returns a 'succeeded' run because
    # the runner returned cleanly. So we exercise skip via the public
    # surface: directly inject a previous 'running' run via the
    # scheduler's run-history helper.
    pass_marker = object()
    # The implementation exposes a way to inspect the last run. Use it
    # to verify the skip path.
    runner = lambda t: None  # noqa: E731
    scheduler.tick(
        runner=runner, now=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    )

    # Force a "previous run still active" by overriding the run history
    # directly. The cleanest way: write a JSONL run record with
    # status='running' to the audit log via the scheduler's helper.
    prior = ScheduledJobRun(
        run_id="run-prior",
        trigger_id="tr-1",
        kind="data_refresh",
        started_at="2026-06-17T12:00:00Z",
        status="running",
    )
    scheduler.record_run(prior)

    runs = scheduler.tick(
        runner=runner, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    assert len(runs) == 1
    assert runs[0].status == "skipped"
    assert runs[0].error is not None
    assert "previous run still active" in runs[0].error


def test_tick_skips_when_previous_run_awaiting_approval(
    scheduler: LocalScheduler,
) -> None:
    """Same skip behaviour when the previous run is parked at
    'awaiting_approval' — the scheduler must not stack a second run on
    top of one that is waiting for a human decision."""
    scheduler.register(make_trigger(cron="every 5m"))
    prior = ScheduledJobRun(
        run_id="run-prior",
        trigger_id="tr-1",
        kind="data_refresh",
        started_at="2026-06-17T12:00:00Z",
        status="awaiting_approval",
    )
    scheduler.record_run(prior)

    runs = scheduler.tick(
        runner=lambda t: None,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(runs) == 1
    assert runs[0].status == "skipped"


def test_tick_does_not_skip_when_previous_run_finished(
    scheduler: LocalScheduler,
) -> None:
    """A previous run that succeeded (or failed) is terminal — the
    next tick fires normally."""
    scheduler.register(make_trigger(cron="every 5m"))
    prior = ScheduledJobRun(
        run_id="run-prior",
        trigger_id="tr-1",
        kind="data_refresh",
        started_at="2026-06-17T12:00:00Z",
        finished_at="2026-06-17T12:00:30Z",
        status="succeeded",
    )
    scheduler.record_run(prior)

    runs = scheduler.tick(
        runner=lambda t: None,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(runs) == 1
    assert runs[0].status == "succeeded"


# ---------------------------------------------------------------------------
# Audit log (per-run JSONL at outputs/{run_id}/scheduler_runs.jsonl)
# ---------------------------------------------------------------------------


def test_tick_writes_runs_to_per_run_audit_log(
    scheduler: LocalScheduler, state_path: Path
) -> None:
    """Every ScheduledJobRun is written to
    outputs/{run_id}/scheduler_runs.jsonl. run_id is read from the
    trigger (default: created_run_id if absent, else 'default')."""
    scheduler.register(make_trigger(trigger_id="tr-1", cron="every 5m"))
    scheduler.tick(
        runner=lambda t: None,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    log_path = state_path.parent / "outputs" / "default" / "scheduler_runs.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["trigger_id"] == "tr-1"
    assert rec["status"] == "succeeded"


def test_tick_with_explicit_run_id_writes_to_that_run_log(
    scheduler: LocalScheduler, state_path: Path
) -> None:
    """A trigger with a non-null run_id writes its runs to the
    matching per-run audit log path."""
    trigger = make_trigger(trigger_id="tr-1", cron="every 5m")
    trigger_with_run = trigger.model_copy(update={"run_id": "run-X"})
    scheduler.register(trigger_with_run)
    scheduler.tick(
        runner=lambda t: None,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    log_path = state_path.parent / "outputs" / "run-X" / "scheduler_runs.jsonl"
    assert log_path.exists()


# ---------------------------------------------------------------------------
# End-to-end: register, tick, persist, tick again
# ---------------------------------------------------------------------------


def test_full_round_trip(scheduler: LocalScheduler) -> None:
    """A real usage: register an hourly trigger, tick at 02:00 (fires),
    tick at 02:30 (does not), tick at 03:00 (fires)."""
    scheduler.register(make_trigger(cron="hourly"))
    runner_calls: list[str] = []
    runner: Callable[[ScheduledJobTrigger], None] = lambda t: runner_calls.append(
        t.trigger_id
    )
    # (hour, minute, expected_fires). Hourly fires at minute=0.
    cases = [
        (2, 0, 1),
        (2, 30, 0),
        (3, 0, 1),
        (3, 15, 0),
    ]
    total_expected_fires = 0
    for hour, minute, expected in cases:
        runs = scheduler.tick(
            runner=runner,
            now=datetime(2026, 6, 17, hour, minute, tzinfo=timezone.utc),
        )
        assert len(runs) == expected
        total_expected_fires += expected
    assert runner_calls == ["tr-1"] * total_expected_fires
