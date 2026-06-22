import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import clsx from "clsx";
import { PageHeader } from "@/components/PageHeader";
import { StatusChip, type StatusChipTone } from "@/components/StatusChip";
import {
  useAdvanceRun,
  useCockpitState,
  usePostMessage,
  useUploadCsv,
} from "@/api/hooks";
import type { CockpitStateResponse, MessageResponse, Possibility } from "@/api/client";

/**
 * RunConsole — Phase 10 CB6.
 *
 * The cockpit's primary page for driving a Run through the
 * lifecycle. Owns the upload form, the chat loop with
 * Meridian, and the driver-button advance. Three layouts by URL:
 *
 * - ``/`` (no runId) → upload form. Submitting the form
 *   navigates to ``/runs/${runId}/console`` so the same page
 *   takes over with the live state.
 * - ``/runs/:runId/console`` with ``:runId`` → two-tab
 *   surface: left rail polls ``GET /cockpit-state/:runId``;
 *   main panel has the Scope (chat) and Run (advance) tabs.
 *
 * The page is a thin orchestrator over the cb6.2 hooks
 * (``useUploadCsv`` / ``usePostMessage`` / ``useAdvanceRun``
 * / ``useCockpitState``); the visual logic lives in
 * <ScopeTab> + <RunTab> + <CockpitRail> below.
 */
export function RunConsole(): JSX.Element {
  const { runId } = useParams<{ runId?: string }>();
  const navigate = useNavigate();

  if (!runId) {
    return <UploadView onUploaded={(id) => navigate(`/runs/${id}/console`)} />;
  }
  return <ConsoleView runId={runId} />;
}

// ---------------------------------------------------------------------------
// Upload view — shown at "/"
// ---------------------------------------------------------------------------

function UploadView(props: {
  onUploaded: (runId: string) => void;
}): JSX.Element {
  const upload = useUploadCsv();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [domain, setDomain] = useState<string>("fmcg");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      return;
    }
    upload.mutate(
      { file, domain },
      {
        onSuccess: (response) => props.onUploaded(response.run_id),
      },
    );
  };

  return (
    <div className="flex flex-col gap-stack-lg">
      <PageHeader
        eyebrow="Run Console"
        title="Start a new run"
        subtitle="Upload an FMCG CSV — preflight runs synchronously, then the chat loop opens."
      />
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-stack-md rounded-card bg-surface-container-lowest p-stack-lg shadow-card"
        aria-label="Upload CSV form"
      >
        <label className="flex flex-col gap-stack-xs text-body-md">
          <span className="font-medium">Domain</span>
          <select
            value={domain}
            onChange={(event) => setDomain(event.target.value)}
            className="rounded-input border border-border-slate bg-surface-container-lowest px-stack-sm py-stack-xs"
            aria-label="Domain"
          >
            {/* Phase 10 ships FMCG only; future domains land here. */}
            <option value="fmcg">FMCG (fast-moving consumer goods)</option>
          </select>
        </label>
        <label className="flex flex-col gap-stack-xs text-body-md">
          <span className="font-medium">CSV file</span>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="text-body-sm"
            aria-label="CSV file"
          />
        </label>
        <div className="flex items-center gap-stack-md">
          <button
            type="submit"
            disabled={upload.isPending}
            className="rounded-button bg-primary px-stack-md py-stack-sm font-medium text-on-primary shadow-card transition-colors hover:bg-primary/90 disabled:bg-text-muted"
            aria-label="Upload"
          >
            {upload.isPending ? "Uploading…" : "Upload"}
          </button>
          {upload.isError && (
            <StatusChip
              label={`Upload failed: ${upload.error.message}`}
              tone="critical"
            />
          )}
        </div>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Console view — shown at "/runs/:runId/console"
// ---------------------------------------------------------------------------

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  possibilities?: Possibility[];
}

function ConsoleView(props: { runId: string }): JSX.Element {
  const { runId } = props;
  const cockpit = useCockpitState(runId);
  const post = usePostMessage();
  const advance = useAdvanceRun();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"scope" | "run">("scope");

  // When the chat-loop reply returns, append it to the message
  // stack. The conductor's reply carries the new possibilities
  // list, which the Scope tab renders as chips.
  useEffect(() => {
    if (!post.data) {
      return;
    }
    setTurns((current) => [
      ...current,
      {
        role: "assistant",
        content: post.data.reply,
        possibilities: post.data.possibilities,
      },
    ]);
    setDraft("");
  }, [post.data]);

  const submitMessage = (text: string): void => {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    setTurns((current) => [
      ...current,
      { role: "user", content: trimmed },
    ]);
    post.mutate({ run_id: runId, user_message: trimmed });
  };

  // The /messages reply may include a refreshed RunState; when
  // it does, ``useCockpitState`` is still polling so we don't
  // need to manually invalidate the cache here — the next poll
  // tick (≤5s) will pick up the new phase.
  const handleAdvance = (): void => {
    advance.mutate(
      { runId },
      {
        onSuccess: (response) => {
          // After a successful advance, jump to the Run tab so the
          // operator sees the updated report links.
          if (response.advanced_to === "report_ready") {
            setActiveTab("run");
          }
        },
      },
    );
  };

  const currentPhase = cockpit.data?.phase ?? "unknown";

  return (
    <div className="grid grid-cols-1 gap-stack-lg lg:grid-cols-[18rem_1fr]">
      <CockpitRail runId={runId} cockpit={cockpit.data ?? null} />
      <div className="flex flex-col gap-stack-md">
        <PageHeader
          eyebrow="Run Console"
          title={`Run ${runId}`}
          subtitle={
            currentPhase === "report_ready"
              ? "Report ready — open the surfaces from the Run tab."
              : "Scope the run with Meridian, then advance the pipeline."
          }
        />
        <TabBar active={activeTab} onChange={setActiveTab} />
        {activeTab === "scope" ? (
          <ScopeTab
            runId={runId}
            turns={turns}
            draft={draft}
            setDraft={setDraft}
            onSubmit={submitMessage}
            isPending={post.isPending}
            error={post.error}
          />
        ) : (
          <RunTab
            runId={runId}
            phase={currentPhase}
            onAdvance={handleAdvance}
            isAdvancing={advance.isPending}
            advanceError={advance.error}
            advanceReply={advance.data?.reply ?? null}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Left rail — live cockpit state, polled every 5s.
// ---------------------------------------------------------------------------

function CockpitRail(props: {
  runId: string;
  cockpit: CockpitStateResponse | null;
}): JSX.Element {
  const { cockpit } = props;
  const phaseTone = useMemo<StatusChipTone>(() => {
    if (!cockpit) return "neutral";
    if (cockpit.phase === "report_ready") return "success";
    if (cockpit.phase === "halted") return "critical";
    if (cockpit.phase === "foundry_modelling") return "warning";
    return "info";
  }, [cockpit]);

  return (
    <aside
      className="flex flex-col gap-stack-md rounded-card bg-surface-container-lowest p-stack-md shadow-card"
      aria-label="Cockpit state"
    >
      <h3 className="text-label-caps uppercase tracking-caps text-text-muted">
        Live state
      </h3>
      <div className="flex flex-col gap-stack-xs">
        <Row label="Phase">
          <StatusChip
            label={cockpit?.phase ?? "loading…"}
            tone={phaseTone}
          />
        </Row>
        <Row label="Step">{cockpit?.current_step ?? "—"}</Row>
        <Row label="Agent">{cockpit?.active_agent ?? "—"}</Row>
        <Row label="Confidence">
          {cockpit?.confidence ?? <span className="text-text-muted">—</span>}
        </Row>
      </div>
      {cockpit?.blockers && cockpit.blockers.length > 0 && (
        <div className="flex flex-col gap-stack-xs border-t border-border-slate pt-stack-sm">
          <h4 className="text-label-caps uppercase tracking-caps text-text-muted">
            Blockers
          </h4>
          <ul className="flex flex-col gap-stack-xs text-body-sm">
            {cockpit.blockers.map((blocker, index) => (
              <li
                key={`${blocker}-${index}`}
                className="flex items-start gap-stack-xs"
              >
                <StatusChip label="blocked" tone="critical" />
                <span>{blocker}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
}

function Row(props: { label: string; children: React.ReactNode }): JSX.Element {
  return (
    <div className="flex items-center justify-between gap-stack-sm text-body-sm">
      <span className="text-text-muted">{props.label}</span>
      <span className="font-medium">{props.children}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function TabBar(props: {
  active: "scope" | "run";
  onChange: (next: "scope" | "run") => void;
}): JSX.Element {
  return (
    <div
      className="flex gap-stack-xs rounded-card bg-surface-container p-stack-xs shadow-card"
      role="tablist"
      aria-label="Run console tabs"
    >
      {(["scope", "run"] as const).map((id) => (
        <button
          key={id}
          type="button"
          role="tab"
          aria-selected={props.active === id}
          onClick={() => props.onChange(id)}
          className={clsx(
            "flex-1 rounded-input px-stack-sm py-stack-xs text-body-sm font-medium transition-colors",
            props.active === id
              ? "bg-surface-container-lowest text-primary shadow-card"
              : "text-text-muted hover:text-on-surface",
          )}
        >
          {id === "scope" ? "Scope" : "Run"}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scope tab — chat loop with Meridian.
// ---------------------------------------------------------------------------

function ScopeTab(props: {
  runId: string;
  turns: ChatTurn[];
  draft: string;
  setDraft: (value: string) => void;
  onSubmit: (text: string) => void;
  isPending: boolean;
  error: Error | null;
}): JSX.Element {
  const { turns, draft, setDraft, onSubmit, isPending, error } = props;
  // The last assistant turn drives the possibility chips
  // rendered below the input. ``findLast`` is the cleanest way
  // to express "find the most recent assistant message" without
  // the noUncheckedIndexedAccess dance.
  const lastAssistantTurn = useMemo<ChatTurn | null>(() => {
    for (let index = turns.length - 1; index >= 0; index--) {
      const turn = turns[index];
      if (turn && turn.role === "assistant") {
        return turn;
      }
    }
    return null;
  }, [turns]);

  return (
    <section
      className="flex flex-col gap-stack-md rounded-card bg-surface-container-lowest p-stack-md shadow-card"
      aria-label="Chat with Meridian"
    >
      <div
        className="flex max-h-[60vh] flex-col gap-stack-sm overflow-y-auto"
        aria-live="polite"
        data-testid="chat-history"
      >
        {turns.length === 0 && (
          <p className="text-body-sm text-text-muted">
            Say hi to Meridian to start scoping this run.
          </p>
        )}
        {turns.map((turn, index) => (
          <ChatBubble key={`${turn.role}-${index}`} turn={turn} />
        ))}
        {isPending && <StatusChip label="Meridian is typing…" tone="info" />}
      </div>
      <form
        className="flex flex-col gap-stack-sm"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(draft);
        }}
        aria-label="Send message"
      >
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          rows={3}
          placeholder="Type a scope answer or paste a Meridian chip label…"
          className="w-full resize-y rounded-input border border-border-slate bg-surface-container-lowest p-stack-sm text-body-md"
          aria-label="Message"
          data-testid="chat-input"
        />
        <div className="flex items-center gap-stack-md">
          <button
            type="submit"
            disabled={isPending || draft.trim().length === 0}
            className="rounded-button bg-primary px-stack-md py-stack-sm font-medium text-on-primary shadow-card transition-colors hover:bg-primary/90 disabled:bg-text-muted"
            data-testid="chat-send"
          >
            {isPending ? "Sending…" : "Send"}
          </button>
          {error && (
            <StatusChip
              label={`Send failed: ${error.message}`}
              tone="critical"
            />
          )}
        </div>
      </form>
      {lastAssistantTurn?.possibilities &&
        lastAssistantTurn.possibilities.length > 0 && (
          <div
            className="flex flex-wrap items-center gap-stack-xs border-t border-border-slate pt-stack-sm"
            data-testid="possibility-chips"
          >
            <span className="text-label-caps uppercase tracking-caps text-text-muted">
              Choices
            </span>
            {lastAssistantTurn.possibilities.map((possibility, index) => (
              <button
                key={`${possibility.label}-${index}`}
                type="button"
                onClick={() => onSubmit(possibility.label)}
                className="rounded-full border border-border-slate bg-surface-container-lowest px-stack-sm py-stack-xs text-body-sm transition-colors hover:border-primary hover:text-primary"
                data-testid="possibility-chip"
              >
                {possibility.label}
              </button>
            ))}
          </div>
        )}
    </section>
  );
}

function ChatBubble(props: { turn: ChatTurn }): JSX.Element {
  const { turn } = props;
  const isAssistant = turn.role === "assistant";
  return (
    <div
      className={clsx(
        "flex max-w-[80%] flex-col gap-stack-xs rounded-card p-stack-sm text-body-sm shadow-card",
        isAssistant
          ? "self-start bg-primary/5 text-on-surface"
          : "self-end bg-secondary-container/15 text-on-surface",
      )}
      data-testid={`chat-bubble-${turn.role}`}
    >
      <span className="text-label-caps uppercase tracking-caps text-text-muted">
        {turn.role === "assistant" ? "Meridian" : "You"}
      </span>
      <span>{turn.content}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run tab — advance button + report link when ready.
// ---------------------------------------------------------------------------

function RunTab(props: {
  runId: string;
  phase: string;
  onAdvance: () => void;
  isAdvancing: boolean;
  advanceError: Error | null;
  advanceReply: string | null;
}): JSX.Element {
  const { runId, phase, onAdvance, isAdvancing, advanceError, advanceReply } =
    props;
  const isReportReady = phase === "report_ready";
  const isMeridianScoping = phase === "meridian_scoping";
  const advanceDisabled = isAdvancing || isMeridianScoping;

  return (
    <section
      className="flex flex-col gap-stack-md rounded-card bg-surface-container-lowest p-stack-md shadow-card"
      aria-label="Driver controls"
    >
      <div className="flex flex-col gap-stack-xs">
        <span className="text-label-caps uppercase tracking-caps text-text-muted">
          Current phase
        </span>
        <StatusChip label={phase} tone="info" />
      </div>
      {isReportReady ? (
        <div
          className="flex flex-col gap-stack-sm"
          data-testid="report-links"
        >
          <span className="text-body-md">
            Report ready — open a surface to review the run.
          </span>
          <div className="flex flex-wrap gap-stack-md">
            <a
              href={`/surfaces/forecast_review/${runId}`}
              className="rounded-button bg-primary px-stack-md py-stack-sm font-medium text-on-primary shadow-card"
            >
              Forecast review
            </a>
            <a
              href={`/surfaces/replenishment_board/${runId}`}
              className="rounded-button bg-secondary-container px-stack-md py-stack-sm font-medium text-on-surface shadow-card"
            >
              Replenishment board
            </a>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-stack-sm">
          <button
            type="button"
            onClick={onAdvance}
            disabled={advanceDisabled}
            data-testid="advance-button"
            className="self-start rounded-button bg-primary px-stack-md py-stack-sm font-medium text-on-primary shadow-card transition-colors hover:bg-primary/90 disabled:bg-text-muted"
          >
            {isAdvancing
              ? "Advancing…"
              : isMeridianScoping
                ? "Chat with Meridian to advance"
                : "Advance to next phase"}
          </button>
          {isMeridianScoping && (
            <p className="text-body-sm text-text-muted">
              The chat loop is the gate for this phase. Once you've
              answered Meridian, the &ldquo;Advance&rdquo; button becomes
              available again.
            </p>
          )}
          {advanceError && (
            <StatusChip
              label={`Advance failed: ${advanceError.message}`}
              tone="critical"
            />
          )}
          {advanceReply && (
            <StatusChip label={advanceReply} tone="info" />
          )}
        </div>
      )}
    </section>
  );
}

// Silence unused-import warning for MessageResponse — used as a
// type-only reference in the surrounding file.
export type { MessageResponse };