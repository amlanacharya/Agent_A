import clsx from "clsx";

/**
 * StatusChip — pill badge for status / tier indicators.
 *
 * Variants map to the DESIGN.md semantic palette (success-teal,
 * warning-amber, critical-rose, primary). Low-saturation backgrounds
 * + saturated text per DESIGN.md "Chips & Badges" section.
 */
export type StatusChipTone =
  | "success"
  | "warning"
  | "critical"
  | "info"
  | "neutral";

const TONE_CLASSES: Record<StatusChipTone, string> = {
  success: "bg-success-teal/10 text-success-teal",
  warning: "bg-warning-amber/10 text-warning-amber",
  critical: "bg-critical-rose/10 text-critical-rose",
  info: "bg-primary/10 text-primary",
  neutral: "bg-surface-container-high text-text-muted",
};

export interface StatusChipProps {
  /** Visible text (usually 1–3 words). */
  label: string;
  /** Tone drives background + text color. */
  tone?: StatusChipTone;
  /** Optional left-side icon. */
  icon?: string;
}

export function StatusChip({
  label,
  tone = "neutral",
  icon,
}: StatusChipProps): JSX.Element {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-stack-sm rounded-full px-stack-md py-stack-sm font-mono text-label-caps uppercase",
        TONE_CLASSES[tone],
      )}
    >
      {icon ? <span aria-hidden>{icon}</span> : null}
      {label}
    </span>
  );
}
