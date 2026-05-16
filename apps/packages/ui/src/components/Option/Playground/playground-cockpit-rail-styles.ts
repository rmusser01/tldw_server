export type CockpitRailTone =
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "focus"
  | "muted";

export const cockpitRailStyles = {
  stack: "flex min-w-0 flex-col gap-2 text-sm",
  section: "rounded-md border border-border/70 bg-surface px-3 py-2",
  heading: "text-[11px] font-semibold uppercase text-text-muted",
  value: "mt-1 text-sm font-medium text-text",
  muted: "mt-1 text-xs text-text-muted",
  action:
    "inline-flex min-h-[30px] items-center rounded-md border border-border/70 bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
  clearAction:
    "inline-flex shrink-0 items-center rounded border border-border/70 bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
  collapseAction:
    "inline-flex h-6 w-6 shrink-0 items-center justify-center rounded border border-border/70 bg-surface2 text-text-muted hover:bg-surface hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
  inset: "rounded-md border border-border/60 bg-bg/70 px-2.5 py-2",
  compactInset: "rounded-md border border-border/60 bg-bg/70 px-2 py-1.5",
  emptyInset:
    "rounded-md border border-dashed border-border/70 bg-bg/60 px-2.5 py-2 text-xs text-text-muted",
  tag: "rounded border border-border/70 bg-surface2 px-2 py-0.5",
  inlineTag:
    "flex items-center gap-1 rounded border border-border/70 bg-surface2 px-2 py-0.5",
  pill: "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold",
  smallPill: "rounded-full border px-1.5 py-0.5 text-[10px]",
};

export const cockpitRailDisabledActionClass =
  `${cockpitRailStyles.action} disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:bg-surface2`;

export const cockpitRailToneClass = (tone: CockpitRailTone): string => {
  if (tone === "success") return "border-success/40 bg-success/10 text-success";
  if (tone === "warning") return "border-warning/40 bg-warning/10 text-warning";
  if (tone === "danger") return "border-danger/40 bg-danger/10 text-danger";
  if (tone === "info") return "border-info/40 bg-info/10 text-info";
  if (tone === "focus") return "border-focus/40 bg-focus/10 text-focus";
  return "border-border bg-surface2 text-text-muted";
};
