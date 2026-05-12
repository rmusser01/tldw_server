import type { PlaygroundCockpitMode } from "./PlaygroundCockpitShell";

export type PlaygroundStatusStripProps = {
  mode: PlaygroundCockpitMode;
  streaming: boolean;
  selectedModel: string | null | undefined;
  messageCount: number;
  serverChatId: string | null | undefined;
  temporaryChat: boolean | undefined;
  hasContext: boolean;
};

const pillClass =
  "inline-flex min-h-[24px] max-w-full items-center rounded-md border border-border bg-surface px-2 text-xs font-medium text-text";

export const PlaygroundStatusStrip = ({
  mode,
  streaming,
  selectedModel,
  messageCount,
  serverChatId,
  temporaryChat,
  hasContext,
}: PlaygroundStatusStripProps) => {
  const sessionLabel = temporaryChat
    ? "Temporary"
    : serverChatId
      ? "Server chat"
      : "Local chat";

  return (
    <footer
      role="status"
      aria-label="Chat status"
      aria-live="polite"
      aria-atomic="false"
      className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-t border-border bg-surface px-3 py-2 text-xs text-text-muted"
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className={pillClass}>{streaming ? "Streaming" : "Ready"}</span>
        <span className={pillClass}>
          {mode === "focus" ? "Focus" : "Cockpit"}
        </span>
        <span className={pillClass}>{sessionLabel}</span>
        {hasContext ? <span className={pillClass}>Context active</span> : null}
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className="max-w-[18rem] truncate">
          {selectedModel || "No model selected"}
        </span>
        <span>
          {messageCount} message{messageCount === 1 ? "" : "s"}
        </span>
      </div>
    </footer>
  );
};
