const railSectionClass = "rounded-md border border-border bg-surface px-3 py-2";
const railHeadingClass = "text-[11px] font-semibold uppercase text-text-muted";
const railValueClass = "mt-1 text-sm font-medium text-text";
const railMutedClass = "mt-1 text-xs text-text-muted";
const railActionClass =
  "inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export type PlaygroundRuntimeInspectorProps = {
  streaming: boolean;
  selectedModel: string | null | undefined;
  messageCount: number;
  threadSearchOpen: boolean;
  selectedCharacterName: string | null | undefined;
};

const dispatchEvent = (eventName: string) => {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(eventName));
};

export const PlaygroundRuntimeInspector = ({
  streaming,
  selectedModel,
  messageCount,
  threadSearchOpen,
  selectedCharacterName,
}: PlaygroundRuntimeInspectorProps) => {
  const messageLabel = `${messageCount} ${
    messageCount === 1 ? "message" : "messages"
  }`;

  return (
    <div
      data-testid="playground-runtime-inspector"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section className={railSectionClass} aria-label="Runtime state">
        <h2 className={railHeadingClass}>Runtime</h2>
        <p className={railValueClass}>{streaming ? "Streaming" : "Ready"}</p>
        <p className={railMutedClass}>
          {selectedModel || "No model selected"}
        </p>
      </section>

      <section className={railSectionClass} aria-label="Model and character">
        <h2 className={railHeadingClass}>Model & character</h2>
        <div className="mt-2 flex flex-col gap-2">
          <button
            type="button"
            onClick={() => dispatchEvent("tldw:open-model-settings")}
            className={railActionClass}
            aria-label="Open model settings"
          >
            Model settings
          </button>
          <button
            type="button"
            onClick={() => dispatchEvent("tldw:open-actor-settings")}
            className={railActionClass}
            aria-label="Open character settings"
          >
            Character
          </button>
        </div>
        <p className={railMutedClass}>
          {selectedCharacterName || "No character selected"}
        </p>
      </section>

      <section className={railSectionClass} aria-label="Conversation volume">
        <h2 className={railHeadingClass}>Timeline</h2>
        <p className={railValueClass}>{messageLabel}</p>
        <p className={railMutedClass}>
          {threadSearchOpen ? "Search open" : "Search closed"}
        </p>
      </section>
    </div>
  );
};
