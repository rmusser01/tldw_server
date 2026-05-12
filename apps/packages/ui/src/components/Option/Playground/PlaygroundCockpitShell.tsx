import React from "react";
import { Maximize2, PanelLeftOpen } from "lucide-react";
import { useTranslation } from "react-i18next";

export type PlaygroundCockpitMode = "cockpit" | "focus";

export type PlaygroundCockpitShellProps = {
  mode: PlaygroundCockpitMode;
  onModeChange: (mode: PlaygroundCockpitMode) => void;
  leftRail: React.ReactNode;
  rightRail: React.ReactNode;
  statusStrip: React.ReactNode;
  children: React.ReactNode;
};

export const PlaygroundCockpitShell = ({
  mode,
  onModeChange,
  leftRail,
  rightRail,
  statusStrip,
  children,
}: PlaygroundCockpitShellProps) => {
  const { t } = useTranslation("playground");
  const focusMode = mode === "focus";
  const nextMode: PlaygroundCockpitMode = focusMode ? "cockpit" : "focus";
  const toggleLabel = focusMode
    ? t("cockpit.showPanels", "Show cockpit panels")
    : t("cockpit.enterFocus", "Enter focus chat");

  return (
    <div
      data-testid="playground-cockpit-shell"
      data-mode={mode}
      className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-bg text-text"
    >
      <header
        aria-label={t("cockpit.layoutControls", "Chat layout controls")}
        className="flex min-h-[42px] shrink-0 items-center justify-between gap-3 border-b border-border bg-surface/80 px-3 py-2 text-xs text-text-muted"
      >
        <div className="min-w-0">
          <p className="truncate font-semibold text-text">
            {focusMode
              ? t("cockpit.focusChat", "Focus chat")
              : t("cockpit.chatCockpit", "Chat cockpit")}
          </p>
        </div>
        <button
          type="button"
          aria-label={toggleLabel}
          aria-pressed={focusMode}
          title={toggleLabel}
          onClick={() => onModeChange(nextMode)}
          className="inline-flex min-h-[30px] items-center gap-1.5 rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          {focusMode ? (
            <PanelLeftOpen className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          <span>
            {focusMode
              ? t("cockpit.cockpit", "Cockpit")
              : t("cockpit.focus", "Focus")}
          </span>
        </button>
      </header>

      {!focusMode && (
        <div
          data-testid="playground-cockpit-mobile-rails"
          className="grid shrink-0 grid-cols-1 gap-2 border-b border-border bg-surface2/40 p-2 text-xs lg:hidden"
        >
          <details className="rounded-md border border-border bg-surface">
            <summary className="cursor-pointer px-3 py-2 font-medium text-text">
              {t("cockpit.context", "Context")}
            </summary>
            <div className="border-t border-border p-2">{leftRail}</div>
          </details>
          <details className="rounded-md border border-border bg-surface">
            <summary className="cursor-pointer px-3 py-2 font-medium text-text">
              {t("cockpit.runtime", "Runtime")}
            </summary>
            <div className="border-t border-border p-2">{rightRail}</div>
          </details>
        </div>
      )}

      <div
        className={
          focusMode
            ? "flex min-h-0 flex-1 justify-center overflow-hidden"
            : "grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(220px,280px)_minmax(0,1fr)_minmax(240px,300px)]"
        }
      >
        {!focusMode && (
          <aside
            data-testid="playground-cockpit-left-rail"
            aria-label={t("cockpit.contextLandmark", "Chat cockpit context")}
            className="hidden min-h-0 overflow-y-auto border-r border-border bg-surface2/30 p-2 lg:block"
          >
            {leftRail}
          </aside>
        )}

        <main
          data-testid="playground-cockpit-main"
          className={`min-h-0 min-w-0 overflow-hidden bg-bg ${
            focusMode ? "w-full max-w-[72rem]" : ""
          }`}
        >
          {children}
        </main>

        {!focusMode && (
          <aside
            data-testid="playground-cockpit-right-rail"
            aria-label={t("cockpit.runtimeLandmark", "Chat cockpit runtime")}
            className="hidden min-h-0 overflow-y-auto border-l border-border bg-surface2/30 p-2 lg:block"
          >
            {rightRail}
          </aside>
        )}
      </div>

      <div data-testid="playground-cockpit-status-strip">{statusStrip}</div>
    </div>
  );
};
