import React from "react";
import {
  Maximize2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-react";
import { useTranslation } from "react-i18next";

export type PlaygroundCockpitMode = "cockpit" | "focus";

export type PlaygroundCockpitShellProps = {
  mode: PlaygroundCockpitMode;
  onModeChange: (mode: PlaygroundCockpitMode) => void;
  leftRailVisible?: boolean;
  rightRailVisible?: boolean;
  onLeftRailVisibleChange?: (visible: boolean) => void;
  onRightRailVisibleChange?: (visible: boolean) => void;
  leftRail: React.ReactNode;
  rightRail: React.ReactNode;
  statusStrip: React.ReactNode;
  children: React.ReactNode;
};

export const PlaygroundCockpitShell = ({
  mode,
  onModeChange,
  leftRailVisible = true,
  rightRailVisible = true,
  onLeftRailVisibleChange,
  onRightRailVisibleChange,
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
  const showLeftRail = !focusMode && leftRailVisible;
  const showRightRail = !focusMode && rightRailVisible;
  const leftRailLabel = leftRailVisible
    ? t("cockpit.hideContextRail", "Hide context rail")
    : t("cockpit.showContextRail", "Show context rail");
  const rightRailLabel = rightRailVisible
    ? t("cockpit.hideRuntimeRail", "Hide runtime rail")
    : t("cockpit.showRuntimeRail", "Show runtime rail");
  const bodyClassName = focusMode
    ? "flex min-h-0 flex-1 justify-center overflow-hidden"
    : showLeftRail && showRightRail
      ? "grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(220px,280px)_minmax(0,1fr)_minmax(240px,300px)]"
      : showLeftRail
        ? "grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(220px,280px)_minmax(0,1fr)]"
        : showRightRail
          ? "grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,1fr)_minmax(240px,300px)]"
          : "grid min-h-0 flex-1 grid-cols-1 overflow-hidden";

  return (
    <div
      data-testid="playground-cockpit-shell"
      data-mode={mode}
      data-left-rail={showLeftRail ? "visible" : "hidden"}
      data-right-rail={showRightRail ? "visible" : "hidden"}
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
        <div className="flex shrink-0 items-center gap-1.5">
          {!focusMode && (
            <div
              aria-label={t("cockpit.railVisibility", "Rail visibility")}
              className="flex items-center gap-1"
            >
              <button
                type="button"
                aria-label={leftRailLabel}
                aria-pressed={leftRailVisible}
                title={leftRailLabel}
                onClick={() =>
                  onLeftRailVisibleChange?.(!leftRailVisible)
                }
                className="inline-flex min-h-[30px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                {leftRailVisible ? (
                  <PanelLeftClose className="h-3.5 w-3.5" aria-hidden="true" />
                ) : (
                  <PanelLeftOpen className="h-3.5 w-3.5" aria-hidden="true" />
                )}
                <span className="hidden sm:inline">
                  {t("cockpit.context", "Context")}
                </span>
              </button>
              <button
                type="button"
                aria-label={rightRailLabel}
                aria-pressed={rightRailVisible}
                title={rightRailLabel}
                onClick={() =>
                  onRightRailVisibleChange?.(!rightRailVisible)
                }
                className="inline-flex min-h-[30px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                {rightRailVisible ? (
                  <PanelRightClose className="h-3.5 w-3.5" aria-hidden="true" />
                ) : (
                  <PanelRightOpen className="h-3.5 w-3.5" aria-hidden="true" />
                )}
                <span className="hidden sm:inline">
                  {t("cockpit.runtime", "Runtime")}
                </span>
              </button>
            </div>
          )}
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
        </div>
      </header>

      {!focusMode && (leftRailVisible || rightRailVisible) && (
        <div
          data-testid="playground-cockpit-mobile-rails"
          className="grid shrink-0 grid-cols-1 gap-2 border-b border-border bg-surface2/40 p-2 text-xs lg:hidden"
        >
          {leftRailVisible && (
            <details className="rounded-md border border-border bg-surface">
              <summary className="cursor-pointer px-3 py-2 font-medium text-text">
                {t("cockpit.context", "Context")}
              </summary>
              <div className="border-t border-border p-2">{leftRail}</div>
            </details>
          )}
          {rightRailVisible && (
            <details className="rounded-md border border-border bg-surface">
              <summary className="cursor-pointer px-3 py-2 font-medium text-text">
                {t("cockpit.runtime", "Runtime")}
              </summary>
              <div className="border-t border-border p-2">{rightRail}</div>
            </details>
          )}
        </div>
      )}

      <div className={bodyClassName}>
        {showLeftRail && (
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
            focusMode || (!showLeftRail && !showRightRail)
              ? "w-full max-w-[72rem]"
              : ""
          }`}
        >
          {children}
        </main>

        {showRightRail && (
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
