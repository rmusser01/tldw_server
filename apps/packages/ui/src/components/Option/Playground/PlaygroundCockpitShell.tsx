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
export type PlaygroundCockpitMobilePanel = "context" | "runtime" | null;

export type PlaygroundCockpitShellProps = {
  mode: PlaygroundCockpitMode;
  onModeChange: (mode: PlaygroundCockpitMode) => void;
  leftRailVisible?: boolean;
  rightRailVisible?: boolean;
  onLeftRailVisibleChange?: (visible: boolean) => void;
  onRightRailVisibleChange?: (visible: boolean) => void;
  mobilePanel?: PlaygroundCockpitMobilePanel;
  onMobilePanelChange?: (panel: PlaygroundCockpitMobilePanel) => void;
  leftRail: React.ReactNode;
  rightRail: React.ReactNode;
  statusStrip: React.ReactNode;
  children: React.ReactNode;
};

type CockpitTooltipPlacement = "bottom" | "left" | "right" | "top";

type CockpitTooltipButtonProps =
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    children: React.ReactNode;
    tooltip: string;
    tooltipId?: string;
    tooltipPlacement?: CockpitTooltipPlacement;
    wrapperClassName?: string;
  };

const tooltipPlacementClassNames: Record<CockpitTooltipPlacement, string> = {
  bottom: "left-1/2 top-full mt-2 -translate-x-1/2",
  left: "right-full top-1/2 mr-2 -translate-y-1/2",
  right: "left-full top-1/2 ml-2 -translate-y-1/2",
  top: "bottom-full left-1/2 mb-2 -translate-x-1/2",
};

const CockpitTooltipButton = ({
  children,
  className,
  tooltip,
  tooltipId: tooltipIdProp,
  tooltipPlacement = "top",
  wrapperClassName,
  ...buttonProps
}: CockpitTooltipButtonProps) => {
  const generatedTooltipId = React.useId();
  const tooltipId = tooltipIdProp ?? generatedTooltipId;
  const describedBy = buttonProps["aria-describedby"];
  const ariaDescribedBy =
    typeof describedBy === "string" && describedBy.trim().length > 0
      ? `${describedBy} ${tooltipId}`
      : tooltipId;

  const resolvedWrapperClassName = wrapperClassName
    ? `group inline-flex ${wrapperClassName}`
    : "group relative inline-flex";

  return (
    <span className={resolvedWrapperClassName}>
      <button
        {...buttonProps}
        aria-describedby={ariaDescribedBy}
        className={className}
      >
        {children}
      </button>
      <span
        id={tooltipId}
        role="tooltip"
        className={`pointer-events-none absolute z-50 whitespace-nowrap rounded-md border border-border bg-surface px-2 py-1 text-[11px] font-medium text-text opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 ${tooltipPlacementClassNames[tooltipPlacement]}`}
      >
        {tooltip}
      </span>
    </span>
  );
};

const resolveVisibleMobilePanel = (
  resolvedMobilePanel: PlaygroundCockpitMobilePanel,
  leftRailVisible: boolean,
  rightRailVisible: boolean,
): PlaygroundCockpitMobilePanel => {
  if (resolvedMobilePanel === "runtime" && rightRailVisible) return "runtime";
  if (resolvedMobilePanel === "context" && leftRailVisible) return "context";
  if (leftRailVisible) return "context";
  if (rightRailVisible) return "runtime";
  return null;
};

const buildModeSummaryKey = (
  focusMode: boolean,
  leftRailVisible: boolean,
  rightRailVisible: boolean,
):
  | "focus"
  | "both-visible"
  | "context-hidden"
  | "runtime-hidden"
  | "both-hidden" => {
  if (focusMode) return "focus";
  if (leftRailVisible && rightRailVisible) return "both-visible";
  if (!leftRailVisible && rightRailVisible) return "context-hidden";
  if (leftRailVisible && !rightRailVisible) return "runtime-hidden";
  return "both-hidden";
};

export const PlaygroundCockpitShell = ({
  mode,
  onModeChange,
  leftRailVisible = true,
  rightRailVisible = true,
  onLeftRailVisibleChange,
  onRightRailVisibleChange,
  mobilePanel,
  onMobilePanelChange,
  leftRail,
  rightRail,
  statusStrip,
  children,
}: PlaygroundCockpitShellProps) => {
  const { t } = useTranslation("playground");
  const [uncontrolledMobilePanel, setUncontrolledMobilePanel] =
    React.useState<PlaygroundCockpitMobilePanel>("context");
  const mobilePanelIdPrefix = React.useId();
  const focusMode = mode === "focus";
  const nextMode: PlaygroundCockpitMode = focusMode ? "cockpit" : "focus";
  const toggleLabel = focusMode
    ? t("cockpit.showPanels", "Show cockpit panels")
    : t("cockpit.enterFocus", "Enter focus chat");
  const mobileReturnToFocusLabel = t(
    "cockpit.mobileReturnToFocus",
    "Return to focus chat",
  );
  const mobileContextTabId = `${mobilePanelIdPrefix}-mobile-context-tab`;
  const mobileRuntimeTabId = `${mobilePanelIdPrefix}-mobile-runtime-tab`;
  const mobileContextPanelId = `${mobilePanelIdPrefix}-mobile-context-panel`;
  const mobileRuntimePanelId = `${mobilePanelIdPrefix}-mobile-runtime-panel`;
  const mobilePanelSummaryId = `${mobilePanelIdPrefix}-mobile-panel-summary`;
  const showLeftRail = !focusMode && leftRailVisible;
  const showRightRail = !focusMode && rightRailVisible;
  const resolvedMobilePanel =
    mobilePanel !== undefined ? mobilePanel : uncontrolledMobilePanel;
  const setMobilePanel = React.useCallback(
    (panel: PlaygroundCockpitMobilePanel) => {
      if (onMobilePanelChange) {
        onMobilePanelChange(panel);
        return;
      }
      setUncontrolledMobilePanel(panel);
    },
    [onMobilePanelChange],
  );
  const visibleMobilePanel = resolveVisibleMobilePanel(
    resolvedMobilePanel,
    leftRailVisible,
    rightRailVisible,
  );
  const mobilePanelSummary =
    visibleMobilePanel === "context"
      ? t(
          "cockpit.mobileContextPanelSummary",
          "Context panel active. Composer draft remains available below.",
        )
      : visibleMobilePanel === "runtime"
        ? t(
            "cockpit.mobileRuntimePanelSummary",
            "Runtime panel active. Composer draft remains available below.",
          )
        : t(
            "cockpit.mobilePanelsHiddenSummary",
            "Cockpit panels hidden. Composer draft remains available below.",
          );
  const leftRailLabel = leftRailVisible
    ? t("cockpit.hideContextRail", "Hide context rail")
    : t("cockpit.showContextRail", "Show context rail");
  const rightRailLabel = rightRailVisible
    ? t("cockpit.hideRuntimeRail", "Hide runtime rail")
    : t("cockpit.showRuntimeRail", "Show runtime rail");
  const collapseContextSidechannelLabel = t(
    "cockpit.collapseContextSidechannel",
    "Collapse context sidechannel",
  );
  const collapseRuntimeSidechannelLabel = t(
    "cockpit.collapseRuntimeSidechannel",
    "Collapse runtime sidechannel",
  );
  const restoreContextSidechannelLabel = t(
    "cockpit.restoreContextSidechannel",
    "Restore context sidechannel",
  );
  const restoreRuntimeSidechannelLabel = t(
    "cockpit.restoreRuntimeSidechannel",
    "Restore runtime sidechannel",
  );
  const contextRailTabLabel = t("cockpit.contextRailTab", "Context rail");
  const runtimeRailTabLabel = t("cockpit.runtimeRailTab", "Runtime rail");
  const modeSummaryKey = buildModeSummaryKey(
    focusMode,
    leftRailVisible,
    rightRailVisible,
  );
  const modeSummary =
    modeSummaryKey === "focus"
      ? t(
          "cockpit.focusModeSummary",
          "Focus mode hides rails. Chat and composer remain active.",
        )
      : modeSummaryKey === "both-visible"
        ? t(
            "cockpit.cockpitRailsVisibleSummary",
            "Context and runtime rails visible.",
          )
        : modeSummaryKey === "context-hidden"
          ? t(
              "cockpit.contextRailHiddenSummary",
              "Context rail hidden. Runtime rail visible.",
            )
          : modeSummaryKey === "runtime-hidden"
            ? t(
                "cockpit.runtimeRailHiddenSummary",
                "Runtime rail hidden. Context rail visible.",
              )
            : t(
                "cockpit.cockpitRailsHiddenSummary",
                "Cockpit rails hidden. Status remains visible.",
              );
  const bodyClassName = focusMode
    ? "flex min-h-0 w-full min-w-0 flex-1 justify-center overflow-hidden"
    : "grid min-h-0 w-full min-w-0 flex-1 grid-cols-1 overflow-hidden lg:[grid-template-columns:var(--cockpit-grid-columns)]";
  const cockpitGridColumns = showLeftRail
    ? showRightRail
      ? "minmax(220px,280px) minmax(0,1fr) minmax(240px,300px)"
      : "minmax(220px,280px) minmax(0,1fr)"
    : showRightRail
      ? "minmax(0,1fr) minmax(240px,300px)"
      : "minmax(0,1fr)";
  const cockpitGridStyle = focusMode
    ? undefined
    : ({
        "--cockpit-grid-columns": cockpitGridColumns,
      } as React.CSSProperties);

  return (
    <div
      data-testid="playground-cockpit-shell"
      data-mode={mode}
      data-left-rail={showLeftRail ? "visible" : "hidden"}
      data-right-rail={showRightRail ? "visible" : "hidden"}
      className="flex h-full min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden bg-bg text-text"
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
          <p
            data-testid="playground-cockpit-mode-summary"
            className="mt-0.5 max-w-[44rem] truncate text-[11px] text-text-muted"
          >
            {modeSummary}
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
                onClick={() => onLeftRailVisibleChange?.(!leftRailVisible)}
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
                onClick={() => onRightRailVisibleChange?.(!rightRailVisible)}
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
          data-mobile-panel={visibleMobilePanel ?? "none"}
          className="grid shrink-0 grid-cols-1 gap-2 border-b border-border bg-surface2/40 p-2 text-xs lg:hidden"
        >
          <div className="flex items-center justify-between gap-2 rounded-md border border-border bg-surface px-2.5 py-2">
            <span className="font-semibold text-text">
              {t("cockpit.cockpit", "Cockpit")}
            </span>
            <button
              type="button"
              aria-label={mobileReturnToFocusLabel}
              onClick={() => onModeChange("focus")}
              className="inline-flex min-h-[32px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
              <span>{t("cockpit.focus", "Focus")}</span>
            </button>
          </div>
          <div
            role="tablist"
            aria-label={t("cockpit.mobilePanelTabs", "Mobile cockpit panels")}
            className="grid grid-cols-2 gap-1 rounded-md border border-border bg-surface p-1"
          >
            {leftRailVisible ? (
              <button
                id={mobileContextTabId}
                type="button"
                role="tab"
                aria-selected={visibleMobilePanel === "context"}
                aria-controls={mobileContextPanelId}
                onClick={() => setMobilePanel("context")}
                className={`min-h-[44px] rounded px-3 py-2 text-xs font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
                  visibleMobilePanel === "context"
                    ? "bg-bg text-text shadow-sm"
                    : "text-text-muted hover:bg-bg"
                }`}
              >
                {t("cockpit.context", "Context")}
              </button>
            ) : null}
            {rightRailVisible ? (
              <button
                id={mobileRuntimeTabId}
                type="button"
                role="tab"
                aria-selected={visibleMobilePanel === "runtime"}
                aria-controls={mobileRuntimePanelId}
                onClick={() => setMobilePanel("runtime")}
                className={`min-h-[44px] rounded px-3 py-2 text-xs font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
                  visibleMobilePanel === "runtime"
                    ? "bg-bg text-text shadow-sm"
                    : "text-text-muted hover:bg-bg"
                }`}
              >
                {t("cockpit.runtime", "Runtime")}
              </button>
            ) : null}
          </div>
          <p
            id={mobilePanelSummaryId}
            data-testid="playground-cockpit-mobile-panel-summary"
            className="sr-only"
          >
            {mobilePanelSummary}
          </p>
          {leftRailVisible ? (
            <section
              id={mobileContextPanelId}
              role="tabpanel"
              aria-labelledby={mobileContextTabId}
              aria-describedby={mobilePanelSummaryId}
              hidden={visibleMobilePanel !== "context"}
              aria-hidden={visibleMobilePanel !== "context"}
              className={`max-h-[30vh] overflow-y-auto rounded-md border border-border bg-surface p-2 ${
                visibleMobilePanel !== "context" ? "hidden" : ""
              }`}
            >
              {leftRail}
            </section>
          ) : null}
          {rightRailVisible ? (
            <section
              id={mobileRuntimePanelId}
              role="tabpanel"
              aria-labelledby={mobileRuntimeTabId}
              aria-describedby={mobilePanelSummaryId}
              hidden={visibleMobilePanel !== "runtime"}
              aria-hidden={visibleMobilePanel !== "runtime"}
              className={`max-h-[30vh] overflow-y-auto rounded-md border border-border bg-surface p-2 ${
                visibleMobilePanel !== "runtime" ? "hidden" : ""
              }`}
            >
              {rightRail}
            </section>
          ) : null}
        </div>
      )}

      <div className={`${bodyClassName} relative`} style={cockpitGridStyle}>
        {showLeftRail && (
          <aside
            id="playground-cockpit-left-rail"
            data-testid="playground-cockpit-left-rail"
            aria-label={t("cockpit.contextLandmark", "Chat cockpit context")}
            className="hidden min-h-0 overflow-y-auto border-r border-border bg-surface2/30 p-2 lg:block"
          >
            <div className="mb-2 flex min-h-[32px] items-center justify-between gap-2 rounded-md border border-border/70 bg-bg px-2 py-1 text-xs text-text-muted">
              <span className="font-semibold text-text">
                {t("cockpit.context", "Context")}
              </span>
              <CockpitTooltipButton
                type="button"
                aria-label={collapseContextSidechannelLabel}
                aria-controls="playground-cockpit-left-rail"
                aria-expanded="true"
                onClick={() => onLeftRailVisibleChange?.(false)}
                tooltip={collapseContextSidechannelLabel}
                tooltipPlacement="left"
                className="inline-flex min-h-[28px] min-w-[28px] items-center justify-center rounded-md border border-border bg-surface2 text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <PanelLeftClose className="h-3.5 w-3.5" aria-hidden="true" />
              </CockpitTooltipButton>
            </div>
            <div className="min-w-0">{leftRail}</div>
          </aside>
        )}

        <main
          data-testid="playground-cockpit-main"
          className={`min-h-0 min-w-0 overflow-hidden bg-bg ${
            focusMode
              ? "w-full max-w-[72rem]"
              : !showLeftRail && !showRightRail
                ? "w-full"
                : ""
          }`}
        >
          {children}
        </main>

        {showRightRail && (
          <aside
            id="playground-cockpit-right-rail"
            data-testid="playground-cockpit-right-rail"
            aria-label={t("cockpit.runtimeLandmark", "Chat cockpit runtime")}
            className="hidden min-h-0 overflow-y-auto border-l border-border bg-surface2/30 p-2 lg:block"
          >
            <div className="mb-2 flex min-h-[32px] items-center justify-between gap-2 rounded-md border border-border/70 bg-bg px-2 py-1 text-xs text-text-muted">
              <span className="font-semibold text-text">
                {t("cockpit.runtime", "Runtime")}
              </span>
              <CockpitTooltipButton
                type="button"
                aria-label={collapseRuntimeSidechannelLabel}
                aria-controls="playground-cockpit-right-rail"
                aria-expanded="true"
                onClick={() => onRightRailVisibleChange?.(false)}
                tooltip={collapseRuntimeSidechannelLabel}
                tooltipPlacement="left"
                className="inline-flex min-h-[28px] min-w-[28px] items-center justify-center rounded-md border border-border bg-surface2 text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <PanelRightClose className="h-3.5 w-3.5" aria-hidden="true" />
              </CockpitTooltipButton>
            </div>
            <div className="min-w-0">{rightRail}</div>
          </aside>
        )}

        {!focusMode && !leftRailVisible ? (
          <CockpitTooltipButton
            type="button"
            data-testid="playground-cockpit-left-rail-restore"
            aria-label={restoreContextSidechannelLabel}
            aria-controls="playground-cockpit-left-rail"
            aria-expanded="false"
            onClick={() => onLeftRailVisibleChange?.(true)}
            tooltip={restoreContextSidechannelLabel}
            tooltipPlacement="right"
            wrapperClassName="absolute left-10 top-1/2 z-50 hidden -translate-y-1/2 lg:inline-flex"
            className="inline-flex h-32 w-9 flex-col items-center justify-center gap-2 rounded-r-md border-y border-r border-border bg-surface2/95 py-2 text-[11px] font-semibold text-text shadow-md backdrop-blur-sm hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <PanelLeftOpen className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="rotate-180 whitespace-nowrap leading-none [writing-mode:vertical-rl]">
              {contextRailTabLabel}
            </span>
          </CockpitTooltipButton>
        ) : null}

        {!focusMode && !rightRailVisible ? (
          <CockpitTooltipButton
            type="button"
            data-testid="playground-cockpit-right-rail-restore"
            aria-label={restoreRuntimeSidechannelLabel}
            aria-controls="playground-cockpit-right-rail"
            aria-expanded="false"
            onClick={() => onRightRailVisibleChange?.(true)}
            tooltip={restoreRuntimeSidechannelLabel}
            tooltipPlacement="left"
            wrapperClassName="absolute right-0 top-1/2 z-50 hidden -translate-y-1/2 lg:inline-flex"
            className="inline-flex h-32 w-9 flex-col items-center justify-center gap-2 rounded-l-md border-y border-l border-border bg-surface2/95 py-2 text-[11px] font-semibold text-text shadow-md backdrop-blur-sm hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <span className="whitespace-nowrap leading-none [writing-mode:vertical-rl]">
              {runtimeRailTabLabel}
            </span>
            <PanelRightOpen className="h-3.5 w-3.5" aria-hidden="true" />
          </CockpitTooltipButton>
        ) : null}
      </div>

      <div data-testid="playground-cockpit-status-strip">{statusStrip}</div>
    </div>
  );
};
