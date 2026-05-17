import React from "react";
import { Popover } from "antd";
import { Settings2 } from "lucide-react";
import { Button as TldwButton } from "@/components/Common/Button";
import type { KnowledgeTab } from "@/components/Knowledge";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlaygroundModeLauncherProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;

  compareModeActive: boolean;
  compareFeatureEnabled: boolean;
  onToggleCompare: () => void;

  selectedCharacterName: string | null;
  onOpenActorSettings: () => void;

  contextToolsOpen: boolean;
  onToggleKnowledgePanel: () => void;

  voiceChatEnabled: boolean;
  voiceChatAvailable: boolean;
  voiceChatUnavailableReason?: string | null;
  isSending: boolean;
  onVoiceChatToggle: () => void;

  webSearch: boolean;
  hasWebSearch: boolean;
  onToggleWebSearch: () => void;

  onModeAnnouncement: (msg: string) => void;
  t: (key: string, defaultValue?: string, options?: any) => any;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundModeLauncher: React.FC<PlaygroundModeLauncherProps> =
  React.memo(function PlaygroundModeLauncher(props) {
    const {
      open,
      onOpenChange,
      compareModeActive,
      compareFeatureEnabled,
      onToggleCompare,
      selectedCharacterName,
      onOpenActorSettings,
      contextToolsOpen,
      onToggleKnowledgePanel,
      voiceChatEnabled,
      voiceChatAvailable,
      voiceChatUnavailableReason,
      isSending,
      onVoiceChatToggle,
      webSearch,
      hasWebSearch,
      onToggleWebSearch,
      onModeAnnouncement,
      t,
    } = props;

    const content = (
      <div className="flex w-72 flex-col gap-1 p-1">
        <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          {t("playground:composer.modes", "Modes")}
        </div>
        <button
          type="button"
          onClick={() => {
            const next = !compareModeActive;
            onToggleCompare();
            onModeAnnouncement(
              next
                ? t(
                    "playground:composer.modeCompareEnabled",
                    "Compare mode enabled.",
                  )
                : t(
                    "playground:composer.modeCompareDisabled",
                    "Compare mode disabled.",
                  ),
            );
            onOpenChange(false);
          }}
          disabled={!compareFeatureEnabled}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span>
            {t("playground:composer.modeCompare", "Compare responses")}
          </span>
          <span className="text-xs text-text-muted">
            {compareModeActive ? t("common:on", "On") : t("common:off", "Off")}
          </span>
        </button>
        <button
          type="button"
          onClick={() => {
            onOpenActorSettings();
            onModeAnnouncement(
              t(
                "playground:composer.modeCharacterNotice",
                "Character / scene settings opened.",
              ),
            );
            onOpenChange(false);
          }}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
        >
          <span>
            {t("playground:composer.modeCharacter", "Character / Scene")}
          </span>
          <span className="truncate text-xs text-text-muted">
            {selectedCharacterName
              ? t(
                  "playground:composer.modeCharacterActive",
                  "Active: {{name}}",
                  { name: selectedCharacterName },
                )
              : t("common:off", "Off")}
          </span>
        </button>
        <button
          type="button"
          onClick={() => {
            const nextOpen = !contextToolsOpen;
            onToggleKnowledgePanel();
            onModeAnnouncement(
              nextOpen
                ? t(
                    "playground:composer.modeKnowledgeOpened",
                    "Search & Context panel opened.",
                  )
                : t(
                    "playground:composer.modeKnowledgeClosed",
                    "Search & Context panel closed.",
                  ),
            );
            onOpenChange(false);
          }}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
        >
          <span>
            {t("playground:composer.modeKnowledge", "Knowledge panel")}
          </span>
          <span className="text-xs text-text-muted">
            {contextToolsOpen
              ? t("common:open", "Open")
              : t("common:closed", "Closed")}
          </span>
        </button>
        <button
          type="button"
          onClick={() => {
            onVoiceChatToggle();
            onModeAnnouncement(
              voiceChatEnabled
                ? t(
                    "playground:composer.modeVoiceDisabled",
                    "Voice mode disabled.",
                  )
                : t(
                    "playground:composer.modeVoiceEnabled",
                    "Voice mode enabled.",
                  ),
            );
            onOpenChange(false);
          }}
          disabled={!voiceChatAvailable || isSending}
          title={
            !voiceChatAvailable
              ? (voiceChatUnavailableReason ?? undefined)
              : undefined
          }
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span>{t("playground:composer.modeVoice", "Voice mode")}</span>
          <span className="text-xs text-text-muted">
            {voiceChatEnabled ? t("common:on", "On") : t("common:off", "Off")}
          </span>
        </button>
        <button
          type="button"
          onClick={() => {
            if (!hasWebSearch) return;
            onToggleWebSearch();
            onModeAnnouncement(
              webSearch
                ? t(
                    "playground:composer.modeWebSearchDisabled",
                    "Web search disabled.",
                  )
                : t(
                    "playground:composer.modeWebSearchEnabled",
                    "Web search enabled.",
                  ),
            );
            onOpenChange(false);
          }}
          disabled={!hasWebSearch}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span>{t("playground:composer.modeWebSearch", "Web search")}</span>
          <span className="text-xs text-text-muted">
            {webSearch ? t("common:on", "On") : t("common:off", "Off")}
          </span>
        </button>
      </div>
    );

    return (
      <Popover
        trigger="click"
        placement="topLeft"
        content={content}
        open={open}
        onOpenChange={onOpenChange}
      >
        <TldwButton
          variant="outline"
          size="sm"
          shape="pill"
          ariaLabel={t("playground:composer.modes", "Modes") as string}
          title={t("playground:composer.modes", "Modes") as string}
          className="min-h-[44px]"
        >
          <span className="inline-flex items-center gap-1.5">
            <Settings2 className="h-4 w-4" aria-hidden="true" />
            <span>{t("playground:composer.modes", "Modes")}</span>
          </span>
        </TldwButton>
      </Popover>
    );
  });
