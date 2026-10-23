import React from "react";
import { useTranslation } from "react-i18next";
import { Settings2, SlidersHorizontal, UserRound } from "lucide-react";
import { cockpitRailStyles } from "./playground-cockpit-rail-styles";

export type PlaygroundAssistantSummary = {
  mode: "none" | "character" | "persona";
  name?: string | null;
  detail?: string | null;
  compositionTitle?: string | null;
  compositionDetail?: string | null;
};

export type PlaygroundAssistantContextProps = {
  selectedCharacterName?: string | null;
  assistantSummary?: PlaygroundAssistantSummary;
  onOpenCharacterSettings?: () => void;
  onOpenAssistantSelect?: () => void;
  onClearAssistant?: () => void;
  onInspectAssistant?: () => void;
  onOpenSceneDirector?: () => void;
};

export const PlaygroundAssistantContext = ({
  selectedCharacterName,
  assistantSummary,
  onOpenCharacterSettings,
  onOpenAssistantSelect,
  onClearAssistant,
  onInspectAssistant,
  onOpenSceneDirector,
}: PlaygroundAssistantContextProps) => {
  const { t } = useTranslation("playground");
  const effectiveAssistantSummary: PlaygroundAssistantSummary =
    assistantSummary ||
    (selectedCharacterName
      ? {
          mode: "character",
          name: selectedCharacterName,
          detail: t("cockpit.characterSelected", "Character selected"),
        }
      : {
          mode: "none",
          name: null,
          detail: t(
            "cockpit.noAssistantDetail",
            "No persona or character will shape replies.",
          ),
        });
  const noAssistantSelectedLabel = t(
    "cockpit.noAssistantSelected",
    "No assistant selected",
  );
  const assistantLabel =
    effectiveAssistantSummary.name ||
    (effectiveAssistantSummary.mode === "none"
      ? t("cockpit.noAssistantSelected", "No assistant selected")
      : noAssistantSelectedLabel);
  const noAssistantDetail = t(
    "cockpit.noAssistantDetail",
    "No persona or character will shape replies.",
  );
  const assistantModeLabel =
    effectiveAssistantSummary.mode === "persona"
      ? t("cockpit.personaMode", "Persona")
      : effectiveAssistantSummary.mode === "character"
        ? t("cockpit.characterMode", "Character")
        : t("cockpit.noAssistantMode", "None");
  const assistantDetail =
    effectiveAssistantSummary.mode === "none" &&
    (effectiveAssistantSummary.detail === assistantLabel ||
      effectiveAssistantSummary.detail === noAssistantSelectedLabel)
      ? noAssistantDetail
      : effectiveAssistantSummary.detail ||
        (effectiveAssistantSummary.mode === "none"
          ? noAssistantDetail
          : assistantModeLabel);
  const openAssistant = onOpenAssistantSelect || onOpenCharacterSettings;
  const openSceneDirector = onOpenSceneDirector || onOpenCharacterSettings;

  return (
    <section
      aria-label={t("cockpit.assistant", "Assistant")}
      className="mt-4 border-t border-border/70 pt-3"
    >
      <h3 className="text-xs font-medium text-text-muted">
        {t("cockpit.assistant", "Assistant")}
      </h3>
      <div className="mt-1 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className={cockpitRailStyles.value}>{assistantLabel}</p>
          {effectiveAssistantSummary.mode !== "none" && (
            <p className={cockpitRailStyles.muted}>{assistantDetail}</p>
          )}
        </div>
        {effectiveAssistantSummary.mode !== "none" && (
          <span className="shrink-0 rounded-full border border-border bg-surface2 px-2 py-0.5 text-[10px] font-semibold text-text-muted">
            {assistantModeLabel}
          </span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {openAssistant ? (
          <button
            type="button"
            onClick={openAssistant}
            data-cockpit-assistant-select-trigger
            className={`${cockpitRailStyles.action} justify-start gap-1.5`}
            aria-label={t(
              "cockpit.selectCharacterPersona",
              "Select character or persona",
            )}
          >
            <UserRound className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.selectAssistant", "Select assistant")}
          </button>
        ) : null}
        {effectiveAssistantSummary.mode !== "none" && onInspectAssistant ? (
          <button
            type="button"
            onClick={onInspectAssistant}
            className={`${cockpitRailStyles.action} justify-start gap-1.5`}
            aria-label={t("cockpit.manageAssistant", "Manage assistant")}
          >
            <Settings2 className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.manageAssistant", "Manage assistant")}
          </button>
        ) : null}
        {effectiveAssistantSummary.mode !== "none" && onClearAssistant ? (
          <button
            type="button"
            onClick={onClearAssistant}
            className={`${cockpitRailStyles.action} justify-start gap-1.5`}
            aria-label={t("cockpit.clearAssistant", "Clear assistant")}
          >
            <UserRound className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.clearAssistant", "Clear assistant")}
          </button>
        ) : null}
        {effectiveAssistantSummary.mode === "character" && openSceneDirector ? (
          <button
            type="button"
            onClick={openSceneDirector}
            className={`${cockpitRailStyles.action} justify-start gap-1.5`}
            aria-label={t("cockpit.openSceneDirector", "Open Scene Director")}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
            {t("cockpit.sceneDirector", "Scene Director")}
          </button>
        ) : null}
      </div>
      {effectiveAssistantSummary.mode === "persona" ? (
        <p className={cockpitRailStyles.muted}>
          {t(
            "cockpit.personaActorUnavailable",
            "Scene Director is available for character-backed chats.",
          )}
        </p>
      ) : null}
    </section>
  );
};
