import React from "react";
import { AlertTriangle, RefreshCw, Settings2, UserCircle2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  getCharacterChatReadinessCopy,
  type CharacterChatReadiness,
  type CharacterChatReadinessAction,
} from "@/utils/chat-model-availability";

export type MissingCharacterRecovery = {
  id: string;
  reason: "missing" | "load-error";
};

type CharacterChatReadinessPanelProps = {
  readiness: CharacterChatReadiness;
  characterName?: string | null;
  missingCharacter?: MissingCharacterRecovery | null;
  onAction: (action: CharacterChatReadinessAction) => void;
  onChooseCharacter?: () => void;
  onRetryMissingCharacter?: () => void;
};

const panelClass =
  "mx-auto mt-2 flex w-full max-w-[64rem] flex-wrap items-start justify-between gap-3 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning";
const actionClass =
  "inline-flex min-h-[28px] items-center gap-1 rounded-md border border-warning/40 bg-surface px-2 text-xs font-medium text-text hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export const CharacterChatReadinessPanel = ({
  readiness,
  characterName,
  missingCharacter,
  onAction,
  onChooseCharacter,
  onRetryMissingCharacter,
}: CharacterChatReadinessPanelProps) => {
  const { t } = useTranslation("playground");
  const statusLabel = t(
    "characterChatReadiness.statusLabel",
    "Character Chat setup status",
  );

  if (missingCharacter) {
    const titleTemplate = t(
      "characterChatReadiness.missingRestoredCharacter.title",
      "Character {{id}} could not be loaded",
    ) as string;
    const title = titleTemplate.replace("{{id}}", missingCharacter.id);
    const description = t(
      "characterChatReadiness.missingRestoredCharacter.description",
      "Choose another character or retry loading it.",
    ) as string;

    return (
      <section
        role="status"
        aria-live="polite"
        aria-atomic="true"
        aria-label={statusLabel}
        data-testid="character-chat-readiness-panel"
        className={panelClass}
      >
        <div className="flex min-w-0 flex-1 items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
          <div className="min-w-0">
            <p className="font-semibold text-text">{title}</p>
            <p className="mt-0.5 text-text-muted">{description}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className={actionClass}
            onClick={onChooseCharacter}
          >
            <UserCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            {t("characterChatReadiness.character.action", "Choose character")}
          </button>
          <button
            type="button"
            className={actionClass}
            onClick={onRetryMissingCharacter}
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            {t("common:retry", "Retry")}
          </button>
        </div>
      </section>
    );
  }

  if (readiness.status === "ready") {
    return null;
  }

  const copy = getCharacterChatReadinessCopy(readiness, t, {
    characterName,
  });

  return (
    <section
      role="status"
      aria-live="polite"
      aria-atomic="true"
      aria-label={statusLabel}
      data-testid="character-chat-readiness-panel"
      className={panelClass}
    >
      <div className="flex min-w-0 flex-1 items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
        <div className="min-w-0">
          <p className="font-semibold text-text">{copy.title}</p>
          <p className="mt-0.5 text-text-muted">{copy.description}</p>
        </div>
      </div>
      {readiness.recommendedAction ? (
        <button
          type="button"
          className={actionClass}
          onClick={() => onAction(readiness.recommendedAction!)}
        >
          {readiness.recommendedAction === "choose-character" ? (
            <UserCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <Settings2 className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          {copy.actionLabel}
        </button>
      ) : null}
    </section>
  );
};
