import {
  buildRecipePromptFields,
  classifyPromptRecipe,
  cloneSavedRecipeSource,
  getRecipePersistenceState,
} from "@/components/Option/Prompt/prompt-recipe-library";
import {
  getAllPrompts,
  markPromptSyncError,
  permanentlyDeletePrompt,
  restorePromptSnapshot,
  savePrompt,
  updatePrompt,
} from "@/db/dexie/helpers";
import type { Prompt } from "@/db/dexie/types";
import { useServerOnline } from "@/hooks/useServerOnline";
import {
  autoSyncPrompt,
  shouldAutoSyncWorkspacePrompts,
} from "@/services/prompt-sync";
import {
  isRecipePersistenceUncertain,
  markRecipePersistenceUncertain,
} from "@/services/recipe-persistence-uncertainty";
import type { PromptCapabilities } from "@/services/prompts-api";
import { isFireFoxPrivateMode } from "@/utils/is-private-mode";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import React from "react";
import { useTranslation } from "react-i18next";

import { SingleFieldRecipeEditor } from "./SingleFieldRecipeEditor";
import type { RecipeTarget, SingleTextRecipeDefinition } from "./types";

export type PromptRecipeBuilderProps = {
  target: RecipeTarget;
  capabilities: PromptCapabilities | undefined;
  persistenceScope: string | null;
  onApply: (compiledText: string) => void;
  onBack: () => void;
};

const acceptSyncResult = (
  result: Awaited<ReturnType<typeof autoSyncPrompt>>,
) => {
  if (result.success) return false;
  if (result.failureKind === "transient" && result.syncStatus === "pending") {
    return true;
  }
  throw new Error(result.error || "recipe_sync_failed");
};

const rollbackFailure = (action: "save" | "update") =>
  new Error(`recipe_${action}_rollback_failed`);

const uncertainSyncFailure = new Error("recipe_sync_uncertain");

const isUncertainSyncFailure = (error: unknown) =>
  error instanceof Error && error.message === uncertainSyncFailure.message;

export function PromptRecipeBuilder({
  target,
  capabilities,
  persistenceScope,
  onApply,
  onBack,
}: PromptRecipeBuilderProps) {
  const { t } = useTranslation(["common"]);
  const queryClient = useQueryClient();
  const isOnline = useServerOnline();
  const [syncPendingNotice, setSyncPendingNotice] = React.useState(false);
  const { data: prompts = [] } = useQuery({
    queryKey: ["getAllPromptsForSelect"],
    queryFn: getAllPrompts,
  });
  const persistence = getRecipePersistenceState(isOnline, capabilities);
  const basePersistenceAvailable =
    persistence.available && !isFireFoxPrivateMode && Boolean(persistenceScope);
  const savePersistenceAvailable =
    basePersistenceAvailable &&
    capabilities?.prompt_persistence?.create_authorized === true;
  const updatePersistenceAvailable =
    basePersistenceAvailable &&
    capabilities?.prompt_persistence?.update_authorized === true;
  const persistenceAuthorizationDenied =
    capabilities?.prompt_persistence?.create_authorized === false ||
    capabilities?.prompt_persistence?.update_authorized === false;
  const persistenceUnavailableReason = isFireFoxPrivateMode
    ? t(
        "common:promptAssist.recipePrivateMode",
        "Recipe saving is unavailable in private browsing. You can still edit, preview, and apply.",
      )
    : !isOnline
      ? t(
          "common:promptAssist.recipeOffline",
          "Recipe saving is unavailable offline. You can still edit, preview, and apply this local draft.",
        )
      : !capabilities || !persistenceScope
        ? t(
            "common:promptAssist.recipeChecking",
            "Checking whether this server supports recipe saving. You can still edit, preview, and apply.",
          )
        : !persistence.available
          ? capabilities.availability === "available"
            ? t(
                "common:promptAssist.recipeUnsupported",
                "This server does not support recipe saving yet. You can still edit, preview, and apply.",
              )
            : t(
                "common:promptAssist.recipeUnknown",
                "Recipe saving is unavailable because server capabilities could not be confirmed. You can still edit, preview, and apply.",
              )
          : t(
              persistenceAuthorizationDenied
                ? "common:promptAssist.recipeAuthorizationDenied"
                : "common:promptAssist.recipeAuthorizationUnavailable",
              persistenceAuthorizationDenied
                ? "Recipe saving or updating is unavailable because this account is not authorized. You can still edit, preview, and apply."
                : "Recipe saving or updating is unavailable because authorization could not be confirmed. You can still edit, preview, and apply.",
            );

  const savedRecipes = React.useMemo(
    () =>
      prompts.flatMap((prompt) => {
        const classification = classifyPromptRecipe(prompt);
        const expectedTarget = target === "system" ? "system" : "user";
        if (
          classification.kind !== "recipe" ||
          classification.target !== expectedTarget
        ) {
          return [];
        }
        const source = cloneSavedRecipeSource(prompt);
        return [
          isRecipePersistenceUncertain(source.id, persistenceScope)
            ? { ...source, syncStatus: "error" as const }
            : source,
        ];
      }),
    [prompts, target, persistenceScope],
  );

  const refreshPromptQueries = React.useCallback(async () => {
    await Promise.allSettled([
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] }),
      queryClient.invalidateQueries({ queryKey: ["getAllPromptsForSelect"] }),
    ]);
  }, [queryClient]);

  const syncIfEnabled = React.useCallback(
    async (id: string) => {
      if (!(await shouldAutoSyncWorkspacePrompts())) return false;
      const result = await autoSyncPrompt(id);
      if (!result.success && result.failureKind === "invalid_server_payload") {
        markRecipePersistenceUncertain(id, persistenceScope);
        try {
          await markPromptSyncError(id);
          // Keep the scoped marker: another backend can overwrite the shared
          // durable status without reconciling this owner's remote outcome.
        } catch {
          // The remote write is still uncertain, so local rollback is never safe.
        }
        throw uncertainSyncFailure;
      }
      // Authoritative success is cleared by sync under its dispatch scope, which
      // can differ from this editor's owner if the connection changed mid-save.
      return acceptSyncResult(result);
    },
    [persistenceScope],
  );

  const saveAsNew = React.useCallback(
    async (definition: SingleTextRecipeDefinition) => {
      if (!savePersistenceAvailable) throw new Error("recipe_save_unavailable");
      setSyncPendingNotice(false);
      try {
        const saved = await savePrompt({
          title: t("common:promptAssist.untitledRecipe", "Untitled recipe"),
          ...buildRecipePromptFields(definition),
        });
        try {
          setSyncPendingNotice(await syncIfEnabled(saved.id));
        } catch (error) {
          if (isUncertainSyncFailure(error)) throw error;
          try {
            await permanentlyDeletePrompt(saved.id, persistenceScope);
          } catch {
            throw rollbackFailure("save");
          }
          throw error;
        }
      } finally {
        await refreshPromptQueries();
      }
    },
    [
      persistenceScope,
      refreshPromptQueries,
      savePersistenceAvailable,
      syncIfEnabled,
      t,
    ],
  );

  const updateSaved = React.useCallback(
    async (savedSourceId: string, definition: SingleTextRecipeDefinition) => {
      if (!updatePersistenceAvailable) {
        throw new Error("recipe_update_unavailable");
      }
      setSyncPendingNotice(false);
      const current = prompts.find(
        (prompt) => String(prompt.id) === savedSourceId,
      ) as Prompt | undefined;
      const classification = classifyPromptRecipe(current);
      const expectedTarget = target === "system" ? "system" : "user";
      if (
        !current ||
        classification.kind !== "recipe" ||
        classification.target !== expectedTarget ||
        current.syncStatus === "conflict"
      ) {
        throw new Error("recipe_update_conflict");
      }
      const snapshot = structuredClone(current);
      try {
        const id = await updatePrompt({
          ...current,
          ...buildRecipePromptFields(definition),
          id: savedSourceId,
        });
        if (id !== savedSourceId) throw new Error("recipe_identity_changed");
        setSyncPendingNotice(await syncIfEnabled(savedSourceId));
      } catch (error) {
        if (isUncertainSyncFailure(error)) throw error;
        try {
          await restorePromptSnapshot(snapshot);
        } catch {
          throw rollbackFailure("update");
        }
        throw error;
      } finally {
        await refreshPromptQueries();
      }
    },
    [
      prompts,
      refreshPromptQueries,
      syncIfEnabled,
      target,
      updatePersistenceAvailable,
    ],
  );

  return (
    <section
      aria-label={t("common:promptAssist.recipeRegion", "Recipe builder")}
      className="min-w-0 space-y-4"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-text">
            {t("common:promptAssist.recipeTitle", "Build from recipe")}
          </h2>
          <p className="text-sm text-text-muted">
            {t(
              "common:promptAssist.recipeDescription",
              "Compile locally, then replace only the current draft when you apply.",
            )}
          </p>
        </div>
        <button type="button" className="min-h-11 px-3" onClick={onBack}>
          {t("common:back", "Back")}
        </button>
      </div>
      <SingleFieldRecipeEditor
        target={target}
        savedRecipes={savedRecipes}
        savePersistenceAvailable={savePersistenceAvailable}
        updatePersistenceAvailable={updatePersistenceAvailable}
        persistenceUnavailableReason={persistenceUnavailableReason}
        onApply={onApply}
        onSaveAsNew={saveAsNew}
        onUpdate={updateSaved}
      />
      {syncPendingNotice ? (
        <p role="status" className="text-sm text-warn">
          {t(
            "common:promptAssist.recipeSavedPending",
            "Recipe saved locally and will sync when the server is available.",
          )}
        </p>
      ) : null}
    </section>
  );
}
