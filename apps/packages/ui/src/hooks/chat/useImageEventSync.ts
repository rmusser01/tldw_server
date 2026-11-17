import React from "react";
import { updateMessageMedia } from "@/db/dexie/helpers";
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  resolveImageGenerationEventId,
  normalizeImageGenerationEventSyncPolicy,
  normalizeImageGenerationEventSyncMode,
  resolveImageGenerationEventSyncMode,
  type ImageGenerationEventSyncPolicy,
  type ImageGenerationEventSyncMode,
} from "@/utils/image-generation-chat";
import type { Message } from "@/store/option";
import type { SaveMessagePayload } from "./chat-action-utils";

type ChatSettings = {
  imageEventSyncMode?: ImageGenerationEventSyncMode;
} | null;

type UseImageEventSyncOptions = {
  chatSettings: ChatSettings;
  imageEventSyncGlobalDefault: ImageGenerationEventSyncMode;
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[]),
  ) => void;
};

export const useImageEventSync = ({
  chatSettings,
  imageEventSyncGlobalDefault,
  setMessages,
}: UseImageEventSyncOptions) => {
  const resolveImageEventSyncModeForPayload = React.useCallback(
    (payload?: SaveMessagePayload): ImageGenerationEventSyncMode => {
      const requestPolicy = normalizeImageGenerationEventSyncPolicy(
        payload?.imageEventSyncPolicy,
        "inherit",
      );
      const chatMode = normalizeImageGenerationEventSyncMode(
        chatSettings?.imageEventSyncMode,
        "off",
      );
      const globalMode = normalizeImageGenerationEventSyncMode(
        imageEventSyncGlobalDefault,
        "off",
      );
      return resolveImageGenerationEventSyncMode({
        requestPolicy,
        chatMode,
        globalMode,
      });
    },
    [chatSettings?.imageEventSyncMode, imageEventSyncGlobalDefault],
  );

  const updateImageEventSyncMetadata = React.useCallback(
    async (
      payload: SaveMessagePayload,
      update: {
        status: "pending" | "synced" | "failed";
        policy: ImageGenerationEventSyncPolicy;
        mode: ImageGenerationEventSyncMode;
        serverMessageId?: string;
        error?: string;
      },
    ) => {
      const targetMessageId = payload.assistantMessageId;
      if (!targetMessageId) return;
      const now = Date.now();

      let nextGenerationInfo: Record<string, unknown> | null = null;
      let nextImages: string[] = [];

      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== targetMessageId) return entry;
          const currentGenerationInfo =
            entry.generationInfo &&
            typeof entry.generationInfo === "object" &&
            !Array.isArray(entry.generationInfo)
              ? (entry.generationInfo as Record<string, unknown>)
              : {};
          const currentImageGeneration =
            currentGenerationInfo.image_generation &&
            typeof currentGenerationInfo.image_generation === "object" &&
            !Array.isArray(currentGenerationInfo.image_generation)
              ? (currentGenerationInfo.image_generation as Record<
                  string,
                  unknown
                >)
              : {};

          nextGenerationInfo = {
            ...currentGenerationInfo,
            image_generation: {
              ...currentImageGeneration,
              sync: {
                mode: update.mode,
                policy: update.policy,
                status: update.status,
                serverMessageId: update.serverMessageId,
                error: update.error,
                lastAttemptAt: now,
                mirroredAt: update.status === "synced" ? now : undefined,
              },
            },
          };
          nextImages = Array.isArray(entry.images)
            ? entry.images.filter(
                (image): image is string =>
                  typeof image === "string" && image.length > 0,
              )
            : [];
          return {
            ...entry,
            generationInfo: nextGenerationInfo,
          };
        }),
      );

      if (!nextGenerationInfo) return;
      await updateMessageMedia(targetMessageId, {
        images: nextImages,
        generationInfo: nextGenerationInfo,
      }).catch((err) =>
        console.warn(
          "[updateImageEventSyncMetadata] Failed to persist media metadata:",
          err,
        ),
      );
    },
    [setMessages],
  );

  return {
    resolveImageEventSyncModeForPayload,
    updateImageEventSyncMetadata,
  };
};


export const updateImageEventSyncMetadataForPayload = async (
      payload: SaveMessagePayload,
      update: {
        status: "pending" | "synced" | "failed"
        policy: ImageGenerationEventSyncPolicy
        mode: ImageGenerationEventSyncMode
        serverMessageId?: string
        error?: string
      },
      sourceIsCurrent: () => boolean,
      setMessages: UseImageEventSyncOptions["setMessages"]
) => {
      const targetMessageId = payload.assistantMessageId
      if (!targetMessageId) return
      const targetEventId = resolveImageGenerationEventId({
        messageId: targetMessageId,
        messageGenerationInfo: payload.generationInfo
      })
      const now = Date.now()

      const generationInfo =
        payload.generationInfo &&
        typeof payload.generationInfo === "object" &&
        !Array.isArray(payload.generationInfo)
          ? (payload.generationInfo as Record<string, unknown>)
          : {}
      const imageGeneration =
        generationInfo.image_generation &&
        typeof generationInfo.image_generation === "object" &&
        !Array.isArray(generationInfo.image_generation)
          ? (generationInfo.image_generation as Record<string, unknown>)
          : {}
      let nextGenerationInfo: Record<string, unknown> = {
        ...generationInfo,
        image_generation: {
          ...imageGeneration,
          sync: {
            mode: update.mode,
            policy: update.policy,
            status: update.status,
            serverMessageId: update.serverMessageId,
            error: update.error,
            lastAttemptAt: now,
            mirroredAt: update.status === "synced" ? now : undefined
          }
        }
      }
      let nextImages = Array.isArray(payload.assistantImages)
        ? payload.assistantImages.filter(
            (image): image is string =>
              typeof image === "string" && image.length > 0
          )
        : []
      // A late acknowledgement belongs to its original payload, even after same-ID reopening.
      const publishSyncToCurrentEvent = () => {
        if (sourceIsCurrent())
          setMessages((prev) => {
            if (!sourceIsCurrent()) return prev
            return prev.map((entry) => {
              if (
                entry.id !== targetMessageId &&
                (entry.messageType !== IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE ||
                  resolveImageGenerationEventId({
                    messageId: entry.id,
                    messageGenerationInfo: entry.generationInfo,
                    variants: entry.variants
                  }) !== targetEventId)
              )
                return entry
              const currentInfo =
                entry.generationInfo &&
                typeof entry.generationInfo === "object" &&
                !Array.isArray(entry.generationInfo)
                  ? (entry.generationInfo as Record<string, unknown>)
                  : generationInfo
              const currentGeneration =
                currentInfo.image_generation &&
                typeof currentInfo.image_generation === "object" &&
                !Array.isArray(currentInfo.image_generation)
                  ? (currentInfo.image_generation as Record<string, unknown>)
                  : imageGeneration
              nextGenerationInfo = {
                ...currentInfo,
                image_generation: {
                  ...currentGeneration,
                  sync: (
                    nextGenerationInfo.image_generation as Record<string, unknown>
                  ).sync
                }
              }
              nextImages = Array.isArray(entry.images)
                ? entry.images.filter(
                    (image): image is string =>
                      typeof image === "string" && image.length > 0
                  )
                : nextImages
              return { ...entry, generationInfo: nextGenerationInfo }
            })
          })
      }
      publishSyncToCurrentEvent()

      await updateMessageMedia(targetMessageId, {
        images: nextImages,
        generationInfo: nextGenerationInfo
      }).catch(() => null)
      publishSyncToCurrentEvent()
      return {
        generationInfo: nextGenerationInfo,
        assistantImages: nextImages
      }
}
