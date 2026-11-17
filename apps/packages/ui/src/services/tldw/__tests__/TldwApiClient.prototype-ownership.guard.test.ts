import { describe, expect, it } from "vitest"

import { TldwApiClient, tldwClient } from "../TldwApiClient"
import { adminMethods } from "../domains/admin"
import { characterMethods } from "../domains/characters"
import { chatRagMethods } from "../domains/chat-rag"
import { collectionsMethods } from "../domains/collections"
import { mediaMethods } from "../domains/media"
import { modelsAudioMethods } from "../domains/models-audio"
import { presentationsMethods } from "../domains/presentations"
import { prototypeWorkspaceMethods } from "../domains/prototype-workspaces"
import { servicePromptMethods } from "../domains/service-prompts"
import { setupOnboardingMethods } from "../domains/setup-onboarding"
import { visualIdentityMethods } from "../domains/visual-identities"
import { webClipperMethods } from "../domains/web-clipper"
import { workspaceApiMethods } from "../domains/workspace-api"

/**
 * TASK-13526 Stage 3: the composition root applies domain mixins via
 * Object.assign, so every domain method must actually be the runtime
 * implementation on the prototype — the declaration-merged interface alone
 * can claim an override that never bound. This guard pins each claimed
 * override to its domain source, which is the precondition for safely
 * deleting the shadowed base copies.
 */
const mixinGroups = [
  ["admin", adminMethods],
  ["characters", characterMethods],
  ["chat-rag", chatRagMethods],
  ["collections", collectionsMethods],
  ["media", mediaMethods],
  ["models-audio", modelsAudioMethods],
  ["presentations", presentationsMethods],
  ["prototype-workspaces", prototypeWorkspaceMethods],
  ["service-prompts", servicePromptMethods],
  ["setup-onboarding", setupOnboardingMethods],
  ["visual-identities", visualIdentityMethods],
  ["web-clipper", webClipperMethods],
  ["workspace-api", workspaceApiMethods],
] as const

describe("TldwApiClient prototype ownership (stage 3 guard)", () => {
  it("binds every domain mixin method as the runtime implementation", () => {
    for (const [name, methods] of mixinGroups) {
      for (const key of Object.keys(methods) as Array<keyof typeof methods>) {
        if (key === "createChatCompletion") {
          // Documented carve-out: the composition root re-pins
          // createChatCompletion to the base implementation AFTER the
          // mixins are applied (see the tail of TldwApiClient.ts); the
          // second test pins that explicitly.
          continue
        }
        expect(
          TldwApiClient.prototype[key as keyof TldwApiClient.prototype],
          `${name}.${String(key)} must own the prototype slot`,
        ).toBe(methods[key])
      }
    }
  })

  it("createChatCompletion remains canonical on the base", () => {
    expect(tldwClient.createChatCompletion).toBe(
      (TldwApiClient.prototype as unknown as Record<string, unknown>)
        .createChatCompletion,
    )
  })
})
