import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

/** The same credential-free locator is used by composer records and session handles. */
export const chatDraftOwnerKey = (scope: ServicePromptRequestScope) =>
  bytesToHex(
    sha256(
      utf8ToBytes(
        JSON.stringify([
          scope.config.serverUrl.replace(/\/+$/, ""),
          scope.config.authMode,
          scope.config.authSource || "manual",
          scope.config.orgId ?? null,
          scope.userId,
          scope.config.expectedSingleUserApiKeyScope ?? null
        ])
      )
    )
  )
