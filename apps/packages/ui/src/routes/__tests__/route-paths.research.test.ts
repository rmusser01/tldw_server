import { describe, expect, it } from "vitest"
import {
  CHAT_PATH,
  RESEARCH_PATH,
  buildCharacterChatPath,
  buildChatThreadPath,
  buildResearchLaunchPath
} from "../route-paths"
import { SETTINGS_SERVER_CHAT_ID_PARAM } from "../../utils/settings-return"

describe("route-paths deep research launch", () => {
  it("exports the canonical research path", () => {
    expect(RESEARCH_PATH).toBe("/research")
  })

  it("builds a launch path with encoded query and launch options", () => {
    const href = buildResearchLaunchPath({
      query: "Investigate local evidence & timeline",
      sourcePolicy: "balanced",
      autonomyMode: "checkpointed",
      autorun: true,
      from: "chat",
      chatId: "chat_123"
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(RESEARCH_PATH)
    expect(parsed.searchParams.get("query")).toBe(
      "Investigate local evidence & timeline"
    )
    expect(parsed.searchParams.get("source_policy")).toBe("balanced")
    expect(parsed.searchParams.get("autonomy_mode")).toBe("checkpointed")
    expect(parsed.searchParams.get("autorun")).toBe("1")
    expect(parsed.searchParams.get("from")).toBe("chat")
    expect(parsed.searchParams.get("chat_id")).toBe("chat_123")
  })

  it("includes bounded follow-up context for research launches", () => {
    const followUp = {
      question: "Verify the peer coaching retention hypothesis.",
      background: {
        question: "Verify the peer coaching retention hypothesis.",
        outline: [{ title: "Hypothesis 1", focus_area: "hypothesis_1" }],
        key_claims: [
          {
            claim_id: "artifact-hypotheses:finding-1",
            text: "Evidence-supported finding: Paper A reports higher completion."
          }
        ],
        unresolved_questions: [
          "Which parts are evidence-supported versus proposed work?"
        ],
        verification_summary: {
          supported_claim_count: 1,
          unsupported_claim_count: 1
        },
        source_trust_summary: {
          high_trust_count: 2,
          low_trust_count: 1
        }
      }
    }
    const href = buildResearchLaunchPath({
      query: "Verify the peer coaching retention hypothesis.",
      followUp
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.searchParams.get("follow_up")).toBe(JSON.stringify(followUp))
  })

  it("drops oversized follow-up context instead of building an unsafe URL", () => {
    const href = buildResearchLaunchPath({
      query: "Verify a large proposal seed.",
      followUp: {
        question: "Verify a large proposal seed.",
        background: {
          question: "Verify a large proposal seed.",
          outline: [{ title: "Proposal", focus_area: "proposal" }],
          key_claims: [
            {
              claim_id: "oversized-proposal-excerpt",
              text: "x".repeat(12000)
            }
          ],
          unresolved_questions: ["Which claims still need verification?"],
          verification_summary: {
            supported_claim_count: 1,
            unsupported_claim_count: 1
          },
          source_trust_summary: {
            high_trust_count: 1,
            low_trust_count: 0
          }
        }
      }
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.searchParams.get("query")).toBe("Verify a large proposal seed.")
    expect(parsed.searchParams.get("follow_up")).toBeNull()
  })

  it("omits empty launch fields", () => {
    const href = buildResearchLaunchPath({
      query: "   ",
      sourcePolicy: "",
      autonomyMode: "",
      autorun: false,
      from: ""
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(RESEARCH_PATH)
    expect(parsed.searchParams.get("query")).toBeNull()
    expect(parsed.searchParams.get("source_policy")).toBeNull()
    expect(parsed.searchParams.get("autonomy_mode")).toBeNull()
    expect(parsed.searchParams.get("autorun")).toBeNull()
    expect(parsed.searchParams.get("from")).toBeNull()
  })

  it("builds an exact chat-thread path for server-backed return handoff", () => {
    const href = buildChatThreadPath({
      serverChatId: "chat_123",
      researchReturnRunId: "rs_1"
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get(SETTINGS_SERVER_CHAT_ID_PARAM)).toBe("chat_123")
    expect(parsed.searchParams.get("researchReturnRunId")).toBe("rs_1")
  })

  it("builds a first-class character chat intent path", () => {
    const href = buildCharacterChatPath({ characterId: "char 123" })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get("mode")).toBe("character")
    expect(parsed.searchParams.get("characterId")).toBe("char 123")
  })
})
