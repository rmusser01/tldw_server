import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  RagTerminalStreamError,
  mayReplayNonStream,
  parseRagStreamLine,
  parseRagTerminalEvent,
} from "../stream-contract"

type ContractCase = {
  name: string
  event: unknown
  valid: boolean
  may_replay: boolean
}

const fixture = JSON.parse(
  readFileSync(
    resolve(
      process.cwd(),
      "../../../tldw_Server_API/tests/fixtures/rag_terminal_stream_events.json"
    ),
    "utf8"
  )
) as { schema_version: number; cases: ContractCase[] }

describe("RAG terminal stream contract", () => {
  it("uses the shared schema version", () => {
    expect(fixture.schema_version).toBe(1)
  })

  for (const contractCase of fixture.cases) {
    it(`${contractCase.name} validates and replays conservatively`, () => {
      if (!contractCase.valid) {
        expect(() => parseRagTerminalEvent(contractCase.event)).toThrow(
          RagTerminalStreamError
        )
        return
      }

      const event = parseRagTerminalEvent(contractCase.event)
      expect(event).not.toBeNull()
      expect(mayReplayNonStream(event!)).toBe(contractCase.may_replay)
    })
  }

  it("does not treat ordinary stream events as terminal", () => {
    expect(parseRagTerminalEvent({ type: "delta", text: "answer" })).toBeNull()
  })

  it("fails closed instead of dropping malformed JSON stream lines", () => {
    expect(() => parseRagStreamLine('{"type":"error"')).toThrow(
      RagTerminalStreamError
    )
  })

  it("strips unknown terminal fields before returning a stream event", () => {
    const sentinel = "sk-untrusted-terminal-field"
    const parsed = parseRagStreamLine(
      JSON.stringify({
        schema_version: 1,
        type: "error",
        code: "credential_store_unavailable",
        status_code: 503,
        upstream_dispatched: true,
        output_emitted: false,
        allow_non_stream_fallback: false,
        message: "Provider credential storage is temporarily unavailable.",
        untrusted_detail: sentinel
      })
    )

    expect(parsed).toEqual({
      schema_version: 1,
      type: "error",
      code: "credential_store_unavailable",
      status_code: 503,
      upstream_dispatched: true,
      output_emitted: false,
      allow_non_stream_fallback: false,
      message: "Provider credential storage is temporarily unavailable."
    })
    expect(JSON.stringify(parsed)).not.toContain(sentinel)
  })
})
