import { describe, expect, it, vi } from "vitest"
import {
  buildServerSlashCommands,
  mergeSlashCommands,
  normalizeServerCommandsPayload
} from "../server-slash-commands"

// Shared by the side panel and the full-page composer (CC-11).
describe("server slash commands", () => {
  it("reads the server's { commands: [...] } list", () => {
    expect(
      normalizeServerCommandsPayload({ commands: [{ name: "time", description: "Time" }, 5] })
    ).toEqual({ commands: [{ name: "time", description: "Time" }] })
    expect(normalizeServerCommandsPayload({ commands: [] })).toEqual({ commands: [] })
    expect(normalizeServerCommandsPayload("nope")).toBeNull()
    expect(normalizeServerCommandsPayload(null)).toBeNull()
  })

  it("drops disabled commands and ones the user lacks permission for", () => {
    const commands = buildServerSlashCommands(
      {
        commands: [
          { name: "time", description: "Time" },
          { name: "/Weather", description: "Weather" },
          { name: "off", enabled: false },
          { name: "admin", permissions: ["chat.admin"] }
        ],
        permissions: ["chat.basic"]
      },
      {},
      {}
    )
    expect(commands.map((command) => command.command)).toEqual(["time", "weather"])
    expect(commands[0]).toMatchObject({ source: "server", label: "time", description: "Time" })
  })

  it("keeps a local command's action when the server lists the same name", () => {
    const search = vi.fn()
    const [command] = buildServerSlashCommands(
      { commands: [{ name: "search" }] },
      { search: { label: "Search", description: "Local search", keywords: ["rag"] } },
      { search }
    )
    expect(command).toMatchObject({ label: "search", description: "Local search", keywords: ["rag"] })
    expect(command.action).toBe(search)
  })

  it("lists server commands first and fills in local ones", () => {
    const local = [
      { id: "slash-search", command: "search", label: "Local search" },
      { id: "slash-new", command: "new", label: "New chat" }
    ]
    const server = [
      { id: "s-search", command: "search", label: "Server search" },
      { id: "s-time", command: "time", label: "time" }
    ]
    expect(mergeSlashCommands([], local)).toBe(local)
    expect(mergeSlashCommands(server, local).map((command) => command.label)).toEqual([
      "Server search",
      "time",
      "New chat"
    ])
  })
})
