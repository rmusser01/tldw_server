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
      []
    )
    expect(commands.map((command) => command.command)).toEqual(["time", "weather"])
    expect(commands[0]).toMatchObject({ source: "server", label: "time", description: "Time" })
  })

  it("keeps a local command's action and behaviour when the server lists the same name", () => {
    const search = vi.fn()
    const local = [
      { id: "slash-search", command: "search", label: "Search", description: "Local search", keywords: ["rag"], action: search },
      { id: "slash-new", command: "new", label: "New chat", action: vi.fn(), standalone: true }
    ]
    const [searchCommand, newCommand] = buildServerSlashCommands(
      { commands: [{ name: "search" }, { name: "new", description: "Server new" }] },
      local
    )
    expect(searchCommand).toMatchObject({ label: "search", description: "Local search", keywords: ["rag"], source: "server" })
    expect(searchCommand.action).toBe(search)
    expect(newCommand).toMatchObject({ standalone: true, description: "Server new", source: "server" })
    expect(newCommand.action).toBe(local[1].action)
  })

  it("describes a server command with its usage, or builds one from its arguments", () => {
    const [weather, forecast] = buildServerSlashCommands(
      {
        commands: [
          { name: "weather", description: "Weather for a place", usage: "/weather <place>" },
          { name: "forecast", description: "forecast", args: ["city", "days"] }
        ]
      } as never,
      []
    )
    expect(weather.description).toBe("Weather for a place · /weather <place>")
    expect(forecast.description).toBe("/forecast <city> <days>")
  })

  it("can list the local commands first, so server commands do not move the highlighted row", () => {
    const local = [
      { id: "slash-search", command: "search", label: "Local search" },
      { id: "slash-new", command: "new", label: "New chat" }
    ]
    const server = [
      { id: "s-search", command: "search", label: "Server search" },
      { id: "s-time", command: "time", label: "time" }
    ]
    expect(mergeSlashCommands(server, local, { localFirst: true }).map((command) => command.label)).toEqual([
      "Server search",
      "New chat",
      "time"
    ])
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
