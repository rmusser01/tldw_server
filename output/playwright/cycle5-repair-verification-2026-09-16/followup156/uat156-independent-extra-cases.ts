

describe("independent timestamp boundary probes", () => {
  it.each(["1000", "", true, false, {}, [], new Date(1000), Object(1000), 1000n].map(value => [value]))("rejects a nonnumeric timestamp without coercing local epoch zero: %s", async invalid => {
    await saveLocalTurn()
    const row = state.messages.get("local-user")!
    state.messages.set(row.id, { ...row, createdAt: 0 })
    const incoming = canonicalTurn()
    incoming[0].createdAt = invalid as number
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), incoming)
    const result = await mirror(incoming, memory)
    expect(memory[0]).toMatchObject({ createdAt: 0, message: rawQuestion })
    expect(formatToMessage(result.rows)[0]).toMatchObject({ createdAt: 0, message: rawQuestion })
  })

  it("gives new invalid-dated rows a stable persisted fallback that reaches memory", async () => {
    vi.spyOn(Date, "now").mockReturnValue(9000)
    const incoming = [{ ...remote("new-server", "user", "New canonical row", 1000), createdAt: NaN }]
    const firstMemory = reconcileServerChatMessages([], incoming)
    const first = await mirror(incoming, firstMemory)
    const reloaded = formatToMessage(first.rows)
    const finalMemory = reconcileServerChatMessages(firstMemory, reloaded, firstMemory)
    expect(finalMemory[0].createdAt).toBe(9000)
    vi.spyOn(Date, "now").mockReturnValue(9500)
    expect(formatToMessage((await mirror(incoming, finalMemory)).rows)[0].createdAt).toBe(9000)
  })

  it("uses the captured local fallback when no durable row exists", async () => {
    const local = { ...remote("server-user", "user", "Captured raw edit", 3001), id: "captured-local" }
    const incoming = [{ ...remote("server-user", "user", "Canonical wrapped content", 1000), createdAt: Infinity }]
    const first = await mirror(incoming, [local])
    expect(formatToMessage(first.rows)[0]).toMatchObject({ id:"captured-local", message:"Captured raw edit", createdAt:3001 })
  })

  it.each([500, 6000])("accepts canonical time %s even for a newer protected edit", async timestamp => {
    await saveLocalTurn()
    const local = state.messages.get("local-user")!
    state.messages.set(local.id, { ...local, content:"Newer edit", serverMessageVersion:8 })
    const incoming = canonicalTurn()
    incoming[0].createdAt = timestamp
    const memory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]), incoming)
    const result = await mirror(incoming, memory)
    expect(memory[0]).toMatchObject({ message:"Newer edit", createdAt:timestamp, serverMessageVersion:8 })
    expect(result.rows.find(row => row.id === "local-user")).toMatchObject({ content:"Newer edit", createdAt:timestamp, serverMessageVersion:8 })
  })

  it("keeps a timestamp-only local mutation from replacing canonical chronology", () => {
    const before = [{ ...remote("server-user", "user", "Same text", 3001), id:"local-user" }]
    const current = [{ ...before[0], createdAt: 9999 }]
    const incoming = [remote("server-user", "user", "Same text", 1000)]
    expect(reconcileServerChatMessages(current, incoming, before)[0]).toMatchObject({ id:"local-user", createdAt:1000, message:"Same text" })
  })
})


describe("independent real loader fallback shape", () => {
  it("assigns a stable new-row fallback after invalid Date.parse maps to undefined", async () => {
    vi.spyOn(Date, "now").mockReturnValue(9000)
    const incoming = [{ ...remote("new-server", "user", "New canonical row", 1000), createdAt: undefined }]
    const firstMemory = reconcileServerChatMessages([], incoming)
    const first = await mirror(incoming, firstMemory)
    const reloaded = formatToMessage(first.rows)
    const finalMemory = reconcileServerChatMessages(firstMemory, reloaded, firstMemory)
    expect(finalMemory[0].createdAt).toBe(9000)
    vi.spyOn(Date, "now").mockReturnValue(9500)
    expect(formatToMessage((await mirror(incoming, finalMemory)).rows)[0].createdAt).toBe(9000)
  })
})
