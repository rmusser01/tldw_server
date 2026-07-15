import type { Page, Request, Route } from "@playwright/test"

export type SkillsFixtureRequestContract = {
  origin: string
  apiKey?: string
}

export type SkillsFixtureOptions = {
  requestContract?: SkillsFixtureRequestContract
}

export function validateSkillsFixtureRequest(
  request: Pick<Request, "headers" | "method" | "url">,
  expectedMethod: string | readonly string[],
  contract?: SkillsFixtureRequestContract,
): void {
  const expectedMethods = typeof expectedMethod === "string"
    ? [expectedMethod]
    : expectedMethod
  const method = request.method()

  if (!expectedMethods.includes(method)) {
    throw new Error(
      `Expected ${expectedMethods.join(" or ")} request, received ${method}`,
    )
  }
  if (!contract) return

  if (new URL(request.url()).origin !== contract.origin) {
    throw new Error(`Expected request origin ${contract.origin}`)
  }
  if (contract.apiKey === undefined) return

  const apiKey = request.headers()["x-api-key"]
  if (!apiKey) {
    throw new Error("Expected x-api-key request header")
  }
  if (apiKey !== contract.apiKey) {
    throw new Error("Unexpected x-api-key request header")
  }
}

type SkillFixtureSummary = typeof seededSkillSummary & {
  allowed_tools?: string[] | null
  model?: string | null
  version?: number
  runtime?: {
    execution_mode?: "inline" | "fork"
    declares_tools?: boolean
    declared_tool_count?: number
  }
}

export const seededSkillSummary = {
  name: "summarize",
  description: "Summarize source material",
  argument_hint: "[text]",
  user_invocable: true,
  disable_model_invocation: false,
  context: "inline",
}

const seededSkillResponse = {
  ...seededSkillSummary,
  id: "skill-summarize",
  allowed_tools: null,
  model: null,
  content:
    "---\ndescription: Summarize source material\nargument-hint: \"[text]\"\ncontext: inline\n---\n\nSummarize this source: $ARGUMENTS",
  raw_content: null,
  supporting_files: null,
  directory_path: "/mock/skills/summarize",
  created_at: "2026-06-01T00:00:00Z",
  last_modified: "2026-06-01T00:00:00Z",
  version: 1,
}

type TldwConnectionRootStore = {
  state: Record<string, unknown>
  checkOnce: (options?: { force?: boolean }) => Promise<void>
}

type TldwConnectionStore = {
  getState: () => TldwConnectionRootStore
  setState: (value: Partial<TldwConnectionRootStore>) => void
}

type TldwConnectionStoreWindow = Window & {
  __tldw_useConnectionStore?: TldwConnectionStore
}

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

async function mockSkillsCapabilityRoutes(
  page: Page,
  requestContract?: SkillsFixtureRequestContract,
) {
  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/api\/v1\/health\/live(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/api\/v1\/rag\/health(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, {
      openapi: "3.0.0",
      info: { title: "Mock tldw API", version: "test" },
      paths: {
        "/api/v1/skills": { get: {}, post: {} },
        "/api/v1/skills/trash": { get: {} },
        "/api/v1/skills/bulk-delete": { post: {} },
        "/api/v1/skills/context": { get: {} },
        "/api/v1/skills/seed": { post: {} },
        "/api/v1/skills/{name}": { get: {}, put: {}, delete: {} },
        "/api/v1/skills/{name}/execute": { post: {} },
        "/api/v1/skills/{name}/restore": { post: {} },
        "/api/v1/skills/{name}/purge": { delete: {} },
        "/api/v1/skills/import": { post: {} },
        "/api/v1/skills/import/file": { post: {} },
      },
    })
  })
}

export async function mockSkillsBeginnerApi(
  page: Page,
  options: SkillsFixtureOptions & { seeded?: boolean } = {},
) {
  let seeded = Boolean(options.seeded)
  const seedRequests: URL[] = []
  const executeRequests: Array<{ args?: string; dry_run?: boolean }> = []
  const { requestContract } = options

  await mockSkillsCapabilityRoutes(page, requestContract)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const url = new URL(route.request().url())
    const query = (url.searchParams.get("q") ?? "").trim().toLowerCase()
    const skills = seeded
      && (!query
        || seededSkillSummary.name.toLowerCase().includes(query)
        || seededSkillSummary.description.toLowerCase().includes(query))
      ? [seededSkillSummary]
      : []
    await fulfillJson(route, {
      skills,
      count: skills.length,
      total: skills.length,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, {
      available_skills: seeded ? [seededSkillSummary] : [],
      context_text: seeded ? "/skill summarize [text]" : "",
    })
  })

  await page.route(/\/api\/v1\/skills\/seed(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "POST", requestContract)
    seedRequests.push(new URL(route.request().url()))
    seeded = true
    await fulfillJson(route, { seeded: ["summarize"], count: 1 })
  })

  await page.route(/\/api\/v1\/skills\/summarize(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, seededSkillResponse)
  })

  await page.route(
    /\/api\/v1\/skills\/summarize\/execute(?:\/)?(?:\?.*)?$/,
    async (route) => {
      validateSkillsFixtureRequest(route.request(), "POST", requestContract)
      const body = (route.request().postDataJSON() ?? {}) as {
        args?: string
        dry_run?: boolean
      }
      executeRequests.push(body)
      const args = body.args ?? "A long article about Skills UX"
      await fulfillJson(route, {
        skill_name: "summarize",
        rendered_prompt: `Summarize this source: ${args}`,
        allowed_tools: null,
        model_override: null,
        execution_mode: "inline",
        fork_output: null,
        dry_run: Boolean(body.dry_run),
      })
    }
  )

  return { executeRequests, seedRequests }
}

export async function mockSkillsTrashWorkflow(
  page: Page,
  options: SkillsFixtureOptions = {},
) {
  let state: "active" | "trash" | "purged" = "active"
  let version = 1
  const operations: string[] = []
  const { requestContract } = options

  await mockSkillsCapabilityRoutes(page, requestContract)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const skills = state === "active"
      ? [{ ...seededSkillSummary, version }]
      : []
    await fulfillJson(route, {
      skills,
      count: skills.length,
      total: skills.length,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/trash(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const skills = state === "trash"
      ? [{
          ...seededSkillSummary,
          allowed_tools: null,
          model: null,
          deleted_at: "2026-07-14T12:00:00Z",
          version,
          restorable: true,
          restore_unavailable_reason: null,
        }]
      : []
    await fulfillJson(route, {
      skills,
      count: skills.length,
      total: skills.length,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const availableSkills = state === "active"
      ? [{ ...seededSkillSummary, version }]
      : []
    await fulfillJson(route, {
      available_skills: availableSkills,
      context_text: availableSkills.length ? "/skill summarize [text]" : "",
    })
  })

  await page.route(
    /\/api\/v1\/skills\/summarize\/restore(?:\/)?(?:\?.*)?$/,
    async (route) => {
      validateSkillsFixtureRequest(route.request(), "POST", requestContract)
      if (state !== "trash") {
        await fulfillJson(
          route,
          { detail: "Skill is not in Trash" },
          state === "active" ? 409 : 404,
        )
        return
      }
      operations.push("restore")
      state = "active"
      version += 1
      await fulfillJson(route, { ...seededSkillResponse, version })
    }
  )

  await page.route(
    /\/api\/v1\/skills\/summarize\/purge(?:\/)?(?:\?.*)?$/,
    async (route) => {
      validateSkillsFixtureRequest(route.request(), "DELETE", requestContract)
      if (state !== "trash") {
        await fulfillJson(
          route,
          { detail: "Skill must be moved to Trash first" },
          state === "active" ? 409 : 404,
        )
        return
      }
      operations.push("purge")
      state = "purged"
      await fulfillJson(route, { deleted: true })
    }
  )

  await page.route(/\/api\/v1\/skills\/summarize(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(
      route.request(),
      ["GET", "DELETE"],
      requestContract,
    )
    if (route.request().method() === "DELETE") {
      if (state !== "active") {
        await fulfillJson(route, { detail: "Skill not found" }, 404)
        return
      }
      operations.push("delete")
      state = "trash"
      version += 1
      await fulfillJson(route, { deleted: true })
      return
    }
    if (state !== "active") {
      await fulfillJson(route, { detail: "Skill not found" }, 404)
      return
    }
    await fulfillJson(route, { ...seededSkillResponse, version })
  })

  return {
    operations,
    state: () => state,
  }
}

const largeLibrarySkills: SkillFixtureSummary[] = [
  ...Array.from({ length: 28 }, (_, index) => ({
    name: `archive-helper-${String(index + 1).padStart(2, "0")}`,
    description: `Archive helper ${index + 1}`,
    argument_hint: "[text]",
    user_invocable: true,
    disable_model_invocation: false,
    context: "inline",
    allowed_tools: null,
    model: null,
    version: index + 1,
  })),
  {
    name: "target-research-formatter",
    description: "Target research formatter",
    argument_hint: "[source]",
    user_invocable: true,
    disable_model_invocation: false,
    context: "fork",
    allowed_tools: ["search"],
    model: "gpt-4.1-mini",
    version: 31,
    runtime: {
      execution_mode: "fork",
      declares_tools: true,
      declared_tool_count: 1,
    },
  },
  {
    name: "batch-cleanup-helper",
    description: "Batch cleanup helper",
    argument_hint: "[items]",
    user_invocable: true,
    disable_model_invocation: false,
    context: "fork",
    allowed_tools: ["filesystem"],
    model: null,
    version: 32,
    runtime: {
      execution_mode: "fork",
      declares_tools: true,
      declared_tool_count: 1,
    },
  },
]

const filterLargeLibrarySkills = (url: URL) => {
  const query = (url.searchParams.get("q") ?? "").trim().toLowerCase()
  const context = url.searchParams.get("context")
  const hasTools = url.searchParams.get("has_tools")
  const model = (url.searchParams.get("model") ?? "").trim().toLowerCase()
  const sort = url.searchParams.get("sort")
  const order = url.searchParams.get("order")
  const limit = Number(url.searchParams.get("limit") ?? 10)
  const offset = Number(url.searchParams.get("offset") ?? 0)

  let skills = largeLibrarySkills.filter((skill) => {
    if (
      query
      && !skill.name.toLowerCase().includes(query)
      && !skill.description.toLowerCase().includes(query)
    ) {
      return false
    }
    if (context && skill.context !== context) return false
    if (hasTools === "true" && !(skill.allowed_tools?.length)) return false
    if (hasTools === "false" && skill.allowed_tools?.length) return false
    if (model && (skill.model ?? "").trim().toLowerCase() !== model) return false
    return true
  })

  if (sort === "name") {
    skills = [...skills].sort((a, b) => a.name.localeCompare(b.name))
    if (order === "desc") skills.reverse()
  }

  return {
    skills: skills.slice(offset, offset + limit),
    total: skills.length,
    limit,
    offset,
  }
}

export async function mockPowerUserSkillsLibrary(
  page: Page,
  options: SkillsFixtureOptions = {},
) {
  const listUrls: URL[] = []
  const deleteRequests: unknown[] = []
  const exportRequests: Array<{ method: string; name: string }> = []
  const { requestContract } = options

  await mockSkillsCapabilityRoutes(page, requestContract)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const url = new URL(route.request().url())
    listUrls.push(url)
    const result = filterLargeLibrarySkills(url)
    await fulfillJson(route, {
      skills: result.skills,
      count: result.skills.length,
      total: result.total,
      limit: result.limit,
      offset: result.offset,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, {
      available_skills: largeLibrarySkills,
      context_text: "/skill target-research-formatter [source]",
    })
  })

  await page.route(/\/api\/v1\/skills\/([^/?]+)(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(
      route.request(),
      ["GET", "DELETE"],
      requestContract,
    )
    const segments = new URL(route.request().url()).pathname.split("/").filter(Boolean)
    const name = decodeURIComponent(segments.pop() ?? "")
    const skill = largeLibrarySkills.find((candidate) => candidate.name === name)
    if (!skill) {
      await fulfillJson(route, { detail: "Skill not found" }, 404)
      return
    }
    if (route.request().method() === "DELETE") {
      deleteRequests.push({ name })
      await fulfillJson(route, { deleted: true, count: 1 })
      return
    }
    await fulfillJson(route, {
      ...skill,
      id: `skill-${name}`,
      content: `---\ndescription: ${skill.description}\n---\n\nUse ${name}.`,
      raw_content: null,
      supporting_files: null,
      directory_path: `/mock/skills/${name}`,
      created_at: "2026-06-01T00:00:00Z",
      last_modified: "2026-06-01T00:00:00Z",
    })
  })

  await page.route(/\/api\/v1\/skills\/([^/?]+)\/export(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    const segments = new URL(route.request().url()).pathname.split("/").filter(Boolean)
    const rawName = segments.at(-2) ?? ""
    let decodedName: string | undefined
    try {
      decodedName = decodeURIComponent(rawName)
    } catch {
      decodedName = undefined
    }

    const method = route.request().method()
    exportRequests.push({ method, name: decodedName ?? rawName })
    if (
      decodedName === undefined
      || !largeLibrarySkills.some((candidate) => candidate.name === decodedName)
    ) {
      await fulfillJson(route, { detail: "Skill not found" }, 404)
      return
    }

    const name = decodedName
    await route.fulfill({
      status: 200,
      headers: {
        "content-type": "application/zip",
        "content-disposition": `attachment; filename="${name}.zip"`,
      },
      body: Buffer.from([
        0x50, 0x4b, 0x05, 0x06,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
      ]),
    })
  })

  return {
    deleteRequests,
    exportRequests,
    lastListUrl: () => listUrls.at(-1),
    listRequestCount: () => listUrls.length,
  }
}

async function mockSingleSkillLibrary(
  page: Page,
  options: {
    executeStatus?: number
    executePayload?: unknown
    deleteStatus?: number
    deletePayload?: unknown
  } = {}
) {
  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      skills: [seededSkillSummary],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      available_skills: [seededSkillSummary],
      context_text: "/skill summarize [text]",
    })
  })

  await page.route(/\/api\/v1\/skills\/summarize(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() === "DELETE") {
      await fulfillJson(
        route,
        options.deletePayload ?? { detail: "Version conflict" },
        options.deleteStatus ?? 200
      )
      return
    }
    await fulfillJson(route, seededSkillResponse)
  })

  await page.route(
    /\/api\/v1\/skills\/summarize\/execute(?:\/)?(?:\?.*)?$/,
    async (route) => {
      const status = options.executeStatus ?? 200
      if (status >= 400) {
        await fulfillJson(
          route,
          options.executePayload ?? { detail: "Model unavailable" },
          status
        )
        return
      }
      await fulfillJson(route, {
        skill_name: "summarize",
        rendered_prompt: "Summarize this source: failure test",
        allowed_tools: null,
        model_override: null,
        execution_mode: "inline",
        fork_output: null,
        dry_run: false,
      })
    }
  )
}

export async function mockSkillsExecutionFailure(page: Page) {
  await mockSingleSkillLibrary(page, {
    executeStatus: 500,
    executePayload: { detail: "Model unavailable" },
  })
}

export async function mockSkillsStaleVersionConflict(page: Page) {
  await mockSingleSkillLibrary(page, {
    deleteStatus: 409,
    deletePayload: { detail: "Version conflict" },
  })
}

export async function mockSkillsImportValidationFailure(page: Page) {
  const importRequests: unknown[] = []

  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      skills: [],
      count: 0,
      total: 0,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      available_skills: [],
      context_text: "",
    })
  })

  await page.route(/\/api\/v1\/skills\/import\/preview(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      valid: false,
      errors: ["Missing skill description"],
      name: null,
      description: null,
      argument_hint: null,
      context: "inline",
      conflict: false,
      existing_version: null,
      supporting_file_count: 0,
      model: null,
      allowed_tools: null,
    })
  })

  await page.route(/\/api\/v1\/skills\/import(?:\/)?(?:\?.*)?$/, async (route) => {
    importRequests.push(route.request().postDataJSON())
    await fulfillJson(route, { detail: "Import should not be called" }, 500)
  })

  return { importRequests }
}

export async function mockSkillsSlowList(page: Page) {
  let resolveList: () => void = () => {}
  const listReleased = new Promise<void>((resolve) => {
    resolveList = resolve
  })
  let listRequests = 0

  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    listRequests += 1
    await listReleased
    await fulfillJson(route, {
      skills: [seededSkillSummary],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      available_skills: [seededSkillSummary],
      context_text: "/skill summarize [text]",
    })
  })

  return {
    listRequests: () => listRequests,
    resolveList,
  }
}

export async function mockSkillsListRecovery(
  page: Page,
  options: SkillsFixtureOptions = {},
) {
  let releaseFirst: () => void = () => {}
  const firstRequestReleased = new Promise<void>((resolve) => {
    releaseFirst = resolve
  })
  let listRequests = 0
  const { requestContract } = options

  await mockSkillsCapabilityRoutes(page, requestContract)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)

    listRequests += 1
    if (listRequests === 1) {
      await firstRequestReleased
    }

    if (listRequests <= 2) {
      const apiKey = route.request().headers()["x-api-key"]
      if (!apiKey) {
        throw new Error("Expected the Skills list request to include x-api-key")
      }
      await fulfillJson(
        route,
        {
          detail:
            `api_key=${apiKey} path=/Users/skills-parity/private-list.log\n`
            + "RAW_SKILLS_LIST_503_BODY",
        },
        503,
      )
      return
    }

    if (listRequests === 3) {
      await fulfillJson(route, {
        skills: [seededSkillSummary],
        count: 1,
        total: 1,
        limit: 10,
        offset: 0,
      })
      return
    }

    await fulfillJson(route, { detail: "Unexpected Skills list request" }, 500)
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    validateSkillsFixtureRequest(route.request(), "GET", requestContract)
    await fulfillJson(route, {
      available_skills: [seededSkillSummary],
      context_text: "/skill summarize [text]",
    })
  })

  return {
    releaseFirst,
    listRequestCount: () => listRequests,
  }
}

export async function forceSkillsConnectionState(
  page: Page,
  state: "connected" | "unreachable" = "connected",
) {
  await page.waitForFunction(
    () =>
      typeof (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
        ?.getState === "function",
    null,
    { timeout: 15_000 }
  )
  await page.evaluate((targetState) => {
    const store = (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
    if (!store) throw new Error("Connection store is unavailable")
    const prev = store.getState().state

    if (targetState === "unreachable") {
      // Keep the active poller from reconnecting this disposable test context.
      store.setState({ checkOnce: async () => {} })
      store.setState({
        state: {
          ...prev,
          phase: "error",
          isConnected: false,
          isChecking: false,
          errorKind: "unreachable",
        },
      })
      return
    }

    const now = Date.now()
    store.setState({
      state: {
        ...prev,
        phase: "connected",
        isConnected: true,
        isChecking: false,
        offlineBypass: true,
        errorKind: "none",
        lastError: null,
        lastStatusCode: null,
        lastCheckedAt: now,
        knowledgeStatus: "ready",
        knowledgeLastCheckedAt: now,
        knowledgeError: null,
        configStep: "health",
        hasCompletedFirstRun: true,
      },
    })
  }, state)
}
