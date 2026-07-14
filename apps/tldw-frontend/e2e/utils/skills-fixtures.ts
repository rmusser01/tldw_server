import type { Page, Route } from "@playwright/test"

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

type TldwConnectionStore = {
  getState: () => { state: Record<string, unknown> }
  setState: (value: { state: Record<string, unknown> }) => void
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

async function mockSkillsCapabilityRoutes(page: Page) {
  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
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
  options: { seeded?: boolean } = {}
) {
  let seeded = Boolean(options.seeded)
  const executeRequests: Array<{ args?: string; dry_run?: boolean }> = []

  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() !== "GET") {
      await fulfillJson(route, {}, 405)
      return
    }
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
    await fulfillJson(route, {
      available_skills: seeded ? [seededSkillSummary] : [],
      context_text: seeded ? "/skill summarize [text]" : "",
    })
  })

  await page.route(/\/api\/v1\/skills\/seed(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() !== "POST") {
      await fulfillJson(route, {}, 405)
      return
    }
    seeded = true
    await fulfillJson(route, { seeded: ["summarize"], count: 1 })
  })

  await page.route(/\/api\/v1\/skills\/summarize(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, seededSkillResponse)
  })

  await page.route(
    /\/api\/v1\/skills\/summarize\/execute(?:\/)?(?:\?.*)?$/,
    async (route) => {
      if (route.request().method() !== "POST") {
        await fulfillJson(route, {}, 405)
        return
      }
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

  return { executeRequests }
}

export async function mockSkillsTrashWorkflow(page: Page) {
  let state: "active" | "trash" | "purged" = "active"
  let version = 1
  const operations: string[] = []

  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
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
    if (route.request().method() !== "GET") {
      await fulfillJson(route, {}, 405)
      return
    }
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
      if (route.request().method() !== "POST") {
        await fulfillJson(route, {}, 405)
        return
      }
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
      if (route.request().method() !== "DELETE") {
        await fulfillJson(route, {}, 405)
        return
      }
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
    if (route.request().method() !== "GET") {
      await fulfillJson(route, {}, 405)
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

export async function mockPowerUserSkillsLibrary(page: Page) {
  const listUrls: URL[] = []
  const deleteRequests: unknown[] = []

  await mockSkillsCapabilityRoutes(page)

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() !== "GET") {
      await fulfillJson(route, {}, 405)
      return
    }
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
    await fulfillJson(route, {
      available_skills: largeLibrarySkills,
      context_text: "/skill target-research-formatter [source]",
    })
  })

  await page.route(/\/api\/v1\/skills\/([^/?]+)(?:\/)?(?:\?.*)?$/, async (route) => {
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

  return {
    deleteRequests,
    lastListUrl: () => listUrls.at(-1),
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

export async function forceSkillsConnectionState(page: Page) {
  await page.waitForFunction(
    () =>
      typeof (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
        ?.getState === "function",
    null,
    { timeout: 15_000 }
  )
  await page.evaluate(() => {
    const store = (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
    if (!store) throw new Error("Connection store is unavailable")
    const prev = store.getState().state
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
  })
}
