import type { Page, Route } from "@playwright/test"

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

export async function mockSkillsBeginnerApi(
  page: Page,
  options: { seeded?: boolean } = {}
) {
  let seeded = Boolean(options.seeded)

  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      openapi: "3.0.0",
      info: { title: "Mock tldw API", version: "test" },
      paths: {
        "/api/v1/skills": { get: {}, post: {} },
        "/api/v1/skills/context": { get: {} },
        "/api/v1/skills/seed": { post: {} },
        "/api/v1/skills/{name}": { get: {}, put: {}, delete: {} },
        "/api/v1/skills/{name}/execute": { post: {} },
      },
    })
  })

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
      const body = route.request().postDataJSON() as { args?: string; dry_run?: boolean }
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
