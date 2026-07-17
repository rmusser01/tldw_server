import type { SkillListOrder, SkillListSort } from "@/types/skill"

export type SkillsContextFilter = "all" | "inline" | "fork"
export type SkillsVisibilityFilter = "visible" | "hidden" | "all"
export type SkillsToolsFilter = "any" | "with-tools" | "without-tools"
export type SkillsView = "library" | "trash"

export interface SkillsQueryState {
  view: SkillsView
  search: string
  context: SkillsContextFilter
  visibility: SkillsVisibilityFilter
  tools: SkillsToolsFilter
  model: string
  sort?: SkillListSort
  order?: SkillListOrder
  page: number
  pageSize: 10 | 20 | 50
}

export const DEFAULT_SKILLS_QUERY_STATE: SkillsQueryState = {
  view: "library",
  search: "",
  context: "all",
  visibility: "visible",
  tools: "any",
  model: "",
  page: 1,
  pageSize: 10
}

const isContext = (value: string | null): value is SkillsContextFilter =>
  value === "all" || value === "inline" || value === "fork"

const isVisibility = (value: string | null): value is SkillsVisibilityFilter =>
  value === "visible" || value === "hidden" || value === "all"

const isTools = (value: string | null): value is SkillsToolsFilter =>
  value === "any" || value === "with-tools" || value === "without-tools"

const isSort = (value: string | null): value is SkillListSort =>
  value === "name" || value === "context" || value === "created_at" || value === "last_modified"

const isOrder = (value: string | null): value is SkillListOrder =>
  value === "asc" || value === "desc"

const isView = (value: string | null): value is SkillsView =>
  value === "library" || value === "trash"

const parsePositiveInteger = (value: string | null, fallback: number): number => {
  if (!value || !/^\d+$/.test(value)) return fallback
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback
}

export const parseSkillsQueryState = (params: URLSearchParams): SkillsQueryState => {
  const view = params.get("view")
  const context = params.get("mode")
  const visibility = params.get("visibility")
  const tools = params.get("tools")
  const sort = params.get("sort")
  const order = params.get("order")
  const hasSortPair = isSort(sort) && isOrder(order)
  const parsedPageSize = parsePositiveInteger(
    params.get("pageSize"),
    DEFAULT_SKILLS_QUERY_STATE.pageSize
  )

  return {
    view: isView(view) ? view : DEFAULT_SKILLS_QUERY_STATE.view,
    search: params.get("q")?.trim() ?? "",
    context: isContext(context) ? context : DEFAULT_SKILLS_QUERY_STATE.context,
    visibility: isVisibility(visibility)
      ? visibility
      : DEFAULT_SKILLS_QUERY_STATE.visibility,
    tools: isTools(tools) ? tools : DEFAULT_SKILLS_QUERY_STATE.tools,
    model: params.get("model")?.trim() ?? "",
    ...(hasSortPair ? { sort, order } : {}),
    page: parsePositiveInteger(params.get("page"), DEFAULT_SKILLS_QUERY_STATE.page),
    pageSize: parsedPageSize === 20 || parsedPageSize === 50 ? parsedPageSize : 10
  }
}

export const serializeSkillsQueryState = (state: SkillsQueryState): URLSearchParams => {
  const params = new URLSearchParams()
  const search = state.search.trim()
  const model = state.model.trim()

  if (state.view !== DEFAULT_SKILLS_QUERY_STATE.view) params.set("view", state.view)
  if (search) params.set("q", search)
  if (state.context !== DEFAULT_SKILLS_QUERY_STATE.context) params.set("mode", state.context)
  if (state.visibility !== DEFAULT_SKILLS_QUERY_STATE.visibility) {
    params.set("visibility", state.visibility)
  }
  if (state.tools !== DEFAULT_SKILLS_QUERY_STATE.tools) params.set("tools", state.tools)
  if (model) params.set("model", model)
  if (state.sort && state.order) {
    params.set("sort", state.sort)
    params.set("order", state.order)
  }
  if (state.page !== DEFAULT_SKILLS_QUERY_STATE.page) params.set("page", String(state.page))
  if (state.pageSize !== DEFAULT_SKILLS_QUERY_STATE.pageSize) {
    params.set("pageSize", String(state.pageSize))
  }
  return params
}
