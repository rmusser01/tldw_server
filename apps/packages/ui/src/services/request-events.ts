export const BACKEND_UNREACHABLE_EVENT = "tldw:backend-unreachable"

export type BackendUnreachableDetail = {
  method: string
  path: string
  status?: number
  code?: string
  message: string
  source: "background" | "direct"
  timestamp: number
}
