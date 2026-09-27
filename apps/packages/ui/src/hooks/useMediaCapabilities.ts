import { useEffect, useState } from "react"
import { bgRequest } from "@/services/background-proxy"

/** Keep permission-based affordances scoped to the active authenticated session. */
export function useMediaCapabilities() {
  const [state, setState] = useState({ canDelete: false, loading: true })
  useEffect(() => {
    let requestId = 0
    let disposed = false
    const refresh = async () => {
      const currentRequest = ++requestId
      setState({ canDelete: false, loading: true })
      try {
        const result = await bgRequest<{ can_delete: boolean }>({ path: "/api/v1/media/capabilities", method: "GET" })
        if (!disposed && currentRequest === requestId) {
          setState({ canDelete: result?.can_delete === true, loading: false })
        }
      } catch {
        if (!disposed && currentRequest === requestId) setState({ canDelete: false, loading: false })
      }
    }
    void refresh()
    window.addEventListener("tldw:auth-principal-changed", refresh)
    window.addEventListener("tldw:config-updated", refresh)
    return () => {
      disposed = true
      window.removeEventListener("tldw:auth-principal-changed", refresh)
      window.removeEventListener("tldw:config-updated", refresh)
    }
  }, [])
  return state
}
