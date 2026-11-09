import { useMemo } from "react"
import {
  quickIngestAuthority,
  useQuickIngestAuthority
} from "@/services/tldw/quick-ingest-authority"

/** Reuse verified account ownership without binding documents to an ingest draft. */
export function useDocumentOperation() {
  const authority = useQuickIngestAuthority()
  return useMemo(() => {
    if (!authority) return null
    try {
      return quickIngestAuthority.capture({ sessionBound: false })
    } catch {
      return null
    }
  }, [authority])
}
