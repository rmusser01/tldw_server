import React, { Dispatch, SetStateAction, createContext } from "react"

// Note: messages state has been moved to useStoreMessageOption (Zustand store)
// This context now only handles abort controllers for streaming requests
interface PageAssistContext {
  controller: AbortController | null
  setController: Dispatch<SetStateAction<AbortController | null>>

  embeddingController: AbortController | null
  setEmbeddingController: Dispatch<SetStateAction<AbortController | null>>
}

export const PageAssistContext = createContext<PageAssistContext>({
  controller: null,
  setController: () => {},

  embeddingController: null,
  setEmbeddingController: () => {}
})

export const usePageAssist = () => {
  const context = React.useContext(PageAssistContext)
  if (!context) {
    throw new Error("usePageAssist must be used within a PageAssistContext")
  }
  return context
}
