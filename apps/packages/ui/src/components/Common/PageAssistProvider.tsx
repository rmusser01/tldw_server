import { PageAssistContext } from "@/context"
import { useStoreMessageOption } from "@/store/option"
import React from "react"

// Note: messages state has been moved to useStoreMessageOption (Zustand store)
// This provider now only manages abort controllers for streaming requests
export const PageAssistProvider = ({
  children
}: {
  children: React.ReactNode
}) => {
  const [controller, setControllerState] =
    React.useState<AbortController | null>(null)
  const controllerRef = React.useRef<AbortController | null>(null)
  const setController = React.useCallback<
    React.Dispatch<React.SetStateAction<AbortController | null>>
  >((next) => {
    const value =
      typeof next === "function" ? next(controllerRef.current) : next
    controllerRef.current = value
    setControllerState(value)
  }, [])
  const getController = React.useCallback(() => controllerRef.current, [])
  const [embeddingController, setEmbeddingController] =
    React.useState<AbortController | null>(null)

  React.useEffect(() => {
    if (process.env.NODE_ENV === "production") return
    if (typeof window === "undefined") return

    const w = window as any
    w.__tldw_pageAssist = {
      setMessages: useStoreMessageOption.getState().setMessages,
      getMessages: () => useStoreMessageOption.getState().messages
    }

    return () => {
      delete w.__tldw_pageAssist
    }
  }, [])

  return (
    <PageAssistContext.Provider
      value={{
        controller,
        setController,
        getController,

        embeddingController,
        setEmbeddingController
      }}>
      {children}
    </PageAssistContext.Provider>
  )
}
