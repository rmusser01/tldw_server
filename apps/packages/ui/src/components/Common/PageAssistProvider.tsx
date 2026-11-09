import { PageAssistContext } from "@/context"
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
    let active = true
    let bridge: typeof w.__tldw_pageAssist
    void import("@/store/option").then(({ useStoreMessageOption }) => {
      if (!active) return
      bridge = {
        setMessages: useStoreMessageOption.getState().setMessages,
        getMessages: () => useStoreMessageOption.getState().messages
      }
      w.__tldw_pageAssist = bridge
    }).catch((error) => {
      console.warn("Unable to load development Chat helpers", error)
    })

    return () => {
      active = false
      if (w.__tldw_pageAssist === bridge) delete w.__tldw_pageAssist
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
