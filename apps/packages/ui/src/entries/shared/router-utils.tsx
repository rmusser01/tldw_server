import React from "react"
import {
  RouterProvider,
  UNSAFE_DataRouterContext,
  createHashRouter,
  createMemoryRouter,
  unstable_usePrompt,
  useBlocker
} from "react-router-dom"

const dataRouterFutureConfig = {
  v7_relativeSplatPath: true
}

const providerFutureConfig = { v7_startTransition: true }
const RouterChildrenContext = React.createContext<React.ReactNode>(null)

const RouterChildrenHost = () => <>{React.useContext(RouterChildrenContext)}</>

type RouteLeavePromptProps = {
  when: boolean
  message: string
}

const ShimRouteLeavePrompt: React.FC<RouteLeavePromptProps> = (props) => {
  unstable_usePrompt(props)
  return null
}

export const RouteLeavePrompt: React.FC<RouteLeavePromptProps> = ({
  when,
  message
}) => {
  const dataRouterContext = React.useContext(UNSAFE_DataRouterContext)
  const blocker = useBlocker(Boolean(dataRouterContext) && when)
  const blockerRef = React.useRef(blocker)
  const proceedTimerRef = React.useRef<number | null>(null)
  blockerRef.current = blocker

  const cancelProceed = React.useCallback(() => {
    if (proceedTimerRef.current !== null) {
      window.clearTimeout(proceedTimerRef.current)
      proceedTimerRef.current = null
    }
  }, [])

  React.useEffect(() => {
    cancelProceed()
    if (blocker.state !== "blocked") return
    if (!when || !window.confirm(message)) {
      try {
        blocker.reset?.()
      } catch {
        // A stale router blocker is already safely unblocked.
      }
      return
    }
    const ownedBlocker = blocker
    const timer = window.setTimeout(() => {
      if (proceedTimerRef.current !== timer) return
      proceedTimerRef.current = null
      if (blockerRef.current !== ownedBlocker || blockerRef.current.state !== "blocked") return
      try {
        ownedBlocker.proceed?.()
      } catch {
        // Cleanup or a newer navigation already retired this blocker.
      }
    }, 0)
    proceedTimerRef.current = timer
    return cancelProceed
  }, [blocker, cancelProceed, message, when])

  React.useEffect(() => () => {
    cancelProceed()
    const current = blockerRef.current
    if (current.state === "blocked") {
      try {
        current.reset?.()
      } catch {
        // The owning router may already be disposed.
      }
    }
  }, [cancelProceed])

  return dataRouterContext ? null : <ShimRouteLeavePrompt when={when} message={message} />
}

const DataRouterWithChildren: React.FC<{
  children: React.ReactNode
  createRouter: () => ReturnType<typeof createMemoryRouter>
}> = ({ children, createRouter }) => {
  const createRouterRef = React.useRef(createRouter)
  createRouterRef.current = createRouter
  const [router, setRouter] = React.useState<ReturnType<typeof createMemoryRouter> | null>(null)
  React.useEffect(() => {
    const ownedRouter = createRouterRef.current()
    setRouter(ownedRouter)
    return () => {
      ownedRouter.dispose()
      setRouter((current) => current === ownedRouter ? null : current)
    }
  }, [])
  if (!router) return <div aria-hidden="true" />
  return (
    <RouterChildrenContext.Provider value={children}>
      <RouterProvider router={router} future={providerFutureConfig} />
    </RouterChildrenContext.Provider>
  )
}

const hostRoutes = [{ path: "*", element: <RouterChildrenHost /> }]

export const HashRouterWithFuture: React.FC<{ children: React.ReactNode }> = ({
  children
}) => (
  <DataRouterWithChildren
    createRouter={() => createHashRouter(hostRoutes, { future: dataRouterFutureConfig })}
  >
    {children}
  </DataRouterWithChildren>
)

export const MemoryRouterWithFuture: React.FC<{
  children: React.ReactNode
}> = ({ children }) => (
  <DataRouterWithChildren
    createRouter={() =>
      createMemoryRouter(hostRoutes, {
        initialEntries: ["/"],
        future: dataRouterFutureConfig
      })
    }
  >
    {children}
  </DataRouterWithChildren>
)

const resolveMemoryInitialEntry = () => {
  if (typeof window === "undefined") {
    return "/"
  }
  const rawHash = window.location.hash || ""
  const trimmed = rawHash.startsWith("#") ? rawHash.slice(1) : rawHash
  if (!trimmed || trimmed === "/") {
    return "/"
  }
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`
}

/** MemoryRouter that seeds its initial route from window.location.hash (for deep links). */
export const HashAwareMemoryRouter: React.FC<{
  children: React.ReactNode
}> = ({ children }) => {
  const initialEntries = React.useMemo(() => [resolveMemoryInitialEntry()], [])
  return (
    <DataRouterWithChildren
      createRouter={() =>
        createMemoryRouter(hostRoutes, {
          initialEntries,
          future: dataRouterFutureConfig
        })
      }
    >
      {children}
    </DataRouterWithChildren>
  )
}

/** @deprecated Use HashAwareMemoryRouter instead. */
export const SidepanelMemoryRouter = HashAwareMemoryRouter

export const resolveRouter = (mode: "hash" | "memory") =>
  mode === "hash" ? HashRouterWithFuture : HashAwareMemoryRouter
