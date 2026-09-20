import React from "react"
import {
  RouterProvider,
  createHashRouter,
  createMemoryRouter
} from "react-router-dom"

export { RouteLeavePrompt, type RouteLeavePromptProps } from "./route-leave-prompt"

const dataRouterFutureConfig = {
  v7_relativeSplatPath: true
}

const providerFutureConfig = { v7_startTransition: true }
const RouterChildrenContext = React.createContext<React.ReactNode>(null)

const RouterChildrenHost = () => <>{React.useContext(RouterChildrenContext)}</>

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
