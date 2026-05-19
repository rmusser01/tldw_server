import React from "react"
import { HashRouter, MemoryRouter } from "react-router-dom"

const routerFutureConfig = {
  v7_startTransition: true,
  v7_relativeSplatPath: true
}

export const HashRouterWithFuture: React.FC<{ children: React.ReactNode }> = ({
  children
}) => <HashRouter future={routerFutureConfig}>{children}</HashRouter>

export const MemoryRouterWithFuture: React.FC<{
  children: React.ReactNode
}> = ({ children }) => (
  <MemoryRouter future={routerFutureConfig}>{children}</MemoryRouter>
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
    <MemoryRouter initialEntries={initialEntries} future={routerFutureConfig}>
      {children}
    </MemoryRouter>
  )
}

/** @deprecated Use HashAwareMemoryRouter instead. */
export const SidepanelMemoryRouter = HashAwareMemoryRouter

export const resolveRouter = (mode: "hash" | "memory") =>
  mode === "hash" ? HashRouterWithFuture : HashAwareMemoryRouter
