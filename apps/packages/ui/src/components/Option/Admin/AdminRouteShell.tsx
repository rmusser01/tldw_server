import React from "react"
import {
  ADMIN_MODULES,
  adminModuleForRoute,
  isAdminRoute
} from "./admin-modules"

/**
 * Chrome shared by every admin page (2026-09 UX audit findings S1/S5):
 * a skip link, a nav landmark linking all admin modules, and a document
 * title per module. Rendered by the host app around admin route content.
 *
 * Kept dependency-free (React only) so host apps can mount it outside the
 * shared component tree without pulling browser-specific modules into SSR.
 */
export const AdminRouteShell: React.FC<{
  path: string
  children: React.ReactNode
}> = ({ path, children }) => {
  const current = adminModuleForRoute(path)
  const onOverview = isAdminRoute(path) && !current

  React.useEffect(() => {
    if (typeof document === "undefined") return
    const previous = document.title
    document.title = current
      ? `${current.label} · Admin · tldw`
      : "Admin · tldw"
    return () => {
      document.title = previous
    }
  }, [current])

  return (
    <div className="flex min-h-full flex-col">
      <a
        href="#admin-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:text-text focus:shadow"
      >
        Skip to admin content
      </a>
      <nav
        aria-label="Admin modules"
        className="border-b border-border bg-surface px-4 py-2"
      >
        <div className="flex items-center gap-1 overflow-x-auto whitespace-nowrap text-sm">
          <a
            href="/admin"
            aria-current={onOverview ? "page" : undefined}
            className={
              onOverview
                ? "rounded-md bg-surface2 px-2.5 py-1 font-semibold text-text"
                : "rounded-md px-2.5 py-1 font-medium text-text-muted hover:bg-surface2 hover:text-text"
            }
          >
            Admin
          </a>
          <span aria-hidden="true" className="px-1 text-text-muted">
            /
          </span>
          {ADMIN_MODULES.map((module) => {
            const isCurrent = module.route === current?.route
            return (
              <a
                key={module.route}
                href={module.route}
                aria-current={isCurrent ? "page" : undefined}
                title={module.description}
                className={
                  isCurrent
                    ? "rounded-md bg-surface2 px-2.5 py-1 font-semibold text-text"
                    : "rounded-md px-2.5 py-1 text-text-muted hover:bg-surface2 hover:text-text"
                }
              >
                {module.label}
              </a>
            )
          })}
        </div>
      </nav>
      {/* tabIndex={-1} makes the skip-link target programmatically focusable,
          so activating the link actually moves keyboard focus past the nav. */}
      <div id="admin-content" tabIndex={-1} className="min-h-0 flex-1 outline-none">
        {children}
      </div>
    </div>
  )
}

export default AdminRouteShell
