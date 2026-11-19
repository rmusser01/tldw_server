import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render as renderView, type RenderOptions } from "@testing-library/react"

export const render = (ui: React.ReactNode, options?: RenderOptions) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const Wrapper = options?.wrapper
  return renderView(ui, {
    ...options,
    wrapper: ({ children }) => (
      <QueryClientProvider client={client}>
        {Wrapper ? <Wrapper>{children}</Wrapper> : children}
      </QueryClientProvider>
    )
  })
}
