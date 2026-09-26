import React from "react"
import Head from "next/head"

/** Web metadata can render on the server while the shared route loads in the browser. */
export function withPageTitle(title: string, Route: React.ComponentType) {
  return function TitledRoute() {
    return <><Head><title>{title} | tldw</title></Head><Route /></>
  }
}
