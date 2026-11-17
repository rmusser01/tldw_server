import { act } from "@testing-library/react"
import { createRequire } from "node:module"

const localRequire = createRequire(import.meta.url)
const antDRequire = createRequire(localRequire.resolve("antd"))
type Container = Element | DocumentFragment
type Renderer = {
  render: (node: unknown, container: Container) => void
  unmount: (container: Container) => Promise<void>
}
type NoticeModule = { actDestroy: () => void }

// AntD's native CJS calls bypass Vitest mocks; wrap its tiny real renderer, not its UI barrel.
const renderer = antDRequire("@rc-component/util/lib/React/render") as Renderer
const originalRender = renderer.render
const originalUnmount = renderer.unmount
const containers = new Set<Container>()

renderer.render = (node, container) => {
  containers.add(container)
  originalRender(node, container)
}
renderer.unmount = async (container) => {
  await originalUnmount(container)
  containers.delete(container)
}

const cachedModule = <T>(path: string): T | undefined =>
  antDRequire.cache[antDRequire.resolve(path)]?.exports as T | undefined

export const cleanupStaticRoots = async (): Promise<void> => {
  const queuedModals = cachedModule<{ default: unknown[] }>("./modal/destroyFns")
  if (!containers.size && !queuedModals?.default.length) return

  await act(async () => {
    cachedModule<{ default: { destroyAll: () => void } }>("./modal")?.default.destroyAll()
    // Modal.info/confirm schedules its separate root on a zero-delay timer.
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
  })
  await act(async () => {
    for (const container of containers) await renderer.unmount(container)
  })
  // Unmounted singleton holders must not receive notices from the next test.
  cachedModule<NoticeModule>("./message")?.actDestroy()
  cachedModule<NoticeModule>("./notification")?.actDestroy()
}

export const restoreStaticRootRenderer = (): void => {
  renderer.render = originalRender
  renderer.unmount = originalUnmount
}
