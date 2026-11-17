export const isMenuPointerReady = async (item: Element): Promise<boolean> => {
  const before = item.getBoundingClientRect()
  await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  for (let ancestor: Element | null = item; ancestor; ancestor = ancestor.parentElement) {
    if (ancestor.getAnimations().some(animation => animation.playState === "running" || animation.pending)) return false
  }
  const rect = item.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0 || getComputedStyle(item).pointerEvents === "none") return false
  if (before.x !== rect.x || before.y !== rect.y || before.width !== rect.width || before.height !== rect.height) return false
  const target = item.ownerDocument.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)
  return target !== null && item.contains(target)
}
