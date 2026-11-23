/**
 * Bring an element to the top of its nearest scrolling ancestor only, without
 * scrolling the page or outer containers (CC-10 review). The composer's
 * Search & Context box is capped and scrolls inside itself, so after a search
 * its results header is scrolled up to the top of that box.
 */
const isScroller = (element: HTMLElement) => {
  const overflowY = window.getComputedStyle(element).overflowY
  return (
    (overflowY === "auto" || overflowY === "scroll") &&
    element.scrollHeight > element.clientHeight
  )
}

export const scrollToTopWithin = (element: HTMLElement, gap = 8) => {
  let scroller = element.parentElement
  while (scroller && !isScroller(scroller)) scroller = scroller.parentElement
  if (!scroller) return
  const offset =
    element.getBoundingClientRect().top - scroller.getBoundingClientRect().top
  scroller.scrollTop += offset - gap
}
