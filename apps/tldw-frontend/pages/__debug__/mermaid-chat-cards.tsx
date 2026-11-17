import dynamic from "next/dynamic"

export default dynamic(() => import("@web/components/debug/MermaidChatCardsHarness"), {
  ssr: false
})
