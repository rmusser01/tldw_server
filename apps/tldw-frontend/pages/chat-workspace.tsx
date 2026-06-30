import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-chat-workspace"), {
  ssr: false
})
