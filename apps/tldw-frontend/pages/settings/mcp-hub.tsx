import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-settings-mcp-hub"), { ssr: false })
