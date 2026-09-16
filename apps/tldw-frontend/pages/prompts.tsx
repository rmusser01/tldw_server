import dynamic from "next/dynamic"
import Head from "next/head"

const Prompts = dynamic(() => import("@/routes/option-prompts"), { ssr: false })

export default function PromptsPage() {
  return <><Head><title>Prompts | tldw</title></Head><Prompts /></>
}
