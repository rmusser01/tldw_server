import dynamic from "next/dynamic"
import Head from "next/head"

const Media = dynamic(() => import("@/routes/option-media"), { ssr: false })

export default function MediaPage() {
  return <><Head><title>Media | tldw</title></Head><Media /></>
}
