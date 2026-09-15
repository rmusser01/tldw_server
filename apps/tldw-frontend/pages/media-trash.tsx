import dynamic from "next/dynamic"
import Head from "next/head"

const MediaTrash = dynamic(() => import("@/routes/option-media-trash"), { ssr: false })

export default function MediaTrashPage() {
  return <><Head><title>Trash | tldw</title></Head><MediaTrash /></>
}
