import React from "react"
import { useParams } from "react-router-dom"
import { PublicShare } from "@/components/Option/PublicShare"

const OptionPublicShare: React.FC = () => {
  const { token = "" } = useParams()
  return <PublicShare token={token} />
}

export default OptionPublicShare
