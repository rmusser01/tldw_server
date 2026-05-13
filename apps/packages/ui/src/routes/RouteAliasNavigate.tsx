import { Navigate, useLocation } from "react-router-dom"

type RouteAliasNavigateProps = {
  to: string
}

export const RouteAliasNavigate = ({ to }: RouteAliasNavigateProps) => {
  const location = useLocation()

  return (
    <Navigate
      to={{
        pathname: to,
        search: location.search,
        hash: location.hash
      }}
      replace
    />
  )
}
