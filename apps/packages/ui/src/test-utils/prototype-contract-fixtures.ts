import { readFileSync } from "node:fs"
import path from "node:path"

export interface PrototypeContractErrorDetail {
  category: string
  message: string
  frontend_state: string
  retryable: boolean
}

export interface PrototypeContractStateFixture {
  state: string
  httpStatus: number
  stableErrorCategory: string
  frontendStateBucket: string
  retryable: boolean
  mockResponse: {
    detail: PrototypeContractErrorDetail
  }
}

const contractFixturePath = path.resolve(
  __dirname,
  "../../../../tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json"
)

const contractFixture = JSON.parse(
  readFileSync(contractFixturePath, "utf8")
) as {
  states: PrototypeContractStateFixture[]
}

export const getPrototypeContractState = (
  state: string
): PrototypeContractStateFixture => {
  const fixture = contractFixture.states.find((item) => item.state === state)
  if (!fixture) {
    throw new Error(`Unknown prototype contract fixture state: ${state}`)
  }
  return fixture
}

export const getPrototypeContractErrorDetail = (
  state: string
): PrototypeContractErrorDetail =>
  getPrototypeContractState(state).mockResponse.detail
