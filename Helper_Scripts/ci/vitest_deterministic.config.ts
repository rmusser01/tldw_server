import { relative, sep } from "node:path"

import { mergeConfig } from "vitest/config"
import { BaseSequencer, type TestSpecification } from "vitest/node"

import baseConfig from "./vitest.config"

class DeterministicSequencer extends BaseSequencer {
  private key(specification: TestSpecification): string {
    return relative(this.ctx.config.root, specification.moduleId)
      .split(sep)
      .join("/")
  }

  async sort(files: TestSpecification[]): Promise<TestSpecification[]> {
    return [...files].sort((leftFile, rightFile) => {
      const left = this.key(leftFile)
      const right = this.key(rightFile)
      return left < right ? -1 : left > right ? 1 : 0
    })
  }
}

export default mergeConfig(baseConfig, {
  test: {
    sequence: {
      sequencer: DeterministicSequencer
    }
  }
})
