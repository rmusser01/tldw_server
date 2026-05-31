export interface PostBuildTask {
  args: string[]
  script: string
}

export interface RunPostBuildTasksOptions {
  cwd?: string
  outDir: string
  targetName: string
}

export function getPostBuildTasks(
  targetName: string,
  outDir: string
): PostBuildTask[]

export function runPostBuildTasks(options: RunPostBuildTasksOptions): void

export function getWxtTargetName(
  browser: string,
  manifestVersion: string | number
): string
