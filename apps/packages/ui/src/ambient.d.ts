// Ambient declarations for shared UI code when compiled outside the extension build.

declare const browser: any

declare function defineBackground<T>(definition: T): T
declare function defineContentScript<T>(definition: T): T

declare namespace chrome {
  namespace storage {
    type StorageArea = any
  }
  namespace runtime {
    type Port = any
    type MessageSender = any
  }
  namespace tabs {
    type Tab = any
  }
  const storage: any
  const runtime: any
  const tabs: any
  const alarms: any
  const action: any
  const i18n: any
  const scripting: any
  const permissions: any
  const notifications: any
  const tts: any
  const sidePanel: any
  namespace sidePanel {
    type OpenOptions = any
  }
}

declare var chrome: typeof chrome

interface ImportMetaEnv {
  DEV?: boolean
  MODE?: string
  PROD?: boolean
  VITE_TLDW_DOCS_URL?: string
  VITE_TLDW_API_KEY?: string
  BROWSER?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
  readonly main?: boolean
  glob?: {
    <T = unknown>(
      pattern: string,
      options?: {
        query?: string
        import?: string
        eager?: false
      }
    ): Record<string, () => Promise<T>>
    <T = unknown>(
      pattern: string,
      options: {
        query?: string
        import?: string
        eager: true
      }
    ): Record<string, T>
  }
}

declare module "*.png" {
  const src: string
  export default src
}

declare module "pa-tesseract.js" {
  export const createWorker: any
}

declare module "rehype-highlight" {
  const rehypeHighlight: any
  export default rehypeHighlight
}

declare module "xterm/css/xterm.css" {
  const css: string
  export default css
}

declare module "exceljs" {
  export class Workbook {
    addWorksheet(name: string): any
    xlsx: {
      writeBuffer(): Promise<import("node:buffer").Buffer>
    }
  }
}

declare module "xterm" {
  export interface IDisposable {
    dispose(): void
  }

  export interface IEvent<T, U = void> {
    (
      listener: (arg1: T, arg2: U) => any,
      thisArg?: any,
      disposables?: IDisposable[]
    ): IDisposable
  }

  export interface ITerminal extends IDisposable {
    open(element: HTMLElement): void
    loadAddon(addon: ITerminalAddon): void
    write(data: string | Uint8Array, callback?: () => void): void
    focus(): void
    onResize: IEvent<{ cols: number; rows: number }>
    onData: IEvent<string>
  }

  export interface ITerminalAddon extends IDisposable {
    activate(terminal: ITerminal): void
  }

  export class Terminal implements ITerminal {
    constructor(options?: any)
    open(element: HTMLElement): void
    loadAddon(addon: ITerminalAddon): void
    write(data: string | Uint8Array, callback?: () => void): void
    focus(): void
    onResize: IEvent<{ cols: number; rows: number }>
    onData: IEvent<string>
    dispose(): void
  }
}

declare module "@xterm/addon-fit" {
  import type { ITerminal, ITerminalAddon } from "xterm"

  export class FitAddon implements ITerminalAddon {
    constructor()
    activate(terminal: ITerminal): void
    fit(): void
    dispose(): void
  }
}
