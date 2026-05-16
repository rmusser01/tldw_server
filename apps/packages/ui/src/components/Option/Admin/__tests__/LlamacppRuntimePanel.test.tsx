import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { LlamacppRuntimePanel } from "../LlamacppRuntimePanel"

const profiles = [
  {
    profile_id: "default",
    name: "Qwen fixed port",
    enabled: true,
    mode: "chat" as const,
    model_id: "gguf:qwen",
    model_path: "/models/qwen.gguf",
    host: "127.0.0.1",
    port: 8181,
    port_policy: "explicit" as const,
    server_args: {},
    autostart: false,
    restart_policy: {},
    tags: []
  },
  {
    profile_id: "vision",
    name: "Vision draft",
    enabled: false,
    mode: "vision" as const,
    model_id: "gguf:vision",
    model_path: "/models/vision.gguf",
    mmproj_model_id: "mmproj:vision",
    host: "127.0.0.1",
    port: 8182,
    port_policy: "explicit" as const,
    server_args: {},
    autostart: false,
    restart_policy: {},
    tags: ["mmproj"]
  }
]

const runtimes = [
  {
    profile_id: "default",
    state: "running" as const,
    pid: 4242,
    host: "127.0.0.1",
    port: 8181,
    endpoint: "http://127.0.0.1:8181",
    model_id: "gguf:qwen",
    model_path: "/models/qwen.gguf",
    resolved_args: [],
    restart_count: 1,
    warnings: [],
    health: { ready: true },
    log_tail_available: true
  },
  {
    profile_id: "vision",
    state: "stopped" as const,
    host: "127.0.0.1",
    port: 8182,
    endpoint: null,
    model_id: "gguf:vision",
    model_path: "/models/vision.gguf",
    resolved_args: [],
    restart_count: 0,
    warnings: ["mmproj pairing is saved but not loaded."],
    health: {},
    log_tail_available: false
  }
]

describe("LlamacppRuntimePanel", () => {
  it("renders multiple llama.cpp runtimes with independent actions", () => {
    const onStart = vi.fn()
    const onStop = vi.fn()
    const onPause = vi.fn()
    const onResume = vi.fn()
    const onUseInChat = vi.fn()

    render(
      <LlamacppRuntimePanel
        profiles={profiles}
        runtimes={runtimes}
        loading={false}
        actionProfileId={null}
        onRefresh={vi.fn()}
        onStart={onStart}
        onStop={onStop}
        onPause={onPause}
        onResume={onResume}
        onUseInChat={onUseInChat}
      />
    )

    expect(screen.getByText("Qwen fixed port")).toBeTruthy()
    expect(screen.getByText("Vision draft")).toBeTruthy()
    expect(screen.getByText("http://127.0.0.1:8181")).toBeTruthy()
    expect(screen.getByText("8182")).toBeTruthy()
    expect(screen.getByText("stopped")).toBeTruthy()
    expect(screen.getByText("mmproj pairing is saved but not loaded.")).toBeTruthy()

    fireEvent.click(screen.getByLabelText("Stop Qwen fixed port"))
    fireEvent.click(screen.getByLabelText("Start Vision draft"))
    fireEvent.click(screen.getByLabelText("Use Qwen fixed port in Chat"))

    expect(onStop).toHaveBeenCalledWith("default")
    expect(onStart).toHaveBeenCalledWith("vision")
    expect(onUseInChat).toHaveBeenCalledWith("default")
    expect(onPause).not.toHaveBeenCalled()
    expect(onResume).not.toHaveBeenCalled()
  })

  it("does not offer a duplicate start action while a runtime is starting", () => {
    const onStart = vi.fn()

    render(
      <LlamacppRuntimePanel
        profiles={profiles}
        runtimes={[
          {
            ...runtimes[1],
            state: "starting" as const
          }
        ]}
        loading={false}
        actionProfileId={null}
        onRefresh={vi.fn()}
        onStart={onStart}
        onStop={vi.fn()}
        onPause={vi.fn()}
        onResume={vi.fn()}
        onUseInChat={vi.fn()}
      />
    )

    expect(screen.queryByLabelText("Start Vision draft")).toBeNull()
    expect(screen.getByLabelText("Vision draft is starting")).toBeDisabled()
    expect(onStart).not.toHaveBeenCalled()
  })

  it("renders runtime load errors through the design-system alert primitive", () => {
    render(
      <LlamacppRuntimePanel
        profiles={profiles}
        runtimes={[]}
        loading={false}
        error="Runtime inventory failed to load."
        actionProfileId={null}
        onRefresh={vi.fn()}
        onStart={vi.fn()}
        onStop={vi.fn()}
        onPause={vi.fn()}
        onResume={vi.fn()}
        onUseInChat={vi.fn()}
      />
    )

    expect(
      screen.getByText("Runtime inventory failed to load.").closest(
        '[data-ds-component="Alert"]'
      )
    ).toHaveAttribute("role", "status")
  })
})
