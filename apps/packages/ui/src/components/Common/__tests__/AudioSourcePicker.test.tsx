// @vitest-environment jsdom

import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AudioSourcePicker } from "../AudioSourcePicker";
import { SSTSettings } from "@/components/Option/Settings/SSTSettings";

const { storageValues, useStorageMock } = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  useStorageMock: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key,
  }),
}));

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>();

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key);

        return {
          ...state,
          label:
            key === "unavailable" ? "Unavailable via registry" : state.label,
        };
      },
    ),
  };
});

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: useStorageMock,
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: vi.fn(async () => ({ all_models: ["whisper-1"] })),
    getTranscriptionModelHealth: vi.fn(async () => ({ status: "ok" })),
  },
}));

vi.mock("antd", () => ({
  Badge: ({
    children,
    count,
  }: {
    children: React.ReactNode;
    count?: React.ReactNode;
  }) => (
    <div>
      {children}
      {count ? <span>{count}</span> : null}
    </div>
  ),
  Button: ({ children, loading: _loading, ...props }: any) => (
    <button {...props}>{children}</button>
  ),
  Collapse: ({
    items,
  }: {
    items: Array<{
      key: string;
      label: React.ReactNode;
      children: React.ReactNode;
    }>;
  }) => (
    <div>
      {items.map((item) => (
        <section key={item.key}>
          <h2>{item.label}</h2>
          <div>{item.children}</div>
        </section>
      ))}
    </div>
  ),
  Input: (props: any) => <input {...props} />,
  InputNumber: ({ value, onChange, ...props }: any) => (
    <input
      {...props}
      type="number"
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.target.value))}
    />
  ),
  Select: ({
    value,
    onChange,
    options,
    placeholder,
    disabled,
    ...props
  }: any) => (
    <label>
      <span>{props["aria-label"] || placeholder || "select"}</span>
      <select
        aria-label={props["aria-label"] || placeholder || "select"}
        disabled={disabled}
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value)}
      >
        {(options ?? []).map((option: any) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  ),
  Switch: ({ checked, onChange }: any) => (
    <input
      type="checkbox"
      role="switch"
      checked={Boolean(checked)}
      onChange={(event) => onChange?.(event.target.checked)}
    />
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("AudioSourcePicker", () => {
  beforeEach(() => {
    storageValues.clear();
    useStorageMock.mockReset();
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      (nextValue: unknown) => storageValues.set(key, nextValue),
      { isLoading: false },
    ]);
  });

  it("renders default mic plus enumerated devices", () => {
    render(
      <AudioSourcePicker
        requestedSourceKind="default_mic"
        resolvedSourceKind="default_mic"
        devices={[
          { deviceId: "default", label: "Default microphone" },
          { deviceId: "usb-1", label: "USB microphone" },
        ]}
      />,
    );

    expect(screen.getByText("Default microphone")).toBeInTheDocument();
    expect(screen.getByText("USB microphone")).toBeInTheDocument();
  });

  it("calls onChange with explicit device details when a device is selected", () => {
    const onChange = vi.fn();

    render(
      <AudioSourcePicker
        requestedSourceKind="default_mic"
        resolvedSourceKind="default_mic"
        devices={[{ deviceId: "usb-1", label: "USB microphone" }]}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Audio input source"), {
      target: { value: "mic_device:usb-1" },
    });

    expect(onChange).toHaveBeenCalledWith({
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone",
    });
  });

  it("calls onChange with default mic when reset to the default option", () => {
    const onChange = vi.fn();

    render(
      <AudioSourcePicker
        requestedSourceKind="mic_device"
        requestedDeviceId="usb-1"
        resolvedSourceKind="mic_device"
        devices={[{ deviceId: "usb-1", label: "USB microphone" }]}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Audio input source"), {
      target: { value: "default_mic" },
    });

    expect(onChange).toHaveBeenCalledWith({
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null,
    });
  });

  it("renders unavailable remembered devices and shows the fallback message", () => {
    render(
      <AudioSourcePicker
        requestedSourceKind="mic_device"
        requestedDeviceId="usb-missing"
        lastKnownLabel="Studio microphone"
        resolvedSourceKind="default_mic"
        devices={[{ deviceId: "usb-1", label: "USB microphone" }]}
      />,
    );

    expect(
      screen.getByText("Studio microphone (Unavailable via registry)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Source fallback active")).toBeInTheDocument();
  });

  it("disables the select when disabled is true", () => {
    render(
      <AudioSourcePicker
        requestedSourceKind="default_mic"
        resolvedSourceKind="default_mic"
        devices={[{ deviceId: "usb-1", label: "USB microphone" }]}
        disabled
      />,
    );

    expect(screen.getByLabelText("Audio input source")).toBeDisabled();
  });

  it("shows a speech input source management section in settings", async () => {
    render(<SSTSettings />);

    expect(
      await screen.findByText("Audio input source preferences"),
    ).toBeInTheDocument();
    expect(screen.getByText("Dictation")).toBeInTheDocument();
    expect(screen.getByText("Live voice")).toBeInTheDocument();
    expect(screen.getByText("Speech Playground")).toBeInTheDocument();
  });
});
