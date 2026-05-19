import { Select } from "antd";
import { useTranslation } from "react-i18next";
import { getDesignSystemState } from "@/design-system";
import type { AudioSourceKind } from "@/audio";
import type { AudioInputDeviceOption } from "@/hooks/useAudioSourceCatalog";

export type AudioSourcePickerValue = {
  sourceKind: AudioSourceKind;
  deviceId?: string | null;
  lastKnownLabel?: string | null;
};

type AudioSourcePickerProps = {
  requestedSourceKind: AudioSourceKind;
  resolvedSourceKind: AudioSourceKind;
  requestedDeviceId?: string | null;
  lastKnownLabel?: string | null;
  devices: AudioInputDeviceOption[];
  onChange?: (nextValue: AudioSourcePickerValue) => void;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
};

const DEFAULT_MIC_VALUE = "default_mic";
const UNAVAILABLE_STATE_LABEL = getDesignSystemState("unavailable").label;

export function AudioSourcePicker({
  requestedSourceKind,
  resolvedSourceKind,
  requestedDeviceId,
  lastKnownLabel,
  devices,
  onChange,
  disabled,
  className,
  ariaLabel,
}: AudioSourcePickerProps) {
  const { t } = useTranslation(["settings", "playground"]);
  const selectedValue =
    requestedSourceKind === "mic_device" &&
    requestedDeviceId &&
    requestedDeviceId !== "default"
      ? `mic_device:${requestedDeviceId}`
      : DEFAULT_MIC_VALUE;

  const options = [
    {
      label: t("audioSourcePicker.defaultMic", "Default microphone"),
      value: DEFAULT_MIC_VALUE,
    },
    ...devices
      .filter((device) => device.deviceId !== "default")
      .map((device) => ({
        label: device.label,
        value: `mic_device:${device.deviceId}`,
      })),
  ];

  const requestedDeviceMissing =
    requestedSourceKind === "mic_device" &&
    Boolean(requestedDeviceId) &&
    requestedDeviceId !== "default" &&
    !devices.some((device) => device.deviceId === requestedDeviceId);

  if (requestedDeviceMissing && requestedDeviceId) {
    options.splice(1, 0, {
      label: `${lastKnownLabel || requestedDeviceId} (${t("audioSourcePicker.unavailable", UNAVAILABLE_STATE_LABEL)})`,
      value: `mic_device:${requestedDeviceId}`,
    });
  }

  return (
    <div className="flex flex-col gap-2">
      <Select
        aria-label={
          ariaLabel || t("audioSourcePicker.selectLabel", "Audio input source")
        }
        className={className}
        value={selectedValue}
        onChange={(value: string) => {
          if (!onChange) {
            return;
          }

          if (value === DEFAULT_MIC_VALUE) {
            onChange({
              sourceKind: "default_mic",
              deviceId: null,
              lastKnownLabel: null,
            });
            return;
          }

          const deviceId = value.replace(/^mic_device:/, "");
          const matchingDevice = devices.find(
            (device) => device.deviceId === deviceId,
          );
          onChange({
            sourceKind: "mic_device",
            deviceId,
            lastKnownLabel: matchingDevice?.label ?? lastKnownLabel ?? null,
          });
        }}
        options={options}
        disabled={disabled}
      />
      {requestedSourceKind !== resolvedSourceKind ? (
        <span className="text-xs text-text-subtle">
          {t(
            "audioSourcePicker.sourceFallbackActive",
            "Source fallback active",
          )}
        </span>
      ) : null}
    </div>
  );
}
