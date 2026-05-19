import { useStorage } from "@plasmohq/storage/hook";
import {
  Alert,
  Collapse,
  Input,
  InputNumber,
  Select,
  Switch,
  Button,
  Tooltip,
} from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
import type { AudioFeatureGroup } from "@/audio";
import { AudioSourcePicker } from "@/components/Common/AudioSourcePicker";
import type { AudioInputDeviceOption } from "@/hooks/useAudioSourceCatalog";
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog";
import { useAudioSourcePreferences } from "@/hooks/useAudioSourcePreferences";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { SUPPORTED_LANGUAGES } from "~/utils/supported-languages";

const AudioSourcePreferenceRow = ({
  featureGroup,
  label,
  devices,
  hideBorder,
}: {
  featureGroup: AudioFeatureGroup;
  label: string;
  devices: AudioInputDeviceOption[];
  hideBorder?: boolean;
}) => {
  const { preference, setPreference } = useAudioSourcePreferences(featureGroup);

  return (
    <div className="flex flex-row justify-between items-start gap-4">
      <span className="text-text">{label}</span>
      <AudioSourcePicker
        ariaLabel={`${label} audio input source`}
        className={hideBorder ? "w-full" : "!min-w-[240px]"}
        requestedSourceKind={preference.sourceKind}
        resolvedSourceKind={preference.sourceKind}
        requestedDeviceId={preference.deviceId}
        lastKnownLabel={preference.lastKnownLabel}
        devices={devices}
        onChange={(nextValue) =>
          setPreference({
            featureGroup,
            sourceKind: nextValue.sourceKind,
            deviceId: nextValue.deviceId ?? null,
            lastKnownLabel: nextValue.lastKnownLabel ?? null,
          })
        }
      />
    </div>
  );
};

export const SSTSettings = ({ hideBorder }: { hideBorder?: boolean }) => {
  const { t } = useTranslation("settings");
  const [speechToTextLanguage, setSpeechToTextLanguage] = useStorage(
    "speechToTextLanguage",
    "en-US",
  );

  const [autoSubmitVoiceMessage, setAutoSubmitVoiceMessage] = useStorage(
    "autoSubmitVoiceMessage",
    false,
  );

  const [autoStopTimeout, setAutoStopTimeout] = useStorage(
    "autoStopTimeout",
    2000,
  );
  const resolvedAutoStopTimeout =
    typeof autoStopTimeout === "number" ? autoStopTimeout : 2000;

  const [sttModel, setSttModel] = useStorage("sttModel", "whisper-1");
  const [sttUseSegmentation, setSttUseSegmentation] = useStorage(
    "sttUseSegmentation",
    false,
  );
  const [sttTimestampGranularities, setSttTimestampGranularities] = useStorage(
    "sttTimestampGranularities",
    "segment",
  );

  const [sttPrompt, setSttPrompt] = useStorage("sttPrompt", "");
  const [sttTask, setSttTask] = useStorage("sttTask", "transcribe");
  const [sttResponseFormat, setSttResponseFormat] = useStorage(
    "sttResponseFormat",
    "json",
  );
  const [sttTemperature, setSttTemperature] = useStorage("sttTemperature", 0);
  const [sttSegK, setSttSegK] = useStorage("sttSegK", 6);
  const [sttSegMinSegmentSize, setSttSegMinSegmentSize] = useStorage(
    "sttSegMinSegmentSize",
    5,
  );
  const [sttSegLambdaBalance, setSttSegLambdaBalance] = useStorage(
    "sttSegLambdaBalance",
    0.01,
  );
  const [sttSegUtteranceExpansionWidth, setSttSegUtteranceExpansionWidth] =
    useStorage("sttSegUtteranceExpansionWidth", 2);
  const [sttSegEmbeddingsProvider, setSttSegEmbeddingsProvider] = useStorage(
    "sttSegEmbeddingsProvider",
    "",
  );
  const [sttSegEmbeddingsModel, setSttSegEmbeddingsModel] = useStorage(
    "sttSegEmbeddingsModel",
    "",
  );

  const [serverModels, setServerModels] = React.useState<string[]>([]);
  const [serverModelsLoading, setServerModelsLoading] = React.useState(true);
  const [serverModelsFetchFailed, setServerModelsFetchFailed] = React.useState(false);
  const [modelHealth, setModelHealth] = React.useState<
    "idle" | "checking" | "ok" | "error"
  >("idle");
  const { devices: audioInputDevices } = useAudioSourceCatalog();

  React.useEffect(() => {
    let cancelled = false;
    const fetchModels = async () => {
      setServerModelsLoading(true);
      setServerModelsFetchFailed(false);
      try {
        const res = await tldwClient.getTranscriptionModels();
        const all = Array.isArray(res?.all_models)
          ? (res.all_models as string[])
          : [];
        if (!cancelled && all.length > 0) {
          const unique = Array.from(new Set(all)).sort();
          setServerModels(unique);
        }
      } catch (e) {
        if (!cancelled) {
          setServerModelsFetchFailed(true);
        }
        if ((import.meta as any)?.env?.DEV) {
          // eslint-disable-next-line no-console
          console.warn("Failed to load transcription models from server", e);
        }
      } finally {
        if (!cancelled) {
          setServerModelsLoading(false);
        }
      }
    };
    fetchModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCheckModelHealth = async () => {
    const model = (sttModel || "").trim();
    if (!model) {
      return;
    }
    setModelHealth("checking");
    try {
      const res = await tldwClient.getTranscriptionModelHealth(model);
      const status =
        typeof res === "object" && res && "status" in res
          ? (res as any).status
          : undefined;
      if (status && typeof status === "string") {
        setModelHealth(status.toLowerCase() === "ok" ? "ok" : "error");
      } else {
        setModelHealth("ok");
      }
    } catch (e) {
      if ((import.meta as any)?.env?.DEV) {
        // eslint-disable-next-line no-console
        console.warn("Transcription model health check failed", e);
      }
      setModelHealth("error");
    }
  };

  const collapseItems = [
    {
      key: "basic",
      label: t("generalSettings.stt.basicSettings", "Basic Settings"),
      className: "!border-0",
      children: (
        <div className="space-y-4">
          <div className="flex flex-row justify-between">
            <span className="text-text">
              {t("generalSettings.settings.speechRecognitionLang.label")}
            </span>
            <Select
              placeholder={t(
                "generalSettings.settings.speechRecognitionLang.placeholder",
              )}
              allowClear
              showSearch
              options={SUPPORTED_LANGUAGES}
              value={speechToTextLanguage}
              filterOption={(input, option) =>
                option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
                option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
              }
              onChange={(value) => {
                setSpeechToTextLanguage(value);
              }}
              className={hideBorder ? "w-full" : "!min-w-[200px]"}
            />
          </div>

          {!serverModelsLoading && !serverModelsFetchFailed && serverModels.length === 0 && (
            <Alert
              type="info"
              showIcon
              message={t("generalSettings.stt.noModelsAlert", "No STT models available on your server. Configure a transcription engine to enable speech-to-text.")}
              className="mb-3"
            />
          )}
          <div className="flex flex-row justify-between">
            <span className="text-text">
              {t("generalSettings.stt.model.label")}
            </span>
            <div
              className={
                hideBorder
                  ? "w-full flex flex-col items-end"
                  : "!min-w-[200px] flex flex-col items-end"
              }
            >
              <Select
                className="w-full"
                showSearch
                placeholder="whisper-1, parakeet, canary..."
                loading={serverModelsLoading}
                value={sttModel}
                onChange={(value) => setSttModel(value)}
                options={
                  serverModels.length > 0
                    ? serverModels.map((model) => ({
                        label: model,
                        value: model,
                      }))
                    : sttModel
                      ? [
                          {
                            label: sttModel,
                            value: sttModel,
                          },
                        ]
                      : []
                }
                allowClear
                onClear={() => setSttModel("")}
                popupMatchSelectWidth
              />
              {serverModels.length > 0 && (
                <span className="mt-1 text-[11px] text-text-subtle self-start">
                  {t(
                    "generalSettings.stt.model.helpFromServer",
                    "Models provided by your tldw server ({{count}} total).",
                    { count: serverModels.length },
                  )}
                </span>
              )}
              <Tooltip
                title={
                  !sttModel
                    ? t(
                        "generalSettings.stt.model.healthCheckSelectFirst",
                        "Select a model first",
                      )
                    : ""
                }
              >
                <Button
                  type="default"
                  size="small"
                  className="mt-1 self-start"
                  onClick={handleCheckModelHealth}
                  loading={modelHealth === "checking"}
                  disabled={!sttModel}
                >
                  {modelHealth === "checking"
                    ? t(
                        "generalSettings.stt.model.healthChecking",
                        "Checking model health…",
                      )
                    : modelHealth === "ok"
                      ? t(
                          "generalSettings.stt.model.healthOk",
                          "Model appears healthy",
                        )
                      : modelHealth === "error"
                        ? t(
                            "generalSettings.stt.model.healthError",
                            "Health check failed",
                          )
                        : t(
                            "generalSettings.stt.model.healthCheck",
                            "Check model health",
                          )}
                </Button>
              </Tooltip>
            </div>
          </div>

          <div className="flex flex-row justify-between">
            <span className="text-text">
              {t("generalSettings.stt.task.label")}
            </span>
            <Select
              className={hideBorder ? "w-full" : "!min-w-[200px]"}
              value={sttTask}
              onChange={(value) => setSttTask(value)}
              options={[
                {
                  value: "transcribe",
                  label: t(
                    "generalSettings.stt.task.transcribe",
                    "Transcribe (same language)",
                  ),
                },
                {
                  value: "translate",
                  label: t(
                    "generalSettings.stt.task.translate",
                    "Translate to English",
                  ),
                },
              ]}
            />
          </div>

          <div className="flex flex-row justify-between">
            <span className="text-text">
              {t("generalSettings.stt.autoSubmitVoiceMessage.label")}
            </span>
            <Switch
              checked={autoSubmitVoiceMessage}
              onChange={(checked) => {
                setAutoSubmitVoiceMessage(checked);
              }}
            />
          </div>

          <div className="flex flex-row justify-between">
            <span className="text-text">
              {t("generalSettings.stt.autoStopTimeout.label")}
            </span>
            <InputNumber
              className={hideBorder ? "w-full" : "!min-w-[200px]"}
              type="number"
              placeholder={t("generalSettings.stt.autoStopTimeout.placeholder")}
              value={resolvedAutoStopTimeout}
              suffix="ms"
              onChange={(e) => {
                setAutoStopTimeout(
                  typeof e === "number" ? e : resolvedAutoStopTimeout,
                );
              }}
            />
          </div>
        </div>
      ),
    },
    {
      key: "sources",
      label: t(
        "generalSettings.stt.sourcePreferences.title",
        "Audio input source preferences",
      ),
      className: "!border-0",
      children: (
        <div className="space-y-4">
          <p className="text-xs text-text-subtle">
            {t(
              "generalSettings.stt.sourcePreferences.description",
              "Choose which microphone each speech surface should prefer by default.",
            )}
          </p>
          <AudioSourcePreferenceRow
            featureGroup="dictation"
            label={t(
              "generalSettings.stt.sourcePreferences.dictation",
              "Dictation",
            )}
            devices={audioInputDevices}
            hideBorder={hideBorder}
          />
          <AudioSourcePreferenceRow
            featureGroup="live_voice"
            label={t(
              "generalSettings.stt.sourcePreferences.liveVoice",
              "Live voice",
            )}
            devices={audioInputDevices}
            hideBorder={hideBorder}
          />
          <AudioSourcePreferenceRow
            featureGroup="speech_playground"
            label={t(
              "generalSettings.stt.sourcePreferences.speechPlayground",
              "Speech Playground",
            )}
            devices={audioInputDevices}
            hideBorder={hideBorder}
          />
        </div>
      ),
    },
    {
      key: "advanced",
      label: t("generalSettings.stt.advancedSettings", "Advanced Settings"),
      className: "!border-0",
      children: (
        <>
          <p className="text-xs text-text-subtle mb-4">
            {t(
              "generalSettings.stt.advancedSettingsHelp",
              "These settings are for advanced users. Most users can leave these at their default values.",
            )}
          </p>
          <div className="space-y-4">
            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.useSegmentation.label")}
              </span>
              <Switch
                checked={sttUseSegmentation}
                onChange={(checked) => setSttUseSegmentation(checked)}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.timestampGranularities.label")}
              </span>
              <Select
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                value={sttTimestampGranularities}
                onChange={(value) => setSttTimestampGranularities(value)}
                options={[
                  {
                    value: "segment",
                    label: t(
                      "generalSettings.stt.timestampGranularities.segment",
                      "Per segment",
                    ),
                  },
                  {
                    value: "word",
                    label: t(
                      "generalSettings.stt.timestampGranularities.word",
                      "Per word",
                    ),
                  },
                  {
                    value: "segment,word",
                    label: t(
                      "generalSettings.stt.timestampGranularities.segmentWord",
                      "Segment + word",
                    ),
                  },
                ]}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.responseFormat.label")}
              </span>
              <Select
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                value={sttResponseFormat}
                onChange={(value) => setSttResponseFormat(value)}
                options={[
                  {
                    value: "json",
                    label: t(
                      "generalSettings.stt.responseFormat.json",
                      "JSON (text + segments)",
                    ),
                  },
                  {
                    value: "verbose_json",
                    label: t(
                      "generalSettings.stt.responseFormat.verboseJson",
                      "Verbose JSON",
                    ),
                  },
                  {
                    value: "text",
                    label: t(
                      "generalSettings.stt.responseFormat.text",
                      "Plain text",
                    ),
                  },
                  {
                    value: "srt",
                    label: t("generalSettings.stt.responseFormat.srt", "SRT"),
                  },
                  {
                    value: "vtt",
                    label: t("generalSettings.stt.responseFormat.vtt", "VTT"),
                  },
                ]}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.temperature.label")}
              </span>
              <InputNumber
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                min={0}
                max={1}
                step={0.1}
                value={sttTemperature}
                onChange={(value) => {
                  setSttTemperature(typeof value === "number" ? value : 0);
                }}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.prompt.label")}
              </span>
              <Input
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                placeholder={t(
                  "generalSettings.stt.prompt.placeholder",
                  "Optional text to guide style",
                )}
                value={sttPrompt}
                onChange={(e) => setSttPrompt(e.target.value)}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segK.label")}
              </span>
              <InputNumber
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                min={1}
                value={sttSegK}
                onChange={(value) => {
                  setSttSegK(typeof value === "number" ? value : 6);
                }}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segMinSegmentSize.label")}
              </span>
              <InputNumber
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                min={1}
                value={sttSegMinSegmentSize}
                onChange={(value) => {
                  setSttSegMinSegmentSize(
                    typeof value === "number" ? value : 5,
                  );
                }}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segLambdaBalance.label")}
              </span>
              <InputNumber
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                min={0}
                step={0.01}
                value={sttSegLambdaBalance}
                onChange={(value) => {
                  setSttSegLambdaBalance(
                    typeof value === "number" ? value : 0.01,
                  );
                }}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segUtteranceExpansionWidth.label")}
              </span>
              <InputNumber
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                min={0}
                value={sttSegUtteranceExpansionWidth}
                onChange={(value) => {
                  setSttSegUtteranceExpansionWidth(
                    typeof value === "number" ? value : 2,
                  );
                }}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segEmbeddingsProvider.label")}
              </span>
              <Input
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                value={sttSegEmbeddingsProvider}
                onChange={(e) => setSttSegEmbeddingsProvider(e.target.value)}
              />
            </div>

            <div className="flex flex-row justify-between">
              <span className="text-text">
                {t("generalSettings.stt.segEmbeddingsModel.label")}
              </span>
              <Input
                className={hideBorder ? "w-full" : "!min-w-[200px]"}
                value={sttSegEmbeddingsModel}
                onChange={(e) => setSttSegEmbeddingsModel(e.target.value)}
              />
            </div>
          </div>
        </>
      ),
    },
  ];

  return (
    <div>
      <div className="mb-5">
        <h2
          className={`${
            !hideBorder ? "text-base font-semibold leading-7" : "text-md"
          } text-text`}
        >
          {t("generalSettings.stt.heading")}
        </h2>
        {!hideBorder && <div className="border-b border-border mt-3"></div>}
        <p className="mt-2 text-xs text-text-muted">
          {t(
            "generalSettings.stt.usedByChat",
            "These Speech-to-Text defaults are used by the chat dictation button in the Playground and Sidebar.",
          )}
        </p>
      </div>
      <Collapse
        defaultActiveKey={["basic"]}
        className="bg-transparent border-0"
        items={collapseItems}
      />
    </div>
  );
};
