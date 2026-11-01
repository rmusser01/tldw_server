import React from "react";
import { Alert, Button, Input, Select } from "antd";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useChatModelCatalog } from "@/hooks/useChatModelCatalog";
import {
  getCanonicalModelKey,
  getModelId,
  getModelProvider,
} from "@/hooks/playground/modelSelectorUtils";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import {
  classifyModeration,
  type ModerationClassificationItem,
} from "@/services/moderation";

const EMPTY_FORM = { modelKey: "", criterion: "", text: "" };
const SUPPORTED_PROVIDERS = new Set([
  "llamacpp",
  "llama.cpp",
  "llama-cpp",
  "llama_cpp",
]);

export default function ModelClassificationPanel({
  configSnapshot,
}: {
  configSnapshot: TldwConfig;
}) {
  const { t } = useTranslation("option");
  const tr = (key: string, fallback: string) =>
    t("moderationClassification." + key, { defaultValue: fallback });
  const catalog = useChatModelCatalog();
  const models = React.useMemo(
    () =>
      (catalog.data ?? [])
        .filter((model) => SUPPORTED_PROVIDERS.has(getModelProvider(model)))
        .map((model) => ({
          value: getCanonicalModelKey(model),
          label: getModelId(model) + " (llama.cpp)",
          model: getModelId(model),
        })),
    [catalog.data],
  );
  const [form, setForm] = React.useState(EMPTY_FORM);
  const [requestId, setRequestId] = React.useState(() => crypto.randomUUID());
  const [result, setResult] =
    React.useState<ModerationClassificationItem | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [running, setRunning] = React.useState(false);
  const [stopping, setStopping] = React.useState(false);
  const [attempted, setAttempted] = React.useState(false);
  const [retired, setRetired] = React.useState(false);
  const controller = React.useRef<AbortController | null>(null);
  const lifetime = React.useRef({ active: true, revision: 0 });
  const id = React.useId();

  React.useEffect(() => {
    lifetime.current.active = true;
    const retire = () => {
      lifetime.current.active = false;
      lifetime.current.revision += 1;
      controller.current?.abort();
    };
    const unsubscribe = watchChatAccountChanges((invalidated) => {
      if (!invalidated) return;
      retire();
      setRetired(true);
      setRunning(false);
      setForm(EMPTY_FORM);
      setResult(null);
      setError(null);
      setAttempted(false);
    });
    return () => {
      retire();
      unsubscribe();
    };
  }, []);

  const change = (patch: Partial<typeof form>) => {
    setForm((previous) => ({ ...previous, ...patch }));
    setRequestId(crypto.randomUUID());
    setResult(null);
    setError(null);
    setAttempted(false);
  };

  const run = async (event: React.FormEvent) => {
    event.preventDefault();
    const selected = models.find((model) => model.value === form.modelKey);
    if (
      !lifetime.current.active ||
      controller.current ||
      !selected ||
      !form.criterion.trim() ||
      !form.text.trim()
    )
      return;
    const own = new AbortController();
    const revision = lifetime.current.revision;
    controller.current = own;
    setRunning(true);
    setStopping(false);
    setAttempted(true);
    setError(null);
    const current = () =>
      lifetime.current.active && lifetime.current.revision === revision;
    try {
      const item = await classifyModeration(
        {
          request_id: requestId,
          provider: "llama.cpp",
          model: selected.model,
          criterion: form.criterion,
          text: form.text,
        },
        { configSnapshot, abortSignal: own.signal },
      );
      if (current() && !own.signal.aborted) setResult(item);
      else if (current())
        setError(
          tr(
            "stopped",
            "Classification stopped. A result may already be saved; Retry checks the same receipt.",
          ),
        );
    } catch (failure) {
      if (current())
        setError(
          own.signal.aborted
            ? tr(
                "stopped",
                "Classification stopped. A result may already be saved; Retry checks the same receipt.",
              )
            : failure instanceof Error
              ? failure.message
              : tr(
                  "failed",
                  "Classification failed. Keep this sample and explicitly retry the saved request.",
                ),
        );
    } finally {
      if (controller.current === own) controller.current = null;
      if (current()) {
        setRunning(false);
        setStopping(false);
      }
    }
  };

  const classification = result?.context.classification;
  return (
    <section className="space-y-6 max-w-3xl" aria-labelledby={id + "-heading"}>
      <div className="space-y-2">
        <h3 id={id + "-heading"} className="text-lg font-semibold text-text">
          {tr("title", "Model classification")}
        </h3>
        <p className="text-sm text-text-muted">
          {tr(
            "intro",
            "Ask a configured model to check one criterion. The sample and result are saved to the staff Review queue. A human makes the final decision; this does not change rules or Chat enforcement.",
          )}
        </p>
      </div>
      {retired && (
        <Alert
          type="warning"
          message={tr(
            "retired",
            "The account changed. Reopen this panel after account verification.",
          )}
        />
      )}
      {!catalog.isFetching && (catalog.isError || models.length === 0) && (
        <Alert
          type={catalog.isError ? "error" : "info"}
          message={
            catalog.isError
              ? tr(
                  "catalogFailed",
                  "Could not load configured models. Check the connection, then reload models.",
                )
              : tr(
                  "noModel",
                  "No configured llama.cpp model is available. Check provider settings, then reload models.",
                )
          }
          action={
            <Button onClick={() => void catalog.refetch()}>
              {tr("reload", "Reload models")}
            </Button>
          }
        />
      )}
      <form onSubmit={run} className="space-y-4">
        <div>
          <label
            htmlFor={id + "-model"}
            className="block text-sm font-medium text-text mb-1"
          >
            {tr("model", "Classification model")}
          </label>
          <Select
            id={id + "-model"}
            className="w-full"
            showSearch
            optionFilterProp="label"
            value={form.modelKey || undefined}
            options={models}
            placeholder={tr("choose", "Choose a configured model")}
            loading={catalog.isFetching}
            disabled={running || retired}
            onChange={(value) => change({ modelKey: value })}
          />
        </div>
        <div>
          <label
            htmlFor={id + "-criterion"}
            className="block text-sm font-medium text-text mb-1"
          >
            {tr("criterion", "Flag when")}
          </label>
          <Input.TextArea
            id={id + "-criterion"}
            rows={2}
            maxLength={500}
            value={form.criterion}
            disabled={running || retired}
            onChange={(event) => change({ criterion: event.target.value })}
            placeholder={tr(
              "criterionHint",
              "Example: the sample discloses an unpublished observatory access code.",
            )}
          />
        </div>
        <div>
          <label
            htmlFor={id + "-sample"}
            className="block text-sm font-medium text-text mb-1"
          >
            {tr("sample", "Sample for classification")}
          </label>
          <Input.TextArea
            id={id + "-sample"}
            rows={5}
            maxLength={4000}
            value={form.text}
            disabled={running || retired}
            onChange={(event) => change({ text: event.target.value })}
          />
        </div>
        {error && <Alert type="error" showIcon message={error} />}
        <div className="flex flex-wrap gap-3">
          <Button
            type="primary"
            htmlType="submit"
            loading={running}
            disabled={
              retired ||
              running ||
              !!result ||
              catalog.isFetching ||
              !models.some((model) => model.value === form.modelKey) ||
              !form.criterion.trim() ||
              !form.text.trim()
            }
          >
            {running
              ? tr("pending", "Classifying…")
              : attempted && !result
                ? tr("retry", "Retry saved request")
                : tr("classify", "Classify and save for review")}
          </Button>
          {running && (
            <Button
              disabled={stopping}
              onClick={() => {
                setStopping(true);
                controller.current?.abort();
              }}
            >
              {stopping
                ? tr("stopping", "Stopping…")
                : tr("stop", "Stop classification")}
            </Button>
          )}
        </div>
      </form>
      {classification && (
        <section
          className="space-y-3 border-t border-border pt-6"
          aria-live="polite"
          aria-label={tr("result", "Model suggestion")}
        >
          <h4 className="font-semibold text-text">
            {classification.label === "flagged"
              ? tr("flagged", "Model suggestion: flagged")
              : tr("benign", "Model suggestion: benign")}
          </h4>
          <p className="text-text">{classification.reason}</p>
          {classification.evidence && (
            <blockquote className="rounded-lg bg-surface-secondary p-3 text-text">
              {classification.evidence}
            </blockquote>
          )}
          <p className="text-sm text-text-muted">
            {classification.provider} · {classification.model}
          </p>
          <p className="text-sm text-text-muted">
            {tr("status", "Review status")}: {result.status}
          </p>
          <Link
            to="/moderation"
            className="text-primary underline underline-offset-4"
          >
            {tr("review", "Open Review queue")}
          </Link>
        </section>
      )}
    </section>
  );
}
