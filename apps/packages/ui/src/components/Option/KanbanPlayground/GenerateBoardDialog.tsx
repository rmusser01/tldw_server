import React from "react";
import { Alert, Button, Input, Modal, Typography } from "antd";
import { ModelSelect } from "@/components/Common/ModelSelect";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { useChatModelCatalog } from "@/hooks/useChatModelCatalog";
import {
  isConfiguredUsableModel,
  resolveModelSelectorSelection,
} from "@/hooks/playground/modelSelectorUtils";
import { useStoreChatModelSettings } from "@/store/model";
import { generateClientId } from "@/services/kanban";
import type { GeneratedBoardDraft } from "@/services/kanban-generation";
import { formatErrorMessage } from "@/utils/format-error-message";
import { useKanbanScope } from "./KanbanScope";

export default function GenerateBoardDialog({
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved: (boardId: number) => void;
}) {
  const { api, isActive } = useKanbanScope();
  const { selectedModel, selectedModelIsLoading } = useSelectedModel();
  const catalog = useChatModelCatalog();
  const apiProvider = useStoreChatModelSettings((state) => state.apiProvider);
  const temperature = useStoreChatModelSettings((state) => state.temperature);
  const topP = useStoreChatModelSettings((state) => state.topP);
  const maxTokens = useStoreChatModelSettings((state) => state.numPredict);
  const selection = resolveModelSelectorSelection(
    catalog.data,
    selectedModel,
    apiProvider,
  );
  const ready = Boolean(
    selection.modelId &&
    selection.provider &&
    isConfiguredUsableModel(selection.model) &&
    catalog.isSuccess &&
    !catalog.isFetching &&
    !selectedModelIsLoading,
  );
  const [instructions, setInstructions] = React.useState("");
  const [draft, setDraft] = React.useState<{
    value: GeneratedBoardDraft;
    clientId: string;
  } | null>(null);
  const [generating, setGenerating] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<string | null>(null);
  const generation = React.useRef<AbortController | null>(null);
  const savePending = React.useRef(false);
  const instructionsId = React.useId();

  React.useEffect(
    () => () => {
      generation.current?.abort();
    },
    [],
  );

  const stop = () => {
    generation.current?.abort();
    generation.current = null;
    setGenerating(false);
    setNotice(
      draft
        ? "Generation stopped. The previous draft is still available."
        : "Generation stopped. Generate a new draft when ready.",
    );
  };
  const generate = async () => {
    if (
      generation.current ||
      savePending.current ||
      !ready ||
      !instructions.trim()
    )
      return;
    const controller = new AbortController();
    generation.current = controller;
    setGenerating(true);
    setError(null);
    setNotice(null);
    try {
      const value = await api.generateBoardDraft(
        {
          instructions,
          model: selection.modelId,
          provider: selection.provider!,
          temperature,
          topP,
          maxTokens,
        },
        controller.signal,
      );
      if (
        isActive() &&
        generation.current === controller &&
        !controller.signal.aborted
      )
        setDraft({ value, clientId: generateClientId() });
    } catch (failure) {
      if (
        isActive() &&
        generation.current === controller &&
        !controller.signal.aborted
      )
        setError(
          formatErrorMessage(failure, "Draft generation failed. Try again."),
        );
    } finally {
      if (generation.current === controller) {
        generation.current = null;
        if (isActive()) setGenerating(false);
      }
    }
  };
  const save = async () => {
    if (!draft || generation.current || savePending.current || !isActive())
      return;
    savePending.current = true;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.importBoard({
        format: "tldw_kanban_draft_v1",
        client_id: draft.clientId,
        ...draft.value,
      });
      if (isActive()) onSaved(result.board.id);
    } catch (failure) {
      if (isActive())
        setError(
          "The draft could not be saved. Try Save board again. " +
            formatErrorMessage(failure, ""),
        );
    } finally {
      savePending.current = false;
      if (isActive()) setSaving(false);
    }
  };
  return (
    <Modal
      title="Generate board"
      open
      centered
      width={720}
      onCancel={onClose}
      styles={{ body: { maxHeight: "60vh", overflowY: "auto" } }}
      closable={!saving}
      keyboard={!saving}
      maskClosable={!saving}
      footer={[
        <Button key="cancel" onClick={onClose} disabled={saving}>
          Cancel
        </Button>,
        generating ? (
          <Button key="stop" onClick={stop}>
            Stop
          </Button>
        ) : (
          <Button
            key="generate"
            onClick={() => void generate()}
            disabled={!ready || !instructions.trim() || saving}
          >
            Generate draft
          </Button>
        ),
        <Button
          key="save"
          type="primary"
          onClick={() => void save()}
          loading={saving}
          disabled={!draft || generating || saving}
        >
          Save board
        </Button>,
      ]}
    >
      <div className="space-y-4">
        <Typography.Paragraph>
          Generate a draft, review its lists and cards, then select Save board
          to create a new board.
        </Typography.Paragraph>
        <ModelSelect showSelectedName showPicker={!generating && !saving} />
        {!ready && (
          <p role="status">Choose a configured model to generate a draft.</p>
        )}
        <div>
          <label htmlFor={instructionsId}>Board instructions</label>
          <Input.TextArea
            id={instructionsId}
            value={instructions}
            maxLength={4000}
            rows={4}
            disabled={generating || saving}
            onChange={(event) => setInstructions(event.target.value)}
          />
        </div>
        {error && <Alert type="error" showIcon message={error} />}
        {notice && <Alert type="info" showIcon message={notice} />}
        {generating && (
          <p role="status">
            Generating draft… Stop preserves your saved boards and any previous
            draft.
          </p>
        )}
        {draft && (
          <section aria-label="Generated board preview">
            <Typography.Title level={3}>
              {draft.value.board.name}
            </Typography.Title>
            {draft.value.board.description && (
              <Typography.Paragraph>
                {draft.value.board.description}
              </Typography.Paragraph>
            )}
            <ol>
              {draft.value.lists.map((list, index) => (
                <li key={index}>
                  <Typography.Title level={4}>{list.name}</Typography.Title>
                  <ul>
                    {list.cards.map((card, cardIndex) => (
                      <li key={cardIndex}>
                        <strong>{card.title}</strong>
                        {card.description && (
                          <Typography.Paragraph>
                            {card.description}
                          </Typography.Paragraph>
                        )}
                      </li>
                    ))}
                  </ul>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </Modal>
  );
}
