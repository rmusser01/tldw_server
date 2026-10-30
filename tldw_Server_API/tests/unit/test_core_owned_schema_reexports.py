"""api/v1/schemas re-exports the models core owns (TASK-13950.15).

Core must not import api/v1/schemas (tests/lint/test_layering_contract.py), so the models core
reads are defined in core and re-exported here, as api/v1/schemas/history_selection_schemas.py
does. A row names an api module, the core module that defines the names, and every name that
moved. The re-export must be the same object, so FastAPI, isinstance checks and the OpenAPI
components see one definition.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

API = "tldw_Server_API.app.api.v1.schemas"
CORE = "tldw_Server_API.app.core"
REPO_ROOT = Path(__file__).resolve().parents[3]

# (module under API, module under CORE, the names that moved). Slices append rows in order.
CORE_OWNED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("evaluation_schemas_unified", "Evaluations.run_state", ("RunStatus",)),
    ("evaluation_recipe_schemas", "Evaluations.evaluation_recipe_schemas", ("ConfidenceSummary", "RecipeManifest", "RecipeRunRecord", "RecommendationSlot", "ReviewState")),
    ("embeddings_abtest_schemas", "Evaluations.embeddings_abtest_schemas", ("ABTestArm", "ABTestChunking", "ABTestCleanupPolicy", "ABTestLimits", "ABTestQuery", "ABTestReRanker", "ABTestRetrieval", "EmbeddingsABTestConfig")),
    ("synthetic_eval_schemas", "Evaluations.synthetic_eval_schemas", ("SyntheticEvalDraftSampleCreate", "SyntheticEvalPromotionCreate", "SyntheticEvalProvenance", "SyntheticEvalReviewActionCreate", "SyntheticEvalReviewActionType", "SyntheticEvalReviewState")),
    ("rag_schemas_unified", "RAG.rag_schemas_unified", ("KnowledgeSourceEmbeddingStatus", "KnowledgeSourceHealthEntry", "KnowledgeSourceHealthResponse", "KnowledgeSourceIndexStatus", "UnifiedBatchRequest", "UnifiedRAGRequest", "UnifiedRAGResponse")),
    ("_compat", "Utils.pydantic_compat", ("Field",)),
    ("audio_schemas", "TTS.speech_schemas", ("NormalizationOptions", "OpenAISpeechRequest")),
    ("audio_schemas", "Streaming.speech_chat_schemas", ("SpeechChatLLMConfig", "SpeechChatRequest", "SpeechChatResponse", "SpeechChatSTTConfig", "SpeechChatTTSConfig", "SpeechChatTiming", "SpeechChatTokenUsage")),
    ("research_workspace_capabilities", "Research_Workspace.research_workspace_capabilities", ("ResearchWorkspaceCapabilitiesResponse", "ResearchWorkspaceCapability", "ResearchWorkspaceCapabilityId", "ResearchWorkspaceCapabilityMode", "ResearchWorkspaceCapabilityStatus", "ResearchWorkspaceOverallStatus")),
    ("research_workspace_outputs", "Research_Workspace.research_workspace_outputs", ("ResearchWorkspaceOutputArtifactType", "ResearchWorkspaceOutputSettings", "ResearchWorkspaceOutputStatus", "ResearchWorkspaceOutputStatusResponse", "ResearchWorkspaceOutputSubmitRequest", "ResearchWorkspaceOutputSubmitResponse")),
    ("workspace_schemas", "Workspaces.workspace_schemas", ("WorkspaceArtifactRedaction", "WorkspaceArtifactResponse", "WorkspaceArtifactReviewState", "WorkspaceAssistantDefaultDegradedReason", "WorkspaceAssistantDefaults", "WorkspaceAssistantKind", "WorkspaceEffectiveAssistantDefault", "WorkspaceEffectiveAssistantDefaultSource", "WorkspaceEffectiveAssistantDefaultStatus", "WorkspaceOperationBase", "WorkspaceOperationStatus", "WorkspacePersonaMemoryMode")),
    ("chat_session_schemas", "Workspaces.chat_session_schemas", ("ALLOWED_ASSISTANT_KINDS", "ALLOWED_PERSONA_MEMORY_MODES", "ChatSessionCreate")),
    ("shared_workspace_recipient_schemas", "Sharing.shared_workspace_recipient_schemas", ("CloneWarningCode", "SharedWorkspaceCloneCounts", "SharedWorkspaceCloneError", "SharedWorkspaceCloneOperationResponse", "SharedWorkspaceCloneProgress", "SharedWorkspaceCloneReadiness", "SharedWorkspaceCloneResult", "SharedWorkspaceCloneWarning", "ShortCode", "_RecipientModel")),
    ("vn_asset_schemas", "VN_Assets.vn_asset_schemas", ("ReviewStatus", "SlotStatus", "VALID_REVIEW_STATUSES", "VNAssetBulkReviewRequest", "VNAssetCleanupRequest", "VNAssetCleanupResponse", "VNAssetGenerationPreflightResponse", "VNAssetGenerationRequest", "VNAssetGenerationStatusResponse", "VNAssetItemResponse", "VNAssetManifestResponse", "VNAssetPackCreate", "VNAssetPackResponse", "VNAssetPackUpdate", "VNAssetPromptPreviewRequest", "VNAssetPromptPreviewResponse", "VNAssetReadinessResponse", "VNAssetReviewRequest", "VNAssetSlotCreate", "VNAssetSlotPreflight", "VNAssetSlotResponse", "VNAssetSlotUpdate")),
    ("vn_play_schemas", "VN_Play.vn_play_schemas", ("VNPlayMode", "VNPlaySetupAssetPackOption", "VNPlaySetupCharacterOption", "VNPlaySetupCompatibility", "VNPlaySetupCompatibilityStatus", "VNPlaySetupDefaults", "VNPlaySetupEmptyState", "VNPlaySetupEmptyStateScope", "VNPlaySetupOptionsResponse", "VNPlaySetupPagination", "VNPlaySetupPaginationSet", "VNPlaySetupScriptVersionOption", "VNPlaySetupTrustLevel", "VNPlaySetupTrustSource", "VNPlaySetupWarning", "VNPlaySetupWarningSeverity", "VNPlaySetupWarningSummary")),
    ("flashcards", "Flashcards.flashcard_schemas", ("FlashcardCardType", "FlashcardGenerateRequest", "FlashcardGenerationType", "FlashcardPlanItem")),
    ("study_packs", "StudyPacks.study_pack_schemas", ("StudyPackCreateJobRequest", "StudyPackSourceSelection", "StudyPackSourceType")),
)


@pytest.mark.parametrize(("api_module", "core_module", "names"), CORE_OWNED, ids=[f"{a}->{c}" for a, c, _ in CORE_OWNED])
def test_api_schema_module_re_exports_the_core_owned_definition(
    api_module: str, core_module: str, names: tuple[str, ...]
) -> None:
    api = importlib.import_module(f"{API}.{api_module}")
    core = importlib.import_module(f"{CORE}.{core_module}")
    for name in names:
        assert getattr(api, name) is getattr(core, name), f"{API}.{api_module}.{name} is not {core.__name__}.{name}"
        obj = getattr(core, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            assert obj.__module__ == core.__name__, f"{name} is defined in {obj.__module__}"


def test_the_evaluations_models_load_no_db_management_code() -> None:
    """S1 raises core.DB_Management -> core.Evaluations to 3; the raise must stay a model-only import."""
    mods = ("run_state", "evaluation_recipe_schemas", "embeddings_abtest_schemas", "synthetic_eval_schemas")
    code = "import importlib, sys\n" + "".join(f"importlib.import_module('{CORE}.Evaluations.{m}')\n" for m in mods)
    code += "print('\\n'.join(sorted(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert [m for m in run.stdout.split() if ".core.DB_Management" in m] == []
