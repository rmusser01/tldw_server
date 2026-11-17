"""The registered benchmarks run on correctly shaped data (TASK-15487).

Rows reach the evaluator in the benchmark's own fields (the registry's field mappings are applied by
load_benchmark_dataset), the two HuggingFace benchmarks load with the right config and shape, every
registered benchmark builds an evaluator, and every judge prompt asks for the 1-10 scale the manager reads.
No test downloads a dataset: ``datasets`` is a stub module, and OpenAI's CSV fetch is stubbed.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tldw_Server_API.app.core.Evaluations import benchmark_utils
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import DatasetLoader, load_benchmark_dataset
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.qa_benchmark_helper import QAEvaluation

pytestmark = pytest.mark.unit

_TEMPLATE = Path(benchmark_utils.__file__).with_name("configs") / "simple_qa_template.yaml"

# One row of TIGER-Lab/MMLU-Pro (test split), in its real fields
_MMLU_PRO_ROW = {
    "question_id": 70,
    "question": "Which of the following is the body cavity that contains the pituitary gland?",
    "options": ["Abdominal", "Cranial", "Pleural", "Spinal", "Thoracic", "Pelvic", "Buccal", "Pericardial", "Ventral", "Dorsal"],
    "answer": "B",
    "answer_index": 1,
    "cot_content": "",
    "category": "health",
    "src": "ori_mmlu-anatomy",
}
# One row of Idavidrein/gpqa (config gpqa_diamond, split train), in its real columns
_GPQA_ROW = {
    "Record ID": "recAAHX3nMOaCODRe",
    "Question": "What is the speed of light in vacuum, in km/s?",
    "Correct Answer": "299,792",
    "Incorrect Answer 1": "150,000",
    "Incorrect Answer 2": "1,079,252,848",
    "Incorrect Answer 3": "3,000",
    "Subdomain": "Physics (general)",
    "High-level domain": "Physics",
}


def _stub_datasets(monkeypatch, row):
    """A ``datasets`` module whose load_dataset records its call and returns one row; nothing is downloaded."""
    calls = []

    def load_dataset(path, name=None, **kwargs):
        calls.append((path, name, kwargs.get("split")))
        return [row]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return calls


@pytest.mark.asyncio
async def test_an_mmlu_pro_row_reaches_the_judge_with_its_choices_and_answer(monkeypatch, evaluation_manager):
    """An MMLU-Pro row's options and answer letter reach the judge in the evaluator's fields (it was told the
    correct answer is None, so its scores carried no information)."""
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

    _stub_datasets(monkeypatch, _MMLU_PRO_ROW)
    prompts: list[str] = []

    def chat(request, timeout=None):
        prompts.append(request["messages"][-1]["content"])
        return {"choices": [{"message": {"content": "B" if len(prompts) == 1 else "SCORE: 10 EXPLANATION: B."}}]}

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )

    (row,) = load_benchmark_dataset("mmlu_pro", limit=1)
    data = get_registry().create_evaluator("mmlu_pro").format_for_custom_metric(row)
    result = await evaluation_manager.evaluate_custom_metric(
        metric_name=data["name"],
        description=data["description"],
        evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"],
        scoring_criteria=data["scoring_criteria"],
        api_name="openai",
        api_key="sk-test",
    )

    assert len(prompts) == 2, prompts  # the model's answer, then the judge
    assert "briefly explain your reasoning" in prompts[0].lower()
    assert "Compare with the correct answer: B" in prompts[1]
    assert "B. Cranial" in prompts[1] and "J. Dorsal" in prompts[1]
    assert data["metadata"]["category"] == "health"
    assert result["score"] == 1.0


def test_simpleqa_verified_loads_its_rows_from_the_shipped_zip():
    """The curated set ships as Helper_Scripts/Evals/simpleqa-verified.zip (one CSV, a 'problem' column)."""
    rows = load_benchmark_dataset("simpleqa_verified", limit=2)

    assert len(rows) == 2
    assert rows[0]["question"].startswith("How much money, in euros")
    assert rows[0]["answer"] == "120,000 euros"
    assert rows[0]["topic"] == "Politics"


def test_verified_benchmark_is_not_advertised_without_the_shipped_archive(monkeypatch):
    from tldw_Server_API.app.core.Evaluations import benchmark_registry

    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: path.name != "simpleqa-verified.zip" and real_is_file(path))
    registry = benchmark_registry.BenchmarkRegistry()
    monkeypatch.setattr(benchmark_registry, "_registry", registry)
    monkeypatch.setattr(DatasetLoader, "load_csv", lambda *_args: pytest.fail("do not fall back to unverified data"))

    assert "simpleqa_verified" not in registry.list_benchmarks()
    assert load_benchmark_dataset("simpleqa_verified") == []


def test_remote_csv_keeps_embedded_line_breaks_and_closes_the_response(monkeypatch):
    from tldw_Server_API.app.core.Evaluations import benchmark_loaders

    closed = []
    response = SimpleNamespace(text='problem,answer\r\nExample,"One\r\nTwo"\r\n', close=lambda: closed.append(True))
    monkeypatch.setattr(benchmark_loaders, "fetch", lambda **_kwargs: response)
    assert DatasetLoader.load_csv("https://benchmark.invalid/example.csv") == [{"problem": "Example", "answer": "One\r\nTwo"}]
    assert closed == [True]


def test_custom_benchmark_loads_its_registered_format_and_source(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Evaluations import benchmark_registry
    from tldw_Server_API.app.core.Evaluations.benchmark_registry import BenchmarkConfig, BenchmarkRegistry

    source = tmp_path / "dataset"
    source.write_text('[{"problem":"one"},{"problem":"two"}]', encoding="utf-8")
    registry = BenchmarkRegistry()
    registry.register(BenchmarkConfig(
        name="custom", description="local fixture", evaluation_type="simpleqa",
        dataset_source=str(source), dataset_format="json", field_mappings={"question": "problem"},
    ))
    monkeypatch.setattr(benchmark_registry, "_registry", registry)

    assert [row["question"] for row in load_benchmark_dataset("custom")] == ["one", "two"]
    assert [row["question"] for row in registry.load_dataset("custom", limit=1)] == ["one"]


def test_registered_huggingface_format_accepts_a_plain_dataset_id(monkeypatch):
    from tldw_Server_API.app.core.Evaluations.benchmark_registry import BenchmarkConfig, BenchmarkRegistry

    calls = _stub_datasets(monkeypatch, {"problem": "example"})
    registry = BenchmarkRegistry()
    registry.register(BenchmarkConfig(
        name="custom_hf", description="dataset fixture", evaluation_type="simpleqa",
        dataset_source="fixture-owner/example", dataset_format="huggingface", field_mappings={"question": "problem"},
    ))

    assert [row["question"] for row in registry.load_dataset("custom_hf")] == ["example"]
    assert calls == [("fixture-owner/example", None, "test")]


def test_local_and_zipped_csv_preserve_embedded_line_breaks(tmp_path):
    from zipfile import ZipFile

    from tldw_Server_API.app.core.Evaluations.benchmark_loaders import BenchmarkDatasetLoader

    raw = b'problem,answer\r\nExample,"One\r\nTwo"\r\n'
    local = tmp_path / "fixture.csv"
    local.write_bytes(raw)
    archived = tmp_path / "fixture.zip"
    with ZipFile(archived, "w") as archive:
        archive.writestr("fixture.csv", raw)

    expected = [{"problem": "Example", "answer": "One\r\nTwo"}]
    assert DatasetLoader.load_csv(str(local)) == expected
    assert BenchmarkDatasetLoader.load_simpleqa(str(archived)) == expected


def test_gpqa_diamond_loads_its_config_and_shapes_the_choices(monkeypatch):
    """GPQA is one gated dataset with a config per subset (train split); a row holds the correct answer and three
    incorrect ones in named columns, so the loader builds the choices and the answer letter."""
    calls = _stub_datasets(monkeypatch, _GPQA_ROW)

    (row,) = load_benchmark_dataset("gpqa_diamond")

    assert calls == [("Idavidrein/gpqa", "gpqa_diamond", "train")]
    assert sorted(row["choices"]) == sorted(["299,792", "150,000", "1,079,252,848", "3,000"])
    assert row["choices"][ord(row["correct_answer"]) - 65] == "299,792"
    assert row["question"] == _GPQA_ROW["Question"]
    assert row["category"] == "Physics (general)"


def test_gpqa_diamond_loads_a_local_csv_in_its_columns(tmp_path):
    """A local file in GPQA's own columns loads through the CSV reader (which read its rows after closing the file)."""
    path = tmp_path / "gpqa_diamond.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_GPQA_ROW))
        writer.writeheader()
        writer.writerow(_GPQA_ROW)

    (row,) = load_benchmark_dataset("gpqa_diamond", source=str(path))

    assert row["question"] == _GPQA_ROW["Question"]
    assert row["choices"][ord(row["correct_answer"]) - 65] == "299,792"


def test_simpleqa_loads_openais_csv_in_the_benchmarks_fields(monkeypatch):
    """SimpleQA is OpenAI's CSV (problem, answer); its rows reach the evaluator as question and answer."""
    sources: list[str] = []
    monkeypatch.setattr(
        DatasetLoader,
        "load_csv",
        staticmethod(
            lambda source, delimiter=",": sources.append(source)
            or [{"metadata": "{'topic': 'Art'}", "problem": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}]
        ),
    )

    (row,) = load_benchmark_dataset("simpleqa")

    assert sources == ["https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"]
    assert (row["question"], row["answer"]) == ("Who painted the Mona Lisa?", "Leonardo da Vinci")


def test_every_registered_benchmark_builds_an_evaluator():
    """The registry lists only benchmarks that run: each builds an evaluator (six could not, and answered 501)."""
    registry = get_registry()

    assert set(registry.list_benchmarks()) == {
        "simpleqa", "simpleqa_verified", "bullshit_benchmark", "mmlu_pro", "gpqa_diamond", "simple_bench",
    }
    assert [name for name in registry.list_benchmarks() if registry.create_evaluator(name) is None] == []


_MC = {"question": "q", "choices": ["a", "b"], "correct_answer": 1}
_FORMATTERS = {
    "multiple_choice": lambda: benchmark_utils.MultipleChoiceEvaluation().format_for_custom_metric(_MC),
    "multiple_choice_reasoning": lambda: benchmark_utils.MultipleChoiceEvaluation(
        require_reasoning=True
    ).format_for_custom_metric(_MC),
    "code_generation": lambda: benchmark_utils.CodeGenerationEvaluation().format_for_custom_metric(
        {"problem": "p", "test_cases": []}
    ),
    "code_generation_no_tests": lambda: benchmark_utils.CodeGenerationEvaluation(
        run_tests=False
    ).format_for_custom_metric({"problem": "p", "test_cases": []}),
    "instruction_following": lambda: benchmark_utils.InstructionFollowingEvaluation().format_for_custom_metric(
        {"instruction": "i", "constraints": ["c"]}
    ),
    "honesty": lambda: benchmark_utils.HonestyEvaluation().format_for_custom_metric(
        {"question": "q", "variations": ["v"]}
    ),
    "bullshit_detection": lambda: benchmark_utils.BullshitDetectionEvaluation().format_for_custom_metric(
        {"question": "q", "nonsensical_element": "n"}
    ),
    "mmlu_helper": lambda: benchmark_utils.create_mmlu_evaluation_data("q", ["a", "b"], 1),
    "function_calling_helper": lambda: benchmark_utils.create_function_calling_evaluation_data(
        "q", {"name": "f", "parameters": {}}, []
    ),
    "code_generation_helper": lambda: benchmark_utils.create_code_generation_evaluation_data("p", []),
    # the custom-QA helper's judge prompt and the template users copy it from
    "qa_helper": lambda: {
        "evaluation_prompt": QAEvaluation().format_for_llm_judge("q", "a", "e"),
        "scoring_criteria": {},
    },
    "simple_qa_template": lambda: {
        "evaluation_prompt": yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))["evaluation_params"][
            "judge_prompt_template"
        ],
        "scoring_criteria": {},
    },
}


@pytest.mark.parametrize("formatter", sorted(_FORMATTERS))
def test_every_formatter_asks_for_the_1_to_10_scale_the_manager_reads(formatter):
    """The manager reads a 1-10 score (1 is the worst, 0.0); a prompt asking for X.X, 0-10 or a percentage made a
    perfect reply parse as the worst, or leave the item unscored."""
    data = _FORMATTERS[formatter]()
    text = data["evaluation_prompt"] + "\n" + "\n".join(str(v) for v in data["scoring_criteria"].values())

    assert "from 1 (worst) to 10 (best)" in text or "SCORE: [1-10]" in text, text
    for stale in ("X.X", "0-10", "1.0", "0.0", "% of score"):
        assert stale not in text, (stale, text)


def test_a_simpleqa_row_in_openais_csv_shape_formats_for_the_judge(monkeypatch):
    """OpenAI's CSV has a string 'metadata' column; the row must still format (it raised TypeError: every run was a 500)."""
    monkeypatch.setattr(
        DatasetLoader,
        "load_csv",
        staticmethod(
            lambda source, delimiter=",": [
                {"metadata": "{'topic': 'Art'}", "problem": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}
            ]
        ),
    )

    (row,) = load_benchmark_dataset("simpleqa")
    data = get_registry().create_evaluator("simpleqa").format_for_custom_metric(row)

    assert data["input_data"]["question"] == "Who painted the Mona Lisa?"
