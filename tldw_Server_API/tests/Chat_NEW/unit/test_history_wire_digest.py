"""Real ECMAScript canonical byte/hash parity and raw-body projection."""

import json
import math
import random
import shutil
import struct

# Fixed local Node parity oracle, no shell.
import subprocess  # nosec B404
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
FRONTEND = Path(__file__).resolve().parents[4] / "apps/packages/ui/src/db/dexie/history-selection.ts"


def node_vectors(values):
    """Execute the actual frontend canonicalizer after erasing its two type annotations."""
    source = FRONTEND.read_text()
    canonicalizer = source.split("export const canonicalHistoryJson = ", 1)[1].split("\nexport const historyDigest", 1)[
        0
    ]
    canonicalizer = canonicalizer.replace("(value: unknown): string", "(value)").replace("(v: any): any", "(v)")
    script = (
        "const crypto = require('node:crypto'); const fail = code => {throw Error(code)}; const canonical = "
        + canonicalizer
        + ";\n"
        + """
const values = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
values.push({z:undefined,nested:{omitted:undefined,nullable:null},default:false});
process.stdout.write(JSON.stringify(values.map(value => {
  const bytes = canonical(value);
  const hash = text => crypto.createHash('sha256').update(text,'utf8').digest('hex');
  const projection = value && value.tldw_turn ? {...value,tldw_turn:{...value.tldw_turn}} : undefined;
  if (projection) delete projection.tldw_turn.history_v1;
  return {wire:JSON.stringify(value), bytes, hash:hash(bytes),
    ...(projection ? {request_hash:hash(canonical(projection))} : {})};
})));
"""
    )
    node = shutil.which("node")
    assert node, "Node is required for the approved byte-parity gate"
    # Fixed executable/script and JSON-only stdin.
    completed = subprocess.run(  # nosec B603
        [node, "-e", script],
        input=json.dumps(values, ensure_ascii=True),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


def test_canonical_bytes_and_hash_match_real_node_for_ecmascript_edge_cases():
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json, history_wire_digest

    values = [
        [0.1, 333333333.33333329, 1e30, 4.50, 2e-3, 1e-27, -0.0],
        [1e-6, 1e-7, 1e20, 1e21, 5e-324, 1.7976931348623157e308],
        [9007199254740991, -9007199254740991, 295147905179352830000.0, 999999999999999700000.0],
        {
            "10": "ten",
            "2": "two",
            "1": "one",
            "0": "zero",
            "01": "not-index",
            "4294967295": "not-index",
            "4294967294": "index",
            "-0": "not-index",
        },
        {"\ue000": "bmp", "\U0001f600": "astral", "a": '\u2028\u2029\n"\\\u0000', "nested": {"10": 1, "2": 2}},
        {"null": None, "default": False, "empty": [], "empty_object": {}},
        {"x": 0.0},
        {"x": -0.0},
        {},
        {"x": None},
        {"x": False},
    ]
    expected = node_vectors(values)
    values.append({"nested": {"nullable": None}, "default": False})
    for value, oracle in zip(values, expected, strict=True):
        assert canonical_history_json(value) == oracle["bytes"]
        assert history_wire_digest(value) == oracle["hash"]


@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
        9007199254740993,
        -9007199254740993,
        10**400,
        "\ud800",
        {"\udfff": "value"},
        {"value": "\ud800"},
        {1: "not-string-key"},
        (1, 2),
        {1, 2},
        Decimal("0.1"),
        object(),
        b"bytes",
    ],
)
def test_canonicalizer_rejects_nonwire_lossy_or_opaque_inputs(value):
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json

    with pytest.raises(ValueError):
        canonical_history_json(value)


def test_request_projection_deletes_only_nested_history_without_defaults_or_mutation():
    from tldw_Server_API.app.core.Chat.history_wire import history_request_projection, selected_durable_request_digest

    body = {
        "stream": False,
        "messages": [{"role": "user", "content": " original "}],
        "model": "m",
        "api_provider": "openai",
        "conversation_id": "chat",
        "save_to_db": True,
        "tldw_turn": {
            "user_message_id": "id",
            "history_v1": {"request_context_digest": "cycle"},
            "result_v1": {"version": 1, "sources": []},
        },
        "explicit_null": None,
        "nested": {"history_v1": "keep"},
        "history_v1": "keep",
    }
    before = deepcopy(body)
    expected = deepcopy(body)
    del expected["tldw_turn"]["history_v1"]
    projection = history_request_projection(body)
    assert projection == expected
    assert body == before
    assert selected_durable_request_digest(body) == node_vectors([expected])[0]["hash"]
    projection["tldw_turn"]["result_v1"]["sources"].append("detached")
    assert body == before
    variants = [deepcopy(body) for _ in range(3)]
    variants[0]["stream"] = True
    variants[1]["temperature"] = None
    variants[2]["temperature"] = 1
    assert len({selected_durable_request_digest(value) for value in [body, *variants]}) == 4
    body["tldw_turn"]["history_v1"] = {"anything": "excluded"}
    assert selected_durable_request_digest(body) == selected_durable_request_digest(before)


def test_cyclic_non_json_input_fails_closed():
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json

    value = []
    value.append(value)
    with pytest.raises(ValueError):
        canonical_history_json(value)


def test_request_projection_preserves_finite_float_scalar_types():
    from tldw_Server_API.app.core.Chat.history_wire import selected_durable_request_digest

    body = {"tldw_turn": {"history_v1": {}}, "extension_number": 1e20}
    assert (
        selected_durable_request_digest(body) == node_vectors([{"tldw_turn": {}, "extension_number": 1e20}])[0]["hash"]
    )


def test_seeded_finite_ieee754_corpus_matches_real_node():
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json, history_wire_digest

    # Deterministic test corpus, never a credential or security primitive.
    rng = random.Random(13398121)  # nosec B311
    values = [struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0] for _ in range(4096)]
    values = [value for value in values if math.isfinite(value)]
    for value, oracle in zip(values, node_vectors(values)[:-1], strict=True):
        assert canonical_history_json(value) == oracle["bytes"]
        assert history_wire_digest(value) == oracle["hash"]


def test_node_stringify_decoded_raw_body_corpus_preserves_number_bytes_and_request_hash():
    from tldw_Server_API.app.core.Chat.history_wire import (
        canonical_history_json,
        history_wire_digest,
        selected_durable_request_digest,
    )

    rng = random.Random(13398121)  # nosec B311 - deterministic numeric corpus only
    values = [struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0] for _ in range(4096)]
    values = [value for value in values if math.isfinite(value)]
    values += [
        1e20,
        -1e20,
        1e20 + 16384,
        295147905179352830000.0,
        999999999999999700000.0,
        9007199254740992.0,
        -9007199254740992.0,
        1e-6,
        1e-7,
        1e21,
        -0.0,
    ]
    bodies = [
        {
            "seed": value,
            "tldw_turn": {"history_v1": {"excluded": True}, "result_v1": {"version": 1, "sources": []}},
            "null": None,
        }
        for value in values
    ]
    for oracle in node_vectors(bodies)[:-1]:
        decoded = json.loads(oracle["wire"])
        assert canonical_history_json(decoded) == oracle["bytes"]
        assert history_wire_digest(decoded) == oracle["hash"]
        assert selected_durable_request_digest(decoded) == oracle["request_hash"]


def test_large_finite_source_scores_survive_node_wire_decode_schema_and_digest():
    from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import HistoryResultPayloadV1
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json, selected_durable_request_digest

    results = [
        {
            "version": 1,
            "sources": [
                {
                    "name": "Document",
                    "type": "pdf",
                    "mode": "rag",
                    "url": "",
                    "pageContent": "evidence",
                    "metadata": {"score": score},
                }
            ],
        }
        for score in [1e20, -1e20, 1e20 + 16384, 1.7976931348623157e308]
    ]
    bodies = [{"tldw_turn": {"history_v1": {"excluded": True}, "result_v1": result}} for result in results]
    for oracle, result_oracle in zip(node_vectors(bodies)[:-1], node_vectors(results)[:-1], strict=True):
        decoded = json.loads(oracle["wire"])
        payload = HistoryResultPayloadV1.model_validate(decoded["tldw_turn"]["result_v1"])
        assert canonical_history_json(payload.model_dump(mode="json")) == result_oracle["bytes"]
        assert selected_durable_request_digest(decoded) == oracle["request_hash"]


def test_source_locator_metadata_survives_schema_and_real_node_request_digest():
    from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import HistoryResultPayloadV1
    from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json, selected_durable_request_digest

    result = {
        "version": 1,
        "sources": [
            {
                "name": "Field memo",
                "type": "document",
                "mode": "rag",
                "url": "",
                "pageContent": "evidence",
                "metadata": {
                    "media_id": "1",
                    "author": " Mira Chen ",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "start_char": 0,
                    "end_char": 8,
                    "chunk_start": 2,
                    "chunk_end": 10,
                },
            }
        ],
    }
    body = {"tldw_turn": {"history_v1": {"excluded": True}, "result_v1": result}}
    oracle = node_vectors([body])[0]
    payload = HistoryResultPayloadV1.model_validate(result).model_dump(mode="json")
    assert payload == result
    assert canonical_history_json(payload) == node_vectors([result])[0]["bytes"]
    assert selected_durable_request_digest(body) == oracle["request_hash"]
    for key, value in result["sources"][0]["metadata"].items():
        changed = deepcopy(body)
        changed["tldw_turn"]["result_v1"]["sources"][0]["metadata"][key] = (
            f"{value}x" if isinstance(value, str) else value + 1
        )
        assert selected_durable_request_digest(changed) != oracle["request_hash"]
