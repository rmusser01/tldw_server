"""Nonmutating replay of only the 13 original UAT222 methods in pytest."""
import ast
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def restore_original_study_pack_methods():
    import tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB as module

    tree = ast.parse(Path(__file__).with_name("baseline-ChaChaNotes_DB.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CharactersRAGDB")
    names = {
        "_require_selected_owner_row", "create_study_pack", "get_study_pack", "add_study_pack_cards",
        "list_study_pack_cards", "add_flashcard_citations", "replace_flashcard_citations",
        "replace_flashcard_citations_and_source_reference_summary", "list_flashcard_citations",
        "set_flashcard_source_reference_summary", "get_study_pack_for_flashcard",
        "soft_delete_study_pack", "supersede_study_pack",
    }
    originals = {}
    try:
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                namespace = dict(module.__dict__)
                exec(compile(ast.Module(body=[node], type_ignores=[]), "uat222-baseline-method", "exec"), namespace)
                originals[node.name] = getattr(module.CharactersRAGDB, node.name)
                setattr(module.CharactersRAGDB, node.name, namespace[node.name])
        assert originals.keys() == names
        yield
    finally:
        for name, method in originals.items():
            setattr(module.CharactersRAGDB, name, method)
