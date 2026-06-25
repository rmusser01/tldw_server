from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupItem


@dataclass(frozen=True, slots=True)
class SessionContext:
    text: str
    diagnostics: dict[str, object]


class SessionContextBuilder:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(1, int(max_chars))

    def build(
        self,
        adapter_key: str,
        session_title: str,
        snapshot: RPGSnapshotState,
        rules_results: list[RuleLookupItem],
    ) -> SessionContext:
        lines: list[str] = []
        omitted_sections: list[str] = []

        def append_line(line: str, section: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return True
            candidate_length = len(stripped) if not lines else sum(len(existing) + 1 for existing in lines) + len(stripped)
            if candidate_length > self.max_chars:
                if section not in omitted_sections:
                    omitted_sections.append(section)
                return False
            lines.append(stripped)
            return True

        append_line(f"RPG session: {session_title}", "header")
        append_line(f"Rules adapter: {adapter_key}", "header")

        scene_summary = str(snapshot.scene.get("summary") or "").strip()
        if scene_summary:
            append_line(f"Scene: {scene_summary}", "scene")

        npc_names = sorted(str(npc.get("name") or npc_id) for npc_id, npc in list(snapshot.npcs.items())[:20])
        if npc_names:
            append_line(f"NPCs: {', '.join(npc_names)}", "npcs")

        if snapshot.notes:
            notes_open = append_line("Recent notes:", "notes")
            for note in snapshot.notes[-5:]:
                text = str(note.get("text") or note.get("summary") or "").strip()
                if text and (not notes_open or not append_line(f"- {text}", "notes")):
                    break

        if snapshot.unresolved_rulings:
            append_line(f"Open rulings: {len(snapshot.unresolved_rulings)}", "rulings")

        if rules_results:
            citations_open = append_line("Rules citations:", "rules")
            for item in rules_results:
                citation = item.citation
                if not citations_open or not append_line(f"- {citation.source_title}: {citation.source_url}", "rules"):
                    break

        text = "\n".join(line for line in lines if line.strip())
        original_length = len(text)
        truncated = bool(omitted_sections)

        return SessionContext(
            text=text,
            diagnostics={
                "truncated": truncated,
                "max_chars": self.max_chars,
                "original_chars": original_length,
                "returned_chars": len(text),
                "rules_result_count": len(rules_results),
                "omitted_sections": omitted_sections,
            },
        )
