from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any, ClassVar

from tldw_Server_API.app.core.exceptions import FileArtifactsValidationError
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult
from tldw_Server_API.app.core.File_Artifacts.adapters.table_adapter_base import TableAdapterBase
from tldw_Server_API.app.core.File_Artifacts.adapters.xlsx_adapter import XlsxAdapter


class DataTableAdapter(TableAdapterBase):
    """Adapter for data_table payloads with CSV/JSON/XLSX exports."""

    file_type: ClassVar[str] = "data_table"
    export_formats: ClassVar[set[str]] = {"csv", "json", "xlsx"}
    validation_error: ClassVar[type[Exception]] = FileArtifactsValidationError

    def export(self, structured: dict[str, Any], *, format: str) -> ExportResult:
        """Export structured table data in the requested format."""
        if format == "csv":
            return self._export_csv(structured)
        if format == "json":
            return self._export_json(structured)
        if format == "xlsx":
            return self._export_xlsx(structured)
        raise FileArtifactsValidationError("unsupported_format")

    def _export_csv(self, structured: dict[str, Any]) -> ExportResult:
        """Export structured table data to CSV bytes."""
        columns = structured.get("columns") or []
        rows = structured.get("rows") or []
        buf = StringIO(newline="")
        writer = csv.writer(buf)
        writer.writerow([self._sanitize_cell(c) for c in columns])
        for row in rows:
            writer.writerow([self._sanitize_cell(c) for c in row])
        data = buf.getvalue().encode("utf-8")
        return ExportResult(status="ready", content_type="text/csv", bytes_len=len(data), content=data)

    def _export_json(self, structured: dict[str, Any]) -> ExportResult:
        """Export structured table data to JSON bytes."""
        columns = structured.get("columns") or []
        rows = structured.get("rows") or []
        if self._has_duplicate_columns(columns):
            raise FileArtifactsValidationError("duplicate_columns")
        try:
            payload = [dict(zip(columns, row, strict=True)) for row in rows]
        except ValueError as exc:
            raise FileArtifactsValidationError("row_length_mismatch") from exc
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        return ExportResult(status="ready", content_type="application/json", bytes_len=len(data), content=data)

    def _export_xlsx(self, structured: dict[str, Any]) -> ExportResult:
        """Export structured table data to XLSX bytes via XlsxAdapter."""
        columns = structured.get("columns") or []
        rows = structured.get("rows") or []
        adapter = XlsxAdapter()
        normalized = adapter.normalize({"columns": columns, "rows": rows})
        return adapter.export(normalized, format="xlsx")

    @staticmethod
    def _sanitize_cell(value: Any) -> str:
        """Normalize a CSV cell and mitigate formula injection."""
        if value is None:
            return ""
        text = str(value).replace("\n", " ")
        stripped = text.lstrip()
        if stripped.startswith(("=", "+", "-", "@")):
            return "'" + text
        return text
