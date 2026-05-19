from __future__ import annotations

from io import BytesIO
from typing import Any, ClassVar

from tldw_Server_API.app.core.exceptions import FileArtifactsError, FileArtifactsValidationError
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult, ValidationIssue


class XlsxAdapter:
    file_type = "xlsx"
    export_formats: ClassVar[set[str]] = {"xlsx"}

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "sheets" not in payload and "columns" in payload and "rows" in payload:
            payload = {
                "sheets": [
                    {
                        "name": payload.get("name") or "Sheet1",
                        "columns": payload.get("columns"),
                        "rows": payload.get("rows"),
                    }
                ]
            }
        sheets = payload.get("sheets")
        if not isinstance(sheets, list):
            raise FileArtifactsValidationError("sheets_required")
        normalized_sheets = []
        for sheet in sheets:
            if not isinstance(sheet, dict):
                raise FileArtifactsValidationError("sheet_must_be_object")
            name = sheet.get("name") or "Sheet1"
            columns = sheet.get("columns")
            rows = sheet.get("rows")
            if columns is None or rows is None:
                raise FileArtifactsValidationError("columns_and_rows_required")
            if not isinstance(columns, list) or not isinstance(rows, list):
                raise FileArtifactsValidationError("columns_rows_must_be_lists")
            normalized_sheets.append(
                {
                    "name": str(name),
                    "columns": [str(col) for col in columns],
                    "rows": [list(row) if isinstance(row, (list, tuple)) else row for row in rows],
                }
            )
        return {"sheets": normalized_sheets}

    def validate(self, structured: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        sheets = structured.get("sheets")
        if not isinstance(sheets, list) or not sheets:
            issues.append(ValidationIssue(code="sheets_required", message="sheets must be a non-empty list", path="sheets"))
            return issues
        for s_idx, sheet in enumerate(sheets):
            if not isinstance(sheet, dict):
                issues.append(ValidationIssue(code="sheet_invalid", message="sheet must be an object", path=f"sheets[{s_idx}]"))
                continue
            name = sheet.get("name")
            if not name or not isinstance(name, str):
                issues.append(ValidationIssue(code="sheet_name_required", message="sheet name is required", path=f"sheets[{s_idx}].name"))
            elif len(name) > 31:
                issues.append(ValidationIssue(code="sheet_name_too_long", message="sheet name must be <= 31 characters", path=f"sheets[{s_idx}].name"))
            columns = sheet.get("columns")
            rows = sheet.get("rows")
            if not isinstance(columns, list) or not columns:
                issues.append(ValidationIssue(code="columns_required", message="columns must be a non-empty list", path=f"sheets[{s_idx}].columns"))
                continue
            if not isinstance(rows, list):
                issues.append(ValidationIssue(code="rows_required", message="rows must be a list", path=f"sheets[{s_idx}].rows"))
                continue
            if self._has_duplicate_columns(columns):
                issues.append(
                    ValidationIssue(
                        code="duplicate_columns",
                        message="columns contain duplicates",
                        path=f"sheets[{s_idx}].columns",
                        level="warning",
                    )
                )
            col_len = len(columns)
            for r_idx, row in enumerate(rows):
                if not isinstance(row, list):
                    issues.append(ValidationIssue(code="row_invalid", message="row must be a list", path=f"sheets[{s_idx}].rows[{r_idx}]"))
                    continue
                if len(row) != col_len:
                    issues.append(
                        ValidationIssue(
                            code="row_length_mismatch",
                            message="row length must match columns length",
                            path=f"sheets[{s_idx}].rows[{r_idx}]",
                        )
                    )
        return issues

    def export(self, structured: dict[str, Any], *, format: str) -> ExportResult:
        if format != "xlsx":
            raise FileArtifactsValidationError("unsupported_format")
        try:
            from openpyxl import Workbook
        except Exception as exc:
            raise FileArtifactsError("xlsx_export_unavailable", detail="xlsx export backend unavailable") from exc

        sheets = structured.get("sheets") or []
        wb = Workbook()
        default = wb.active
        if sheets:
            # Remove default sheet to avoid empty extra sheet when we add custom sheets.
            wb.remove(default)
        else:
            # Ensure there's always at least one worksheet for empty payloads.
            default.title = "Sheet1"

        for sheet in sheets:
            name = sheet.get("name") or "Sheet1"
            ws = wb.create_sheet(title=str(name)[:31])
            columns = sheet.get("columns") or []
            rows = sheet.get("rows") or []
            ws.append([self._sanitize_spreadsheet_value(c) for c in columns])
            for row in rows:
                ws.append([self._sanitize_spreadsheet_value(c) for c in row])

        buf = BytesIO()
        wb.save(buf)
        data = buf.getvalue()
        return ExportResult(
            status="ready",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            bytes_len=len(data),
            content=data,
        )

    @staticmethod
    def _sanitize_spreadsheet_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            stripped = value.lstrip()
            if stripped.startswith(("=", "+", "-", "@")):
                return "'" + value
        return value

    @staticmethod
    def _has_duplicate_columns(columns: list[Any]) -> bool:
        seen = set()
        for col in columns:
            if col in seen:
                return True
            seen.add(col)
        return False
