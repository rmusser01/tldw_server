"""
Prompt Studio Test Cases API

Owns the test corpus for a project. Test cases provide inputs,
expected outputs, and metadata (tags, golden flag) to enable
repeatable evaluation and optimization of prompts.

Key responsibilities
- CRUD for individual and bulk test cases
- Import/export test cases (CSV/JSON)
- Generate test cases from descriptions or signatures
- Filter/search/paginate lists by tags, golden status, signatures

Security
- Read operations require project access
- Write operations require project write access
- Rate limits applied to generation endpoints
"""

import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    PromptStudioDatabase,
    SecurityConfig,
    check_rate_limit,
    get_prompt_studio_db,
    get_prompt_studio_user,
    get_security_config,
    require_project_access,
    require_project_write_access,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_page_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import (
    ListResponse,
    PageListResponse,
    StandardResponse,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import (
    RunTestCasesSimpleRequest,
    TestCaseBulkCreate,
    TestCaseCreate,
    TestCaseExportRequest,
    TestCaseGenerateRequest,
    TestCaseImportRequest,
    TestCaseResponse,
    TestCaseUpdate,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_generator import TestCaseGenerator
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_io import TestCaseIO
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_manager import TestCaseManager
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

PROMPT_STUDIO_TEST_CASE_EXCEPTIONS = (
    DatabaseError,
    InputError,
    ConflictError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt-studio/test-cases",
    tags=["Prompt Studio"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

########################################################################################################################
# Test Case CRUD Endpoints

# Compatibility: simple POST on base path returns test case object directly
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_test_case_simple(
    test_case_data: TestCaseCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> dict[str, Any]:
    resp = await create_test_case(test_case_data, db, security_config, user_context)  # type: ignore[arg-type]
    # Unwrap StandardResponse regardless of Pydantic/dict
    obj = resp.model_dump() if hasattr(resp, "model_dump") else resp if isinstance(resp, dict) else {}
    data = obj.get("data", obj)
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data

@router.post(
    "/create",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Create a test case",
                            "value": {
                                "project_id": 1,
                                "name": "Short text",
                                "inputs": {"text": "Hello world"},
                                "expected_outputs": {"summary": "Hello world."},
                                "tags": ["smoke"],
                                "is_golden": True
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Test case created",
                "content": {
                    "application/json": {
                        "examples": {
                            "created": {
                                "summary": "Created test case",
                                "value": {
                                    "success": True,
                                    "data": {
                                        "id": 101,
                                        "project_id": 1,
                                        "name": "Short text",
                                        "is_golden": True,
                                        "created_at": "2024-09-20T10:00:00"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_test_case(
    test_case_data: TestCaseCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create a new test case.

    Args:
        test_case_data: Test case creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        Created test case details
    """
    try:
        # Enforce project write access
        await require_project_write_access(test_case_data.project_id, user_context=user_context, db=db)

        # Check test case limit
        manager = TestCaseManager(db)
        current_count = manager.get_test_case_stats(test_case_data.project_id)["total"]

        if current_count >= security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Project has reached maximum of {security_config.max_test_cases} test cases"
            )

        # Create test case
        test_case = manager.create_test_case(
            project_id=test_case_data.project_id,
            name=test_case_data.name,
            description=test_case_data.description,
            inputs=test_case_data.inputs,
            expected_outputs=test_case_data.expected_outputs,
            tags=test_case_data.tags,
            is_golden=test_case_data.is_golden,
            signature_id=test_case_data.signature_id
        )

        logger.info(f"User {user_context['user_id']} created test case: {test_case.get('name', 'Unnamed')}")

        return StandardResponse(
            success=True,
            data=TestCaseResponse(**test_case)
        )

    except ConflictError as e:
        logger.warning("Conflict creating test case")
        raise map_db_error_to_http(
            e,
            conflict_detail="Test case already exists",
        ) from e
    except DatabaseError as e:
        logger.error("Database error creating test case")
        raise map_db_error_to_http(e, default_detail="Failed to create test case") from e

@router.post(
    "/bulk",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "bulk": {
                            "summary": "Bulk create",
                            "value": {
                                "project_id": 1,
                                "test_cases": [
                                    {"name": "Short", "inputs": {"text": "Hi"}, "expected_outputs": {"summary": "Hi."}},
                                    {"name": "Long", "inputs": {"text": "..."}}
                                ]
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Bulk created",
                "content": {
                    "application/json": {
                        "examples": {
                            "created": {
                                "summary": "Bulk response",
                                "value": {"success": True, "data": [{"id": 102}, {"id": 103}]}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_bulk_test_cases(
    bulk_data: TestCaseBulkCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create multiple test cases at once.

    Args:
        bulk_data: Bulk test case creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        List of created test cases
    """
    try:
        # Enforce project write access
        await require_project_write_access(bulk_data.project_id, user_context=user_context, db=db)

        manager = TestCaseManager(db)

        # Check test case limit
        current_count = manager.get_test_case_stats(bulk_data.project_id)["total"]
        new_total = current_count + len(bulk_data.test_cases)

        if new_total > security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Would exceed maximum of {security_config.max_test_cases} test cases"
            )

        # Create test cases
        serialized_cases: list[dict[str, Any]] = []
        for tc in bulk_data.test_cases:
            try:
                serialized_cases.append(model_dump_compat(tc))
            except TypeError:
                encoded_case = jsonable_encoder(tc)
                if not isinstance(encoded_case, dict):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid test case payload"
                    ) from None
                serialized_cases.append(encoded_case)

        test_cases = manager.create_bulk_test_cases(
            project_id=bulk_data.project_id,
            test_cases=serialized_cases,
            signature_id=bulk_data.signature_id
        )

        logger.info(f"User {user_context['user_id']} created {len(test_cases)} test cases in bulk")

        return StandardResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in test_cases]
        )

    except DatabaseError as e:
        logger.error("Database error creating bulk test cases")
        raise map_db_error_to_http(e, default_detail="Failed to create test cases") from e

@router.get(
    "/list/{project_id}",
    response_model=PageListResponse,
    openapi_extra={
        "responses": {
            "200": {
                "description": "Test cases",
                "content": {
                    "application/json": {
                        "examples": {
                            "list": {
                                "summary": "Cases",
                                "value": {
                                    "success": True,
                                    "data": [
                                        {"id": 101, "name": "Short text"}
                                    ],
                                    "metadata": {
                                        "page": 1,
                                        "per_page": 20,
                                        "total": 1,
                                        "total_pages": 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def list_test_cases(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    is_golden: Optional[bool] = Query(None, description="Filter by golden status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    signature_id: Optional[int] = Query(None, description="Filter by signature"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> PageListResponse:
    """
    List test cases in a project.

    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        is_golden: Filter by golden status
        tags: Filter by tags
        search: Search query
        signature_id: Filter by signature
        db: Database instance

    Returns:
        Paginated list of test cases
    """
    try:
        manager = TestCaseManager(db)

        # Parse tags
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Get test cases
        result = manager.list_test_cases(
            project_id=project_id,
            signature_id=signature_id,
            is_golden=is_golden,
            tags=tag_list,
            search=search,
            page=page,
            per_page=per_page,
            return_pagination=True
        )

        return PageListResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in result["test_cases"]],
            metadata=result["pagination"],
            pagination=build_page_pagination_meta(
                page=result["pagination"]["page"],
                per_page=result["pagination"]["per_page"],
                total=result["pagination"]["total"],
                total_pages=result["pagination"]["total_pages"],
            ),
        )

    except DatabaseError as e:
        detail = "Failed to list test cases"
        logger.error("Database error listing test cases")
        raise map_db_error_to_http(e, default_detail=detail) from e
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error listing test cases")
        detail = "Failed to list test cases"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from e

@router.get("/get/{test_case_id}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Test case", "content": {"application/json": {"examples": {"get": {"summary": "Case", "value": {"success": True, "data": {"id": 101, "name": "Short text", "is_golden": True}}}}}}}}
})
async def get_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Get a specific test case by ID.

    Args:
        test_case_id: Test case ID
        db: Database instance

    Returns:
        Test case details
    """
    try:
        manager = TestCaseManager(db)
        test_case = manager.get_test_case(test_case_id)

        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )

        # Check project access
        await require_project_access(test_case["project_id"], user_context=user_context, db=db)

        return StandardResponse(
            success=True,
            data=TestCaseResponse(**test_case)
        )

    except DatabaseError as e:
        logger.error("Database error getting test case")
        raise map_db_error_to_http(e, default_detail="Failed to get test case") from e

@router.put("/update/{test_case_id}", response_model=StandardResponse, openapi_extra={
    "responses": {
        "200": {"description": "Test case updated", "content": {"application/json": {"examples": {"updated": {"summary": "Updated case", "value": {"success": True, "data": {"id": 101, "name": "Short text (v2)", "is_golden": True}}}}}}},
        "404": {"description": "Not found"}
    }
})
async def update_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    updates: TestCaseUpdate = ...,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Update a test case.

    Args:
        test_case_id: Test case ID
        updates: Fields to update
        db: Database instance
        user_context: Current user context

    Returns:
        Updated test case details
    """
    try:
        manager = TestCaseManager(db)

        # Get test case to check project
        test_case = manager.get_test_case(test_case_id)
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )

        # Check project access
        await require_project_write_access(test_case["project_id"], user_context=user_context, db=db)

        # Update test case
        try:
            update_data = model_dump_compat(updates, exclude_none=True)
        except TypeError:
            encoded_update = jsonable_encoder(updates)
            update_data = (
                {k: v for k, v in encoded_update.items() if v is not None}
                if isinstance(encoded_update, dict)
                else {}
            )

        updated = manager.update_test_case(test_case_id, update_data)

        logger.info(f"User {user_context['user_id']} updated test case {test_case_id}")

        return StandardResponse(
            success=True,
            data=TestCaseResponse(**updated)
        )

    except InputError as e:
        raise map_db_error_to_http(
            e,
            input_status_code=status.HTTP_404_NOT_FOUND,
        ) from e
    except DatabaseError as e:
        logger.error("Database error updating test case")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to update test case",
        ) from e

@router.delete("/delete/{test_case_id}", response_model=StandardResponse, openapi_extra={
    "responses": {
        "200": {"description": "Deleted", "content": {"application/json": {"examples": {"deleted": {"value": {"success": True, "data": {"message": "Test case soft deleted"}}}}}}},
        "404": {"description": "Not found"}
    }
})
async def delete_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    permanent: bool = Query(False, description="Permanently delete"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Delete a test case.

    Args:
        test_case_id: Test case ID
        permanent: If True, permanently delete
        db: Database instance
        user_context: Current user context

    Returns:
        Success response
    """
    try:
        manager = TestCaseManager(db)

        # Get test case to check project
        test_case = manager.get_test_case(test_case_id)
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )

        # Check project access
        await require_project_write_access(test_case["project_id"], user_context=user_context, db=db)

        # Delete test case
        success = manager.delete_test_case(test_case_id, hard_delete=permanent)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test case not found or already deleted"
            )

        logger.info(
            f"User {user_context['user_id']} {'permanently' if permanent else 'soft'} "
            f"deleted test case {test_case_id}"
        )

        return StandardResponse(
            success=True,
            data={"message": f"Test case {'permanently' if permanent else 'soft'} deleted"}
        )

    except DatabaseError as e:
        logger.error("Database error deleting test case")
        raise map_db_error_to_http(e, default_detail="Failed to delete test case") from e

########################################################################################################################
# Import/Export Endpoints

@router.post(
    "/import",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "json": {
                            "summary": "Import from JSON",
                            "value": {
                                "project_id": 1,
                                "format": "json",
                                "data": '[{"name":"Short","inputs":{"text":"Hi"}}]',
                                "auto_generate_names": True
                            }
                        },
                        "csv": {
                            "summary": "Import from CSV (inline string)",
                            "value": {
                                "project_id": 1,
                                "format": "csv",
                                "data": 'name,inputs,expected_outputs\nShort,"{""text"":""Hi""}","{""summary"":""Hi.""}"'
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Import result",
                "content": {
                    "application/json": {
                        "examples": {
                            "result": {
                                "summary": "Imported count",
                                "value": {"success": True, "data": {"imported": 2, "errors": [], "total_test_cases": 10}}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def import_test_cases(
    import_data: TestCaseImportRequest,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Import test cases from CSV or JSON.

    Args:
        import_data: Import request data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        Import results
    """
    try:
        # Ensure write access
        await require_project_write_access(import_data.project_id, user_context=user_context, db=db)

        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)

        # Check test case limit
        current_count = manager.get_test_case_stats(import_data.project_id)["total"]

        # Import based on format
        if import_data.format == "csv":
            imported, errors = io_manager.import_from_csv(
                project_id=import_data.project_id,
                csv_data=import_data.data,
                signature_id=import_data.signature_id,
                auto_generate_names=import_data.auto_generate_names
            )
        else:  # json
            imported, errors = io_manager.import_from_json(
                project_id=import_data.project_id,
                json_data=import_data.data,
                signature_id=import_data.signature_id,
                auto_generate_names=import_data.auto_generate_names
            )

        # Check if we exceeded the limit
        new_total = current_count + imported
        if new_total > security_config.max_test_cases:
            logger.warning("Import would exceed test case limit")

        logger.info(f"User {user_context['user_id']} imported {imported} test cases")

        return StandardResponse(
            success=True,
            data={
                "imported": imported,
                "errors": errors,
                "total_test_cases": new_total
            }
        )

    except (DatabaseError, InputError, ConflictError) as e:
        logger.error("Error importing test cases")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to import test cases",
        ) from e
    except PROMPT_STUDIO_TEST_CASE_EXCEPTIONS as e:
        logger.error("Error importing test cases")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import test cases"
        ) from e


@router.post(
    "/import/csv-upload",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "integer"},
                            "signature_id": {"type": "integer"},
                            "auto_generate_names": {"type": "boolean"},
                            "file": {"type": "string", "format": "binary"}
                        },
                        "required": ["project_id", "file"]
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Import result",
                "content": {
                    "application/json": {
                        "examples": {
                            "result": {
                                "summary": "CSV Import",
                                "value": {"success": True, "data": {"imported": 3, "errors": [], "total_test_cases": 15}}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def import_test_cases_csv_upload(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    signature_id: Optional[int] = Form(None),
    auto_generate_names: bool = Form(True),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """Import test cases from a CSV file upload (multipart/form-data)."""
    try:
        await require_project_write_access(project_id, user_context=user_context, db=db)

        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)

        # Read file content
        content_bytes = await file.read()
        try:
            csv_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            csv_text = content_bytes.decode("latin-1")

        imported, errors = io_manager.import_from_csv(
            project_id=project_id,
            csv_data=csv_text,
            signature_id=signature_id,
            auto_generate_names=auto_generate_names
        )

        # Count current total
        current_total = manager.get_test_case_stats(project_id)["total"]

        return StandardResponse(
            success=True,
            data={
                "imported": imported,
                "errors": errors,
                "total_test_cases": current_total
            }
        )
    except (DatabaseError, InputError, ConflictError) as e:
        logger.error("Error importing test cases via upload")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to import CSV test cases",
        ) from e
    except PROMPT_STUDIO_TEST_CASE_EXCEPTIONS as e:
        logger.error("Error importing test cases via upload")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to import CSV test cases") from e


@router.get(
    "/import/template",
    response_class=Response,
    openapi_extra={
        "responses": {
            "200": {
                "description": "CSV template",
                "content": {
                    "text/csv": {
                        "examples": {
                            "template": {
                                "summary": "CSV template example",
                                "value": "name,description,input.text,expected.summary,tags,is_golden\n"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_csv_import_template(
    signature_id: Optional[int] = Query(None, description="Optional signature id to derive input/output columns") ,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> Response:
    """Download a CSV template for test case import.

    If a signature_id is provided, the template will include input.* and expected.*
    columns derived from the signature schemas. Otherwise, a minimal template is returned.
    """
    try:
        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)
        csv_text = io_manager.generate_csv_template(signature_id=signature_id)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prompt_studio_test_cases_template.csv"}
        )
    except (DatabaseError, InputError, ConflictError) as e:
        logger.error("Failed to generate CSV template")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to generate CSV template",
        ) from e
    except PROMPT_STUDIO_TEST_CASE_EXCEPTIONS as e:
        logger.error("Failed to generate CSV template")
        raise HTTPException(status_code=500, detail="Failed to generate CSV template") from e

# Compatibility: run test cases endpoint returning {"results": [...]}
@router.post("/run")
async def run_test_cases_simple(
    payload: RunTestCasesSimpleRequest,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> dict[str, Any]:
    manager = TestCaseManager(db)
    prompt_id = int(payload.prompt_id)
    test_case_ids = payload.test_case_ids or []
    model = payload.model or "gpt-3.5-turbo"
    # Enforce access to the prompt's project before running tests
    project_id: Optional[int] = None
    # Prefer explicit project_id from payload for compatibility with older clients/tests
    if getattr(payload, "project_id", None):
        try:
            project_id = int(payload.project_id)  # type: ignore[attr-defined]
        except (TypeError, ValueError, AttributeError):
            project_id = None
    if project_id is None:
        # Fallback: resolve project via prompt record
        try:
            prompt_row = db.get_prompt(prompt_id)
            project_id = int(prompt_row["project_id"]) if prompt_row and "project_id" in prompt_row else None
        except (DatabaseError, InputError, KeyError, TypeError, ValueError):
            project_id = None
    if project_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Prompt {prompt_id} not found or no project context")
    await require_project_access(project_id, user_context=user_context, db=db)
    results = await manager.run_batch_tests(prompt_id=prompt_id, test_case_ids=test_case_ids, model=model)
    return {"results": results}

@router.post(
    "/export/{project_id}",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "json": {
                            "summary": "Export as JSON",
                            "value": {
                                "format": "json",
                                "include_golden_only": False,
                                "tag_filter": ["smoke"]
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Export payload",
                "content": {
                    "application/json": {
                        "examples": {
                            "json": {
                                "summary": "JSON export",
                                "value": {"success": True, "data": {"format": "json", "data": [{"name": "Short"}]}}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def export_test_cases(
    project_id: int = Path(..., description="Project ID"),
    export_request: TestCaseExportRequest = ...,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Export test cases to CSV or JSON.

    Args:
        project_id: Project ID
        export_request: Export configuration
        db: Database instance

    Returns:
        Exported data
    """
    try:
        # Ensure read access
        await require_project_access(project_id, user_context=user_context, db=db)

        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)

        # Export based on format
        if export_request.format == "csv":
            data = io_manager.export_to_csv(
                project_id=project_id,
                include_golden_only=export_request.include_golden_only,
                tag_filter=export_request.tag_filter
            )
        else:  # json
            data = io_manager.export_to_json(
                project_id=project_id,
                include_golden_only=export_request.include_golden_only,
                tag_filter=export_request.tag_filter
            )

        return StandardResponse(
            success=True,
            data={
                "format": export_request.format,
                "data": data,
                "content_type": "text/csv" if export_request.format == "csv" else "application/json"
            }
        )

    except (DatabaseError, InputError, ConflictError) as e:
        logger.error("Error exporting test cases")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to export test cases",
        ) from e
    except PROMPT_STUDIO_TEST_CASE_EXCEPTIONS as e:
        logger.error("Error exporting test cases")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export test cases"
        ) from e

########################################################################################################################
# Generation Endpoints

@router.post(
    "/generate",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "diverse": {
                            "summary": "Generate cases (diverse)",
                            "value": {
                                "project_id": 1,
                                "signature_id": 2,
                                "num_cases": 5,
                                "generation_strategy": "diverse"
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Generated test cases",
                "content": {
                    "application/json": {
                        "examples": {
                            "generated": {
                                "summary": "Generated examples",
                                "value": {"success": True, "data": [{"id": 110, "is_generated": True}]}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def _rl_generate(
    user_context: dict = Depends(get_prompt_studio_user),
    security_config: SecurityConfig = Depends(get_security_config),
) -> bool:
    return await check_rate_limit("generate", user_context=user_context, security_config=security_config)

async def generate_test_cases(
    generate_request: TestCaseGenerateRequest,
    _rate: bool = Depends(_rl_generate),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Auto-generate test cases.

    Args:
        generate_request: Generation configuration
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        Generated test cases
    """
    try:
        # Enforce project write access
        await require_project_write_access(generate_request.project_id, user_context=user_context, db=db)

        manager = TestCaseManager(db)
        generator = TestCaseGenerator(manager)

        # Check test case limit
        current_count = manager.get_test_case_stats(generate_request.project_id)["total"]
        new_total = current_count + generate_request.num_cases

        if new_total > security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Would exceed maximum of {security_config.max_test_cases} test cases"
            )

        # Generate based on strategy
        if generate_request.generation_strategy == "diverse" and generate_request.signature_id:
            generated = generator.generate_diverse_cases(
                project_id=generate_request.project_id,
                signature_id=generate_request.signature_id,
                num_cases=generate_request.num_cases
            )
        elif generate_request.base_on_description:
            generated = generator.generate_from_description(
                project_id=generate_request.project_id,
                description=generate_request.base_on_description,
                num_cases=generate_request.num_cases,
                signature_id=generate_request.signature_id,
                prompt_id=generate_request.prompt_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either signature_id for diverse generation or base_on_description"
            )

        logger.info(f"User {user_context['user_id']} generated {len(generated)} test cases")

        return StandardResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in generated]
        )

    except ValueError as e:
        logger.warning("Invalid test case generation request")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test case generation request"
        ) from e
    except (DatabaseError, InputError, ConflictError) as e:
        logger.error("Error generating test cases")
        raise map_db_error_to_http(
            e,
            default_detail="Failed to generate test cases",
        ) from e
    except PROMPT_STUDIO_TEST_CASE_EXCEPTIONS as e:
        logger.error("Error generating test cases")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate test cases"
        ) from e
