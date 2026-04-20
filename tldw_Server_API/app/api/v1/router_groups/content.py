"""Content router group — media, RAG, embeddings, and content processing endpoints.

These routers handle content ingestion, search, retrieval, and
related operations.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec

API_V1_PREFIX = "/api/v1"


def iter_content_router_specs() -> Iterable[RouterSpec]:
    """Yield content/media-focused router specs."""
    specs: list[RouterSpec] = []

    # RAG unified endpoints (router has its own /api/v1/rag prefix)
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_unified import router as rag_unified_router

        specs.append(RouterSpec(
            router=rag_unified_router,
            tags=("rag-unified",),
            route_key="rag",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping rag_unified router: {e}")

    # RAG health endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router

        specs.append(RouterSpec(
            router=rag_health_router,
            tags=("rag-health",),
            route_key="rag-health",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping rag_health router: {e}")

    # Embeddings (OpenAI-compatible)
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            router as embeddings_router,
        )

        specs.append(RouterSpec(
            router=embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("embeddings",),
            route_key="embeddings",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping embeddings router: {e}")

    # Media embeddings
    try:
        from tldw_Server_API.app.api.v1.endpoints.media_embeddings import router as media_embeddings_router

        specs.append(RouterSpec(
            router=media_embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("media-embeddings",),
            route_key="media-embeddings",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping media_embeddings router: {e}")

    # Vector stores (OpenAI-compatible)
    try:
        from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router

        specs.append(RouterSpec(
            router=vector_stores_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vector-stores",),
            route_key="vector-stores",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping vector-stores router: {e}")

    # Chunking templates
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router

        specs.append(RouterSpec(
            router=chunking_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("chunking-templates",),
            route_key="chunking-templates",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chunking_templates router: {e}")

    # Prompts
    try:
        from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router

        specs.append(RouterSpec(
            router=prompt_router,
            prefix=f"{API_V1_PREFIX}/prompts",
            tags=("prompts",),
            route_key="prompts",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping prompts router: {e}")

    # Claims
    try:
        from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router

        specs.append(RouterSpec(
            router=claims_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("claims",),
            route_key="claims",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping claims router: {e}")

    # Text2SQL
    try:
        from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router

        specs.append(RouterSpec(
            router=text2sql_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("text2sql",),
            route_key="text2sql",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping text2sql router: {e}")

    # Email search
    try:
        from tldw_Server_API.app.api.v1.endpoints.email import router as email_router

        specs.append(RouterSpec(
            router=email_router,
            prefix=f"{API_V1_PREFIX}/email",
            tags=("email",),
            route_key="email",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping email router: {e}")

    # Outputs and output templates
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router

        specs.append(RouterSpec(
            router=outputs_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs-templates",),
            route_key="outputs-templates",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs_templates router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

        specs.append(RouterSpec(
            router=outputs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs",),
            route_key="outputs",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs router: {e}")

    return specs
