from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, selectinload, sessionmaker

from tldw_Server_API.app.core.DB_Management.db_path_utils import resolve_trusted_database_path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class EmbeddingABTest(Base):
    __tablename__ = "embedding_abtests"

    test_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    stats_json: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    arms: Mapped[list[EmbeddingABTestArm]] = relationship(
        back_populates="test", cascade="all, delete-orphan", order_by="EmbeddingABTestArm.arm_index"
    )
    queries: Mapped[list[EmbeddingABTestQuery]] = relationship(
        back_populates="test", cascade="all, delete-orphan", order_by="EmbeddingABTestQuery.created_at"
    )
    results: Mapped[list[EmbeddingABTestResult]] = relationship(
        back_populates="test", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_abtests_created", created_at.desc()),
        Index("idx_abtests_status", status),
    )


class EmbeddingABTestArm(Base):
    __tablename__ = "embedding_abtest_arms"

    arm_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    test_id: Mapped[str] = mapped_column(ForeignKey("embedding_abtests.test_id"), nullable=False)
    arm_index: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int | None] = mapped_column(Integer)
    collection_hash: Mapped[str | None] = mapped_column(String(128))
    pipeline_hash: Mapped[str | None] = mapped_column(String(128))
    collection_name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    stats_json: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text)

    test: Mapped[EmbeddingABTest] = relationship(back_populates="arms")
    results: Mapped[list[EmbeddingABTestResult]] = relationship(back_populates="arm", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_abtest_arms_test", "test_id"),
    )


class EmbeddingABTestQuery(Base):
    __tablename__ = "embedding_abtest_queries"

    query_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    test_id: Mapped[str] = mapped_column(ForeignKey("embedding_abtests.test_id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_ids: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    test: Mapped[EmbeddingABTest] = relationship(back_populates="queries")
    results: Mapped[list[EmbeddingABTestResult]] = relationship(back_populates="query", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_abtest_queries_test", "test_id"),
    )


class EmbeddingABTestResult(Base):
    __tablename__ = "embedding_abtest_results"

    result_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    test_id: Mapped[str] = mapped_column(ForeignKey("embedding_abtests.test_id"), nullable=False)
    arm_id: Mapped[str] = mapped_column(ForeignKey("embedding_abtest_arms.arm_id"), nullable=False)
    query_id: Mapped[str] = mapped_column(ForeignKey("embedding_abtest_queries.query_id"), nullable=False)
    ranked_ids: Mapped[str] = mapped_column(Text, nullable=False)
    scores: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    ranked_distances: Mapped[str | None] = mapped_column(Text)
    ranked_metadatas: Mapped[str | None] = mapped_column(Text)
    ranked_documents: Mapped[str | None] = mapped_column(Text)
    rerank_scores: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    test: Mapped[EmbeddingABTest] = relationship(back_populates="results")
    arm: Mapped[EmbeddingABTestArm] = relationship(back_populates="results")
    query: Mapped[EmbeddingABTestQuery] = relationship(back_populates="results")

    __table_args__ = (
        Index("idx_abtest_results_test", "test_id"),
        Index("idx_abtest_results_arm", "arm_id"),
        Index("idx_abtest_results_query", "query_id"),
    )


@dataclass
class RepositoryConfig:
    db_url: str
    echo: bool = False


class EmbeddingABTestRepository:
    """SQLAlchemy-backed repository for embeddings A/B tests."""

    def __init__(self, engine: Engine, session_factory: sessionmaker[Session]) -> None:
        self._engine = engine
        self._session_factory = session_factory

    @classmethod
    def from_config(cls, config: RepositoryConfig) -> EmbeddingABTestRepository:
        engine = create_engine(config.db_url, echo=config.echo, future=True)
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        return cls(engine, factory)

    @contextmanager
    def session_scope(self) -> Iterable[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_test(
        self,
        *,
        test_id: str,
        name: str,
        created_by: str | None,
        config: dict[str, Any],
        status: str = "pending",
        notes: str | None = None,
    ) -> EmbeddingABTest:
        with self.session_scope() as session:
            entity = EmbeddingABTest(
                test_id=test_id,
                name=name,
                created_by=created_by,
                status=status,
                notes=notes,
                config_json=json.dumps(config, separators=(",", ":"), sort_keys=True),
            )
            session.add(entity)
            session.flush()
            session.refresh(entity)
            if entity.created_at.tzinfo is None:
                entity.created_at = entity.created_at.replace(tzinfo=timezone.utc)
            return entity

    def add_arm(
        self,
        *,
        arm_id: str,
        test_id: str,
        arm_index: int,
        provider: str,
        model_id: str,
        dimensions: int | None = None,
        collection_hash: str | None = None,
        pipeline_hash: str | None = None,
        collection_name: str | None = None,
        status: str = "pending",
        stats: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingABTestArm:
        with self.session_scope() as session:
            arm = session.get(EmbeddingABTestArm, arm_id)
            payload_stats = json.dumps(stats, sort_keys=True) if stats else None
            payload_meta = json.dumps(metadata, sort_keys=True) if metadata else None
            if arm is None:
                arm = EmbeddingABTestArm(
                    arm_id=arm_id,
                    test_id=test_id,
                    arm_index=arm_index,
                    provider=provider,
                    model_id=model_id,
                    dimensions=dimensions,
                    collection_hash=collection_hash,
                    pipeline_hash=pipeline_hash,
                    collection_name=collection_name,
                    status=status,
                    stats_json=payload_stats,
                    metadata_json=payload_meta,
                )
            else:
                arm.arm_index = arm_index
                arm.provider = provider
                arm.model_id = model_id
                arm.dimensions = dimensions
                arm.collection_hash = collection_hash
                arm.pipeline_hash = pipeline_hash
                arm.collection_name = collection_name
                arm.status = status
                if stats is not None:
                    arm.stats_json = payload_stats
                arm.metadata_json = payload_meta
            session.add(arm)
            session.flush()
            session.refresh(arm)
            return arm

    def add_query(
        self,
        *,
        query_id: str,
        test_id: str,
        text: str,
        ground_truth_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingABTestQuery:
        with self.session_scope() as session:
            query = EmbeddingABTestQuery(
                query_id=query_id,
                test_id=test_id,
                text=text,
                ground_truth_ids=json.dumps(ground_truth_ids) if ground_truth_ids else None,
                metadata_json=json.dumps(metadata, sort_keys=True) if metadata else None,
            )
            session.add(query)
            session.flush()
            session.refresh(query)
            return query

    def record_result(
        self,
        *,
        result_id: str,
        test_id: str,
        arm_id: str,
        query_id: str,
        ranked_ids: list[str],
        scores: list[float] | None = None,
        metrics: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        ranked_distances: list[float] | None = None,
        ranked_metadatas: list[dict[str, Any]] | None = None,
        ranked_documents: list[str] | None = None,
        rerank_scores: list[float] | None = None,
    ) -> EmbeddingABTestResult:
        with self.session_scope() as session:
            result = EmbeddingABTestResult(
                result_id=result_id,
                test_id=test_id,
                arm_id=arm_id,
                query_id=query_id,
                ranked_ids=json.dumps(ranked_ids),
                scores=json.dumps(scores) if scores is not None else None,
                metrics_json=json.dumps(metrics, sort_keys=True) if metrics else None,
                latency_ms=latency_ms,
                ranked_distances=json.dumps(ranked_distances) if ranked_distances else None,
                ranked_metadatas=json.dumps(ranked_metadatas, sort_keys=True) if ranked_metadatas else None,
                ranked_documents=json.dumps(ranked_documents) if ranked_documents else None,
                rerank_scores=json.dumps(rerank_scores) if rerank_scores else None,
            )
            session.add(result)
            session.flush()
            session.refresh(result)
            return result

    def update_test_status(
        self,
        *,
        test_id: str,
        status: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        with self.session_scope() as session:
            stmt = select(EmbeddingABTest).where(EmbeddingABTest.test_id == test_id)
            entity = session.execute(stmt).scalar_one_or_none()
            if not entity:
                raise ValueError(f"Embedding A/B test {test_id} not found")
            entity.status = status
            if stats is not None:
                entity.stats_json = json.dumps(stats, sort_keys=True)
            session.add(entity)

    def get_test_with_children(self, test_id: str) -> EmbeddingABTest | None:
        with self._session_factory() as session:
            stmt = (
                select(EmbeddingABTest)
                .where(EmbeddingABTest.test_id == test_id)
                .options(
                    selectinload(EmbeddingABTest.arms),
                    selectinload(EmbeddingABTest.queries),
                    selectinload(EmbeddingABTest.results),
                )
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                session.expunge(result)
            return result

    def get_test(self, test_id: str) -> EmbeddingABTest | None:
        with self._session_factory() as session:
            stmt = select(EmbeddingABTest).where(EmbeddingABTest.test_id == test_id)
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                session.expunge(result)
            return result

    def list_arms(self, test_id: str) -> list[EmbeddingABTestArm]:
        with self._session_factory() as session:
            stmt = (
                select(EmbeddingABTestArm)
                .where(EmbeddingABTestArm.test_id == test_id)
                .order_by(EmbeddingABTestArm.arm_index.asc())
            )
            rows = session.execute(stmt).scalars().all()
            for row in rows:
                session.expunge(row)
            return rows

    def find_reusable_arm(
        self,
        *,
        test_id: str,
        collection_hash: str,
        created_by: str | None,
    ) -> EmbeddingABTestArm | None:
        if not collection_hash or not created_by:
            return None
        with self._session_factory() as session:
            stmt = (
                select(EmbeddingABTestArm)
                .join(EmbeddingABTest)
                .where(
                    EmbeddingABTestArm.collection_hash == collection_hash,
                    EmbeddingABTestArm.status == "ready",
                    EmbeddingABTestArm.collection_name.is_not(None),
                    EmbeddingABTest.test_id != test_id,
                    EmbeddingABTest.created_by == created_by,
                )
                .order_by(EmbeddingABTest.created_at.desc())
                .limit(1)
            )
            row = session.execute(stmt).scalars().first()
            if row:
                session.expunge(row)
            return row

    def list_queries(self, test_id: str) -> list[EmbeddingABTestQuery]:
        with self._session_factory() as session:
            stmt = select(EmbeddingABTestQuery).where(EmbeddingABTestQuery.test_id == test_id)
            rows = session.execute(stmt).scalars().all()
            for row in rows:
                session.expunge(row)
            return rows

    def list_results(self, test_id: str, limit: int, offset: int) -> tuple[list[EmbeddingABTestResult], int]:
        with self._session_factory() as session:
            count_stmt = (
                select(func.count())
                .select_from(EmbeddingABTestResult)
                .where(EmbeddingABTestResult.test_id == test_id)
            )
            total = int(session.execute(count_stmt).scalar_one() or 0)
            stmt = (
                select(EmbeddingABTestResult)
                .where(EmbeddingABTestResult.test_id == test_id)
                .order_by(EmbeddingABTestResult.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            rows = session.execute(stmt).scalars().all()
            for row in rows:
                session.expunge(row)
            return rows, total

    def delete_test(self, test_id: str) -> int:
        """Delete an A/B test and cascade to related rows."""
        with self._session_factory() as session:
            entity = session.execute(
                select(EmbeddingABTest).where(EmbeddingABTest.test_id == test_id)
            ).scalar_one_or_none()
            if not entity:
                return 0
            session.delete(entity)
            session.commit()
            return 1


def serialize_test(entity: EmbeddingABTest) -> dict[str, Any]:
    return {
        "test_id": entity.test_id,
        "name": entity.name,
        "created_by": entity.created_by,
        "created_at": _iso_or_none(entity.created_at),
        "status": entity.status,
        "config_json": entity.config_json,
        "stats_json": entity.stats_json,
        "notes": entity.notes,
    }


def serialize_arm(entity: EmbeddingABTestArm) -> dict[str, Any]:
    return {
        "arm_id": entity.arm_id,
        "test_id": entity.test_id,
        "arm_index": entity.arm_index,
        "provider": entity.provider,
        "model_id": entity.model_id,
        "dimensions": entity.dimensions,
        "collection_hash": entity.collection_hash,
        "pipeline_hash": entity.pipeline_hash,
        "collection_name": entity.collection_name,
        "status": entity.status,
        "stats_json": entity.stats_json,
        "metadata_json": entity.metadata_json,
    }


def serialize_query(entity: EmbeddingABTestQuery) -> dict[str, Any]:
    return {
        "query_id": entity.query_id,
        "test_id": entity.test_id,
        "text": entity.text,
        "ground_truth_ids": entity.ground_truth_ids,
        "metadata_json": entity.metadata_json,
        "created_at": _iso_or_none(entity.created_at),
    }


def serialize_result(entity: EmbeddingABTestResult) -> dict[str, Any]:
    return {
        "result_id": entity.result_id,
        "test_id": entity.test_id,
        "arm_id": entity.arm_id,
        "query_id": entity.query_id,
        "ranked_ids": entity.ranked_ids,
        "scores": entity.scores,
        "metrics_json": entity.metrics_json,
        "latency_ms": entity.latency_ms,
        "ranked_distances": entity.ranked_distances,
        "ranked_metadatas": entity.ranked_metadatas,
        "ranked_documents": entity.ranked_documents,
        "rerank_scores": entity.rerank_scores,
        "created_at": _iso_or_none(entity.created_at),
    }


class EmbeddingsABTestStore:
    """Compatibility adapter that mirrors EvaluationsDatabase A/B test APIs."""

    def __init__(self, repository: EmbeddingABTestRepository) -> None:
        self._repo = repository

    @staticmethod
    def _created_by_matches(entity_created_by: str | None, created_by: str | None) -> bool:
        if not created_by:
            return True
        if not entity_created_by:
            return False
        raw = str(created_by).strip()
        if not raw:
            return False
        variants = {raw}
        if raw.startswith("user_"):
            core = raw[5:]
            if core:
                variants.add(core)
        elif raw.isdigit():
            variants.add(f"user_{raw}")
        return entity_created_by in variants

    def _authorized(self, test_id: str, created_by: str | None) -> bool:
        if not created_by:
            return True
        entity = self._repo.get_test(test_id)
        return self._created_by_matches(getattr(entity, "created_by", None), created_by)

    def create_abtest(self, name: str, config: dict[str, Any], created_by: str | None = None) -> str:
        test_id = f"abtest_{uuid.uuid4().hex[:12]}"
        self._repo.create_test(
            test_id=test_id,
            name=name,
            created_by=created_by,
            config=config,
        )
        return test_id

    def upsert_abtest_arm(
        self,
        *,
        test_id: str,
        arm_index: int,
        provider: str,
        model_id: str,
        dimensions: int | None = None,
        collection_hash: str | None = None,
        pipeline_hash: str | None = None,
        collection_name: str | None = None,
        status: str = "pending",
        stats_json: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> str:
        arm_id = f"arm_{test_id}_{arm_index}"
        self._repo.add_arm(
            arm_id=arm_id,
            test_id=test_id,
            arm_index=arm_index,
            provider=provider,
            model_id=model_id,
            dimensions=dimensions,
            collection_hash=collection_hash,
            pipeline_hash=pipeline_hash,
            collection_name=collection_name,
            status=status,
            stats=stats_json,
            metadata=metadata_json,
        )
        return arm_id

    def insert_abtest_queries(self, test_id: str, queries: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for payload in queries:
            qid = f"q_{uuid.uuid4().hex[:10]}"
            ids.append(qid)
            ground_truth = payload.get("expected_ids") or payload.get("ground_truth_ids")
            metadata = payload.get("metadata")
            text = payload.get("text") or ""
            self._repo.add_query(
                query_id=qid,
                test_id=test_id,
                text=text,
                ground_truth_ids=list(ground_truth) if isinstance(ground_truth, (list, tuple)) else ground_truth,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        return ids

    def set_abtest_status(self, test_id: str, status: str, stats_json: dict[str, Any] | None = None, *, created_by: str | None = None) -> None:
        if not self._authorized(test_id, created_by):
            return
        self._repo.update_test_status(test_id=test_id, status=status, stats=stats_json)

    def insert_abtest_result(
        self,
        test_id: str,
        arm_id: str,
        query_id: str,
        ranked_ids: list[str],
        scores: list[float] | None = None,
        metrics: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        ranked_distances: list[float] | None = None,
        ranked_metadatas: list[dict[str, Any]] | None = None,
        ranked_documents: list[str] | None = None,
        rerank_scores: list[float] | None = None,
    ) -> str:
        rid = f"res_{uuid.uuid4().hex[:12]}"
        self._repo.record_result(
            result_id=rid,
            test_id=test_id,
            arm_id=arm_id,
            query_id=query_id,
            ranked_ids=ranked_ids,
            scores=scores,
            metrics=metrics,
            latency_ms=latency_ms,
            ranked_distances=ranked_distances,
            ranked_metadatas=ranked_metadatas,
            ranked_documents=ranked_documents,
            rerank_scores=rerank_scores,
        )
        return rid

    def get_abtest(self, test_id: str, *, created_by: str | None = None) -> dict[str, Any] | None:
        entity = self._repo.get_test(test_id)
        if not entity or not self._created_by_matches(entity.created_by, created_by):
            return None
        return serialize_test(entity)

    def get_abtest_arms(self, test_id: str, *, created_by: str | None = None) -> list[dict[str, Any]]:
        if not self._authorized(test_id, created_by):
            return []
        return [serialize_arm(arm) for arm in self._repo.list_arms(test_id)]

    def find_reusable_abtest_arm(
        self,
        *,
        test_id: str,
        collection_hash: str,
        created_by: str | None,
    ) -> dict[str, Any] | None:
        arm = self._repo.find_reusable_arm(
            test_id=test_id,
            collection_hash=collection_hash,
            created_by=created_by,
        )
        if not arm:
            return None
        return serialize_arm(arm)

    def get_abtest_queries(self, test_id: str, *, created_by: str | None = None) -> list[dict[str, Any]]:
        if not self._authorized(test_id, created_by):
            return []
        return [serialize_query(q) for q in self._repo.list_queries(test_id)]

    def list_abtest_results(self, test_id: str, limit: int, offset: int, *, created_by: str | None = None) -> tuple[list[dict[str, Any]], int]:
        if not self._authorized(test_id, created_by):
            return [], 0
        rows, total = self._repo.list_results(test_id, limit, offset)
        return [serialize_result(r) for r in rows], total

    def delete_abtest(self, test_id: str, *, created_by: str | None = None) -> int:
        if not self._authorized(test_id, created_by):
            return 0
        return self._repo.delete_test(test_id)


def _sqlite_url_from_path(path: str) -> str:
    location = resolve_trusted_database_path(
        path,
        label="embeddings A/B test database",
    )
    return f"sqlite:///{location}"


@lru_cache(maxsize=8)
def get_embeddings_abtest_store(db_path_or_url: str) -> EmbeddingsABTestStore:
    db_url = db_path_or_url if "://" in db_path_or_url else _sqlite_url_from_path(db_path_or_url)
    config = RepositoryConfig(db_url=db_url)
    repo = EmbeddingABTestRepository.from_config(config)
    return EmbeddingsABTestStore(repo)
