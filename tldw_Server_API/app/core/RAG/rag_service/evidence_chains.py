"""
Multi-Hop Evidence Chain Builder for RAG.

This module tracks evidence dependencies across documents for complex queries,
building chains of evidence that show how facts from multiple sources
support claims in the generated response.

Design:
- EvidenceNode: Represents a single fact from a source document
- EvidenceChain: Links nodes to show reasoning path
- ChainBuilder: Constructs chains from documents and claims
- Confidence scoring: Product of node confidences for conservative estimates
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from .types import Citation, Document, EvidenceChain, EvidenceNode


@dataclass
class ClaimEvidence:
    """Evidence supporting a specific claim."""
    claim_id: str
    claim_text: str
    evidence_nodes: list[EvidenceNode] = field(default_factory=list)
    aggregated_confidence: float = 0.0
    is_supported: bool = False


@dataclass
class ChainBuildResult:
    """Result from chain building process."""
    chains: list[EvidenceChain]
    claims: list[ClaimEvidence]
    overall_confidence: float
    multi_hop_detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chains": [c.to_dict() for c in self.chains],
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.claim_text,
                    "evidence_count": len(c.evidence_nodes),
                    "confidence": c.aggregated_confidence,
                    "is_supported": c.is_supported,
                }
                for c in self.claims
            ],
            "overall_confidence": self.overall_confidence,
            "multi_hop_detected": self.multi_hop_detected,
            "metadata": self.metadata,
        }


# Claim extraction patterns
CLAIM_PATTERNS = [
    r"(?:According to|Based on|As stated in).*?,\s*(.+?)\.",
    r"(?:The|This|It)\s+(?:shows?|indicates?|suggests?|demonstrates?)\s+that\s+(.+?)\.",
    r"(?:We|I)\s+(?:found|discovered|concluded|determined)\s+that\s+(.+?)\.",
    r"(?:Therefore|Thus|Hence|Consequently),?\s+(.+?)\.",
    r"(?:The (?:data|evidence|results?|findings?))\s+(?:show|indicate|suggest|demonstrate)\s+(.+?)\.",
]

FACT_EXTRACTION_PROMPT = """Extract key facts from the following text that could support answering the query.

Query: {query}

Text:
{text}

Extract 1-5 atomic facts. Each fact should be:
- A single, verifiable statement
- Directly relevant to the query
- Self-contained (understandable without context)

Respond with one fact per line, prefixed with "- ".
"""


def _generate_claim_id(claim_text: str, index: int) -> str:
    """Generate a unique ID for a claim."""
    h = hashlib.sha256(claim_text.encode()).hexdigest()[:8]
    return f"claim_{index}_{h}"


def _generate_node_id(doc_id: str, chunk_id: str, fact: str) -> str:
    """Generate a unique ID for an evidence node."""
    content = f"{doc_id}:{chunk_id}:{fact}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _extract_claims_heuristic(text: str) -> list[str]:
    """Extract claims from text using pattern matching."""
    claims = []
    for pattern in CLAIM_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        claims.extend(matches)

    # Also extract sentences that look like assertions
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 200:
            continue
        # Look for declarative sentences
        if re.match(r"^[A-Z].*[.!]$", sent):
            if not any(sent.startswith(w) for w in ["If ", "When ", "Although ", "While ", "Unless "]):
                claims.append(sent)

    # Deduplicate and limit
    seen = set()
    unique_claims = []
    for c in claims:
        c_normalized = c.lower().strip()
        if c_normalized not in seen and len(c_normalized) > 10:
            seen.add(c_normalized)
            unique_claims.append(c.strip())

    return unique_claims[:10]


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute simple word-overlap similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    words1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
    words2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


class EvidenceChainBuilder:
    """
    Builds evidence chains from documents and generated responses.

    Tracks how facts from source documents support claims in the response,
    enabling transparency in multi-hop reasoning.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_chain_length: int = 5,
        similarity_threshold: float = 0.3,
        enable_llm_extraction: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the chain builder.

        Args:
            min_confidence: Minimum confidence for including evidence
            max_chain_length: Maximum nodes in a single chain
            similarity_threshold: Minimum similarity for fact-claim matching
            enable_llm_extraction: Use LLM for fact extraction
            llm_provider: LLM provider for extraction
            llm_model: LLM model for extraction
        """
        self.min_confidence = min_confidence
        self.max_chain_length = max_chain_length
        self.similarity_threshold = similarity_threshold
        self.enable_llm_extraction = enable_llm_extraction
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    async def build_chains(
        self,
        query: str,
        documents: list[Document],
        generated_answer: Optional[str] = None,
        existing_citations: Optional[list[Citation]] = None,
    ) -> ChainBuildResult:
        """
        Build evidence chains from documents and response.

        Args:
            query: The original query
            documents: Source documents used for retrieval
            generated_answer: The generated response (optional)
            existing_citations: Pre-existing citations to incorporate

        Returns:
            ChainBuildResult with chains and claims
        """
        if not documents:
            return ChainBuildResult(
                chains=[],
                claims=[],
                overall_confidence=0.0,
                multi_hop_detected=False,
                metadata={"error": "No documents provided"},
            )

        # Extract claims from the generated answer
        claims = []
        if generated_answer:
            claim_texts = _extract_claims_heuristic(generated_answer)
            for i, text in enumerate(claim_texts):
                claim_id = _generate_claim_id(text, i)
                claims.append(ClaimEvidence(
                    claim_id=claim_id,
                    claim_text=text,
                ))

        # Extract facts from documents
        all_nodes: list[EvidenceNode] = []
        for doc in documents:
            nodes = await self._extract_facts_from_document(doc, query)
            all_nodes.extend(nodes)

        # Match facts to claims
        for claim in claims:
            matching_nodes = self._find_supporting_nodes(claim, all_nodes)
            claim.evidence_nodes = matching_nodes
            for node in matching_nodes:
                if claim.claim_id not in node.supports:
                    node.supports.append(claim.claim_id)

            # Compute claim confidence
            if matching_nodes:
                claim.aggregated_confidence = sum(n.confidence for n in matching_nodes) / len(matching_nodes)
                claim.is_supported = claim.aggregated_confidence >= self.min_confidence

        # Build chains by grouping nodes that support the same claims
        chains = self._build_chains_from_nodes(query, all_nodes, claims)

        # Compute overall confidence
        overall_confidence = sum(c.chain_confidence for c in chains) / len(chains) if chains else 0.0

        # Detect multi-hop reasoning
        multi_hop = any(c.hop_count > 1 for c in chains)

        return ChainBuildResult(
            chains=chains,
            claims=claims,
            overall_confidence=overall_confidence,
            multi_hop_detected=multi_hop,
            metadata={
                "total_nodes": len(all_nodes),
                "total_claims": len(claims),
                "supported_claims": sum(1 for c in claims if c.is_supported),
            },
        )

    async def _extract_facts_from_document(
        self,
        doc: Document,
        query: str,
    ) -> list[EvidenceNode]:
        """Extract facts from a document."""
        if not doc.content:
            return []

        # Try LLM extraction first
        if self.enable_llm_extraction:
            try:
                return await self._llm_extract_facts(doc, query)
            except Exception:
                logger.debug("LLM fact extraction failed, using heuristic")

        # Fall back to heuristic extraction
        return self._heuristic_extract_facts(doc, query)

    def _heuristic_extract_facts(
        self,
        doc: Document,
        query: str,
    ) -> list[EvidenceNode]:
        """Extract facts using heuristic patterns."""
        facts = []
        content = doc.content or ""

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)

        # Extract query terms for relevance filtering
        query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15 or len(sent) > 300:
                continue

            # Check relevance to query
            sent_terms = set(re.findall(r"\b\w{3,}\b", sent.lower()))
            overlap = len(query_terms & sent_terms)

            if overlap < 1:
                continue

            # Score based on characteristics
            confidence = 0.5

            # Boost for specific indicators
            if re.search(r"\d+%?", sent):  # Contains numbers
                confidence += 0.1
            if re.search(r'"[^"]+"', sent):  # Contains quotes
                confidence += 0.1
            if re.search(r"\b(is|are|was|were|has|have|had)\b", sent, re.I):  # Assertion
                confidence += 0.1

            # Adjust for query term overlap
            confidence += min(0.2, overlap * 0.05)
            confidence = min(1.0, confidence)

            if confidence >= self.min_confidence:
                node = EvidenceNode(
                    document_id=doc.id,
                    chunk_id=doc.id,  # Use doc.id if no separate chunk_id
                    fact=sent,
                    confidence=confidence,
                    source_text=sent,
                    extraction_method="pattern",
                    metadata={
                        "document_title": (doc.metadata or {}).get("title"),
                        "query_overlap": overlap,
                    },
                )
                facts.append(node)

        # Limit to top facts by confidence
        facts.sort(key=lambda n: n.confidence, reverse=True)
        return facts[:8]

    async def _llm_extract_facts(
        self,
        doc: Document,
        query: str,
    ) -> list[EvidenceNode]:
        """Extract facts using LLM."""
        try:
            from .generation import AnswerGenerator

            generator = AnswerGenerator(
                provider=self.llm_provider,
                model=self.llm_model,
            )

            # Truncate document content for prompt
            content = (doc.content or "")[:2000]
            prompt = FACT_EXTRACTION_PROMPT.format(query=query, text=content)

            result = await generator.generate(
                query=query,
                context=content,
                prompt_template=prompt,
                max_tokens=400,
            )

            response = result.get("answer", "") if isinstance(result, dict) else str(result)

            # Parse facts from response
            facts = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    fact_text = line[2:].strip()
                    if len(fact_text) > 10:
                        node = EvidenceNode(
                            document_id=doc.id,
                            chunk_id=doc.id,
                            fact=fact_text,
                            confidence=0.7,  # LLM extraction gets higher base confidence
                            source_text=fact_text,
                            extraction_method="llm",
                            metadata={
                                "document_title": (doc.metadata or {}).get("title"),
                            },
                        )
                        facts.append(node)

            return facts[:5]

        except Exception:
            logger.warning("LLM fact extraction failed")
            raise

    def _find_supporting_nodes(
        self,
        claim: ClaimEvidence,
        nodes: list[EvidenceNode],
    ) -> list[EvidenceNode]:
        """Find nodes that support a claim."""
        supporting = []

        for node in nodes:
            similarity = _compute_text_similarity(claim.claim_text, node.fact)
            if similarity >= self.similarity_threshold:
                # Adjust confidence based on similarity
                adjusted_confidence = min(1.0, node.confidence * (0.5 + 0.5 * similarity))
                supporting.append(
                    EvidenceNode(
                        document_id=node.document_id,
                        chunk_id=node.chunk_id,
                        fact=node.fact,
                        confidence=adjusted_confidence,
                        supports=list(node.supports),
                        source_text=node.source_text,
                        extraction_method=node.extraction_method,
                        metadata=dict(node.metadata),
                    )
                )

        # Sort by confidence and limit
        supporting.sort(key=lambda n: n.confidence, reverse=True)
        return supporting[:self.max_chain_length]

    def _build_chains_from_nodes(
        self,
        query: str,
        nodes: list[EvidenceNode],
        claims: list[ClaimEvidence],
    ) -> list[EvidenceChain]:
        """Build evidence chains from nodes and claims."""
        chains = []

        # Group nodes by the claims they support
        claim_to_nodes: dict[str, list[EvidenceNode]] = defaultdict(list)
        for node in nodes:
            for claim_id in node.supports:
                claim_to_nodes[claim_id].append(node)

        # Build a chain for each supported claim
        for claim in claims:
            if not claim.is_supported:
                continue

            claim_nodes = claim_to_nodes.get(claim.claim_id, [])
            if not claim_nodes:
                continue

            chain = EvidenceChain(
                query=query,
                nodes=claim_nodes[:self.max_chain_length],
                root_claims=[claim.claim_id],
                metadata={
                    "claim_text": claim.claim_text,
                },
            )
            chains.append(chain)

        # Also create an overall chain if multiple claims share evidence
        shared_nodes = [n for n in nodes if len(n.supports) > 1]
        if shared_nodes:
            all_claim_ids = list({
                cid for node in shared_nodes for cid in node.supports
            })
            overall_chain = EvidenceChain(
                query=query,
                nodes=shared_nodes[:self.max_chain_length],
                root_claims=all_claim_ids,
                metadata={"type": "shared_evidence"},
            )
            chains.append(overall_chain)

        return chains

    def compute_chain_confidence(self, chain: EvidenceChain) -> float:
        """
        Compute confidence score for an evidence chain.

        Uses product of node confidences for conservative estimates.
        """
        if not chain.nodes:
            return 0.0

        confidence = 1.0
        for node in chain.nodes:
            confidence *= node.confidence

        return confidence

    def extend_citation_with_chain(
        self,
        citation: Citation,
        chain: EvidenceChain,
    ) -> Citation:
        """
        Extend a citation with evidence chain information.

        Args:
            citation: The original citation
            chain: Evidence chain to link

        Returns:
            Citation with chain metadata added
        """
        # Add chain info to citation metadata
        chain_info = {
            "chain_confidence": chain.chain_confidence,
            "hop_count": chain.hop_count,
            "source_documents": chain.get_source_documents(),
            "facts_count": len(chain.nodes),
        }

        citation.metadata = citation.metadata or {}
        citation.metadata["evidence_chain"] = chain_info

        return citation


# Module-level convenience functions

async def build_evidence_chains(
    query: str,
    documents: list[Document],
    generated_answer: Optional[str] = None,
) -> ChainBuildResult:
    """
    Convenience function to build evidence chains.

    Args:
        query: The search query
        documents: Source documents
        generated_answer: The generated response

    Returns:
        ChainBuildResult with chains and claims
    """
    builder = EvidenceChainBuilder()
    return await builder.build_chains(
        query=query,
        documents=documents,
        generated_answer=generated_answer,
    )


def compute_chain_confidence(chain: EvidenceChain) -> float:
    """Compute confidence score for an evidence chain."""
    return EvidenceChainBuilder().compute_chain_confidence(chain)
