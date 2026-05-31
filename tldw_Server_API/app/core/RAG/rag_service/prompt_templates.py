# prompt_templates.py
"""
Prompt template management for the RAG service.

This module provides customizable prompt templates for different use cases
and contexts, with support for dynamic variable substitution.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt


class TemplateType(Enum):
    """Types of prompt templates."""
    SYSTEM = "system"
    USER = "user"
    CONTEXT = "context"
    QUESTION = "question"
    ANSWER = "answer"
    FULL = "full"


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""
    name: str
    template: str
    type: TemplateType
    description: Optional[str] = None
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            import re
            # Find all {variable} patterns
            self.variables = re.findall(r'\{(\w+)\}', self.template)

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError:
            logger.warning(f"Missing variable in template '{self.name}'")
            # Return template with missing variables as-is
            return self.template

    def validate_variables(self, variables: dict[str, Any]) -> bool:
        """Check if all required variables are provided."""
        return all(var in variables for var in self.variables)


class PromptTemplateLibrary:
    """Library of reusable prompt templates."""

    def __init__(self):
        """Initialize with default templates."""
        self.templates: dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default prompt templates."""

        # System prompts
        self.add_template(PromptTemplate(
            name="default_system",
            template="You are a helpful AI assistant that provides accurate, well-reasoned answers based on the provided context.",
            type=TemplateType.SYSTEM,
            description="Default system prompt for general use"
        ))

        self.add_template(PromptTemplate(
            name="research_system",
            template="You are an expert research assistant with deep knowledge across multiple domains. Analyze information critically and provide comprehensive, well-cited answers.",
            type=TemplateType.SYSTEM,
            description="System prompt for research tasks"
        ))

        self.add_template(PromptTemplate(
            name="technical_system",
            template="You are a technical expert assistant. Provide precise, technically accurate answers with appropriate terminology and detail level for the context.",
            type=TemplateType.SYSTEM,
            description="System prompt for technical questions"
        ))

        # Full prompts with context
        self.add_template(PromptTemplate(
            name="default_full",
            template="""{system_prompt}

Context Information:
{context}

User Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information, clearly state what is missing.

Answer:""",
            type=TemplateType.FULL,
            description="Default full prompt with context",
            variables=["system_prompt", "context", "question"]
        ))

        self.add_template(PromptTemplate(
            name="with_citations",
            template="""{system_prompt}

Source Documents:
{context}

Question: {question}

Instructions:
1. Answer the question based on the provided sources
2. Include [Source N] citations when referencing specific information
3. If multiple sources support a point, cite all relevant sources
4. Clearly indicate if information is not available in the sources

Answer with citations:""",
            type=TemplateType.FULL,
            description="Prompt requesting citations",
            variables=["system_prompt", "context", "question"]
        ))

        self.add_template(PromptTemplate(
            name="comparative",
            template="""{system_prompt}

Multiple perspectives from sources:
{context}

Question: {question}

Analyze the different perspectives presented in the sources and provide a balanced answer that:
1. Acknowledges different viewpoints
2. Identifies areas of agreement and disagreement
3. Synthesizes the information into a coherent response

Comparative Analysis:""",
            type=TemplateType.FULL,
            description="Prompt for comparing multiple sources",
            variables=["system_prompt", "context", "question"]
        ))

        self.add_template(PromptTemplate(
            name="step_by_step",
            template="""{system_prompt}

Reference Material:
{context}

Question: {question}

Please provide a step-by-step answer:
1. Break down the problem/question
2. Address each component systematically
3. Use the context to support each step
4. Provide a clear conclusion

Step-by-Step Answer:""",
            type=TemplateType.FULL,
            description="Prompt for step-by-step reasoning",
            variables=["system_prompt", "context", "question"]
        ))

        self.add_template(PromptTemplate(
            name="code_focused",
            template="""{system_prompt}

Code Examples and Documentation:
{context}

Programming Question: {question}

Provide a technical answer that includes:
1. Explanation of the concept
2. Code examples (if applicable)
3. Best practices and common pitfalls
4. References to the provided documentation

Technical Answer:""",
            type=TemplateType.FULL,
            description="Prompt for code-related questions",
            variables=["system_prompt", "context", "question"]
        ))

        # Context formatting templates
        self.add_template(PromptTemplate(
            name="context_with_metadata",
            template="""[Document {index}]
Title: {title}
Source: {source}
Date: {date}
Content:
{content}
---""",
            type=TemplateType.CONTEXT,
            description="Format for individual context documents",
            variables=["index", "title", "source", "date", "content"]
        ))

        self.add_template(PromptTemplate(
            name="context_simple",
            template="""Document {index}:
{content}
---""",
            type=TemplateType.CONTEXT,
            description="Simple context format",
            variables=["index", "content"]
        ))

        # Question formatting templates
        self.add_template(PromptTemplate(
            name="question_with_context_request",
            template="""{question}

Please base your answer on the provided context and indicate if additional information would be helpful.""",
            type=TemplateType.QUESTION,
            description="Question with context awareness",
            variables=["question"]
        ))

        # Answer formatting templates
        self.add_template(PromptTemplate(
            name="structured_answer",
            template="""## Summary
{summary}

## Detailed Explanation
{details}

## Key Points
{key_points}

## Additional Considerations
{considerations}""",
            type=TemplateType.ANSWER,
            description="Structured answer format",
            variables=["summary", "details", "key_points", "considerations"]
        ))

        # External overrides/extensions from Prompts/rag (YAML/JSON/MD)
        self._load_external_templates()

    def _load_external_templates(self) -> None:
        """Override or extend templates using Prompts/rag if available.

        Recognized keys:
        - default_system, with_citations, retrieval_guidance, reranking_instruction
        """
        key_map = {
            "default_system": ("default_system", TemplateType.SYSTEM),
            "with_citations": ("with_citations", TemplateType.FULL),
            "retrieval_guidance": ("retrieval_guidance", TemplateType.SYSTEM),
            "reranking_instruction": ("reranking_instruction", TemplateType.SYSTEM),
        }
        for ext_key, (name, ttype) in key_map.items():
            content = load_prompt("rag", ext_key)
            if isinstance(content, str) and content.strip():
                if name in self.templates:
                    self.templates[name].template = content
                    self.templates[name].type = ttype
                else:
                    self.add_template(PromptTemplate(
                        name=name,
                        template=content,
                        type=ttype,
                        description=f"Loaded from Prompts/rag: {ext_key}"
                    ))

    def add_template(self, template: PromptTemplate) -> None:
        """Add a template to the library."""
        self.templates[template.name] = template
        logger.debug(f"Added template: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self, type: Optional[TemplateType] = None) -> list[str]:
        """List available template names, optionally filtered by type."""
        if type:
            return [
                name for name, template in self.templates.items()
                if template.type == type
            ]
        return list(self.templates.keys())

    def create_custom_template(
        self,
        name: str,
        template_str: str,
        type: TemplateType = TemplateType.FULL,
        description: Optional[str] = None
    ) -> PromptTemplate:
        """Create and add a custom template."""
        template = PromptTemplate(
            name=name,
            template=template_str,
            type=type,
            description=description
        )
        self.add_template(template)
        return template

    def format_context_documents(
        self,
        documents: list[Any],
        template_name: str = "context_with_metadata"
    ) -> str:
        """Format a list of documents using a context template."""
        template = self.get_template(template_name)
        if not template:
            # Fallback to simple formatting
            return "\n\n".join(
                f"Document {i+1}:\n{doc.content}"
                for i, doc in enumerate(documents)
            )

        formatted_docs = []
        for i, doc in enumerate(documents):
            # Extract metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}

            # Format individual document
            formatted = template.format(
                index=i+1,
                title=metadata.get("title", "Untitled"),
                source=metadata.get("source", "Unknown"),
                date=metadata.get("date", "N/A"),
                content=doc.content if hasattr(doc, 'content') else str(doc)
            )
            formatted_docs.append(formatted)

        return "\n\n".join(formatted_docs)

    def build_full_prompt(
        self,
        question: str,
        context: str,
        template_name: str = "default_full",
        system_prompt_name: str = "default_system",
        **kwargs
    ) -> str:
        """Build a complete prompt from components."""
        # Get templates
        full_template = self.get_template(template_name)
        system_template = self.get_template(system_prompt_name)

        if not full_template:
            # Fallback to basic prompt
            return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Get system prompt
        system_prompt = system_template.template if system_template else ""

        # Format full prompt
        return full_template.format(
            system_prompt=system_prompt,
            context=context,
            question=question,
            **kwargs
        )

    def save_to_file(self, filepath: str) -> None:
        """Save templates to a JSON file."""
        data = {
            name: {
                "template": template.template,
                "type": template.type.value,
                "description": template.description,
                "variables": template.variables,
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.templates)} templates to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load templates from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        for name, template_data in data.items():
            template = PromptTemplate(
                name=name,
                template=template_data["template"],
                type=TemplateType(template_data["type"]),
                description=template_data.get("description"),
                variables=template_data.get("variables"),
                metadata=template_data.get("metadata", {})
            )
            self.add_template(template)

        logger.info(f"Loaded {len(data)} templates from {filepath}")


# Global template library instance
template_library = PromptTemplateLibrary()


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get a template from the global library."""
    return template_library.get_template(name)


def format_prompt(
    question: str,
    context: str,
    template: str = "default_full",
    **kwargs
) -> str:
    """Format a prompt using the global library."""
    return template_library.build_full_prompt(
        question=question,
        context=context,
        template_name=template,
        **kwargs
    )
