# prompt_improver.py
# Prompt improvement and optimization for Prompt Studio

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError, PromptStudioDatabase
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.gateway_call import call_prompt_studio_llm

########################################################################################################################
# Enums and Data Classes

class ImprovementStrategy(Enum):
    """Strategies for improving prompts."""
    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    STRUCTURE = "structure"
    ADD_EXAMPLES = "add_examples"
    ADD_CONSTRAINTS = "add_constraints"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SIMPLIFY = "simplify"
    EXPAND = "expand"
    FORMALIZE = "formalize"
    TECHNICAL = "technical"
    AUTO = "auto"

@dataclass
class PromptAnalysis:
    """Analysis of a prompt's quality."""
    clarity_score: float = 0.0
    specificity_score: float = 0.0
    structure_score: float = 0.0
    completeness_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = [self.clarity_score, self.specificity_score,
                 self.structure_score, self.completeness_score]
        return sum(scores) / len(scores)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clarity_score": self.clarity_score,
            "specificity_score": self.specificity_score,
            "structure_score": self.structure_score,
            "completeness_score": self.completeness_score,
            "overall_score": self.overall_score(),
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metrics": self.metrics
        }

@dataclass
class ImprovementResult:
    """Result of a prompt improvement."""
    original_prompt: str
    improved_prompt: str
    strategy: ImprovementStrategy
    improvements_made: list[str] = field(default_factory=list)
    score_change: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    before_analysis: Optional[PromptAnalysis] = None
    after_analysis: Optional[PromptAnalysis] = None

    def calculate_improvement(self) -> float:
        """Calculate the improvement score."""
        if self.before_analysis and self.after_analysis:
            return self.after_analysis.overall_score() - self.before_analysis.overall_score()
        return self.score_change


def _call_openai_adapter(request: dict[str, Any]) -> dict[str, Any]:
    return call_prompt_studio_llm("openai", request)

########################################################################################################################
# Improvement Strategies

IMPROVEMENT_STRATEGIES = {
    "clarity": {
        "focus": "Improve clarity and reduce ambiguity",
        "techniques": [
            "Use specific, concrete language",
            "Define technical terms",
            "Break complex instructions into steps",
            "Add examples where helpful"
        ]
    },
    "structure": {
        "focus": "Improve prompt structure and organization",
        "techniques": [
            "Use XML or markdown formatting",
            "Create clear sections",
            "Add numbered steps",
            "Use consistent formatting"
        ]
    },
    "specificity": {
        "focus": "Make instructions more specific",
        "techniques": [
            "Add explicit constraints",
            "Specify output format",
            "Include success criteria",
            "Define edge cases"
        ]
    },
    "robustness": {
        "focus": "Make prompt more robust to edge cases",
        "techniques": [
            "Add error handling instructions",
            "Include fallback behaviors",
            "Handle missing inputs",
            "Add validation steps"
        ]
    },
    "efficiency": {
        "focus": "Reduce token usage while maintaining quality",
        "techniques": [
            "Remove redundant instructions",
            "Consolidate similar points",
            "Use concise language",
            "Eliminate unnecessary examples"
        ]
    }
}

########################################################################################################################
# Prompt Improver Class

class PromptImprover:
    """Improves and optimizes prompts for Prompt Studio projects."""

    def __init__(self, db: PromptStudioDatabase, enable_cache: bool = False, enable_chain_of_thought: Optional[bool] = None):
        """
        Initialize PromptImprover.

        Args:
            db: PromptStudioDatabase instance
            enable_cache: Whether to enable caching
        """
        self.db = db
        self.client_id = db.client_id if db else None
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self.strategies = list(ImprovementStrategy)
        self.analyzers = {}
        self.llm_client = None  # Mock LLM client
        # Policy switch for chain-of-thought additions (default enabled unless env says otherwise)
        if enable_chain_of_thought is None:
            env_val = os.getenv("PROMPT_STUDIO_ENABLE_CHAIN_OF_THOUGHT", "true").strip().lower()
            self.enable_chain_of_thought = env_val not in {"0", "false", "no"}
        else:
            self.enable_chain_of_thought = bool(enable_chain_of_thought)

    ####################################################################################################################
    # Core Analysis and Improvement Methods

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt's quality."""
        analysis = PromptAnalysis()

        # Calculate scores (simplified)
        analysis.clarity_score = self._calculate_clarity_score(prompt)
        analysis.specificity_score = self._calculate_specificity_score(prompt)
        analysis.structure_score = self._calculate_structure_score(prompt)
        analysis.completeness_score = self._calculate_completeness_score(prompt)

        # Identify issues
        if analysis.clarity_score < 0.7:
            analysis.issues.append("Prompt lacks clarity")
        if analysis.specificity_score < 0.7:
            analysis.issues.append("Prompt is too vague")

        # Generate suggestions
        if analysis.clarity_score < 0.7:
            analysis.suggestions.append("Simplify language and break into steps")
        if analysis.specificity_score < 0.7:
            analysis.suggestions.append("Add specific constraints and examples")

        return analysis

    def improve(self, prompt: str, strategy: ImprovementStrategy = ImprovementStrategy.AUTO,
                context: dict[str, Any] = None, examples: list[dict] = None,
                constraints: list[str] = None, fallback: bool = False) -> ImprovementResult:
        """Improve a prompt using specified strategy."""
        # Validate input
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            # Analyze before
            before_analysis = self.analyze(prompt)

            # Apply strategy
            if strategy == ImprovementStrategy.AUTO:
                strategy = self._select_best_strategy(prompt, before_analysis)

            improved = self._apply_strategy(prompt, strategy, context, examples, constraints)

            # Analyze after
            after_analysis = self.analyze(improved)

            # Create result
            result = ImprovementResult(
                original_prompt=prompt,
                improved_prompt=improved,
                strategy=strategy,
                before_analysis=before_analysis,
                after_analysis=after_analysis,
                score_change=after_analysis.overall_score() - before_analysis.overall_score(),
                improvements_made=self._list_improvements(prompt, improved)
            )

            # Cache if enabled
            if self.enable_cache:
                cache_key = f"{prompt}:{strategy.value}"
                self.cache[cache_key] = result

            return result

        except Exception as e:
            if fallback:
                # Return original on error, but log the warning
                logger.warning(f"Prompt improvement failed (fallback enabled): strategy={strategy.value}, error={e}")
                return ImprovementResult(
                    original_prompt=prompt,
                    improved_prompt=prompt,
                    strategy=strategy,
                    metadata={"error": str(e)}
                )
            # Log and re-raise when fallback is disabled
            logger.error(f"Prompt improvement failed: strategy={strategy.value}, error={e}")
            raise

    def improve_multi(self, prompt: str, strategies: list[ImprovementStrategy]) -> ImprovementResult:
        """Apply multiple improvement strategies."""
        current = prompt
        all_improvements = []

        for strategy in strategies:
            result = self.improve(current, strategy)
            current = result.improved_prompt
            all_improvements.extend(result.improvements_made)

        return ImprovementResult(
            original_prompt=prompt,
            improved_prompt=current,
            strategy=ImprovementStrategy.AUTO,
            improvements_made=all_improvements
        )

    def auto_improve(self, prompt: str) -> ImprovementResult:
        """Automatically improve a prompt."""
        return self.improve(prompt, strategy=ImprovementStrategy.AUTO)

    def improve_iterative(self, prompt: str, max_iterations: int = 3,
                         target_score: float = 0.8) -> list[ImprovementResult]:
        """Iteratively improve a prompt."""
        results = []
        current = prompt

        for _ in range(max_iterations):
            analysis = self.analyze(current)
            if analysis.overall_score() >= target_score:
                break

            result = self.auto_improve(current)
            results.append(result)
            current = result.improved_prompt

        return results

    async def improve_async(self, prompt: str, strategy: ImprovementStrategy) -> ImprovementResult:
        """Asynchronously improve a prompt."""
        # Simulate async operation
        await asyncio.sleep(0.01)
        return self.improve(prompt, strategy)

    def improve_batch(self, prompts: list[str], strategy: ImprovementStrategy) -> list[ImprovementResult]:
        """Improve multiple prompts."""
        return [self.improve(p, strategy) for p in prompts]

    def compare(self, prompt1: str, prompt2: str) -> dict[str, Any]:
        """Compare two prompts."""
        analysis1 = self.analyze(prompt1)
        analysis2 = self.analyze(prompt2)

        return {
            "prompt1": analysis1.to_dict(),
            "prompt2": analysis2.to_dict(),
            "winner": "prompt2" if analysis2.overall_score() > analysis1.overall_score() else "prompt1"
        }

    def validate_improvement(self, original: str, improved: str) -> bool:
        """Validate that an improvement is actually better."""
        original_analysis = self.analyze(original)
        improved_analysis = self.analyze(improved)
        return improved_analysis.overall_score() > original_analysis.overall_score()

    def improve_with_llm(self, prompt: str, model: str = "gpt-4") -> ImprovementResult:
        """Improve using LLM assistance."""
        improved = self.llm_client.generate(prompt) if self.llm_client else prompt + " (improved with LLM)"

        return ImprovementResult(
            original_prompt=prompt,
            improved_prompt=improved,
            strategy=ImprovementStrategy.AUTO
        )

    ####################################################################################################################
    # Analysis Helper Methods

    def analyze_readability(self, prompt: str) -> float:
        """Analyze readability score."""
        # Simple heuristic based on word complexity
        complex_words = ["endeavor", "synthesize", "comprehensive", "facilitate"]
        score = 1.0
        for word in complex_words:
            if word in prompt.lower():
                score -= 0.1
        return max(0, score)

    def detect_ambiguity(self, prompt: str) -> list[str]:
        """Detect ambiguous language."""
        issues = []
        ambiguous = ["thing", "stuff", "it", "there", "do"]
        for word in ambiguous:
            if word in prompt.lower().split():
                issues.append(f"Ambiguous term: {word}")
        return issues

    def check_completeness(self, prompt: str) -> float:
        """Check prompt completeness."""
        # Higher score for longer, more detailed prompts
        word_count = len(prompt.split())
        if word_count < 3:
            return 0.2
        elif word_count < 10:
            return 0.5
        elif word_count < 30:
            return 0.8
        else:
            return 1.0

    def check_task_alignment(self, prompt: str, task: str) -> float:
        """Check alignment between prompt and task."""
        # Simple keyword matching
        task_words = task.lower().split()
        prompt_words = prompt.lower().split()
        matches = sum(1 for word in task_words if word in prompt_words)
        return min(1.0, matches / max(1, len(task_words)))

    def fuzzy_compare_outputs(self, actual: dict, expected: dict, threshold: float = 0.9) -> bool:
        """Fuzzy comparison of outputs."""
        if "text" in actual and "text" in expected:
            # Simple similarity check
            a = actual["text"].lower()
            b = expected["text"].lower()
            return a in b or b in a or abs(len(a) - len(b)) < 2
        return False

    ####################################################################################################################
    # Strategy Implementation

    def _apply_strategy(self, prompt: str, strategy: ImprovementStrategy,
                       context: dict = None, examples: list = None,
                       constraints: list = None) -> str:
        """Apply improvement strategy."""
        if strategy == ImprovementStrategy.CLARITY:
            return self._improve_clarity(prompt)
        elif strategy == ImprovementStrategy.SPECIFICITY:
            return self._improve_specificity(prompt, context)
        elif strategy == ImprovementStrategy.STRUCTURE:
            return self._improve_structure(prompt)
        elif strategy == ImprovementStrategy.ADD_EXAMPLES and examples:
            return self._add_examples(prompt, examples)
        elif strategy == ImprovementStrategy.ADD_CONSTRAINTS and constraints:
            return self._add_constraints(prompt, constraints)
        elif strategy == ImprovementStrategy.CHAIN_OF_THOUGHT:
            return self._add_chain_of_thought(prompt)
        elif strategy == ImprovementStrategy.SIMPLIFY:
            return self._simplify(prompt)
        elif strategy == ImprovementStrategy.EXPAND:
            return self._expand(prompt)
        elif strategy == ImprovementStrategy.FORMALIZE:
            return self._formalize(prompt)
        elif strategy == ImprovementStrategy.TECHNICAL:
            return self._make_technical(prompt, context)
        else:
            return prompt

    def _improve_clarity(self, prompt: str) -> str:
        """Improve clarity."""
        # Remove hedge words
        improved = prompt
        improved = improved.replace("maybe", "")
        improved = improved.replace("perhaps", "")
        improved = improved.replace("try to", "")
        improved = improved.replace("if you can", "")

        # If no changes were made, add clarity instruction
        if improved == prompt and not prompt.startswith("Please"):
            improved = f"Please {prompt.lower()}"

        return improved.strip()

    def _improve_specificity(self, prompt: str, context: dict = None) -> str:
        """Improve specificity."""
        improved = prompt
        if context:
            if "task" in context:
                improved = f"Task: {context['task']}\n\n{improved}"
            if "data_type" in context:
                improved += f"\n\nData type: {context['data_type']}"

        # If no context or no changes, add specificity instructions
        if improved == prompt:
            # Look for vague terms and replace them
            vague_terms = {
                "stuff": "specific items",
                "things": "particular objects",
                "do": "perform",
                "something": "a specific task",
                "it": "the target item"
            }
            for vague, specific in vague_terms.items():
                if vague in improved.lower():
                    improved = improved.replace(vague, specific)

            # If still no changes, add instruction for specificity
            if improved == prompt:
                improved = f"Specifically: {prompt}"

        return improved

    def _improve_structure(self, prompt: str) -> str:
        """Improve structure."""
        # Add structure markers
        sentences = prompt.split(". ")
        if len(sentences) > 1:
            structured = "Steps:\n"
            for i, sentence in enumerate(sentences, 1):
                structured += f"{i}. {sentence}\n"
            return structured

        # If single sentence, add basic structure
        if "\n" not in prompt:
            return f"Task:\n{prompt}\n\nInstructions:\n1. Complete the task as specified"

        return prompt

    def _add_examples(self, prompt: str, examples: list[dict]) -> str:
        """Add examples."""
        improved = prompt + "\n\nExamples:\n"
        for ex in examples:
            improved += f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}\n\n"
        return improved

    def _add_constraints(self, prompt: str, constraints: list[str]) -> str:
        """Add constraints."""
        improved = prompt + "\n\nConstraints:\n"
        for constraint in constraints:
            improved += f"- {constraint}\n"
        return improved

    def _add_chain_of_thought(self, prompt: str) -> str:
        """Add chain-of-thought if enabled by policy."""
        if not self.enable_chain_of_thought:
            return prompt
        return prompt + "\n\nLet's think step by step:"

    def _simplify(self, prompt: str) -> str:
        """Simplify prompt."""
        # Remove complex words
        replacements = {
            "facilitate": "help with",
            "endeavor": "try",
            "utilize": "use",
            "comprehensive": "complete"
        }
        for old, new in replacements.items():
            prompt = prompt.replace(old, new)
        return prompt

    def _expand(self, prompt: str) -> str:
        """Expand prompt."""
        return f"Please provide a detailed response.\n\n{prompt}\n\nInclude specific examples and explanations."

    def _formalize(self, prompt: str) -> str:
        """Formalize prompt."""
        # Fix informal language
        replacements = {
            "hey": "Greetings",
            "u": "you",
            "plz": "please",
            "ok": "acceptable"
        }
        for old, new in replacements.items():
            prompt = prompt.replace(old, new)
        return prompt

    def _make_technical(self, prompt: str, context: dict = None) -> str:
        """Make prompt more technical."""
        improved = prompt
        if context and "task" in context and "unit testing" in context["task"]:
            improved += "\n\nInclude test cases, assertions, edge cases, exception handling, and validation."
        return improved

    ####################################################################################################################
    # Utility Methods

    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score."""
        # Simple heuristic
        hedge_words = ["maybe", "perhaps", "possibly", "might"]
        score = 1.0
        for word in hedge_words:
            if word in prompt.lower():
                score -= 0.1
        return max(0, score)

    def _calculate_specificity_score(self, prompt: str) -> float:
        """Calculate specificity score."""
        vague_words = ["something", "thing", "stuff", "do"]
        score = 1.0
        for word in vague_words:
            if word in prompt.lower():
                score -= 0.15
        return max(0, score)

    def _calculate_structure_score(self, prompt: str) -> float:
        """Calculate structure score."""
        # Check for structure indicators
        if any(marker in prompt for marker in ["1.", "Step", "First", "Then", "Finally"]):
            return 0.9
        elif "." in prompt:
            return 0.7
        else:
            return 0.5

    def _calculate_completeness_score(self, prompt: str) -> float:
        """Calculate completeness score."""
        return self.check_completeness(prompt)

    def _select_best_strategy(self, prompt: str, analysis: PromptAnalysis) -> ImprovementStrategy:
        """Select best improvement strategy."""
        if analysis.clarity_score < 0.6:
            return ImprovementStrategy.CLARITY
        elif analysis.specificity_score < 0.6:
            return ImprovementStrategy.SPECIFICITY
        elif analysis.structure_score < 0.6:
            return ImprovementStrategy.STRUCTURE
        else:
            return ImprovementStrategy.EXPAND

    def _list_improvements(self, original: str, improved: str) -> list[str]:
        """List improvements made."""
        improvements = []
        if len(improved) > len(original):
            improvements.append("Added detail")
        if "step" in improved.lower() and "step" not in original.lower():
            improvements.append("Added structure")
        if "example" in improved.lower() and "example" not in original.lower():
            improvements.append("Added examples")
        return improvements if improvements else ["General improvements"]

    ####################################################################################################################
    # Cache and Export Methods

    def clear_cache(self):
        """Clear the cache."""
        if self.cache is not None:
            self.cache.clear()

    def export_improvements(self, results: list[ImprovementResult], format: str = "json") -> str:
        """Export improvements."""
        if format == "json":
            data = []
            for result in results:
                data.append({
                    "original_prompt": result.original_prompt,
                    "improved_prompt": result.improved_prompt,
                    "strategy": result.strategy.value,
                    "improvements_made": result.improvements_made,
                    "score_change": result.score_change
                })
            return json.dumps(data, indent=2)
        return ""

    def import_strategies(self, strategies: dict[str, Any]):
        """Import custom strategies."""
        self.custom_strategies = strategies

    def create_pipeline(self, strategies: list[ImprovementStrategy]):
        """Create improvement pipeline."""
        class Pipeline:
            def __init__(self, improver, strategies):
                self.improver = improver
                self.strategies = strategies

            def run(self, prompt: str, **kwargs) -> ImprovementResult:
                return self.improver.improve_multi(prompt, self.strategies)

        return Pipeline(self, strategies)

    ####################################################################################################################
    # Improvement Methods

    def improve_prompt(self, prompt_id: int, strategies: list[str] = None,
                            model_name: str = "gpt-4") -> dict[str, Any]:
        """
        Improve an existing prompt using specified strategies.

        Args:
            prompt_id: ID of prompt to improve
            strategies: List of improvement strategies to apply
            model_name: Model to use for improvement

        Returns:
            Improved prompt data
        """
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()

                # Get existing prompt
                cursor.execute("""
                    SELECT id, project_id, signature_id, name, system_prompt,
                           user_prompt, few_shot_examples
                    FROM prompt_studio_prompts
                    WHERE id = ? AND deleted = 0
                """, (prompt_id,))

                prompt_data = cursor.fetchone()
                if not prompt_data:
                    raise ValueError(f"Prompt {prompt_id} not found")

                # Default strategies if none specified
                if not strategies:
                    strategies = ["clarity", "structure", "specificity"]

                # Apply improvements
                improved_system = self._improve_text(
                    prompt_data[4],  # system_prompt
                    strategies,
                    "system prompt",
                    model_name
                )

                improved_user = self._improve_text(
                    prompt_data[5],  # user_prompt
                    strategies,
                    "user prompt",
                    model_name
                )

                # Create new version
                cursor.execute("""
                    INSERT INTO prompt_studio_prompts (
                        uuid, project_id, signature_id, name, system_prompt,
                        user_prompt, parent_version_id, change_description,
                        version_number, client_id
                    ) VALUES (
                        lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?,
                        (SELECT COALESCE(MAX(version_number), 0) + 1
                         FROM prompt_studio_prompts
                         WHERE project_id = ? AND name = ?),
                        ?
                    )
                """, (
                    prompt_data[1],  # project_id
                    prompt_data[2],  # signature_id
                    prompt_data[3],  # name
                    improved_system,
                    improved_user,
                    prompt_id,
                    f"Improved using strategies: {', '.join(strategies)}",
                    prompt_data[1], prompt_data[3],  # for version subquery
                    self.client_id
                ))

                new_prompt_id = cursor.lastrowid

            logger.info(f"Created improved version {new_prompt_id} of prompt {prompt_id}")

            return {
                "id": new_prompt_id,
                "parent_id": prompt_id,
                "system_prompt": improved_system,
                "user_prompt": improved_user,
                "strategies_applied": strategies,
                "improvements": self._analyze_improvements(
                    prompt_data[4], improved_system,
                    prompt_data[5], improved_user
                )
            }

        except LLMPolicyRefusal:
            raise  # moderation refusals reach the caller as 400, not as a database error
        except Exception as e:
            logger.error(f"Failed to improve prompt: {e}")
            raise DatabaseError(f"Failed to improve prompt: {e}") from e

    def analyze_prompt(self, prompt_id: int, model_name: str = "gpt-4") -> dict[str, Any]:
        """
        Analyze a prompt for potential improvements.

        Args:
            prompt_id: ID of prompt to analyze
            model_name: Model to use for analysis

        Returns:
            Analysis results
        """
        try:
            # Get prompt
            with self.db.transaction() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT system_prompt, user_prompt
                    FROM prompt_studio_prompts
                    WHERE id = ? AND deleted = 0
                """, (prompt_id,))

                prompt_data = cursor.fetchone()
                if not prompt_data:
                    raise ValueError(f"Prompt {prompt_id} not found")

            # Analyze with LLM (outside transaction to avoid holding locks)
            analysis_prompt = f"""Analyze this prompt for potential improvements:

SYSTEM PROMPT:
{prompt_data[0]}

USER PROMPT:
{prompt_data[1]}

Please identify:
1. Strengths of the current prompt
2. Potential weaknesses or areas for improvement
3. Specific suggestions for each weakness
4. Priority of improvements (high/medium/low)

Format as JSON.
"""

            response = _call_openai_adapter(
                {
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "system_message": "You are an expert prompt analyst.",
                    "model": model_name,
                    "temperature": 0.7,
                }
            )

            response_text = self._extract_llm_content(response)

            # Parse response
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                analysis = {"raw_analysis": response_text}

            return {
                "prompt_id": prompt_id,
                "analysis": analysis,
                "recommended_strategies": self._recommend_strategies(analysis)
            }

        except LLMPolicyRefusal:
            raise  # moderation refusals reach the caller as 400, not as a database error
        except Exception as e:
            logger.error(f"Failed to analyze prompt: {e}")
            raise DatabaseError(f"Failed to analyze prompt: {e}") from e

    def standardize_to_xml(self, prompt_id: int) -> dict[str, Any]:
        """
        Convert a prompt to XML format for better structure.

        Args:
            prompt_id: ID of prompt to standardize

        Returns:
            Standardized prompt data
        """
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()

                # Get prompt
                cursor.execute("""
                    SELECT project_id, signature_id, name, system_prompt, user_prompt
                    FROM prompt_studio_prompts
                    WHERE id = ? AND deleted = 0
                """, (prompt_id,))

                prompt_data = cursor.fetchone()
                if not prompt_data:
                    raise ValueError(f"Prompt {prompt_id} not found")

                # Convert to XML format
                xml_system = f"""<role>
    You are an AI assistant that follows structured instructions precisely.
</role>

<capabilities>
    {prompt_data[3]}
</capabilities>"""

                xml_user = self._convert_to_xml_format(prompt_data[4])

                # Create new version
                cursor.execute("""
                    INSERT INTO prompt_studio_prompts (
                        uuid, project_id, signature_id, name, system_prompt,
                        user_prompt, parent_version_id, change_description,
                        version_number, client_id
                    ) VALUES (
                        lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?,
                        (SELECT COALESCE(MAX(version_number), 0) + 1
                         FROM prompt_studio_prompts
                         WHERE project_id = ? AND name = ?),
                        ?
                    )
                """, (
                    prompt_data[0], prompt_data[1], prompt_data[2],
                    xml_system, xml_user, prompt_id,
                    "Standardized to XML format",
                    prompt_data[0], prompt_data[2],
                    self.client_id
                ))

                new_prompt_id = cursor.lastrowid

            return {
                "id": new_prompt_id,
                "parent_id": prompt_id,
                "system_prompt": xml_system,
                "user_prompt": xml_user,
                "format": "xml"
            }

        except Exception as e:
            logger.error(f"Failed to standardize prompt: {e}")
            raise DatabaseError(f"Failed to standardize prompt: {e}") from e

    ####################################################################################################################
    # Helper Methods

    def _improve_text(self, text: str, strategies: list[str],
                           text_type: str, model_name: str) -> str:
        """
        Improve a text using specified strategies.

        Args:
            text: Text to improve
            strategies: Improvement strategies
            text_type: Type of text (for context)
            model_name: Model to use

        Returns:
            Improved text
        """
        # Build improvement instructions
        instructions = []
        for strategy in strategies:
            if strategy in IMPROVEMENT_STRATEGIES:
                strat_info = IMPROVEMENT_STRATEGIES[strategy]
                instructions.append(f"\n{strat_info['focus']}:")
                for technique in strat_info['techniques']:
                    instructions.append(f"  - {technique}")

        improvement_prompt = f"""Improve this {text_type} using these strategies:
{''.join(instructions)}

Original text:
{text}

Provide only the improved text, no explanations.
"""

        response = _call_openai_adapter(
            {
                "messages": [{"role": "user", "content": improvement_prompt}],
                "system_message": "You are an expert prompt engineer focused on improvement.",
                "model": model_name,
                "temperature": 0.7,
            }
        )

        improved_text = self._extract_llm_content(response)
        return improved_text.strip()

    def _extract_llm_content(self, response: Any) -> str:
        """
        Normalize an OpenAI-style chat completion response to raw text content.
        """
        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            choices = response.get("choices") or []
            for choice in choices:
                message = choice.get("message")
                if message:
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                        if parts:
                            return "".join(parts)
                delta = choice.get("delta")
                if delta:
                    delta_content = delta.get("content")
                    if isinstance(delta_content, str):
                        return delta_content
                    if isinstance(delta_content, list):
                        parts = [part.get("text", "") for part in delta_content if isinstance(part, dict)]
                        if parts:
                            return "".join(parts)

        raise ValueError("LLM response did not contain any assistant content.")

    def _analyze_improvements(self, old_system: str, new_system: str,
                            old_user: str, new_user: str) -> dict[str, Any]:
        """
        Analyze the improvements made.

        Args:
            old_system: Original system prompt
            new_system: Improved system prompt
            old_user: Original user prompt
            new_user: Improved user prompt

        Returns:
            Analysis of improvements
        """
        return {
            "system_prompt": {
                "length_change": len(new_system) - len(old_system),
                "length_ratio": len(new_system) / max(len(old_system), 1)
            },
            "user_prompt": {
                "length_change": len(new_user) - len(old_user),
                "length_ratio": len(new_user) / max(len(old_user), 1)
            },
            "total_length_change": (len(new_system) + len(new_user)) - (len(old_system) + len(old_user))
        }

    def _recommend_strategies(self, analysis: dict[str, Any]) -> list[str]:
        """
        Recommend improvement strategies based on analysis.

        Args:
            analysis: Prompt analysis results

        Returns:
            List of recommended strategies
        """
        recommendations = []

        # Simple heuristic-based recommendations
        if "weaknesses" in analysis:
            weaknesses = analysis.get("weaknesses", [])
            if isinstance(weaknesses, list):
                for weakness in weaknesses:
                    weakness_lower = str(weakness).lower()
                    if "unclear" in weakness_lower or "ambiguous" in weakness_lower:
                        recommendations.append("clarity")
                    if "structure" in weakness_lower or "organization" in weakness_lower:
                        recommendations.append("structure")
                    if "specific" in weakness_lower or "vague" in weakness_lower:
                        recommendations.append("specificity")
                    if "edge" in weakness_lower or "error" in weakness_lower:
                        recommendations.append("robustness")
                    if "long" in weakness_lower or "verbose" in weakness_lower:
                        recommendations.append("efficiency")

        # Remove duplicates while preserving order
        seen = set()
        return [x for x in recommendations if not (x in seen or seen.add(x))]

    def _convert_to_xml_format(self, text: str) -> str:
        """
        Convert text to XML format.

        Args:
            text: Text to convert

        Returns:
            XML-formatted text
        """
        import re
        from html import escape as html_escape

        def sanitize_tag_name(name: str) -> str:
            """Sanitize a string to be a valid XML tag name."""
            # Remove leading/trailing whitespace
            name = name.strip()
            # Replace spaces with underscores
            name = name.replace(' ', '_')
            # Remove characters that are invalid in XML tag names
            # Valid: letters, digits, hyphens, underscores, periods (but not at start)
            name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)
            # Tag names can't start with digits, hyphens, or periods
            name = re.sub(r'^[^a-zA-Z_]+', '', name)
            # Default if empty after sanitization
            return name.lower() if name else 'section'

        def escape_xml_content(content: str) -> str:
            """Escape special XML characters in content."""
            return html_escape(content, quote=False)

        # Simple conversion - could be enhanced with more sophisticated parsing
        lines = text.split('\n')
        xml_parts = ["<instructions>"]

        for line in lines:
            line = line.strip()
            if line:
                if line.endswith(':'):
                    # Section header - sanitize tag name
                    tag = sanitize_tag_name(line[:-1])
                    xml_parts.append(f"  <{tag}>")
                else:
                    # Content - escape special characters
                    escaped_line = escape_xml_content(line)
                    xml_parts.append(f"    {escaped_line}")

        xml_parts.append("</instructions>")

        return '\n'.join(xml_parts)
