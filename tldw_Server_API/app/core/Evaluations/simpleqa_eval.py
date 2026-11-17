"""
SimpleQA Evaluation Module

Implements OpenAI's SimpleQA benchmark for measuring short-form factuality
in large language models.

SimpleQA features:
- 4,326 short, fact-seeking questions
- Single, indisputable answers
- Three-category grading: correct, incorrect, not_attempted
- Adversarially collected against GPT-4
- Wide topic coverage
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Evaluations.benchmark_utils import BaseEvaluation

logger = logger


class SimpleQAEvaluation(BaseEvaluation):
    """Evaluation for SimpleQA factuality benchmark."""

    def __init__(self, name: str = "simpleqa",
                 grading_model: str = "openai",
                 strict_grading: bool = True):
        """Initialize SimpleQA evaluation.

        Args:
            name: Evaluation name
            grading_model: Model to use for grading (openai, anthropic, etc.)
            strict_grading: Whether to use strict grading criteria
        """
        super().__init__(name, "SimpleQA factuality evaluation")
        self.grading_model = grading_model
        self.strict_grading = strict_grading

    def format_for_custom_metric(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Format SimpleQA question for evaluation.

        Args:
            input_data: Contains 'question', 'answer', and optional 'topic'

        Returns:
            Formatted evaluation data
        """
        question = input_data.get("question", "")
        expected_answer = input_data.get("answer", "")
        topic = input_data.get("topic", "general")
        metadata = input_data.get("metadata", {})
        if not isinstance(metadata, dict):  # OpenAI's CSV has a string metadata column
            metadata = {}

        # SimpleQA uses a specific prompt format for evaluation
        evaluation_prompt = self._create_grading_prompt(
            question, expected_answer, "{model_response}"
        )

        return {
            "name": f"{self.name}_factuality",
            "description": f"{self.description} - {topic}",
            "evaluation_prompt": evaluation_prompt,
            "require_grade": True,
            "input_data": {
                "question": question,
                "expected_answer": expected_answer,
                "model_response": "{model_response}",
                "topic": topic
            },
            "scoring_criteria": {
                "factuality": "Answer must be factually correct",
                "completeness": "Answer should include all key facts",
                "accuracy": "No incorrect information",
                "attempt": "Model should attempt to answer"
            },
            "metadata": {
                **metadata,
                "evaluation_type": "simpleqa",
                "grading_model": self.grading_model,
                "strict_grading": self.strict_grading
            }
        }

    def _create_grading_prompt(self, question: str, expected_answer: str,
                              model_response: str) -> str:
        """Create the grading prompt for SimpleQA.

        SimpleQA uses a specific grading approach with three categories:
        - correct: The answer is factually correct
        - incorrect: The answer contains factual errors
        - not_attempted: The model didn't attempt to answer
        """
        if self.strict_grading:
            prompt = f"""Grade this answer as 'correct', 'incorrect', or 'not_attempted'.

Question: {question}
Expected Answer: {expected_answer}
Model Answer: {model_response}

Grading Criteria:
1. CORRECT: The model's answer contains the same factual information as the expected answer.
   - Minor phrasing differences are acceptable
   - Additional correct context is acceptable
   - The core fact must be accurate

2. INCORRECT: The model's answer contains factual errors or contradicts the expected answer.
   - Any factual inaccuracy makes it incorrect
   - Missing critical information makes it incorrect
   - Wrong dates, names, numbers, etc. make it incorrect

3. NOT_ATTEMPTED: The model explicitly refuses to answer or says it doesn't know.
   - "I don't know" or similar responses
   - Refusal to answer
   - Completely off-topic response

Respond with exactly one of these three grades followed by a brief explanation:
GRADE: [correct/incorrect/not_attempted]
EXPLANATION: [1-2 sentences explaining the grade]"""
        else:
            # More lenient grading
            prompt = f"""Evaluate this answer for factual accuracy.

Question: {question}
Expected Answer: {expected_answer}
Model Answer: {model_response}

Is the model's answer factually correct? Consider:
- The core fact should match the expected answer
- Minor variations in phrasing are acceptable
- Additional correct details are fine
- Partial answers can be correct if the stated facts are accurate

Grade as:
- CORRECT: Factually accurate (even if partial)
- INCORRECT: Contains factual errors
- NOT_ATTEMPTED: Refuses to answer or says "I don't know"

GRADE: [correct/incorrect/not_attempted]
EXPLANATION: [brief explanation]"""

        return prompt

    def parse_response(self, response: str) -> str:
        """Parse model response for SimpleQA.

        Args:
            response: Raw model response

        Returns:
            Cleaned response text
        """
        # SimpleQA expects short, direct answers
        # Remove any markdown formatting
        response = re.sub(r'```[^`]*```', '', response)
        response = re.sub(r'`([^`]*)`', r'\1', response)

        # Remove common prefixes
        prefixes_to_remove = [
            "The answer is:",
            "Answer:",
            "The correct answer is:",
            "Based on the question,",
            "According to my knowledge,",
        ]

        response_lower = response.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix.lower()):
                response = response[len(prefix):].strip()
                break

        return response.strip()

    def score(self, predicted: str, expected: str) -> tuple[float, str]:
        """Score a SimpleQA answer.

        This is a simplified scoring for when not using LLM grading.

        Args:
            predicted: Model's answer
            expected: Expected answer

        Returns:
            Tuple of (score, grade) where grade is 'correct', 'incorrect', or 'not_attempted'
        """
        # Check for non-attempt
        non_attempt_phrases = [
            "i don't know",
            "i do not know",
            "i cannot answer",
            "i can't answer",
            "i'm not sure",
            "i am not sure",
            "unable to answer",
            "cannot provide",
            "no information",
            "insufficient information"
        ]

        predicted_lower = predicted.lower().strip()

        for phrase in non_attempt_phrases:
            if phrase in predicted_lower:
                return 0.0, "not_attempted"

        # If response is too short, consider it not attempted
        if len(predicted_lower.split()) < 2:
            return 0.0, "not_attempted"

        # Normalize for comparison
        expected_lower = expected.lower().strip()

        # Check for exact match (after normalization)
        if predicted_lower == expected_lower:
            return 1.0, "correct"

        # Check if expected answer is contained in prediction
        if expected_lower in predicted_lower:
            return 1.0, "correct"

        # Check for common variations
        # Numbers
        if self._normalize_numbers(predicted_lower) == self._normalize_numbers(expected_lower):
            return 1.0, "correct"

        # For more complex cases, we'd need LLM grading
        # Default to incorrect if we can't determine
        return 0.0, "incorrect"

    def _normalize_numbers(self, text: str) -> str:
        """Normalize number representations in text."""
        # Convert written numbers to digits
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12'
        }

        for word, digit in number_map.items():
            text = text.replace(word, digit)

        # Remove commas from numbers
        text = re.sub(r'(\d),(\d)', r'\1\2', text)

        return text

    def parse_grading_response(self, grading_response: str) -> dict[str, Any]:
        """Parse the grading response from LLM.

        Args:
            grading_response: Response from grading LLM

        Returns:
            Dict with grade and explanation
        """
        result = {
            "grade": "incorrect",  # Default
            "explanation": "",
            "score": 0.0
        }

        # Look for grade
        grade_match = re.search(
            r'GRADE:\s*(correct|incorrect|not_attempted)',
            grading_response,
            re.IGNORECASE
        )

        if grade_match:
            grade = grade_match.group(1).lower()
            result["grade"] = grade

            # Assign score based on grade
            if grade == "correct":
                result["score"] = 1.0
            elif grade == "not_attempted":
                result["score"] = 0.0  # Could be different if we want to penalize less
            else:  # incorrect
                result["score"] = 0.0

        # Look for explanation
        explanation_match = re.search(
            r'EXPLANATION:\s*(.+?)(?:\n|$)',
            grading_response,
            re.IGNORECASE | re.DOTALL
        )

        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()

        return result


class SimpleQADataset:
    """Helper class for loading and managing SimpleQA dataset."""

    @staticmethod
    def load_from_file(filepath: str) -> list[dict[str, Any]]:
        """Load SimpleQA dataset from file.

        Args:
            filepath: Path to dataset file (JSON or JSONL)

        Returns:
            List of question-answer pairs
        """
        questions = []
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        # Determine format
        if path.suffix == '.jsonl':
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
        else:  # Assume JSON
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict):
                    # Look for common keys
                    if 'questions' in data:
                        questions = data['questions']
                    elif 'data' in data:
                        questions = data['data']
                    elif 'items' in data:
                        questions = data['items']
                    else:
                        # Assume single question
                        questions = [data]

        return questions

    @staticmethod
    def load_from_huggingface(dataset_id: str = "openai/simple-qa",
                            split: str = "test",
                            revision: str | None = None) -> list[dict[str, Any]]:
        """Load SimpleQA from HuggingFace.

        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split to load

        Returns:
            List of question-answer pairs
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.exception("HuggingFace datasets not installed. Install with: pip install datasets")
            return []

        try:
            resolved_revision = revision or os.getenv("HF_DATASETS_DEFAULT_REVISION")
            dataset = load_dataset(dataset_id, split=split, revision=resolved_revision)  # nosec B615

            questions = []
            for item in dataset:
                # Map to standard format
                question = {
                    "question": item.get("question", item.get("prompt", "")),
                    "answer": item.get("answer", item.get("target", "")),
                    "topic": item.get("topic", item.get("category", "general")),
                    "metadata": {
                        "source": "huggingface",
                        "dataset_id": dataset_id
                    }
                }

                # Include any additional fields
                for key in item:
                    if key not in ["question", "answer", "topic", "prompt", "target", "category"]:
                        question["metadata"][key] = item[key]

                questions.append(question)

            return questions

        except Exception as e:
            logger.exception(f"Failed to load SimpleQA from HuggingFace: {e}")
            return []

    @staticmethod
    def create_sample_dataset() -> list[dict[str, Any]]:
        """Create a sample SimpleQA dataset for testing.

        Returns:
            List of sample questions
        """
        return [
            {
                "question": "What year did World War II end?",
                "answer": "1945",
                "topic": "history"
            },
            {
                "question": "Who painted the Mona Lisa?",
                "answer": "Leonardo da Vinci",
                "topic": "art"
            },
            {
                "question": "What is the capital of Japan?",
                "answer": "Tokyo",
                "topic": "geography"
            },
            {
                "question": "What element has the symbol 'Au'?",
                "answer": "Gold",
                "topic": "science"
            },
            {
                "question": "Who wrote '1984'?",
                "answer": "George Orwell",
                "topic": "literature"
            },
            {
                "question": "What is the speed of light in vacuum?",
                "answer": "299,792,458 meters per second",
                "topic": "physics"
            },
            {
                "question": "In what year was the iPhone first released?",
                "answer": "2007",
                "topic": "technology"
            },
            {
                "question": "What is the largest planet in our solar system?",
                "answer": "Jupiter",
                "topic": "astronomy"
            },
            {
                "question": "Who developed the theory of evolution by natural selection?",
                "answer": "Charles Darwin",
                "topic": "science"
            },
            {
                "question": "What is the currency of Brazil?",
                "answer": "Real",
                "topic": "economics"
            }
        ]


class SimpleQAAnalyzer:
    """Analyzer for SimpleQA results."""

    @staticmethod
    def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze SimpleQA evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Analysis dict with metrics
        """
        if not results:
            return {}

        # Count grades
        grade_counts = {
            "correct": 0,
            "incorrect": 0,
            "not_attempted": 0
        }

        topic_performance = {}

        for result in results:
            grade = result.get("grade", "incorrect")
            grade_counts[grade] += 1

            # Track by topic
            topic = result.get("topic", "unknown")
            if topic not in topic_performance:
                topic_performance[topic] = {
                    "correct": 0,
                    "incorrect": 0,
                    "not_attempted": 0,
                    "total": 0
                }

            topic_performance[topic][grade] += 1
            topic_performance[topic]["total"] += 1

        total = len(results)

        # Calculate metrics
        accuracy = grade_counts["correct"] / total if total > 0 else 0
        attempt_rate = 1 - (grade_counts["not_attempted"] / total) if total > 0 else 0

        # Calculate topic accuracies
        topic_accuracies = {}
        for topic, stats in topic_performance.items():
            if stats["total"] > 0:
                topic_accuracies[topic] = stats["correct"] / stats["total"]

        return {
            "total_questions": total,
            "grade_distribution": grade_counts,
            "accuracy": accuracy,
            "attempt_rate": attempt_rate,
            "correct_rate_attempted": (
                grade_counts["correct"] / (total - grade_counts["not_attempted"])
                if (total - grade_counts["not_attempted"]) > 0 else 0
            ),
            "topic_performance": topic_performance,
            "topic_accuracies": topic_accuracies,
            "best_topic": max(topic_accuracies.items(), key=lambda x: x[1])[0] if topic_accuracies else None,
            "worst_topic": min(topic_accuracies.items(), key=lambda x: x[1])[0] if topic_accuracies else None
        }

    @staticmethod
    def generate_report(results: list[dict[str, Any]],
                       benchmark_name: str = "SimpleQA") -> str:
        """Generate a formatted report of SimpleQA results.

        Args:
            results: List of evaluation results
            benchmark_name: Name for the report

        Returns:
            Formatted report string
        """
        analysis = SimpleQAAnalyzer.analyze_results(results)

        if not analysis:
            return f"{benchmark_name}: No results to analyze"

        report = [
            f"{'=' * 60}",
            f"{benchmark_name} Evaluation Report",
            f"{'=' * 60}",
            "",
            f"Total Questions: {analysis['total_questions']}",
            f"Overall Accuracy: {analysis['accuracy']:.1%}",
            f"Attempt Rate: {analysis['attempt_rate']:.1%}",
            f"Accuracy (excluding non-attempts): {analysis['correct_rate_attempted']:.1%}",
            "",
            "Grade Distribution:",
            f"  Correct: {analysis['grade_distribution']['correct']} ({analysis['grade_distribution']['correct']/analysis['total_questions']:.1%})",
            f"  Incorrect: {analysis['grade_distribution']['incorrect']} ({analysis['grade_distribution']['incorrect']/analysis['total_questions']:.1%})",
            f"  Not Attempted: {analysis['grade_distribution']['not_attempted']} ({analysis['grade_distribution']['not_attempted']/analysis['total_questions']:.1%})",
            ""
        ]

        if analysis.get('topic_accuracies'):
            report.extend([
                "Performance by Topic:",
                "-" * 30
            ])

            for topic, accuracy in sorted(analysis['topic_accuracies'].items(),
                                         key=lambda x: x[1], reverse=True):
                stats = analysis['topic_performance'][topic]
                report.append(
                    f"  {topic}: {accuracy:.1%} "
                    f"({stats['correct']}/{stats['total']} correct)"
                )

            report.extend([
                "",
                f"Best Topic: {analysis['best_topic']} ({analysis['topic_accuracies'][analysis['best_topic']]:.1%})",
                f"Worst Topic: {analysis['worst_topic']} ({analysis['topic_accuracies'][analysis['worst_topic']]:.1%})"
            ])

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
