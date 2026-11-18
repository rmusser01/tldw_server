"""
QA Benchmark Helper - Tools for creating and running custom QA benchmarks.

This module provides utilities for easily creating, customizing, and running
question-answer benchmarks for evaluating language models.
"""

import json
from difflib import SequenceMatcher
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Evaluations.benchmark_utils import JUDGE_SCALE_LINE

logger = logger


class QAEvaluation:
    """Simple QA evaluation with multiple scoring methods."""

    def __init__(self, scoring_method: str = "llm_judge", **kwargs):
        """Initialize QA evaluation.

        Args:
            scoring_method: One of 'exact_match', 'contains', 'semantic', 'llm_judge'
            **kwargs: Additional parameters for scoring
        """
        self.scoring_method = scoring_method
        self.case_sensitive = kwargs.get('case_sensitive', False)
        self.normalize_whitespace = kwargs.get('normalize_whitespace', True)
        self.required_keywords = kwargs.get('required_keywords', [])
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.85)
        self.judge_prompt_template = kwargs.get('judge_prompt_template', self._default_judge_prompt())

    def _default_judge_prompt(self) -> str:
        """Default prompt template for LLM judge scoring."""
        return """Question: {question}
Expected Answer: {expected_answer}
Model Answer: {model_answer}

Evaluate if the model answer is correct and complete.
Consider factual accuracy and coverage of key points.

Score from 1 (worst) to 10 (best) where:
- 10 = Perfectly correct and complete
- 7 = Mostly correct with minor issues
- 5 = Partially correct
- 3 = Mostly incorrect but has some correct elements
- 1 = Completely incorrect

""" + JUDGE_SCALE_LINE

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        return text.strip()

    def score_exact_match(self, model_answer: str, expected_answer: str,
                         alternative_answers: list[str] = None) -> float:
        """Score using exact match.

        Args:
            model_answer: The model's answer
            expected_answer: The expected answer
            alternative_answers: List of acceptable alternative answers

        Returns:
            1.0 if match, 0.0 otherwise
        """
        model_norm = self.normalize_text(model_answer)
        expected_norm = self.normalize_text(expected_answer)

        if model_norm == expected_norm:
            return 1.0

        # Check alternative answers
        if alternative_answers:
            for alt in alternative_answers:
                if model_norm == self.normalize_text(alt):
                    return 1.0

        return 0.0

    def score_contains(self, model_answer: str, expected_answer: str,
                      required_keywords: list[str] = None) -> float:
        """Score based on keyword presence.

        Args:
            model_answer: The model's answer
            expected_answer: The expected answer (used to extract keywords if not provided)
            required_keywords: List of keywords that must appear

        Returns:
            Score based on keyword coverage
        """
        model_norm = self.normalize_text(model_answer)

        # Use provided keywords or extract from expected answer
        if required_keywords:
            keywords = required_keywords
        else:
            # Extract important words from expected answer (simple approach)
            keywords = [w for w in expected_answer.split()
                       if len(w) > 3 and w.lower() not in ['the', 'and', 'or', 'but', 'with']]

        if not keywords:
            # Fallback to checking if expected answer is contained
            expected_norm = self.normalize_text(expected_answer)
            return 1.0 if expected_norm in model_norm else 0.0

        # Calculate coverage
        found = 0
        for keyword in keywords:
            if self.normalize_text(keyword) in model_norm:
                found += 1

        return found / len(keywords)

    def score_similarity(self, model_answer: str, expected_answer: str) -> float:
        """Score using string similarity.

        Args:
            model_answer: The model's answer
            expected_answer: The expected answer

        Returns:
            Similarity score between 0 and 1
        """
        model_norm = self.normalize_text(model_answer)
        expected_norm = self.normalize_text(expected_answer)

        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, model_norm, expected_norm).ratio()

        return similarity

    def format_for_llm_judge(self, question: str, model_answer: str,
                            expected_answer: str) -> str:
        """Format QA pair for LLM judge evaluation.

        Args:
            question: The question
            model_answer: The model's answer
            expected_answer: The expected answer

        Returns:
            Formatted prompt for LLM judge
        """
        return self.judge_prompt_template.format(
            question=question,
            model_answer=model_answer,
            expected_answer=expected_answer
        )

    def score(self, question: str, model_answer: str, expected_answer: str,
             alternative_answers: list[str] = None, **kwargs) -> dict[str, Any]:
        """Score a QA pair using the configured method.

        Args:
            question: The question
            model_answer: The model's answer
            expected_answer: The expected answer
            alternative_answers: List of acceptable alternatives
            **kwargs: Additional scoring parameters

        Returns:
            Dict with score and metadata
        """
        result = {
            "question": question,
            "model_answer": model_answer,
            "expected_answer": expected_answer,
            "scoring_method": self.scoring_method,
            "score": 0.0,
            "explanation": ""
        }

        if self.scoring_method == "exact_match":
            result["score"] = self.score_exact_match(
                model_answer, expected_answer, alternative_answers
            )
            result["explanation"] = "Exact match" if result["score"] == 1.0 else "No match"

        elif self.scoring_method == "contains":
            result["score"] = self.score_contains(
                model_answer, expected_answer, self.required_keywords
            )
            result["explanation"] = f"Keyword coverage: {result['score']:.1%}"

        elif self.scoring_method == "similarity":
            result["score"] = self.score_similarity(model_answer, expected_answer)
            result["explanation"] = f"String similarity: {result['score']:.1%}"

        elif self.scoring_method == "llm_judge":
            # For LLM judge, return the formatted prompt
            # Actual LLM call would be made by the runner
            result["judge_prompt"] = self.format_for_llm_judge(
                question, model_answer, expected_answer
            )
            result["explanation"] = "Requires LLM judge evaluation"

        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        return result


class QADatasetBuilder:
    """Helper for building QA datasets."""

    @staticmethod
    def create_from_pairs(qa_pairs: list[tuple],
                         category: str = "general") -> dict[str, Any]:
        """Create dataset from simple Q&A pairs.

        Args:
            qa_pairs: List of (question, answer) tuples
            category: Category for all questions

        Returns:
            Dataset dict
        """
        questions = []
        for i, (q, a) in enumerate(qa_pairs, 1):
            questions.append({
                "id": i,
                "question": q,
                "answer": a,
                "category": category,
                "difficulty": "medium",
                "alternative_answers": [],
                "context": None
            })

        return {
            "benchmark_name": "Custom QA Dataset",
            "version": "1.0",
            "questions": questions
        }

    @staticmethod
    def create_from_dict(data: dict[str, str],
                        category: str = "general") -> dict[str, Any]:
        """Create dataset from question->answer dictionary.

        Args:
            data: Dict mapping questions to answers
            category: Category for all questions

        Returns:
            Dataset dict
        """
        qa_pairs = list(data.items())
        return QADatasetBuilder.create_from_pairs(qa_pairs, category)

    @staticmethod
    def add_metadata(dataset: dict[str, Any],
                     metadata: dict[str, Any]) -> dict[str, Any]:
        """Add or update metadata in dataset.

        Args:
            dataset: Dataset dict
            metadata: Metadata to add

        Returns:
            Updated dataset
        """
        dataset.update(metadata)
        return dataset

    @staticmethod
    def save_dataset(dataset: dict[str, Any], filepath: str):
        """Save dataset to JSON file.

        Args:
            dataset: Dataset dict
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> dict[str, Any]:
        """Load dataset from JSON file.

        Args:
            filepath: Path to dataset file

        Returns:
            Dataset dict
        """
        with open(filepath) as f:
            return json.load(f)


class QABenchmarkAnalyzer:
    """Analyzer for QA benchmark results."""

    @staticmethod
    def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics from results.

        Args:
            results: List of scoring results

        Returns:
            Metrics dict
        """
        if not results:
            return {}

        scores = [r['score'] for r in results]
        categories = {}
        difficulties = {}

        for result in results:
            # By category
            cat = result.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['score'])

            # By difficulty
            diff = result.get('difficulty', 'unknown')
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result['score'])

        metrics = {
            "total_questions": len(results),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "perfect_scores": sum(1 for s in scores if s == 1.0),
            "failed_scores": sum(1 for s in scores if s == 0.0),
            "by_category": {},
            "by_difficulty": {}
        }

        # Category metrics
        for cat, cat_scores in categories.items():
            metrics["by_category"][cat] = {
                "count": len(cat_scores),
                "average": sum(cat_scores) / len(cat_scores),
                "perfect": sum(1 for s in cat_scores if s == 1.0)
            }

        # Difficulty metrics
        for diff, diff_scores in difficulties.items():
            metrics["by_difficulty"][diff] = {
                "count": len(diff_scores),
                "average": sum(diff_scores) / len(diff_scores),
                "perfect": sum(1 for s in diff_scores if s == 1.0)
            }

        return metrics

    @staticmethod
    def identify_weaknesses(results: list[dict[str, Any]],
                           threshold: float = 0.5) -> list[dict[str, Any]]:
        """Identify questions where the model performed poorly.

        Args:
            results: List of scoring results
            threshold: Score threshold for poor performance

        Returns:
            List of poorly performed questions
        """
        weaknesses = []
        for result in results:
            if result['score'] < threshold:
                weaknesses.append({
                    "question": result['question'],
                    "score": result['score'],
                    "expected": result['expected_answer'],
                    "model_answer": result['model_answer'],
                    "category": result.get('category', 'unknown')
                })

        return sorted(weaknesses, key=lambda x: x['score'])

    @staticmethod
    def generate_report(results: list[dict[str, Any]],
                       benchmark_name: str = "QA Benchmark") -> str:
        """Generate a text report of benchmark results.

        Args:
            results: List of scoring results
            benchmark_name: Name of the benchmark

        Returns:
            Formatted report string
        """
        metrics = QABenchmarkAnalyzer.calculate_metrics(results)
        weaknesses = QABenchmarkAnalyzer.identify_weaknesses(results)

        report = [
            f"{'=' * 60}",
            f"{benchmark_name} Results",
            f"{'=' * 60}",
            "",
            f"Total Questions: {metrics['total_questions']}",
            f"Average Score: {metrics['average_score']:.2%}",
            f"Perfect Answers: {metrics['perfect_scores']} ({metrics['perfect_scores']/metrics['total_questions']:.1%})",
            f"Failed Answers: {metrics['failed_scores']} ({metrics['failed_scores']/metrics['total_questions']:.1%})",
            "",
            "Performance by Category:",
            "-" * 30
        ]

        for cat, stats in metrics['by_category'].items():
            report.append(f"  {cat}: {stats['average']:.2%} (n={stats['count']})")

        if metrics['by_difficulty']:
            report.extend([
                "",
                "Performance by Difficulty:",
                "-" * 30
            ])
            for diff, stats in metrics['by_difficulty'].items():
                report.append(f"  {diff}: {stats['average']:.2%} (n={stats['count']})")

        if weaknesses:
            report.extend([
                "",
                f"Top {min(5, len(weaknesses))} Weakest Answers:",
                "-" * 30
            ])
            for w in weaknesses[:5]:
                report.append(f"  Q: {w['question'][:50]}...")
                report.append(f"     Score: {w['score']:.2%} | Category: {w['category']}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


# Example usage script
def create_example_benchmark():
    """Create an example QA benchmark."""

    # Create simple QA pairs
    qa_pairs = [
        ("What is 2 + 2?", "4"),
        ("What color is the sky?", "Blue"),
        ("Who was the first president of the United States?", "George Washington"),
        ("What is the largest mammal?", "Blue whale"),
        ("How many days are in a week?", "7")
    ]

    # Build dataset
    dataset = QADatasetBuilder.create_from_pairs(qa_pairs, category="basic_knowledge")

    # Add metadata
    dataset = QADatasetBuilder.add_metadata(dataset, {
        "benchmark_name": "My First QA Benchmark",
        "author": "Your Name",
        "description": "A simple benchmark to test basic knowledge"
    })

    # Save dataset
    QADatasetBuilder.save_dataset(dataset, "my_qa_benchmark.json")

    print("Benchmark created successfully!")
    print(f"Total questions: {len(dataset['questions'])}")

    return dataset


if __name__ == "__main__":
    # Example: Create a benchmark
    dataset = create_example_benchmark()

    # Example: Score an answer
    evaluator = QAEvaluation(scoring_method="exact_match")
    result = evaluator.score(
        question="What is 2 + 2?",
        model_answer="4",
        expected_answer="4"
    )
    print(f"\nScoring result: {result}")

    # Example: Analyze results
    sample_results = [
        {"question": "Q1", "score": 1.0, "category": "math", "model_answer": "4", "expected_answer": "4"},
        {"question": "Q2", "score": 0.5, "category": "science", "model_answer": "blue", "expected_answer": "blue"},
        {"question": "Q3", "score": 0.0, "category": "math", "model_answer": "wrong", "expected_answer": "right"}
    ]

    report = QABenchmarkAnalyzer.generate_report(sample_results, "Sample Benchmark")
    print(f"\n{report}")
