import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


class AbstractiveSummarizationBenchmark:
    def __init__(self, dataset_name, subset=None, split='test'):
        self.dataset = load_dataset(dataset_name, subset, split=split)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()

    def evaluate_model(self, model, num_samples=100):
        results = {
            'rouge1': [], 'rouge2': [], 'rougeL': [],
            'bleu': [], 'bert_score': [], 'meteor': []
        }

        for i in range(min(num_samples, len(self.dataset))):
            article = self.dataset[i]['article']
            reference = self.dataset[i]['highlights']

            # Generate summary using the provided model
            generated_summary = model(article)

            # Calculate ROUGE scores
            rouge_scores = self.scorer.score(reference, generated_summary)
            results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL'].append(rouge_scores['rougeL'].fmeasure)

            # Calculate BLEU score
            bleu_score = self.bleu.sentence_score(generated_summary, [reference]).score
            results['bleu'].append(bleu_score)

            # Calculate BERTScore
            p, r, f1 = score([generated_summary], [reference], lang='en')
            results['bert_score'].append(f1.item())

            # Calculate METEOR score
            meteor = meteor_score([reference.split()], generated_summary.split())
            results['meteor'].append(meteor)

        # Calculate average scores
        for metric in results:
            results[metric] = np.mean(results[metric])

        return results

    def run_benchmark(self, models):
        benchmark_results = {}
        for model_name, model in models.items():
            print(f"Evaluating model: {model_name}")
            benchmark_results[model_name] = self.evaluate_model(model)
        return benchmark_results


# Example usage:
def dummy_model(article):
    # This is a placeholder for an actual summarization model
    return "This is a dummy summary."


if __name__ == "__main__":
    # Initialize the benchmark with a dataset (e.g., CNN/DailyMail)
    benchmark = AbstractiveSummarizationBenchmark("cnn_dailymail", "3.0.0")

    # Define models to evaluate
    models = {
        "Dummy Model": dummy_model,
        # Add more models here
    }

    # Run the benchmark
    results = benchmark.run_benchmark(models)

    # Print results
    for model_name, scores in results.items():
        print(f"\nResults for {model_name}:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")


#
# To set up this evaluation benchmark effectively, follow these steps:
#
# Dataset Selection:
#
# Choose appropriate datasets for summarization. Popular options include CNN/DailyMail, XSum, or NEWSROOM.
# In the example, we're using the CNN/DailyMail dataset, but you can easily switch to others.
#
#
# Evaluation Metrics:
#
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures overlap between generated and reference summaries.
# BLEU (Bilingual Evaluation Understudy): Originally for machine translation, but also used for summarization.
# BERTScore: Uses contextual embeddings to compute similarity.
# METEOR (Metric for Evaluation of Translation with Explicit ORdering): Considers synonyms and paraphrases.
#
#
# Human Evaluation:
#
# While not included in the code, human evaluation is crucial for assessing readability, coherence, and factual accuracy.
# Create a subset of generated summaries for human raters to score on various aspects.
#
#
# Implementing the Benchmark:
#
# Use the provided AbstractiveSummarizationBenchmark class.
# Replace the dummy_model with actual summarization models you want to evaluate.
#
#
# Running the Benchmark:
#
# Instantiate the benchmark with your chosen dataset.
# Define a dictionary of models to evaluate.
# Run the benchmark and analyze the results.
#
#
# Analyzing Results:
#
# Compare the performance of different models across all metrics.
# Consider the strengths and weaknesses of each model based on different evaluation aspects.
#
#
# Iterative Improvement:
#
# Use benchmark results to identify areas for improvement in your summarization models.
# Continuously update your benchmark with new datasets and metrics as the field evolves.
#
#
#
# To enhance this benchmark further:
#
# Add more datasets to test generalization across different domains.
# Implement a human evaluation component with a user interface for manual scoring.
# Include additional metrics like summary length analysis, readability scores, or factual consistency checks.
# Create visualizations to better compare model performances.
# Implement statistical significance tests to ensure differences between models are meaningful.










################################################################################################################################################################################################










# The code and instructions you've received provide a solid foundation for an abstractive summarization benchmark. It's a well-structured approach that incorporates several important evaluation metrics commonly used in the field. Here's an analysis of its correctness and helpfulness, along with some suggestions for improvement:
# Correctness and Helpfulness:
#
# The code correctly implements a variety of evaluation metrics (ROUGE, BLEU, BERTScore, and METEOR) which are indeed relevant for summarization tasks.
# The use of the Hugging Face datasets library for loading standard datasets is appropriate and convenient.
# The structure allows for easy addition of multiple models for comparison.
# The instructions provide a good overview of the steps needed to set up and run the benchmark, including important considerations like dataset selection and human evaluation.
#
# Suggestions for Improvement:
#
# Error Handling and Logging:
# Add more robust error handling and logging to catch and report issues during evaluation.
# Parallelization:
# Consider adding parallel processing to speed up evaluation, especially for large datasets or multiple models.
# Customizable Metric Weights:
# Allow users to specify weights for different metrics to calculate a weighted overall score.
# Factual Consistency:
# Implement a metric for factual consistency, which is crucial for summarization. You could use models like BERT-Score with semantic similarity for this purpose.
# Length Analysis:
# Add metrics to evaluate summary length and compression ratio.
# Confidence Intervals:
# Include calculation of confidence intervals for the metrics to give a sense of the reliability of the scores.
# Data Preprocessing:
# Add options for text preprocessing (e.g., lowercasing, removing special characters) to ensure fair comparison.
# Model Loading:
# Instead of passing model functions, consider accepting model names and loading them dynamically (e.g., from Hugging Face's model hub).
# Output Formats:
# Provide options to save results in various formats (JSON, CSV, etc.) for easier analysis and visualization.
# Configurable Parameters:
# Make parameters like number of samples configurable through command-line arguments or a config file.
# Progress Tracking:
# Add a progress bar or periodic updates for long-running evaluations.
#
#
# This improved version includes:
#
# Parallelization using ProcessPoolExecutor
# Progress tracking with tqdm
# Confidence interval calculation
# Overall score calculation with customizable metric weights
# Text preprocessing (though currently just lowercasing)
# Dynamic model loading from Hugging Face
# Command-line argument parsing for configurability
# JSON output for results
#
# These improvements make the benchmark more robust, flexible, and informative. You can further enhance it by adding more metrics, implementing factual consistency checks, or integrating a human evaluation component.

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from tqdm import tqdm

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class AbstractiveSummarizationBenchmark:
    def __init__(self, dataset_name, subset=None, split='test', metric_weights=None):
        self.dataset = load_dataset(dataset_name, subset, split=split)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        self.metric_weights = metric_weights or {'rouge1': 1, 'rouge2': 1, 'rougeL': 1, 'bleu': 1, 'bert_score': 1,
                                                 'meteor': 1}

    def preprocess_text(self, text):
        # Add any text preprocessing steps here
        return text.lower()

    def evaluate_model(self, model, tokenizer, num_samples=100):
        results = {
            'rouge1': [], 'rouge2': [], 'rougeL': [],
            'bleu': [], 'bert_score': [], 'meteor': []
        }

        for i in tqdm(range(min(num_samples, len(self.dataset))), desc="Evaluating"):
            article = self.preprocess_text(self.dataset[i]['article'])
            reference = self.preprocess_text(self.dataset[i]['highlights'])

            inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0,
                                         num_beams=4, early_stopping=True)
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            rouge_scores = self.scorer.score(reference, generated_summary)
            results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL'].append(rouge_scores['rougeL'].fmeasure)

            results['bleu'].append(self.bleu.sentence_score(generated_summary, [reference]).score)

            _, _, f1 = score([generated_summary], [reference], lang='en')
            results['bert_score'].append(f1.item())

            results['meteor'].append(meteor_score([reference.split()], generated_summary.split()))

        # Calculate average scores and confidence intervals
        for metric in results:
            scores = np.array(results[metric])
            results[metric] = {
                'mean': np.mean(scores),
                'ci': (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
            }

        return results

    def calculate_overall_score(self, results):
        overall_score = sum(results[metric]['mean'] * weight for metric, weight in self.metric_weights.items())
        return overall_score / sum(self.metric_weights.values())

    def run_benchmark(self, models: Dict[str, Any], num_samples: int = 100):
        benchmark_results = {}
        with ProcessPoolExecutor() as executor:
            future_to_model = {executor.submit(self.evaluate_model, model, tokenizer, num_samples): model_name
                               for model_name, (model, tokenizer) in models.items()}
            for future in tqdm(future_to_model, desc="Models Evaluated"):
                model_name = future_to_model[future]
                try:
                    benchmark_results[model_name] = future.result()
                    benchmark_results[model_name]['overall_score'] = self.calculate_overall_score(
                        benchmark_results[model_name])
                except Exception as exc:
                    print(f'{model_name} generated an exception: {exc}')
        return benchmark_results


def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Abstractive Summarization Benchmark")
    parser.add_argument("--dataset", default="cnn_dailymail", help="Dataset to use for evaluation")
    parser.add_argument("--subset", default="3.0.0", help="Subset of the dataset to use")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    args = parser.parse_args()

    benchmark = AbstractiveSummarizationBenchmark(args.dataset, args.subset)

    models = {
        "t5-small": load_model("t5-small"),
        "bart-large-cnn": load_model("facebook/bart-large-cnn"),
        # Add more models here
    }

    results = benchmark.run_benchmark(models, args.num_samples)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

    for model_name, scores in results.items():
        print(f"\nResults for {model_name}:")
        for metric, score in scores.items():
            if metric != 'overall_score':
                print(f"{metric}: {score['mean']:.4f} (95% CI: {score['ci'][0]:.4f} - {score['ci'][1]:.4f})")
        print(f"Overall Score: {scores['overall_score']:.4f}")
