#######################################################################################################################
#
# Evaluations_Benchmarks_tab.py
#
# Description: This file contains the code to evaluate the generated text using G-Eval metric.
#
# Scripts taken from https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-summarization and modified.
#
import configparser
import inspect
import json
import logging
import os
import re
from typing import Dict, Callable, List, Any

import gradio as gr
from tenacity import (
    RetryError,
    Retrying,
    after_log,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
)
#from tldw_Server_API.app.core.Chat.Chat_Functions import (
#    chat_api_call
#)
#
#######################################################################################################################
#
# Start of G-Eval.py

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the config file
config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
# Read the config file
config = configparser.ConfigParser()
config.read(config_path)


def aggregate(
    fluency_list: List[float],
    consistency_list: List[float],
    relevance_list: List[float],
    coherence_list: List[float],
) -> Dict[str, float]:
    """
    Takes list of scores for 4 dims and outputs average for them.

    Args:
        fluency_list (List(float)): list of fluency scores
        consistency_list (List(float)): list of consistency scores
        relevance_list (List(float)): list of relevance scores
        coherence_list (List(float)): list of coherence scores

    Returns:
        Dict[str, float]: Returns average scores
    """
    average_fluency = sum(fluency_list) / len(fluency_list)
    average_consistency = sum(consistency_list) / len(consistency_list)
    average_relevance = sum(relevance_list) / len(relevance_list)
    average_coherence = sum(coherence_list) / len(coherence_list)

    log_metric("average_fluency", average_fluency)
    log_metric("average_consistency", average_consistency)
    log_metric("average_relevance", average_relevance)
    log_metric("average_coherence", average_coherence)

    return {
        "average_fluency": average_fluency,
        "average_consistency": average_consistency,
        "average_relevance": average_relevance,
        "average_coherence": average_coherence,
    }

def run_geval(transcript: str, summary: str, api_key: str, api_name: str = None, save: bool = False):
    try:
        validate_inputs(transcript, summary, api_name, api_key)
    except ValueError as e:
        return str(e)

    prompts = {
        "coherence": """You will be given one summary written for a source document.

        Your task is to rate the summary on one metric.

        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

        Evaluation Criteria:

        Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

        Evaluation Steps:

        1. Read the source document carefully and identify the main topic and key points.
        2. Read the summary and compare it to the source document. Check if the summary covers the main topic and key points of the source document, and if it presents them in a clear and logical order.
        3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


        Example:


        Source Document:

        {{Document}}

        Summary:

        {{Summary}}


        Evaluation Form (scores ONLY):

        - Coherence:""",
        "consistency": """You will be given a source document. You will then be given one summary written for this source document.

        Your task is to rate the summary on one metric.

        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


        Evaluation Criteria:

        Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

        Evaluation Steps:

        1. Read the source document carefully and identify the main facts and details it presents.
        2. Read the summary and compare it to the source document. Check if the summary contains any factual errors that are not supported by the source document.
        3. Assign a score for consistency based on the Evaluation Criteria.


        Example:


        Source Document: 

        {{Document}}

        Summary: 

        {{Summary}}


        Evaluation Form (scores ONLY):

        - Consistency:""",
        "fluency": """You will be given one summary written for a source document.

        Your task is to rate the summary on one metric.

        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


        Evaluation Criteria:

        Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

        - 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
        - 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
        - 3: Good. The summary has few or no errors and is easy to read and follow.


        Example:

        Summary:

        {{Summary}}


        Evaluation Form (scores ONLY):

        - Fluency (1-3):""",
        "relevance": """You will be given one summary written for a source document.

        Your task is to rate the summary on one metric.

        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

        Evaluation Criteria:

        Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

        Evaluation Steps:

        1. Read the summary and the source document carefully.
        2. Compare the summary to the source document and identify the main points of the source document.
        3. Assess how well the summary covers the main points of the source document, and how much irrelevant or redundant information it contains.
        4. Assign a relevance score from 1 to 5.


        Example:


        Source Document:

        {{Document}}

        Summary:

        {{Summary}}


        Evaluation Form (scores ONLY):

        - Relevance:"""
    }

    scores = {}
    explanations = {}
    for metric, prompt in prompts.items():
        full_prompt = prompt.replace("{{Document}}", transcript).replace("{{Summary}}", summary)
        try:
            score = geval_summarization(full_prompt, 5 if metric != "fluency" else 3, api_name, api_key)
            scores[metric] = score
            explanations[metric] = "Score based on the evaluation criteria."
        except Exception as e:
            error_message = detailed_api_error(api_name, e)
            return error_message

    avg_scores = aggregate([scores['fluency']], [scores['consistency']],
                           [scores['relevance']], [scores['coherence']])

    results = {
        "scores": scores,
        "average_scores": avg_scores
    }
    logging.debug("Results: %s", results)

    if save is not None:
        logging.debug("Saving results to geval_results.json")
        save_eval_results(results)
        logging.debug("Results saved to geval_results.json")

    formatted_result = f"""
    Confabulation Check Results:

    Coherence: {scores['coherence']:.2f} - {explanations['coherence']}
    Consistency: {scores['consistency']:.2f} - {explanations['consistency']}
    Fluency: {scores['fluency']:.2f} - {explanations['fluency']}
    Relevance: {scores['relevance']:.2f} - {explanations['relevance']}

    Overall Assessment: The summary has been evaluated on four key metrics. 
    The average scores are:
      Fluency: {avg_scores['average_fluency']:.2f}
      Consistency: {avg_scores['average_consistency']:.2f}
      Relevance: {avg_scores['average_relevance']:.2f}
      Coherence: {avg_scores['average_coherence']:.2f}

    These scores indicate the overall quality of the summary in terms of its 
    coherence, consistency with the original text, fluency of language, and 
    relevance of content.
    """

    return formatted_result


def create_geval_tab():
    with gr.Tab("G-Eval", id="g-eval"):
        gr.Markdown("# G-Eval Summarization Evaluation")
        with gr.Row():
            with gr.Column():
                document_input = gr.Textbox(label="Source Document", lines=10)
                summary_input = gr.Textbox(label="Summary", lines=5)
                api_name_input = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Cohere", "Groq", "OpenRouter", "DeepSeek", "HuggingFace", "Mistral", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "Local-LLM", "Ollama"],
                    label="Select API"
                )
                api_key_input = gr.Textbox(label="API Key (if required)", type="password")
                save_value = gr.Checkbox(label="Save Results to a JSON file(geval_results.json)")
                evaluate_button = gr.Button("Evaluate Summary")
            with gr.Column():
                output = gr.Textbox(label="Evaluation Results", lines=10)

        evaluate_button.click(
            fn=run_geval,
            inputs=[document_input, summary_input, api_name_input, api_key_input, save_value],
            outputs=output
        )

    return document_input, summary_input, api_name_input, api_key_input, evaluate_button, output


def parse_output(output: str, max: float) -> float:
    """
    Function that extracts numerical score from the beginning of string

    Args:
        output (str): String to search
        max (float): Maximum score allowed

    Returns:
        float: The extracted score
    """
    matched: List[str] = re.findall(r"(?<!\S)\d+(?:\.\d+)?", output)
    if matched:
        if len(matched) == 1:
            score = float(matched[0])
            if score > max:
                raise ValueError(f"Parsed number: {score} was larger than max score: {max}")
        else:
            raise ValueError(f"More than one number detected in input. Input to parser was: {output}")
    else:
        raise ValueError(f'No number detected in input. Input to parser was "{output}". ')
    return score

def geval_summarization(
    prompt_with_src_and_gen: str,
    max_score: float,
    api_endpoint: str,
    api_key: str,
) -> float:
    model = get_model_from_config(api_endpoint)

    try:
        for attempt in Retrying(
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.INFO),
            after=after_log(logger, logging.INFO),
            wait=wait_random_exponential(multiplier=1, min=1, max=120),
            stop=stop_after_attempt(10),
        ):
            with attempt:
                system_message="You are a helpful AI assistant"
                # TEMP setting for Confabulation check
                temp = 0.7
                logging.info(f"Debug - geval_summarization Function - API Endpoint: {api_endpoint}")
                try:
                    response = chat_api_call(api_endpoint, api_key, prompt_with_src_and_gen, "", temp, system_message, streaming=False, minp=None, maxp=None, model=None)
                except Exception as e:
                    raise ValueError(f"Unsupported API endpoint: {api_endpoint}")
    except RetryError:
        logger.exception(f"geval {api_endpoint} call failed\nInput prompt was: {prompt_with_src_and_gen}")
        raise

    try:
        score = parse_output(response, max_score)
    except ValueError as e:
        logger.warning(f"Error parsing output: {e}")
        score = 0

    return score


def get_model_from_config(api_name: str) -> str:
    model = config.get('models', api_name)
    if isinstance(model, dict):
        # If the model is a dictionary, return a specific key or a default value
        return model.get('name', str(model))  # Adjust 'name' to the appropriate key if needed
    return str(model) if model is not None else ""

def aggregate_llm_scores(llm_responses: List[str], max_score: float) -> float:
    """Parse and average valid scores from the generated responses of
    the G-Eval LLM call.

    Args:
        llm_responses (List[str]): List of scores from multiple LLMs
        max_score (float): The maximum score allowed.

    Returns:
        float: The average of all the valid scores
    """
    all_scores = []
    error_count = 0
    for generated in llm_responses:
        try:
            parsed = parse_output(generated, max_score)
            all_scores.append(parsed)
        except ValueError as e:
            logger.warning(e)
            error_count += 1
    if error_count:
        logger.warning(f"{error_count} out of 20 scores were discarded due to corrupt g-eval generation")
    score = sum(all_scores) / len(all_scores)
    return score


def validate_inputs(document: str, summary: str, api_name: str, api_key: str) -> None:
    """
    Validate inputs for the G-Eval function.

    Args:
        document (str): The source document
        summary (str): The summary to evaluate
        api_name (str): The name of the API to use
        api_key (str): The API key

    Raises:
        ValueError: If any of the inputs are invalid
    """
    if not document.strip():
        raise ValueError("Source document cannot be empty")
    if not summary.strip():
        raise ValueError("Summary cannot be empty")
    if api_name.lower() not in ["openai", "anthropic", "cohere", "groq", "openrouter", "deepseek", "huggingface",
                                "mistral", "llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama"]:
        raise ValueError(f"Unsupported API: {api_name}")


def detailed_api_error(api_name: str, error: Exception) -> str:
    """
    Generate a detailed error message for API failures.

    Args:
        api_name (str): The name of the API that failed
        error (Exception): The exception that was raised

    Returns:
        str: A detailed error message
    """
    error_type = type(error).__name__
    error_message = str(error)
    return f"API Failure: {api_name}\nError Type: {error_type}\nError Message: {error_message}\nPlease check your API key and network connection, and try again."


def save_eval_results(results: Dict[str, Any], filename: str = "geval_results.json") -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results (Dict[str, Any]): The evaluation results
        filename (str): The name of the file to save results to
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")




#
#
#######################################################################################################################
#
# Taken from: https://github.com/microsoft/promptflow/blob/b5a68f45e4c3818a29e2f79a76f2e73b8ea6be44/src/promptflow-core/promptflow/_core/metric_logger.py

class MetricLoggerManager:
    _instance = None

    def __init__(self):
        self._metric_loggers = []

    @staticmethod
    def get_instance() -> "MetricLoggerManager":
        if MetricLoggerManager._instance is None:
            MetricLoggerManager._instance = MetricLoggerManager()
        return MetricLoggerManager._instance

    def log_metric(self, key, value, variant_id=None):
        for logger in self._metric_loggers:
            if len(inspect.signature(logger).parameters) == 2:
                logger(key, value)  # If the logger only accepts two parameters, we don't pass variant_id
            else:
                logger(key, value, variant_id)

    def add_metric_logger(self, logger_func: Callable):
        existing_logger = next((logger for logger in self._metric_loggers if logger is logger_func), None)
        if existing_logger:
            return
        if not callable(logger_func):
            return
        sign = inspect.signature(logger_func)
        # We accept two kinds of metric loggers:
        # def log_metric(k, v)
        # def log_metric(k, v, variant_id)
        if len(sign.parameters) not in [2, 3]:
            return
        self._metric_loggers.append(logger_func)

    def remove_metric_logger(self, logger_func: Callable):
        self._metric_loggers.remove(logger_func)


def log_metric(key, value, variant_id=None):
    """Log a metric for current promptflow run.

    :param key: Metric name.
    :type key: str
    :param value: Metric value.
    :type value: float
    :param variant_id: Variant id for the metric.
    :type variant_id: str
    """
    MetricLoggerManager.get_instance().log_metric(key, value, variant_id)


def add_metric_logger(logger_func: Callable):
    MetricLoggerManager.get_instance().add_metric_logger(logger_func)


def remove_metric_logger(logger_func: Callable):
    MetricLoggerManager.get_instance().remove_metric_logger(logger_func)
#
# End of G-Eval.py
#######################################################################################################################