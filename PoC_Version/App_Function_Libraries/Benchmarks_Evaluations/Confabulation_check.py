# Confabulation_check.py
#
# This file contains the functions that are used to check the confabulation of the user's input.
#
#
# Imports
#
# External Imports
#
# Local Imports
#
#
####################################################################################################
#
# Functions:
from PoC_Version.App_Function_Libraries.Chat import chat_api_call
from PoC_Version.App_Function_Libraries.Benchmarks_Evaluations.ms_g_eval import validate_inputs, detailed_api_error


def simplified_geval(transcript: str, summary: str, api_name: str, api_key: str, temp: float = 0.7) -> str:
    """
    Perform a simplified version of G-Eval using a single query to evaluate the summary.

    Args:
        transcript (str): The original transcript
        summary (str): The summary to be evaluated
        api_name (str): The name of the LLM API to use
        api_key (str): The API key for the chosen LLM
        temp (float, optional): The temperature parameter for the API call. Defaults to 0.7.

    Returns:
        str: The evaluation result
    """
    try:
        validate_inputs(transcript, summary, api_name, api_key)
    except ValueError as e:
        return str(e)

    prompt = f"""You are an AI assistant tasked with evaluating the quality of a summary. You will be given an original transcript and a summary of that transcript. Your task is to evaluate the summary based on the following criteria:

1. Coherence (1-5): How well-structured and organized is the summary?
2. Consistency (1-5): How factually aligned is the summary with the original transcript?
3. Fluency (1-3): How well-written is the summary in terms of grammar, spelling, and readability?
4. Relevance (1-5): How well does the summary capture the important information from the transcript?

Please provide a score for each criterion and a brief explanation for your scoring. Then, give an overall assessment of the summary's quality.

Original Transcript:
{transcript}

Summary to Evaluate:
{summary}

Please provide your evaluation in the following format:
Coherence: [score] - [brief explanation]
Consistency: [score] - [brief explanation]
Fluency: [score] - [brief explanation]
Relevance: [score] - [brief explanation]

Overall Assessment: [Your overall assessment of the summary's quality]
"""
    # FIXME - Add g_eval_model to config.txt
    # g_eval_model = loaded_config[][]
    try:
        result = chat_api_call(
            api_name,
            api_key,
            prompt,
            "",
            temp=temp,
            system_message="You are a helpful AI assistant tasked with evaluating summaries.",
            streaming=False,
            minp=None,
            maxp=None,
            model=None,
            topk=None,
            topp=None,
        )
    except Exception as e:
        return detailed_api_error(api_name, e)

    formatted_result = f"""
    Confabulation Check Results:

    {result}
    """

    return formatted_result