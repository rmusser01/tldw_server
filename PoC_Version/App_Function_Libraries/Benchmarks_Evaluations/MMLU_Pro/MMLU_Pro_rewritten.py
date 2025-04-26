# MMLU_Pro_rewritten.py
# Description: Script to perform MMLU-Pro benchmarking
#
####################################################################################################################
# Imports
import os
import threading
import time
import toml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from openai import OpenAI
from datasets import load_dataset
import json
import re
#
##################################################################################################################
#
# Functions:


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mmlu_pro_config(**kwargs):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to config.toml
    config_path = os.path.join(script_dir, 'config.toml')

    # Load the config
    config = toml.load(config_path)

    # Update config with provided kwargs
    for key, value in kwargs.items():
        if key in config["server"]:
            config["server"][key] = value
        elif key in config["test"]:
            config["test"][key] = value
        elif key in config["log"]:
            config["log"][key] = value

    return config

# client_initializer.py
def initialize_client(config):
    try:
        return OpenAI(
            base_url=config["server"]["url"],
            api_key=config["server"]["api_key"],
            timeout=config["server"]["timeout"]
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise

# dataset_loader.py
def load_mmlu_pro():
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        test_df, val_df = dataset["test"], dataset["validation"]
        return preprocess(test_df), preprocess(val_df)
    except Exception as e:
        logger.error(f"Error loading MMLU-Pro dataset: {e}")
        raise

def preprocess(data):
    res = {}
    for item in data:
        options = [opt for opt in item["options"] if opt != "N/A"]
        item["options"] = options
        category = item["category"]
        if category not in res:
            res[category] = []
        res[category].append(item)
    return res

# prompt_creator.py
def create_prompt(cot_examples, question, options, config):
    style = config["inference"]["style"]
    system_prompt = config["inference"]["system_prompt"]

    def format_example(q, opts, cot=""):
        if not cot:
            cot = "Let's think step by step."
        cot = cot[3:] if cot.startswith("A: ") else cot
        example = f"Question: {q}\nOptions: "
        example += "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(opts))
        return example.strip(), cot.strip()

    if style == "multi_chat":
        messages = [{"role": "system", "content": system_prompt}]
        for ex in cot_examples:
            ex_text, cot = format_example(ex["question"], ex["options"], ex["cot_content"])
            messages.extend([
                {"role": "user", "content": ex_text},
                {"role": "assistant", "content": f"Answer: {cot}"}
            ])
        q_text, _ = format_example(question, options)
        messages.append({"role": "user", "content": q_text})
        return messages
    elif style == "single_chat":
        prompt = f"{system_prompt}\n\n"
        for ex in cot_examples:
            ex_text, cot = format_example(ex["question"], ex["options"], ex["cot_content"])
            prompt += f"{ex_text}\nAnswer: {cot}\n\n"
        q_text, _ = format_example(question, options)
        prompt += f"{q_text}\nAnswer: Let's think step by step."
        return [{"role": "user", "content": prompt}]
    else:  # no_chat
        prompt = f"{system_prompt}\n\n"
        for ex in cot_examples:
            ex_text, cot = format_example(ex["question"], ex["options"], ex["cot_content"])
            prompt += f"{ex_text}\nAnswer: {cot}\n\n"
        q_text, _ = format_example(question, options)
        prompt += f"{q_text}\nAnswer: Let's think step by step."
        return prompt

# answer_extractor.py
def extract_answer(text):
    patterns = [
        r"answer is \(?([A-J])\)?",
        r".*[aA]nswer:\s*\(?([A-J])\)?",
        r"\b([A-J])\b(?!.*\b[A-J]\b)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()

    logger.warning(f"Failed to extract answer from: {text}")
    return None

# question_evaluator.py
def run_single_question(question, cot_examples, client, config):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = create_prompt(cot_examples, question['question'], question['options'], config)

            if config["inference"]["style"] == "no_chat":
                response = client.completions.create(
                    model=config["server"]["model"],
                    prompt=prompt,
                    temperature=config["inference"]["temperature"],
                    max_tokens=config["inference"]["max_tokens"],
                    top_p=config["inference"]["top_p"],
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["Question:"],
                    timeout=config["server"]["timeout"],
                )
                response_text = response.choices[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=config["server"]["model"],
                    messages=prompt,
                    temperature=config["inference"]["temperature"],
                    max_tokens=config["inference"]["max_tokens"],
                    top_p=config["inference"]["top_p"],
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["Question:"],
                    timeout=config["server"]["timeout"],
                )
                response_text = response.choices[0].message.content.strip()

            pred = extract_answer(response_text)
            usage = response.usage

            return prompt, response_text, pred, usage

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All attempts failed for question: {question['question_id']}")
                return None, None, None, None
            time.sleep(3)  # Wait before retrying

# result_processor.py
def save_results(results, output_path, lock):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with lock:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to save results failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to save results to {output_path}")
            time.sleep(1)  # Wait before retrying

def save_summary(category_record, output_path, lock):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with lock:
                with open(output_path, 'w') as f:
                    json.dump(category_record, f, indent=2)
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to save summary failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to save summary to {output_path}")
            time.sleep(1)  # Wait before retrying

def update_results(results, category_record, question, pred, answer):
    category = question['category']

    if category not in category_record:
        category_record[category] = {"correct": 0, "total": 0}

    category_record[category]["total"] += 1
    if pred == answer:
        category_record[category]["correct"] += 1

    result = {
        "question_id": question['question_id'],
        "category": category,
        "question": question['question'],
        "options": question['options'],
        "pred": pred,
        "answer": answer,
        "correct": pred == answer
    }
    results.append(result)

    return results, category_record

def process_and_save_results(question, pred, client, config, results, category_record, output_dir, lock):
    results, category_record = update_results(results, category_record, question, pred, question['answer'])

    output_res_path = os.path.join(output_dir, f"{question['category']}_result.json")
    output_summary_path = os.path.join(output_dir, f"{question['category']}_summary.json")

    save_results(results, output_res_path, lock)
    save_summary(category_record, output_summary_path, lock)

    return results, category_record

def generate_final_report(category_record, output_dir):
    total_correct = sum(cat["correct"] for cat in category_record.values())
    total_questions = sum(cat["total"] for cat in category_record.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    report = f"MMLU-Pro Benchmark Final Report\n"
    report += f"================================\n\n"
    report += f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_questions})\n\n"
    report += f"Category Breakdown:\n"
    for category, stats in category_record.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        report += f"  {category}: {accuracy:.2%} ({stats['correct']}/{stats['total']})\n"

    report_path = os.path.join(output_dir, "final_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Final report saved to {report_path}")

def mmlu_pro_main():
    # Load configuration
    config = load_mmlu_pro_config()

    # Initialize OpenAI client
    client = initialize_client(config)

    # Load and preprocess the MMLU-Pro dataset
    test_data, dev_data = load_mmlu_pro()
    if test_data is None or dev_data is None:
        logger.error("Failed to load dataset. Exiting.")
        return

    # Prepare output directory
    output_dir = os.path.join("eval_results", config["server"]["model"].replace("/", "-"))
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results = []
    category_record = {}
    lock = threading.Lock()

    # Set a failure threshold to cancel the benchmark if too many questions fail
    max_failed_questions = 6
    failed_questions = 0

    # Process each subject
    for subject, questions in test_data.items():
        logger.info(f"Processing subject: {subject}")
        cot_examples = dev_data[subject]

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
            futures = []
            for question in questions:
                future = executor.submit(run_single_question, question, cot_examples, client, config)
                futures.append((future, question))

            # Process results as they complete
            for future, question in tqdm(futures, total=len(futures)):
                prompt, response, pred, usage = future.result()

                # Check if the question failed and increment the failure count
                if pred is None:
                    failed_questions += 1
                    logger.warning(f"Failed question count: {failed_questions}/{max_failed_questions}")

                # Stop the entire process if too many questions fail
                if failed_questions >= max_failed_questions:
                    logger.error(f"Too many failed questions. Stopping the benchmark for {subject}.")
                    return

                # Process and save results if the question was answered
                if pred is not None:
                    results, category_record = process_and_save_results(
                        question, pred, client, config, results, category_record, output_dir, lock
                    )

        # Save final results for the subject
        save_results(results, os.path.join(output_dir, f"{subject}_final_result.json"), lock)
        save_summary(category_record, os.path.join(output_dir, f"{subject}_final_summary.json"), lock)

    # Generate and save final report
    generate_final_report(category_record, output_dir)

    logger.info(f"Evaluation complete. Results saved in {output_dir}")

def run_mmlu_pro_benchmark():
    start_time = time.time()
    mmlu_pro_main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
#
# End of file
####################################################################################################
