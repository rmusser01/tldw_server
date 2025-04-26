# Script taken from: https://github.com/chigkim/Ollama-MMLU-Pro
# No changes made
import os
import re
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import codecs
import toml
import argparse
import queue
import numpy as np
import copy

parser = argparse.ArgumentParser(
	prog="python3 run_openai.py",
	description="Run MMLU Pro Benchmark for  a local LLM  via  OpenAI Compatible API.",
	epilog="Specify  options above  to override  one or more settings from config.",
)
parser.add_argument(
	"-c",
	"--config",
	help="Configuration file. Default=config.toml",
	default="config.toml",
)
parser.add_argument(
	"-u",
	"--url",
	help="server url",
)
parser.add_argument("-a", "--api", help="api key")
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument(
	"--timeout",
	type=float,
	help="Request timeout in seconds",
)
parser.add_argument("--category", type=str)
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level 0-2")
parser.add_argument(
	"--log_prompt",
	help="Writes exact prompt and response into log.txt",
	action="store_true",
)
parser.add_argument(
	"--comment", type=str, help="Comment to be included in the final report."
)
args = parser.parse_args()
config = toml.load(open(args.config))
if args.url:
	config["server"]["url"] = args.url
if args.api:
	config["server"]["api_key"] = args.api
if args.model:
	config["server"]["model"] = args.model
if args.timeout:
	config["server"]["timeout"] = args.timeout
if args.category:
	config["test"]["categories"] = [args.category]
if args.parallel:
	config["test"]["parallel"] = args.parallel
if args.verbosity:
	config["log"]["verbosity"] = args.verbosity
if args.log_prompt:
	config["log"]["log_prompt"] = args.log_prompt
if args.comment:
	config["comment"] = args.comment


client = OpenAI(
	base_url=config["server"]["url"],
	api_key=config["server"]["api_key"],
	timeout=config["server"]["timeout"],
)


def log(message):
	print(message)
	with codecs.open(log_path, "a", "utf-8") as file:
		file.write(message + "\n")


def get_chat_completion(messages):
	try:
		response = client.chat.completions.create(
			model=config["server"]["model"],
			messages=messages,
			temperature=config["inference"]["temperature"],
			max_tokens=config["inference"]["max_tokens"],
			top_p=config["inference"]["top_p"],
			frequency_penalty=0,
			presence_penalty=0,
			stop=["Question:"],
			timeout=config["server"]["timeout"],
		)
		try:
			usage_q.put(
				(response.usage.prompt_tokens, response.usage.completion_tokens)
			)
		except:
			pass
		return response.choices[0].message.content.strip()
	except Exception as e:
		print("Resubmitting, Error: ", e)
		time.sleep(3)
		return get_chat_completion(messages)


def get_completion(prompt):
	try:
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
		try:
			usage_q.put(
				(response.usage.prompt_tokens, response.usage.completion_tokens)
			)
		except:
			pass
		if response.choices:
			return response.choices[0].text.strip()
		elif response.content:
			return response.content.strip()
		print("Can't get response.")
		return None
	except Exception as e:
		print("Resubmitting, Error: ", e)
		time.sleep(3)
		return get_completion(prompt)


def load_mmlu_pro():
	dataset = load_dataset("TIGER-Lab/MMLU-Pro")
	test_df, val_df = dataset["test"], dataset["validation"]
	test_df = preprocess(test_df)
	val_df = preprocess(val_df)
	return test_df, val_df


def preprocess(test_df):
	res_df = []
	for each in test_df:
		options = []
		for opt in each["options"]:
			if opt == "N/A":
				continue
			options.append(opt)
		each["options"] = options
		res_df.append(each)
	res = {}
	for each in res_df:
		if each["category"] not in res:
			res[each["category"]] = []
		res[each["category"]].append(each)
	return res


def format_example(question, options, cot_content=""):
	if cot_content == "":
		cot_content = "Let's think step by step."
	if cot_content.startswith("A: "):
		cot_content = cot_content[3:]
	example = "Question: {}\nOptions: ".format(question)
	choice_map = "ABCDEFGHIJ"
	for i, opt in enumerate(options):
		example += "{}. {}\n".format(choice_map[i], opt)
	return example.strip(), cot_content.strip()


def multi_chat_prompt(cot_examples, question, options):
	messages = [
		{
			"role": "system",
			"content": config["inference"]["system_prompt"],
		},
	]
	for each in cot_examples:
		example, cot_content = format_example(
			each["question"], each["options"], each["cot_content"]
		)
		messages.append({"role": "user", "content": example})
		messages.append({"role": "assistant", "content": "Answer: " + cot_content})
	example, cot_content = format_example(question, options)
	messages.append({"role": "user", "content": example})
	return messages


def single_chat_prompt(cot_examples, question, options):
	messages = [
		{
			"role": "system",
			"content": config["inference"]["system_prompt"],
		},
	]
	prompt = no_chat_prompt(cot_examples, question, options, no_system=True)
	messages.append({"role": "user", "content": prompt})
	return messages


def no_chat_prompt(cot_examples, question, options, no_system=False):
	prompt = config["inference"]["system_prompt"] + "\n\n"
	if no_system:
		prompt = ""
	for each in cot_examples:
		example, cot_content = format_example(
			each["question"], each["options"], each["cot_content"]
		)
		prompt += example + "\n"
		prompt += "Answer: " + cot_content + "\n\n"
	example, cot_content = format_example(question, options)
	prompt += example + "\n"
	prompt += "Answer: " + cot_content
	return prompt


def extract_answer(text):
	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_again(text)


def extract_again(text):
	pattern = r".*[aA]nswer:\s*\(?([A-J])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_final(text)


def extract_final(text):
	pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
	match = re.search(pattern, text, re.DOTALL)
	if match:
		return match[0]
	else:
		if config["log"]["verbosity"] >= 1:
			print("Extraction failed:\n", text)
		return None


def run_single_question(single_question, cot_examples_dict, exist_result):
	exist = True
	q_id = single_question["question_id"]
	for each in exist_result:
		if (
			q_id == each["question_id"]
			and single_question["question"] == each["question"]
		):
			if config["log"]["verbosity"] >= 1:
				print("already exists, skipping.")
			return None, None, None, exist
	exist = False
	category = single_question["category"]
	cot_examples = cot_examples_dict[category]
	question = single_question["question"]
	options = single_question["options"]
	try:
		if config["inference"]["style"] == "single_chat":
			prompt = single_chat_prompt(cot_examples, question, options)
			response = get_chat_completion(prompt)
		elif config["inference"]["style"] == "multi_chat":
			prompt = multi_chat_prompt(cot_examples, question, options)
			response = get_chat_completion(prompt)
		elif config["inference"]["style"] == "no_chat":
			prompt = no_chat_prompt(cot_examples, question, options)
			response = get_completion(prompt)
	except Exception as e:
		print("error", e)
		return None, None, None, exist
	pred = extract_answer(response)
	return prompt, response, pred, exist


def update_result(output_res_path, lock):
	category_record = {}
	res = []
	success = False
	while not success:
		try:
			if os.path.exists(output_res_path):
				with lock:
					with open(output_res_path, "r") as fi:
						res = json.load(fi)
						for each in res:
							category = each["category"]
							if category not in category_record:
								category_record[category] = {"corr": 0.0, "wrong": 0.0}
								category_record["random"] = {"corr": 0.0, "wrong": 0.0}
							if not each["pred"]:
								random.seed(12345)
								x = random.randint(0, len(each["options"]) - 1)
								if x == each["answer_index"]:
									category_record[category]["corr"] += 1
									category_record["random"]["corr"] += 1
								else:
									category_record[category]["wrong"] += 1
									category_record["random"]["wrong"] += 1
							elif each["pred"] == each["answer"]:
								category_record[category]["corr"] += 1
							else:
								category_record[category]["wrong"] += 1
			success = True
		except Exception as e:
			print("Error", e)
	return res, category_record


def evaluate(subjects):
	test_df, dev_df = load_mmlu_pro()
	if not subjects:
		subjects = list(test_df.keys())
	print("assigned subjects", subjects)
	lock = threading.Lock()
	system_prompt = config["inference"]["system_prompt"]
	for subject in subjects:
		start = time.time()
		print(f"Testing {subject}...")
		config["inference"]["system_prompt"] = system_prompt.replace(
			"{subject}", subject
		)
		test_data = test_df[subject]
		output_res_path = os.path.join(output_dir, subject + "_result.json")
		output_summary_path = os.path.join(output_dir, subject + "_summary.json")
		res, category_record = update_result(output_res_path, lock)

		with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
			futures = {
				executor.submit(run_single_question, each, dev_df, res): each
				for each in test_data
			}
			for future in tqdm(
				as_completed(futures), total=len(futures), smoothing=0.0, ascii=True
			):
				each = futures[future]
				label = each["answer"]
				category = subject
				prompt, response, pred, exist = future.result()
				if exist:
					continue
				if response is not None:
					res, category_record = update_result(output_res_path, lock)
					if category not in category_record:
						category_record[category] = {"corr": 0.0, "wrong": 0.0}
					if config["log"]["log_prompt"]:
						each["prompt"] = prompt
					each["response"] = response
					each["pred"] = pred
					res.append(each)
					if config["log"]["verbosity"] >= 2:
						log_json = {
							"id": each["question_id"],
							"question": each["question"],
							"response": each["response"],
							"pred": each["pred"],
							"answer": each["answer"],
						}
						print("\n" + json.dumps(log_json, indent="\t"))
					if pred is not None:
						if pred == label:
							category_record[category]["corr"] += 1
						else:
							category_record[category]["wrong"] += 1
					else:
						category_record[category]["wrong"] += 1
					save_res(res, output_res_path, lock)
					save_summary(category_record, output_summary_path, lock)
					res, category_record = update_result(output_res_path, lock)
		save_res(res, output_res_path, lock)
		hours, minutes, seconds = elapsed(start)
		log(
			f"Finished testing {subject} in {hours} hours, {minutes} minutes, {seconds} seconds."
		)
		save_summary(category_record, output_summary_path, lock, report=True)


def save_res(res, output_res_path, lock):
	temp = []
	exist_q_id = []
	for each in res:
		if each["question_id"] not in exist_q_id:
			exist_q_id.append(each["question_id"])
			temp.append(each)
		else:
			continue
	res = temp
	with lock:
		with open(output_res_path, "w") as fo:
			fo.write(json.dumps(res, indent="\t"))


def print_score(label, corr, wrong):
	try:
		corr = int(corr)
		wrong = int(wrong)
		total = corr + wrong
		acc = corr / total * 100
		log(f"{label}, {corr}/{total}, {acc:.2f}%")
	except Exception as e:
		log(f"{label}, {e} error")


def save_summary(category_record, output_summary_path, lock, report=False):
	total_corr = 0.0
	total_wrong = 0.0
	for k, v in category_record.items():
		if k == "total" or k == "random":
			continue
		cat_acc = v["corr"] / (v["corr"] + v["wrong"])
		category_record[k]["acc"] = cat_acc
		total_corr += v["corr"]
		total_wrong += v["wrong"]
	acc = total_corr / (total_corr + total_wrong)
	category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
	if report:
		print_score("Total", total_corr, total_wrong)
		if "random" in category_record:
			random_corr = category_record["random"]["corr"]
			random_wrong = category_record["random"]["wrong"]
			print_score(
				"Random Guess Attempts",
				random_corr + random_wrong,
				total_corr + total_wrong - random_corr - random_wrong,
			)
			print_score("Correct Random Guesses", random_corr, random_wrong)
			print_score(
				"Adjusted Score Without Random Guesses",
				total_corr - random_corr,
				total_wrong - random_wrong,
			)
	with lock:
		with open(output_summary_path, "w") as fo:
			fo.write(json.dumps(category_record, indent="\t"))


def final_report(assigned_subjects):
	total_corr = 0.0
	total_wrong = 0.0
	random_corr = 0.0
	random_wrong = 0.0
	names = ["overall"] + assigned_subjects
	table = "| " + " | ".join(names) + " |\n"
	separators = [re.sub(r".", "-", name) for name in names]
	table += "| " + " | ".join(separators) + " |\n"
	scores = []
	for file in assigned_subjects:
		res = json.load(open(os.path.join(output_dir, file + "_summary.json")))
		cat_corr = res["total"]["corr"]
		total_corr += cat_corr
		cat_wrong = res["total"]["wrong"]
		total_wrong += cat_wrong
		scores.append(cat_corr / (cat_corr + cat_wrong))
		if "random" in res:
			random_corr += res["random"]["corr"]
			random_wrong += res["random"]["wrong"]
	print_score("Total", total_corr, total_wrong)
	if random_corr and random_wrong:
		print_score(
			"Random Guess Attempts",
			random_corr + random_wrong,
			total_corr + total_wrong - random_corr - random_wrong,
		)
		print_score("Correct Random Guesses", random_corr, random_wrong)
		print_score(
			"Adjusted Score Without Random Guesses",
			total_corr - random_corr,
			total_wrong - random_wrong,
		)
	scores.insert(0, total_corr / (total_corr + total_wrong))
	scores = [f"{score*100:.2f}" for score in scores]
	table += "| " + " | ".join(scores) + " |"
	token_report()
	log("Markdown Table:")
	log(table)


def elapsed(start):
	duration = time.time() - start
	duration_td = timedelta(seconds=duration)
	hours, remainder = divmod(duration_td.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	return hours, minutes, seconds


def token_report():
	ptoks = []
	ctoks = []
	while not usage_q.empty():
		usage = usage_q.get()
		ptoks.append(usage[0])
		ctoks.append(usage[1])
	if ptoks and ctoks:
		log("Token Usage:")
		duration = end - start
		ptoks = np.array(ptoks)
		ctoks = np.array(ctoks)
		log(
			f"Prompt tokens: min {ptoks.min()}, average {ptoks.mean():.0f}, max {ptoks.max()}, total {ptoks.sum()}, tk/s {ptoks.sum()/duration:.2f}"
		)
		log(
			f"Completion tokens: min {ctoks.min()}, average {ctoks.mean():.0f}, max {ctoks.max()}, total {ctoks.sum()}, tk/s {ctoks.sum()/duration:.2f}"
		)


if __name__ == "__main__":
	usage_q = queue.Queue()
	output_dir = "eval_results/" + re.sub(r"\W", "-", config["server"]["model"])
	os.makedirs(output_dir, exist_ok=True)
	log_path = os.path.join(output_dir, "report.txt")
	try:
		os.remove(log_path)
	except:
		pass
	config_copy = copy.deepcopy(config)
	del config_copy["server"]["api_key"]
	del config_copy["test"]["categories"]
	log(f"{datetime.now()}")
	log(json.dumps(config_copy, indent="\t"))
	assigned_subjects = config["test"]["categories"]
	start = time.time()
	evaluate(assigned_subjects)
	end = time.time()
	hours, minutes, seconds = elapsed(start)
	log(
		f"Finished the benchmark in {hours} hours, {minutes} minutes, {seconds} seconds."
	)
	final_report(assigned_subjects)
	print("Report saved to:", log_path)
