# Test the load_config function
def test_load_config():
    import sys
    original_argv = sys.argv
    #sys.argv = ["run_openai.py", "-c", "test_config.toml", "-u", "http://test.com", "-m", "test-model"]

    config = load_config()

    assert config["server"]["url"] == "http://test.com"
    assert config["server"]["model"] == "test-model"

    sys.argv = original_argv
    print("load_config test passed")

def test_load_mmlu_pro():
    test_df, val_df = load_mmlu_pro()
    assert test_df is not None
    assert val_df is not None
    assert isinstance(test_df, dict)
    assert isinstance(val_df, dict)
    print("load_mmlu_pro test passed")


def test_initialize_client():
    test_config = {
        "server": {
            "url": "http://test.com",
            "api_key": "test_key",
            "timeout": 30
        }
    }

    client = initialize_client(test_config)

    assert client.base_url == "http://test.com"
    assert client.api_key == "test_key"
    assert client.timeout == 30

    print("initialize_client test passed")


test_initialize_client()

def test_preprocess():
    sample_data = [
        {"category": "math", "options": ["A", "B", "N/A", "C"]},
        {"category": "science", "options": ["X", "Y", "Z"]}
    ]
    processed = preprocess(sample_data)
    assert "math" in processed
    assert "science" in processed
    assert len(processed["math"][0]["options"]) == 3
    assert "N/A" not in processed["math"][0]["options"]
    assert len(processed["science"][0]["options"]) == 3
    print("preprocess test passed")

test_load_mmlu_pro()
test_preprocess()


test_load_config()


def test_create_prompt():
    config = {
        "inference": {
            "style": "multi_chat",
            "system_prompt": "You are a helpful assistant."
        }
    }
    cot_examples = [{
        "question": "What is 2+2?",
        "options": ["3", "4", "5"],
        "cot_content": "Let's add 2 and 2. 2+2 = 4."
    }]
    question = "What is 3+3?"
    options = ["5", "6", "7"]

    # Test multi_chat
    result = create_prompt(cot_examples, question, options, config)
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0]["role"] == "system"
    assert result[-1]["role"] == "user"

    # Test single_chat
    config["inference"]["style"] = "single_chat"
    result = create_prompt(cot_examples, question, options, config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"

    # Test no_chat
    config["inference"]["style"] = "no_chat"
    result = create_prompt(cot_examples, question, options, config)
    assert isinstance(result, str)
    assert "What is 2+2?" in result
    assert "What is 3+3?" in result

    print("create_prompt test passed")

test_create_prompt()


def test_extract_answer():
    test_cases = [
        ("The answer is (B)", "B"),
        ("After careful consideration, I believe the answer is C.", "C"),
        (
        "Let's analyze each option:\nA. Incorrect\nB. Incorrect\nC. Correct\nD. Incorrect\nTherefore, the answer is C.",
        "C"),
        ("A. GHTIS\nB. MCU\nC. UBT\nD. ALIN\n\nThe correct answer is B. MCU.", "B"),
        ("There is no clear answer in this text.", None),
        ("The options are A, B, C, and D. I think B is the best answer.", "B")
    ]

    for text, expected in test_cases:
        result = extract_answer(text)
        assert result == expected, f"Failed on input '{text}'. Expected {expected}, got {result}"

    print("extract_answer test passed")


test_extract_answer()

from unittest.mock import Mock

def test_run_single_question():
    # Mock OpenAI client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(text="The answer is B", message=Mock(content="The answer is B"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    mock_client.completions.create.return_value = mock_response
    mock_client.chat.completions.create.return_value = mock_response

    # Mock configuration
    config = {
        "inference": {
            "style": "no_chat",
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0
        },
        "server": {
            "model": "test-model",
            "timeout": 30
        }
    }

    # Mock question and examples
    question = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5"]
    }
    cot_examples = []

    # Test no_chat style
    prompt, response, pred, usage = run_single_question(question, cot_examples, mock_client, config)
    assert prompt is not None
    assert response == "The answer is B"
    assert pred == "B"
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30

    # Test chat style
    config["inference"]["style"] = "multi_chat"
    prompt, response, pred, usage = run_single_question(question, cot_examples, mock_client, config)
    assert prompt is not None
    assert response == "The answer is B"
    assert pred == "B"
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30

    print("run_single_question test passed")

test_run_single_question()


def test_save_and_update_functions():
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        lock = threading.Lock()
        results = []
        category_record = {}

        # Test question
        question = {
            'question_id': '1',
            'category': 'math',
            'question': 'What is 2+2?',
            'options': ['3', '4', '5'],
            'answer': 'B'
        }

        # Test update_results
        results, category_record = update_results(results, category_record, question, 'B', 'B')
        assert len(results) == 1
        assert category_record['math']['correct'] == 1
        assert category_record['math']['total'] == 1

        # Test save_results and save_summary
        results_path = os.path.join(tmpdir, 'results.json')
        summary_path = os.path.join(tmpdir, 'summary.json')

        save_results(results, results_path, lock)
        save_summary(category_record, summary_path, lock)

        assert os.path.exists(results_path)
        assert os.path.exists(summary_path)

        # Test process_and_save_results
        config = {'server': {'model': 'test-model'}}
        client = None  # We don't need a real client for this test

        results, category_record = process_and_save_results(question, 'B', client, config, results, category_record,
                                                            tmpdir, lock)

        assert len(results) == 2
        assert category_record['math']['correct'] == 2
        assert category_record['math']['total'] == 2

        assert os.path.exists(os.path.join(tmpdir, 'math_result.json'))
        assert os.path.exists(os.path.join(tmpdir, 'math_summary.json'))

    print("save_and_update_functions tests passed")


test_save_and_update_functions()