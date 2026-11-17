"""The root conftest keeps numeric libraries single-threaded under pytest (TASK-13950.19)."""

import os


def test_pytest_runs_with_single_threaded_numeric_libraries():
    assert os.environ["JOBLIB_MULTIPROCESSING"] == "0"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
