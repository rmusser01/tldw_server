"""Single-threaded numeric-library defaults for pytest runs (TASK-13950.19).

Imported first by the root conftest.py and tests/e2e/conftest.py (the e2e pytest.ini moves
the rootdir, so the root conftest is not loaded there). Keeps joblib/loky from leaking semaphores.
Import-light on purpose: only os.
"""

import os

THREAD_ENV_DEFAULTS = {
    "JOBLIB_MULTIPROCESSING": "0",
    "JOBLIB_BACKEND": "threading",
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

for _name, _value in THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)
