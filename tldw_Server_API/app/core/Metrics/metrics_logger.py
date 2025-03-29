# metrics_logger.py
#
# Imports
import functools
import time
from datetime import datetime, timezone

import psutil

#
# Third-party Imports
#
# Local Imports
from App_Function_Libraries.Metrics.logger_config import logger
#
############################################################################################################
#
# Functions:

def log_counter(metric_name, labels=None, value=1):
    log_entry = {
        "event": metric_name,
        "type": "counter",
        "value": value,
        "labels": labels or {},
        # datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
        # FIXME
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }
    logger.info("metric", extra=log_entry)


def log_histogram(metric_name, value, labels=None):
    log_entry = {
        "event": metric_name,
        "type": "histogram",
        "value": value,
        "labels": labels or {},
        # datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
        # FIXME
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }
    logger.info("metric", extra=log_entry)


def timeit(func):
    """
    Decorator that times the execution of the wrapped function
    and logs the result using log_histogram. Optionally, you could also
    log a counter each time the function is called.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        # Print to console (optional)
        print(f"{func.__name__} executed in {elapsed:.2f} seconds.")

        # Log how long the function took (histogram)
        log_histogram(
            metric_name=f"{func.__name__}_duration_seconds",
            value=elapsed,
            labels={"function": func.__name__}
        )

        # (Optional) log how many times the function has been called
        log_counter(
            metric_name=f"{func.__name__}_calls",
            labels={"function": func.__name__}
        )

        return result
    return wrapper
    # Add '@timeit' decorator to functions you want to time


def log_resource_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    cpu = process.cpu_percent(interval=0.1)
    print(f"Memory: {memory:.2f} MB, CPU: {cpu:.2f}%")

#
# End of Functions
############################################################################################################

# # Prometheus
# # metrics_logger.py (Prometheus version)
# from prometheus_client import Counter, Histogram, start_http_server
# import logging
# from functools import wraps
# import time
#
# # Initialize Prometheus metrics
# VIDEOS_PROCESSED = Counter('videos_processed_total', 'Total number of videos processed', ['whisper_model', 'api_name'])
# VIDEOS_FAILED = Counter('videos_failed_total', 'Total number of videos failed to process', ['whisper_model', 'api_name'])
# TRANSCRIPTIONS_GENERATED = Counter('transcriptions_generated_total', 'Total number of transcriptions generated', ['whisper_model'])
# SUMMARIES_GENERATED = Counter('summaries_generated_total', 'Total number of summaries generated', ['whisper_model'])
# VIDEO_PROCESSING_TIME = Histogram('video_processing_time_seconds', 'Time spent processing videos', ['whisper_model', 'api_name'])
# TOTAL_PROCESSING_TIME = Histogram('total_processing_time_seconds', 'Total time spent processing all videos', ['whisper_model', 'api_name'])
#
# def init_metrics_server(port=8000):
#     start_http_server(port)
#
# def log_counter(metric_name, labels=None, value=1):
#     if metric_name == "videos_processed_total":
#         VIDEOS_PROCESSED.labels(**(labels or {})).inc(value)
#     elif metric_name == "videos_failed_total":
#         VIDEOS_FAILED.labels(**(labels or {})).inc(value)
#     elif metric_name == "transcriptions_generated_total":
#         TRANSCRIPTIONS_GENERATED.labels(**(labels or {})).inc(value)
#     elif metric_name == "summaries_generated_total":
#         SUMMARIES_GENERATED.labels(**(labels or {})).inc(value)
#
# def log_histogram(metric_name, value, labels=None):
#     if metric_name == "video_processing_time_seconds":
#         VIDEO_PROCESSING_TIME.labels(**(labels or {})).observe(value)
#     elif metric_name == "total_processing_time_seconds":
#         TOTAL_PROCESSING_TIME.labels(**(labels or {})).observe(value)


# # main.py or equivalent entry point
# from metrics_logger import init_metrics_server
#
#
# def main():
#     # Start Prometheus metrics server on port 8000
#     init_metrics_server(port=8000)
#
#     # Initialize and launch your Gradio app
#     create_video_transcription_tab()
#
#
# if __name__ == "__main__":
#     main()

# prometheus.yml
# scrape_configs:
#   - job_name: 'video_transcription_app'
#     static_configs:
#       - targets: ['localhost:8000']  # Replace with your application's host and port

#
# End of metrics_logger.py
############################################################################################################
