# Diarization_Lib.py
#########################################
# Diarization Library
# This library provides functions to perform speaker diarization using pyannote.audio
# and combine the results with transcription segments.
####################
# Function List
#
# 1. load_pipeline_from_config(config_path: Union[str, Path]) -> SpeakerDiarization
# 2. calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float
# 3. audio_diarization(
#      audio_file_path: Union[str, Path],
#      pipeline: Optional[SpeakerDiarization] = None,
#      config_path: Optional[Union[str, Path]] = None
#    ) -> List[Dict[str, Any]]
# 4. combine_transcription_and_diarization(
#      transcription_segments: List[Dict[str, Any]],
#      diarization_segments: List[Dict[str, Any]]
#    ) -> List[Dict[str, Any]]
#
####################
# Import necessary libraries
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import 3rd Party Libraries
import yaml
#from pyannote.audio import Pipeline
#from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
# Filter out UserWarnings from Pyannote/Torch related to lazy loading or specific model features
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to infer the `batch_size` from an ambiguous collection.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Model was trained with pyannote.audio 0.0.1, yours is .*") # Ignore minor version mismatches if they cause warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to infer the `batch_size` from an ambiguous collection.*") # Broader batch size warning


# Import Local Libraries from tldw_Server_API (assuming integration context)
# For standalone use, replace these with standard logging/timing or dependency injection
try:
    from tldw_Server_API.app.core.Metrics.metrics_logger import timeit
    from tldw_Server_API.app.core.Utils.Utils import logging
except ImportError:
    # Fallback for potential standalone testing or if structure changes
    import logging
    logging.basicConfig(level=logging.INFO)
    # Dummy decorator if timeit is not available
    def timeit(func):
        return func

#######################################################################################################################
# Custom Exception
#
class DiarizationError(Exception):
    """Custom exception for errors during diarization processing."""
    pass

#######################################################################################################################
# Function Definitions
#

@timeit
def load_pipeline_from_config(config_path: Union[str, Path]) -> SpeakerDiarization:
    """
    Loads a PyAnnote SpeakerDiarization pipeline from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        An initialized SpeakerDiarization pipeline instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If required keys are missing in the configuration file.
        yaml.YAMLError: If the configuration file is invalid YAML.
        Exception: For other errors during pipeline initialization or instantiation.
        DiarizationError: Wraps configuration or instantiation errors.
    """
    resolved_path = Path(config_path).resolve()
    logging.debug(f"Attempting to load pyannote pipeline configuration from: {resolved_path}")

    if not resolved_path.exists():
        raise FileNotFoundError(f"Diarization config file not found: {resolved_path}")

    try:
        with open(resolved_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logging.debug(f"Loaded diarization config: {config}")

        # Basic validation of top-level structure
        if not isinstance(config, dict) or 'pipeline' not in config or 'params' not in config:
            raise DiarizationError(f"Invalid configuration structure in {resolved_path}. Missing 'pipeline' or 'params'.")

        # Create the base SpeakerDiarization pipeline using model identifiers
        # These model IDs can be paths or HuggingFace identifiers
        pipeline = SpeakerDiarization(
            segmentation=config['pipeline']['params'].get('segmentation', "pyannote/segmentation"), # Default if missing
            embedding=config['pipeline']['params'].get('embedding', "pyannote/embedding"),       # Default if missing
            clustering=config['pipeline']['params'].get('clustering', "AgglomerativeClustering"), # Default if missing
            # embedding_batch_size=config['pipeline']['params'].get('embedding_batch_size', 32), # Handled by instantiate
            # segmentation_batch_size=config['pipeline']['params'].get('segmentation_batch_size', 32) # Handled by instantiate
        )

        # Prepare hyperparameters for instantiation
        # Use .get() for safer access and provide defaults where appropriate
        hyperparameters = {
            # General pipeline params
            "embedding_batch_size": config.get('pipeline', {}).get('params', {}).get('embedding_batch_size', 32),
            "embedding_exclude_overlap": config.get('pipeline', {}).get('params', {}).get('embedding_exclude_overlap', True),
            "segmentation_batch_size": config.get('pipeline', {}).get('params', {}).get('segmentation_batch_size', 32),
            # Segmentation specific params (example)
            "segmentation": {
                "min_duration_off": config.get('params', {}).get('segmentation', {}).get('min_duration_off', 0.0),
            },
            # Clustering specific params (example)
            "clustering": {
                "method": config.get('params', {}).get('clustering', {}).get('method', 'centroid'), # Or 'affinity_propagation' etc.
                "min_cluster_size": config.get('params', {}).get('clustering', {}).get('min_cluster_size', 15),
                "threshold": config.get('params', {}).get('clustering', {}).get('threshold', 0.715), # Example threshold
                # Add other clustering params as needed from your config
            }
        }

        logging.debug(f"Instantiating pipeline with hyperparameters: {hyperparameters}")
        pipeline.instantiate(hyperparameters)
        logging.info(f"Successfully loaded and instantiated diarization pipeline from {resolved_path}")
        return pipeline

    except FileNotFoundError: # Re-raise specific error
        raise
    except (yaml.YAMLError, KeyError, TypeError, AttributeError) as e:
        logging.error(f"Error processing configuration file {resolved_path}: {e}", exc_info=True)
        raise DiarizationError(f"Failed to load or parse diarization config '{resolved_path}': {e}") from e
    except Exception as e:
        # Catch-all for unexpected errors during pyannote initialization/instantiation
        logging.error(f"Unexpected error loading/instantiating pipeline from {resolved_path}: {e}", exc_info=True)
        raise DiarizationError(f"Failed to initialize diarization pipeline: {e}") from e

@timeit
def audio_diarization(
    audio_file_path: Union[str, Path],
    pipeline: Optional[SpeakerDiarization] = None,
    config_path: Optional[Union[str, Path]] = None,
    num_speakers: Optional[int] = None, # Allow specifying number of speakers
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Performs speaker diarization on an audio file using a PyAnnote pipeline.

    Requires either a pre-loaded pipeline object or a path to a configuration file.

    Args:
        audio_file_path: Path to the input audio file (WAV format recommended).
        pipeline: A pre-loaded and instantiated SpeakerDiarization pipeline.
        config_path: Path to the YAML configuration file to load the pipeline.
        num_speakers: Known number of speakers (optional, overrides clustering threshold).
        min_speakers: Minimum number of speakers expected (optional).
        max_speakers: Maximum number of speakers expected (optional).


    Returns:
        A list of dictionaries, where each dictionary represents a speaker turn
        with keys: 'start' (float seconds), 'end' (float seconds), 'speaker' (str).

    Raises:
        ValueError: If neither a pipeline nor a config_path is provided, or if audio_file_path is invalid.
        FileNotFoundError: If the audio file or config file does not exist.
        DiarizationError: If pipeline loading or diarization processing fails.
    """
    if pipeline is None and config_path is None:
        raise ValueError("Must provide either a pre-loaded 'pipeline' or a 'config_path'.")

    audio_path = Path(audio_file_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logging.info(f"Starting audio diarization for: {audio_path}")

    try:
        # Load pipeline if not provided
        if pipeline is None:
            logging.info(f"Loading diarization pipeline from config: {config_path}")
            pipeline = load_pipeline_from_config(config_path) # Can raise FileNotFoundError, DiarizationError

        # Prepare diarization parameters
        diarization_params = {}
        if num_speakers is not None:
            diarization_params['num_speakers'] = num_speakers
            logging.info(f"Running diarization with fixed num_speakers={num_speakers}")
        elif min_speakers is not None or max_speakers is not None:
             diarization_params['min_speakers'] = min_speakers
             diarization_params['max_speakers'] = max_speakers
             logging.info(f"Running diarization with min_speakers={min_speakers}, max_speakers={max_speakers}")
        else:
             logging.info("Running diarization with parameters from config (e.g., threshold-based clustering).")

        # Perform diarization
        logging.info(f"Applying pipeline to {audio_path}...")
        diarization_result = pipeline(str(audio_path), **diarization_params)
        # Note: If diarization_result is None or empty, the loop below handles it gracefully.

        # Process results into the desired format
        segments = []
        if diarization_result:
            # Use .itertracks for easy iteration over turns
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segment = {
                    "start": round(turn.start, 3), # Round to milliseconds
                    "end": round(turn.end, 3),     # Round to milliseconds
                    "speaker": speaker             # Speaker label (e.g., "SPEAKER_00")
                }
                logging.debug(f"Diarization Segment: {segment}")
                segments.append(segment)
            logging.info(f"Diarization completed. Found {len(segments)} speaker turns.")
        else:
            logging.warning(f"Diarization returned no results for {audio_path}. The audio might be silent or too short.")

        return segments

    except (FileNotFoundError, DiarizationError): # Let specific errors from loading pass through
        raise
    except Exception as e:
        # Catch potential runtime errors from pipeline execution
        logging.error(f"Error during diarization processing for {audio_path}: {e}", exc_info=True)
        raise DiarizationError(f"Failed to perform diarization on {audio_path}: {e}") from e

def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates the duration of overlap between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)

@timeit
def combine_transcription_and_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combines transcription segments with speaker diarization information based on maximum overlap.

    Args:
        transcription_segments: A list of transcription segment dictionaries.
            Expected keys: 'text' (str) and time keys like 'start'/'end' or
            'start_seconds'/'end_seconds' (float or convertible to float).
        diarization_segments: A list of diarization segment dictionaries.
            Expected keys: 'start' (float seconds), 'end' (float seconds), 'speaker' (str).

    Returns:
        A list of combined segment dictionaries with keys:
        'start' (float), 'end' (float), 'text' (str), 'speaker' (str).
        Segments that cannot be reliably assigned a speaker (no overlap)
        will have speaker set to 'UNKNOWN'.
    """
    logging.info(f"Combining {len(transcription_segments)} transcription segments with {len(diarization_segments)} diarization segments.")

    if not transcription_segments:
        logging.warning("Transcription segments list is empty. Returning empty list.")
        return []

    combined_results = []

    # --- Handle case where diarization is missing ---
    if not diarization_segments:
        logging.warning("Diarization segments list is empty. Assigning 'UNKNOWN' speaker to all transcription segments.")
        for i, t_seg in enumerate(transcription_segments):
            # Standardize time keys access - Robustly get start/end times
            t_start_raw = t_seg.get('start', t_seg.get('start_seconds', None))
            t_end_raw = t_seg.get('end', t_seg.get('end_seconds', None))
            t_text = t_seg.get('text', t_seg.get('Text', '')) # Handle potential 'Text' key

            if t_start_raw is None or t_end_raw is None:
                logging.warning(f"[No Diarization Fallback] Skipping transcription segment {i} due to missing time keys: {t_seg}")
                continue

            # --- Attempt conversion and handle potential errors ---
            try:
                t_start = float(t_start_raw)
                t_end = float(t_end_raw)
            except (ValueError, TypeError) as e:
                logging.warning(f"[No Diarization Fallback] Skipping transcription segment {i} due to invalid time format (not convertible to float): start='{t_start_raw}', end='{t_end_raw}'. Error: {e}. Segment: {t_seg}")
                continue
            # --- End conversion block ---

            # Basic validation after conversion
            if t_start > t_end:
                 logging.warning(f"[No Diarization Fallback] Skipping transcription segment {i} due to invalid time (start > end): {t_seg}")
                 continue

            combined_results.append({
                "start": t_start,
                "end": t_end,
                "text": t_text,
                "speaker": "UNKNOWN" # Assign UNKNOWN as diarization is missing
            })
        logging.info(f"[No Diarization Fallback] Generated {len(combined_results)} segments.")
        return combined_results

    # --- Main combination logic ---
    diarization_idx = 0 # Optimization: track the last used diarization segment index

    for i, t_seg in enumerate(transcription_segments):
        # Standardize time keys access - Robustly get start/end times
        t_start_raw = t_seg.get('start', t_seg.get('start_seconds', None))
        t_end_raw = t_seg.get('end', t_seg.get('end_seconds', None))
        t_text = t_seg.get('text', t_seg.get('Text', '')) # Handle potential 'Text' key

        if t_start_raw is None or t_end_raw is None:
            logging.warning(f"Skipping transcription segment {i} due to missing time keys: {t_seg}")
            continue

        # --- Attempt conversion and handle potential errors ---
        try:
            t_start = float(t_start_raw)
            t_end = float(t_end_raw)
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping transcription segment {i} due to invalid time format (not convertible to float): start='{t_start_raw}', end='{t_end_raw}'. Error: {e}. Segment: {t_seg}")
            continue
        # --- End conversion block ---

        # Now t_start and t_end are guaranteed floats if we passed the conversion
        if t_start > t_end:
            logging.warning(f"Skipping transcription segment {i} due to invalid time (start > end): {t_seg}")
            continue
        if not t_text and not t_text.strip(): # Also check for empty strings or whitespace-only
            logging.debug(f"Skipping transcription segment {i} due to empty/whitespace text: {t_seg}")
            continue

        best_speaker = "UNKNOWN"
        max_overlap = 0.0

        # Iterate through potentially relevant diarization segments
        temp_idx = diarization_idx
        while temp_idx < len(diarization_segments):
            d_seg = diarization_segments[temp_idx]
            # Robustly get diarization times, defaulting to 0.0 and converting
            try:
                # Use float() directly here too for consistency and error handling
                d_start = float(d_seg.get('start', 0.0))
                d_end = float(d_seg.get('end', 0.0))
            except (ValueError, TypeError) as e:
                 logging.warning(f"Skipping diarization segment {temp_idx} due to invalid time format: {d_seg}. Error: {e}")
                 temp_idx += 1
                 # Update main index only if we permanently skip this d_seg for *all* future t_segs
                 # This happens if d_end <= t_start (handled below), so maybe don't update diarization_idx here.
                 continue # Skip this d_seg

            d_speaker = d_seg.get('speaker', 'UNKNOWN')

            # --- Comparisons using guaranteed floats ---
            # If the diarization segment starts after the transcription segment ends,
            # we can likely stop searching for this transcription segment (assuming sorted lists)
            if d_start >= t_end:
                 break # No further d_segs will overlap this t_seg

            # If the diarization segment ends before the transcription segment starts, skip it
            if d_end <= t_start:
                temp_idx += 1
                # Only update the main diarization index if we are sure we won't need this d_seg again
                # This optimization assumes t_segs are roughly sorted by start time.
                diarization_idx = temp_idx
                continue # Move to the next d_seg for the *current* t_seg

            # --- Call calculate_overlap with guaranteed floats ---
            overlap = calculate_overlap(t_start, t_end, d_start, d_end)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_speaker
                # Optimization idea kept commented: check if good enough overlap found
                # if max_overlap > 0 and (t_end - t_start > 0) and (max_overlap / (t_end - t_start)) > 0.9:
                #     break # Assume this is the best match

            temp_idx += 1 # Check next d_seg for the *current* t_seg

        # Add segment even if speaker is UNKNOWN, as text is present
        combined_segment = {
            "start": t_start, # Already confirmed as float
            "end": t_end,     # Already confirmed as float
            "text": t_text,
            "speaker": best_speaker # Will be UNKNOWN if max_overlap remained 0
        }
        combined_results.append(combined_segment)
        logging.debug(f"Combined Segment: {combined_segment}")

    logging.info(f"Combination complete. Generated {len(combined_results)} combined segments.")
    return combined_results

# # Example Usage (requires transcription_segments and diarization_segments)
# transcription_data = [
#     {'start_seconds': 0.5, 'end_seconds': 2.1, 'text': 'Hello there.'},
#     {'start': 2.5, 'end': 4.0, 'text': 'How are you?'},
#     {'start': 4.1, 'end': 5.5, 'text': 'Fine thanks.'},
#     {'start_seconds': 'invalid', 'end_seconds': 7.0, 'text': 'Bad data'}, # Will be skipped
#     {'start_seconds': 8.0, 'text': 'Missing end time'}, # Will be skipped
#     {'start': 9.0, 'end': 8.5, 'text': 'Invalid time order'}, # Will be skipped
#     {'start': 10.0, 'end': 11.0, 'text': ''}, # Will be skipped
# ]
#
# diarization_data = [
#     {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
#     {'start': 'bad_start', 'end': 3.5, 'speaker': 'SPEAKER_XX'}, # Will be skipped
#     {'start': 3.8, 'end': 6.0, 'speaker': 'SPEAKER_01'},
#     {'start': 9.5, 'end': 10.5, 'speaker': 'SPEAKER_00'},
# ]
#
# combined = combine_transcription_and_diarization(transcription_data, diarization_data)
# print("\nCombined Results:")
# for segment in combined:
#     print(segment)
#
# print("\n--- Test with empty diarization ---")
# combined_no_diar = combine_transcription_and_diarization(transcription_data, [])
# print("\nCombined Results (No Diarization):")
# for segment in combined_no_diar:
#     print(segment)

#
# End of Diarization_Lib_v2.py
#######################################################################################################################
