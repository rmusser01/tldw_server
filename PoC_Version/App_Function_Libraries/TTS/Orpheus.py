



#     Download the 52 MB SNAC model to the same directory.
#       https://huggingface.co/onnx-community/snac_24khz-ONNX/blob/main/onnx/decoder_model.onnx
#     Download the Q8 or Q4 Orpheus GGUF.
#           Q8: https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/tree/main
#           Q4: https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/tree/main
#     llama-server -m Orpheus-3b-FT-Q8_0.gguf -ngl 99 -c 4096
#
#     python orpheus.py --voice tara --text "Hello from llama.cpp generation<giggle>!"
#
#     Any packages missing? pip install onnxruntime or what ever else might be missing.
#
# This saves and plays output.wav, at least on Windows. Sometimes the generation is randomly messed up. It usually works after a few retries. If it doesn't, then a tag, especially a mistyped tag potentially messed up the generation.
#
# The code itself supports streaming, which is also done with the llama.cpp server, but I don't stream-play the resulting audio as I got slightly below real-time inference on my system. Oh, speaking of performance, you can pip install onnxruntime_gpu to speed things up a little, not sure if needed, but it comes with the drawback that you then also need to install cudnn.

# Based on https://github.com/freddyaboulton/orpheus-cpp/

import argparse
import asyncio
import json
import platform
import requests
import soundfile
import threading
import winsound
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    NotRequired,
    TypedDict,
    cast,
)

import numpy as np
import onnxruntime
from numpy.typing import NDArray


class TTSOptions(TypedDict):
    max_tokens: NotRequired[int]
    """Maximum number of tokens to generate. Default: 2048"""
    temperature: NotRequired[float]
    """Temperature for top-p sampling. Default: 0.8"""
    top_p: NotRequired[float]
    """Top-p sampling. Default: 0.95"""
    top_k: NotRequired[int]
    """Top-k sampling. Default: 40"""
    min_p: NotRequired[float]
    """Minimum probability for top-p sampling. Default: 0.05"""
    pre_buffer_size: NotRequired[float]
    """Seconds of audio to generate before yielding the first chunk. Smoother audio streaming at the cost of higher time to wait for the first chunk."""
    voice_id: NotRequired[
        Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    ]
    """The voice to use for the TTS. Default: "tara"."""


CUSTOM_TOKEN_PREFIX = "<custom_token_"


class OrpheusCpp:
    def __init__(self, verbose: bool = True):
        import importlib.util

        snac_model_path = "snac_decoder_model.onnx"

        # Load SNAC model with optimizations
        self._snac_session = onnxruntime.InferenceSession(
            snac_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        token_string = token_text.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None
        else:
            return None

    def _decode(
        self, token_gen: Generator[str, None, None]
    ) -> Generator[np.ndarray, None, None]:
        """Asynchronous token decoder that converts token stream to audio stream."""
        buffer = []
        count = 0
        for token_text in token_gen:
            token = self._token_to_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = self._convert_to_audio(buffer_to_proc)
                    if audio_samples is not None:
                        yield audio_samples

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:  # Ensure we have enough tokens
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        # Initialize empty numpy arrays instead of torch tensors
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32)
        codes_2 = np.array([], dtype=np.int32)

        for j in range(num_frames):
            i = 7 * j
            # Append values to numpy arrays
            codes_0 = np.append(codes_0, frame[i])

            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])

            codes_2 = np.append(
                codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]
            )

        # Reshape arrays to match the expected input format (add batch dimension)
        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)

        # Check that all tokens are between 0 and 4096
        if (
            np.any(codes_0 < 0)
            or np.any(codes_0 > 4096)
            or np.any(codes_1 < 0)
            or np.any(codes_1 > 4096)
            or np.any(codes_2 < 0)
            or np.any(codes_2 > 4096)
        ):
            return None

        # Create input dictionary for ONNX session

        snac_input_names = [x.name for x in self._snac_session.get_inputs()]

        input_dict = dict(zip(snac_input_names, [codes_0, codes_1, codes_2]))

        # Run inference
        audio_hat = self._snac_session.run(None, input_dict)[0]

        # Process output
        audio_np = audio_hat[:, :, 2048:4096]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        buffer = []
        for _, array in self.stream_tts_sync(text, options):
            buffer.append(array)
        return (24_000, np.concatenate(buffer, axis=1))

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        queue = asyncio.Queue()
        finished = asyncio.Event()

        def strem_to_queue(text, options, queue, finished):
            for chunk in self.stream_tts_sync(text, options):
                queue.put_nowait(chunk)
            finished.set()

        thread = threading.Thread(
            target=strem_to_queue, args=(text, options, queue, finished)
        )
        thread.start()
        while not finished.is_set():
            try:
                yield await asyncio.wait_for(queue.get(), 0.1)
            except (asyncio.TimeoutError, TimeoutError):
                pass
        while not queue.empty():
            chunk = queue.get_nowait()
            yield chunk

    def _token_gen(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[str, None, None]:

        options = options or TTSOptions()
        voice_id = options.get("voice_id", "tara")
        text = f"<|audio|>{voice_id}: {text}<|eot_id|><custom_token_4>"
        completion_url = "http://localhost:8080/completion"
        data = {
            "stream": True,
            "prompt": text,
            "max_tokens": options.get("max_tokens", 2_048),
            "temperature": options.get("temperature", 0.8),
            "top_p": options.get("top_p", 0.95),
            "top_k": options.get("top_k", 40),
            "min_p": options.get("min_p", 0.05),
        }
        response = requests.post(completion_url, json=data, stream=True)
        for line in response.iter_lines():
            line = line.decode("utf-8")

            if line.startswith("data: ") and not line.endswith("[DONE]"):
                data = json.loads(line[len("data: "):])
                yield data["content"]


    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        options = options or TTSOptions()
        token_gen = self._token_gen(text, options)
        pre_buffer = np.array([], dtype=np.int16).reshape(1, 0)
        pre_buffer_size = 24_000 * options.get("pre_buffer_size", 1.5)
        started_playback = False
        for audio_bytes in self._decode(token_gen):
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
            if not started_playback:
                pre_buffer = np.concatenate([pre_buffer, audio_array], axis=1)
                if pre_buffer.shape[1] >= pre_buffer_size:
                    started_playback = True
                    yield (24_000, pre_buffer)
            else:
                yield (24_000, audio_array)
        if not started_playback:
            yield (24_000, pre_buffer)

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with OrpheusCpp")
    parser.add_argument("--text", type=str, help="The text to convert to speech. You can use these tags: <giggle>, <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")
    parser.add_argument("--voice", type=str, choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"], default="tara", help="The voice to use for the TTS")
    args = parser.parse_args()

    orpheus = OrpheusCpp()
    sample_rate, samples = orpheus.tts(args.text.strip(), options={"voice_id": args.voice, "temperature": 0.3})
    soundfile.write("output.wav", samples.squeeze(), sample_rate)
    winsound.PlaySound("output.wav", winsound.SND_FILENAME)

if __name__ == "__main__":
    main()