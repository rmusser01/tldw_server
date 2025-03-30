import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio
import gradio as gr
import tempfile

llasa_3b ='srinivasbilla/llasa-3b'

tokenizer = AutoTokenizer.from_pretrained(llasa_3b)

model = AutoModelForCausalLM.from_pretrained(
    llasa_3b,
    trust_remote_code=True,
    device_map='cuda',
)

model_path = "srinivasbilla/xcodec2"

Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()

whisper_turbo_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device='cuda',
)

def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def split_text_into_chunks(text, max_length=300):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@spaces.GPU(duration=60)
def infer(sample_audio_path, target_text, progress=gr.Progress()):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        progress(0, 'Loading and trimming audio...')
        waveform, sample_rate = torchaudio.load(sample_audio_path)
        if len(waveform[0])/sample_rate > 15:
            waveform = waveform[:, :sample_rate*15]

        # Check if the audio is stereo (i.e., has more than one channel)
        if waveform.size(0) > 1:
            # Convert stereo to mono by averaging the channels
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            # If already mono, just use the original waveform
            waveform_mono = waveform

        prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
        prompt_text = whisper_turbo_pipe(prompt_wav[0].numpy())['text'].strip()
        progress(0.5, 'Transcribed! Generating speech...')

        if len(target_text) == 0:
            return None

        # Split the target text into chunks if necessary
        text_chunks = split_text_into_chunks(target_text)
        generated_wavs = []

        for chunk in text_chunks:
            input_text = prompt_text + ' ' + chunk

            # TTS start!
            with torch.no_grad():
                # Encode the prompt wav
                vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)

                vq_code_prompt = vq_code_prompt[0, 0, :]
                # Convert int 12345 to token <|s_12345|>
                speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

                formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

                # Tokenize the text and the speech prefix
                chat = [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
                ]

                input_ids = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    return_tensors='pt',
                    continue_final_message=True
                )
                input_ids = input_ids.to('cuda')
                speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

                # Generate the speech autoregressively
                outputs = model.generate(
                    input_ids,
                    max_length=2048,  # We trained our model with a max length of 2048
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    top_p=1,
                    temperature=0.8
                )
                # Extract the speech tokens
                generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix):-1]

                speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Convert token <|s_23456|> to int 23456
                speech_tokens = extract_speech_ids(speech_tokens)

                speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

                # Decode the speech tokens to speech waveform
                gen_wav = Codec_model.decode_code(speech_tokens)

                # Keep only the generated part
                gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]

                # Trim the first 0.5 seconds (8000 samples at 16kHz)
                if gen_wav.shape[2] > 3000:  # Ensure there are enough samples to trim
                    gen_wav = gen_wav[:, :, 3000:]

                # Create a silence tensor of 0.5 seconds (8000 samples at 16kHz)
                silence = torch.zeros(1, 1, 3000).cuda()

                # Append silence to the end of the generated audio
                gen_wav = torch.cat([silence, gen_wav], dim=2)

                generated_wavs.append(gen_wav)

        # Concatenate all generated wavs
        final_wav = torch.cat(generated_wavs, dim=2)

        progress(1, 'Synthesized!')

        return (16000, final_wav[0, 0, :].cpu().numpy())

with gr.Blocks() as app_tts:
    gr.Markdown("# Zero Shot Voice Clone TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)

    generate_btn = gr.Button("Synthesize", variant="primary")

    audio_output = gr.Audio(label="Synthesized Audio")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            gen_text_input,
        ],
        outputs=[audio_output],
    )

with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [zhenye234](https://github.com/zhenye234) for the original [repo](https://github.com/zhenye234/LLaSA_training)
* [mrfakename](https://huggingface.co/mrfakename) for the [gradio demo code](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
""")

with gr.Blocks() as app:
    gr.Markdown(
        """
# llasa 3b TTS

This is a local web UI for llasa 3b SOTA(imo) Zero Shot Voice Cloning and TTS model.

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.
"""
    )
    gr.TabbedInterface([app_tts], ["TTS"])


app.launch(ssr_mode=False)
