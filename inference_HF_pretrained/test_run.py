# Copyright (c) 2025 NVIDIA CORPORATION.
#   Licensed under the MIT license.
# Simple test script for Audio Flamingo 2

import os
import yaml
import json
import numpy as np
import soundfile as sf

import torch
import librosa
from pydub import AudioSegment
from safetensors.torch import load_file

from src.factory import create_model_and_transforms
from utils import Dict2Class, get_autocast, get_cast_dtype

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_num_windows(T, sr, clap_config):
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap

    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)

        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))
    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))

        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]

    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))

    assert len(data.shape) == 1, data.shape
    return data


def load_audio(audio_path, clap_config):
    sr = 16000
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config)
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)

    return audio_clips, audio_embed_mask


def predict(filepath, question, clap_config, model, tokenizer, device_id, cast_dtype, inference_kwargs):
    audio_clips, audio_embed_mask = load_audio(filepath, clap_config)
    audio_clips = audio_clips.to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

    text_prompt = str(question).lower()
    sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"

    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device_id, non_blocking=True)
    prompt = input_ids

    with torch.no_grad():
        output = model.generate(
            audio_x=audio_clips.unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            **inference_kwargs,
        )[0]

    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')

    print('Prompt:', question)
    print('Audio Flamingo 2:', output_decoded)
    print()

    return output_decoded


def generate_test_audio():
    """Generate a simple test audio file with some tones."""
    sr = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sr * duration))

    # Generate a simple melody - C major scale notes
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
    audio = np.zeros_like(t)

    note_duration = duration / len(freqs)
    for i, freq in enumerate(freqs):
        start = int(i * note_duration * sr)
        end = int((i + 1) * note_duration * sr)
        audio[start:end] = 0.5 * np.sin(2 * np.pi * freq * t[start:end])

    # Apply fade in/out
    fade_len = int(0.01 * sr)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    output_path = "test_audio.wav"
    sf.write(output_path, audio, sr)
    print(f"Generated test audio: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Flamingo 2 - Test Run")
    print("=" * 60)

    # Use the MP3 file from root directory if available, otherwise generate test audio
    import os
    mp3_path = "../Everyone Else Art Pop.mp3"
    if os.path.exists(mp3_path):
        print("\n1. Using MP3 file: Everyone Else Art Pop.mp3")
        audio_path = mp3_path
    else:
        print("\n1. Generating test audio...")
        audio_path = generate_test_audio()

    # Load config
    print("\n2. Loading configuration...")
    config = yaml.load(open("configs/inference.yaml"), Loader=yaml.FullLoader)

    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    args = Dict2Class(config['train_config'])

    # Use 3B model if available, otherwise fall back to 1.5B
    use_3b = os.path.exists("af2_3b/safe_ckpt/metadata.json")
    if use_3b:
        print("   Using 3B model")
        model_config['lang_encoder_path'] = 'Qwen/Qwen2.5-3B'
        model_config['tokenizer_path'] = 'Qwen/Qwen2.5-3B'
        ckpt_dir = "af2_3b/safe_ckpt"
        clap_ckpt = "af2_3b/clap_ckpt/epoch_16.pt"
    else:
        print("   Using 1.5B model")
        model_config['lang_encoder_path'] = 'Qwen/Qwen2.5-1.5B'
        model_config['tokenizer_path'] = 'Qwen/Qwen2.5-1.5B'
        ckpt_dir = "safe_ckpt"
        clap_ckpt = "clap_ckpt/epoch_16.pt"

    # Update CLAP checkpoint path
    clap_config['checkpoint'] = clap_ckpt

    # Create model
    print("\n3. Creating model...")
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    device_id = 0
    print(f"\n4. Moving model to GPU {device_id}...")
    model = model.to(device_id)
    model.eval()

    # Load checkpoint
    print("\n5. Loading checkpoint...")
    with open(f"{ckpt_dir}/metadata.json", "r") as f:
        metadata = json.load(f)

    state_dict = {}
    for chunk_name in metadata:
        chunk_path = f"{ckpt_dir}/{chunk_name}.safetensors"
        print(f"   Loading {chunk_path}...")
        chunk_tensors = load_file(chunk_path)
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    print(f"   Loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    cast_dtype = get_cast_dtype(args.precision)

    inference_kwargs = {
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    # Test prompts
    print("\n6. Running inference...")
    print("-" * 60)

    prompts = [
        "Caption the input audio.",
        "What sounds can you hear in this audio?",
        "Describe the music in this audio.",
    ]

    for prompt in prompts:
        predict(audio_path, prompt, clap_config, model, tokenizer, device_id, cast_dtype, inference_kwargs)

    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
