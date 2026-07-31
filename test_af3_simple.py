# Simple test for Audio Flamingo 3 using transformers directly
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModel, AutoConfig
from huggingface_hub import snapshot_download
import librosa

print("=" * 60)
print("Audio Flamingo 3 - Simple Test")
print("=" * 60)

model_id = "nvidia/audio-flamingo-3-hf"

print("\n1. Loading model with trust_remote_code=True...")
try:
    # Try loading config first
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"   Config loaded: {config.model_type}")

    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"   Model loaded!")

    # Load processor if available
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"   Processor loaded!")
    except Exception as e:
        print(f"   No processor: {e}")

except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
