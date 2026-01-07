import runpod
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
from pathlib import Path

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Try multiple paths for cache - ensures it works regardless of RunPod setup
# Priority: 1) /workspace (volume mount), 2) /app (container disk), 3) /tmp (fallback)
if os.path.exists("/workspace"):
    PERSISTENT_CACHE_DIR = "/workspace/.cache/huggingface"
elif os.path.exists("/app"):
    PERSISTENT_CACHE_DIR = "/app/.cache/huggingface"
else:
    PERSISTENT_CACHE_DIR = "/tmp/.cache/huggingface"

# Allow override via environment variable
PERSISTENT_CACHE_DIR = os.environ.get("HF_HOME", PERSISTENT_CACHE_DIR)
os.makedirs(PERSISTENT_CACHE_DIR, exist_ok=True)

# Set Hugging Face cache directory
os.environ["HF_HOME"] = PERSISTENT_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = PERSISTENT_CACHE_DIR

print(f"Using cache directory: {PERSISTENT_CACHE_DIR}")

# Check if model is already cached
def is_model_cached(model_name, cache_dir):
    """Check if model files exist in cache"""
    from huggingface_hub import snapshot_download
    try:
        # Try to get cached files
        cached_files = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Only check local cache, don't download
            token=HF_TOKEN
        )
        return True
    except Exception:
        return False

# Check cache status
model_cached = is_model_cached(MODEL_NAME, PERSISTENT_CACHE_DIR)

if model_cached:
    print(f"✅ Model found in cache! Loading from: {PERSISTENT_CACHE_DIR}")
    print("🚀 No download needed - using cached model")
else:
    print(f"📥 Model not in cache. Downloading to: {PERSISTENT_CACHE_DIR}")
    print("⏳ This will only happen ONCE - subsequent cold starts will use cache")

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load processor - will use cache if available
print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
    cache_dir=PERSISTENT_CACHE_DIR,  # This ensures it checks cache first
    local_files_only=False,  # Allow download if not cached
)

# Load model - will use cache if available
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
    cache_dir=PERSISTENT_CACHE_DIR,  # This ensures it checks cache first
    local_files_only=False,  # Allow download if not cached
)

print("✅ Model loaded successfully!")

def handler(event):
    payload = None
    if isinstance(event, dict):
        payload = event.get("input")
        if payload is None:
            payload = event
    if not isinstance(payload, dict):
        return {"error": "Invalid input payload"}

    prompt = payload.get("text_prompt") or payload.get("text")

    if not prompt:
        return {"error": "No prompt provided. Provide `text_prompt` (or `text`)."}

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64
    )

    result = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return {"output": result}

runpod.serverless.start({"handler": handler})
