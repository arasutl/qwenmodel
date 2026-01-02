import runpod
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
)

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