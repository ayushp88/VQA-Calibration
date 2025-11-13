#!/usr/bin/env python3
import os
import io
import csv
import json
from typing import Optional

import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

# ─── 1) Environment & Device Setup ─────────────────────────────────
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
    os.environ[k] = HF_CACHE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Make torchdynamo quiet if present
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
except Exception:
    pass

# ─── 2) Model / Processor ─────────────────────────────────────────
MODEL_ID = "google/gemma-3-4b-it"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
# For chat-style decoding + short generations, right padding is safer
if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
    processor.tokenizer.padding_side = "right"

# ─── 3) Data: VQA-RAD yes/no test split ───────────────────────────
dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")

# Keep only yes/no questions (robust to casing/whitespace)
dataset = dataset.filter(
    lambda ex: isinstance(ex.get("answer"), str)
    and ex["answer"].strip().lower() in ["yes", "no"]
)

def open_image(img_field) -> Optional[Image.Image]:
    if img_field is None:
        return None
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict):
        if img_field.get("path"):
            return Image.open(img_field["path"]).convert("RGB")
        if img_field.get("bytes"):
            return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    return None

# ─── 4) Helper: generate text with (optional) messages/images ─────
@torch.no_grad()
def chat_generate(messages, max_new_tokens=64, temperature=0.7, top_p=0.95):
    # Build model inputs via the processor's chat template (supports vision)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    # For VLMs, images go via the messages; some processors also accept an 'images' kw.
    # Current Gemma-3 processor reads image objects directly from messages.
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature),
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # Decode only the generated tail
    input_len = inputs["input_ids"].shape[1]
    gen_ids = gen.sequences[:, input_len:]
    text = processor.decode(gen_ids[0], skip_special_tokens=True)
    return text, gen

def clean_yn(text: str) -> str:
    t = text.strip().lower()
    # extract first token-ish answer robustly
    # handles things like "yes.", "yes, the ...", "answer: yes"
    if "yes" in t[:6]:
        return "yes"
    if "no" in t[:5]:
        return "no"
    # fallback: pick the first occurrence
    yi = t.find("yes")
    ni = t.find("no")
    if yi == -1 and ni == -1:
        return t.split()[0] if t else ""
    if yi == -1:
        return "no"
    if ni == -1:
        return "yes"
    return "yes" if yi < ni else "no"

# ─── 5) Knowledge step (self-deliberation) ─────────────────────────
@torch.no_grad()
def generate_background_paragraph(question: str, max_words: int = 70) -> str:
    """
    Lightweight 'knowledge agent' without external deps:
    Ask the same model to draft a short background paragraph before answering.
    """
    sys = {
        "role": "system",
        "content": [
            {"type": "text", "text": (
                "You are a clinical VQA assistant. "
                "Generate a brief background paragraph (<=70 words) that would help answer the user's question. "
                "Do not include the final answer; provide only context."
            )}
        ],
    }
    usr = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Question: {question}\nProvide background only."}
        ],
    }
    text, _ = chat_generate([sys, usr], max_new_tokens=96, temperature=0.6)
    # Trim aggressively to ~max_words (model usually respects it)
    words = text.strip().split()
    return " ".join(words[:max_words])

# ─── 6) Temps (4 model samples) ────────────────────────────────────
TEMPS = [0.4, 0.7, 1.0, 1.3]

# ─── 7) CSV logging ────────────────────────────────────────────────
csv_path = "pathrad_gemma4temps_knowledge.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["question_id", "temperature", "predicted_choice", "lm_prob", "correct_answer"]
    )
    writer.writeheader()

# ─── 8) Inference Loop with Knowledge-Augmented Prompt ────────────
for ex in tqdm(dataset, desc="VQA-RAD yes/no with Gemma-3 + knowledge prompt @ 4 temps"):
    qid = ex.get("id") or ex.get("image_id")
    answer = ex.get("answer", "").strip().lower()
    question = (ex.get("question") or "").strip()

    # load image
    img = open_image(ex.get("image"))
    if img is None:
        # Skip if no usable image (VQA-RAD should have one)
        continue

    # --- Step A: background knowledge generation (text-only)
    background = generate_background_paragraph(question)

    # --- Step B: final answer with image + background
    # System prompt carries the guardrails and background context.
    system_msg = {
        "role": "system",
        "content": [
            {"type": "text", "text": (
                "You are a helpful radiology VQA assistant.\n"
                "Use the background context when answering.\n"
                "Respond with a single token: 'yes' or 'no' only. No explanation."
            )},
            {"type": "text", "text": f"Background context: {background}"},
        ],
    }
    user_content = [
        {"type": "image", "image": img},
        {"type": "text", "text": f"Question: {question}\nAnswer 'yes' or 'no' only."},
    ]
    messages = [system_msg, {"role": "user", "content": user_content}]

    try:
        # Precompute inputs once to get input_len for score indexing
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        for temp in TEMPS:
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(temp),
                    top_p=0.95,
                    max_new_tokens=4,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # decode generated piece
            gen_ids = gen_out.sequences[:, input_len:]
            raw_output = processor.decode(gen_ids[0], skip_special_tokens=True)
            yn_pred = clean_yn(raw_output)

            # first generated token probability
            scores = gen_out.scores  # list over generated time steps
            # Defensive: if for some reason no scores, set NaN
            if scores and len(scores) > 0 and gen_ids.numel() > 0:
                probs0 = torch.softmax(scores[0][0], dim=-1)
                token_id0 = gen_ids[0][0].item()
                lm_prob = float(probs0[token_id0].item())
            else:
                lm_prob = float("nan")

            # log one row per temperature
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["question_id", "temperature", "predicted_choice", "lm_prob", "correct_answer"],
                )
                writer.writerow(
                    {
                        "question_id": qid,
                        "temperature": temp,
                        "predicted_choice": yn_pred,
                        "lm_prob": lm_prob,
                        "correct_answer": answer,
                    }
                )

    except Exception as e:
        print(f"Skipped QID={qid} due to error: {e}")

print(f"\nDone. Wrote: {csv_path}")
