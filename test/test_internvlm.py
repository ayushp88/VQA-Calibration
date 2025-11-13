#!/usr/bin/env python3
import os

# ─── 0) HF cache setup ──────────────────────────────────────────────────
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
os.environ["HF_HOME"]               = HF_CACHE
os.environ["TRANSFORMERS_CACHE"]    = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]     = HF_CACHE

import io
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig
)

# ─── 1) Device & Dynamo config ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# ─── 2) Load model & processor ────────────────────────────────────────
model_path = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    cache_dir=HF_CACHE
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    cache_dir=HF_CACHE,
    trust_remote_code=True,
    _attn_implementation="sdpa",  # or "eager" on pre‐Ampere GPUs
).to(device)
generation_config = GenerationConfig.from_pretrained(model_path)

# ─── 3) Load dataset & prepare CSV ─────────────────────────────────────
dataset = load_dataset("derek-thomas/ScienceQA", split="test")  # full, non-streaming
csv_path = "scienceqa_phi4_predictions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "question_id",
        "predicted_choice",
        "lm_prob",
        "correct_choice_number",
        "correct_choice_text"
    ])
    writer.writeheader()

# ─── 4) Image loader helper ────────────────────────────────────────────
def open_image(img_field):
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

# ─── 5) Prompt tokens ──────────────────────────────────────────────────
user_prompt      = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix    = "<|end|>"
image_token      = "<|image_1|>"

# ─── 6) Inference loop ─────────────────────────────────────────────────
for ex in tqdm(dataset, desc="Evaluating ScienceQA"):
    # identifiers & question data
    qid        = ex.get("id") or ex.get("question_id")
    question   = ex["question"]
    choices    = ex["choices"]
    answer_idx = int(ex["answer"])

    # open image if available
    pil_img = open_image(ex.get("image"))

    # build MC prompt
    choices_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(choices))
    mc_prompt = (
        f"{user_prompt}"
        f"{image_token if pil_img is not None else ''}"
        f"Question: {question}\n"
        f"Choices:\n{choices_text}\n"
        f"Please answer by selecting the correct choice (number and full answer). Dont give any explanation, only corect answer from the choices"
        f"{prompt_suffix}"
        f"{assistant_prompt}"
    )

    # tokenize & prepare inputs (with or without image)
    if pil_img is not None:
        inputs = processor(
            text=mc_prompt,
            images=pil_img,
            return_tensors="pt"
        )
    else:
        inputs = processor(
            text=mc_prompt,
            return_tensors="pt"
        )
    inputs = inputs.to(device)

    try:
        # generate
        prompt_len = inputs.input_ids.shape[1]
        gen_out = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=16,
            return_dict_in_generate=True,
            output_scores=True,
            num_logits_to_keep=1
        )

        # decode response
        gen_ids    = gen_out.sequences[:, prompt_len:]
        raw_output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # compute joint probability
        trans_scores = model.compute_transition_scores(
            sequences=gen_out.sequences,
            scores=gen_out.scores,
            normalize_logits=True
        )
        token_probs = torch.exp(trans_scores[0]).cpu().tolist()
        lm_prob = float(np.prod(token_probs) ** (1 / len(token_probs))) if token_probs else 0.0

        # log to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id",
                "predicted_choice",
                "lm_prob",
                "correct_choice_number",
                "correct_choice_text"
            ])
            writer.writerow({
                "question_id":           qid,
                "predicted_choice":      raw_output,
                "lm_prob":               lm_prob,
                "correct_choice_number": answer_idx + 1,
                "correct_choice_text":   choices[answer_idx]
            })

    except Exception as e:
        print(f"Skipped QID={qid} due to error: {e}")
