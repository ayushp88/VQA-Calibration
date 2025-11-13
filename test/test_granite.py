#!/usr/bin/env python3
import os
# ─── 1) Cache & Device Setup ────────────────────────────────────────
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
os.environ["HF_HOME"]               = HF_CACHE
os.environ["TRANSFORMERS_CACHE"]    = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"]     = HF_CACHE
os.environ["HF_ATTN_IMPLEMENTATION"] = "eager"  
import csv
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download  # for single-image example
 # disable FlashAttention-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
# 0) Optional monkey-patch to satisfy any stray HF calls
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.get_default_device = get_default_device
# ─── 2) Load Granite Model & Processor ─────────────────────────────
model_path = "ibm-granite/granite-vision-3.3-2b"
config = AutoConfig.from_pretrained(model_path, cache_dir=HF_CACHE)
processor = AutoProcessor.from_pretrained(
    model_path,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
).to(device).eval()

# ─── 3) Single-Image Sanity-Check (optional) ────────────────────────
# img_path = hf_hub_download(repo_id=model_path, filename="example.png")
# conversation = [
#     {"role": "user", "content": [
#         {"type": "image", "url": img_path},
#         {"type": "text",  "text": "What is the highest scoring model on ChartQA and what is its score?"}
#     ]},
# ]
# inputs = processor.apply_chat_template(
#     conversation,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(device)
# out = model.generate(**inputs, max_new_tokens=100)
# print(processor.decode(out[0], skip_special_tokens=True))

# ─── 4) Load ScienceQA Test Split ───────────────────────────────────
dataset = load_dataset("derek-thomas/ScienceQA", split="test")
# if you only want Language-Science:
# dataset = dataset.filter(lambda ex: ex["subject"] == "language science")

# ─── 5) Prepare CSV for Logging ─────────────────────────────────────
csv_path = "scienceqa_granite_predictions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "question_id", "predicted_choice", "lm_prob",
        "correct_choice_number", "correct_choice_text"
    ])
    writer.writeheader()

# ─── 6) Inference Loop ──────────────────────────────────────────────
for idx, ex in enumerate(tqdm(dataset, desc="ScienceQA→Granite")):
    # 6a) safe QID
    qid = ex.get("id") or ex.get("question_id") or f"idx_{idx}"

    # 6b) text parts
    question = ex["question"]
    choices   = ex["choices"]
    answer_idx= int(ex["answer"])

    # 6c) grab the dataset-cached local image path
    #    HF datasets already downloads image to disk; ex["image"]["path"] holds it.
    img_path = None
    img_field = ex.get("image")
    if isinstance(img_field, dict) and img_field.get("path"):
        img_path = img_field["path"]

    # 6d) build prompt text
    choices_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(choices))
    prompt = (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n"
        "Please answer by selecting the correct choice. Only mention choice number and full answer. Dont give explanation"
    )

    # 6e) assemble chat messages, including image only if we have a path
    user_content = []
    if img_path:
        user_content.append({"type": "image", "url": img_path})
    user_content.append({"type": "text",  "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user",   "content": user_content},
    ]

    # 6f) tokenize & prepare tensors
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    # 6g) generate + decode + score
    try:
        in_len = inputs.input_ids.shape[1]
        gen   = model.generate(
            **inputs,
            max_new_tokens=16,
            return_dict_in_generate=True,
            output_scores=True,
        )
        out_ids     = gen.sequences[:, in_len:]
        raw_answer  = processor.decode(out_ids[0], skip_special_tokens=True).strip()

        trans_scores= model.compute_transition_scores(
            sequences=gen.sequences,
            scores=gen.scores,
            normalize_logits=True
        )
        probs       = torch.exp(trans_scores[0]).cpu().tolist()
        lm_prob     = float(np.prod(probs) ** (1/len(probs))) if probs else 0.0

        # 6h) log to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id", "predicted_choice", "lm_prob",
                "correct_choice_number", "correct_choice_text"
            ])
            writer.writerow({
                "question_id":           qid,
                "predicted_choice":      raw_answer,
                "lm_prob":               lm_prob,
                "correct_choice_number": answer_idx + 1,
                "correct_choice_text":   choices[answer_idx],
            })

    except Exception as e:
        print(f"Skipped QID={qid} due to error: {e}")
