#!/usr/bin/env python3
import os
import io
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

# ─── 0) HF cache setup ──────────────────────────────────────────────────
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
    os.environ[k] = HF_CACHE

# ─── 1) Device & Dynamo config ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


from transformers import Gemma3ForConditionalGeneration, LlavaOnevisionForConditionalGeneration # ensure available
from transformers import Gemma3ForConditionalGeneration, LlavaOnevisionForConditionalGeneration # ensure available
#!/usr/bin/env python3
"""
Fine‑tune Gemma‑3‑4B‑IT on VQA‑RAD with:
 - 4‑bit quantization (bitsandbytes)
 - LoRA (PEFT)
 - Cross‑entropy + Focal Loss + Local Calibration Error (LCE) loss
"""
import os
import io
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from datasets import load_dataset, Image as DatasetsImage, DatasetDict
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model

# ====================== 0) CONFIG / CACHE ======================
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
for env in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
    os.environ[env] = HF_CACHE

MODEL_ID = "google/gemma-3-4b-it"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("Need GPU with compute capability ≥ 8.0 for bfloat16.")

# ====================== 1) MODEL + QUANT ======================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
base_model.to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
processor.tokenizer.padding_side = "right"

FINETUNED_DIR = "gemma-finetune-dca"
#IXED_CKPT = "gemma-finetune-mmce-only/adapter_model_fixed.safetensors"
model = PeftModel.from_pretrained(base_model, FINETUNED_DIR, adapter_name="default")




# ─── 3) Load Path‑RAD test split and filter yes/no ─────────────────────
# Replace "path-rad-dataset-id" with the actual Hugging Face dataset ID
dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
# Keep only yes/no questions and this is the 
dataset = dataset.filter(
    lambda ex: isinstance(ex.get("answer"), str)
               and ex["answer"].strip().lower() in ["yes", "no"]
)

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
            return Image.open(io.BytesIO(img_field["bytes"]))
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    return None

# ─── 5) Prepare CSV for logging ────────────────────────────────────────
csv_path = "dca.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "question_id",
        "predicted_choice",
        "lm_prob",
        "correct_answer",
    ])
    writer.writeheader()

# ─── 6) Inference Loop ─────────────────────────────────────────────────
for ex in tqdm(dataset, desc="Evaluating Path‑RAD yes/no with Gemma-3"):
    qid      = ex.get("id") or ex.get("image_id")
    answer   = ex.get("answer", "").strip().lower()
    print(answer)
    # load image
    img = open_image(ex.get("image"))

    # build prompt
    prompt_text = (
        f"Answer the question in yes or no only\n"
        f"{ex.get('question', '').strip()}"
    )

    system_msg = {"role":"system","content":[{"type":"text","text":"You are a helpful assistant. Choose the correct answer from the given choices"}]}
  
    user_content = [{"type":"image","image":img},{"type":"text","text":prompt_text}]

    messages = [system_msg,{"role":"user","content":user_content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)

    try:
        input_len = inputs.input_ids.shape[1]
        gen_out = model.generate(
            **inputs,
            max_new_tokens=4,
            return_dict_in_generate=True,
            output_scores=True
        )
        # extract text
        gen_ids    = gen_out.sequences[:, input_len:]
        raw_output = processor.decode(gen_ids[0], skip_special_tokens=True).strip().lower()
        print(raw_output+"hello")
        # compute first-token probability
        scores = gen_out.scores
        probs0 = torch.softmax(scores[0][0], dim=-1)
        token_id0 = gen_ids[0][0].item()
        lm_prob = float(probs0[token_id0].item())

        # write to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id","predicted_choice","lm_prob","correct_answer"
            ])
            writer.writerow({
                "question_id":      qid,
                "predicted_choice": raw_output,
                "lm_prob":          lm_prob,
                "correct_answer":   answer
            })
    except Exception as e:
        print(f"Skipped QID={qid} due to error: {e}")
