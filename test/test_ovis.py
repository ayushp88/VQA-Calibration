import numpy as np
import os
HF_CACHE = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
import torch
import torchvision.transforms as T
from PIL import Image
import io
import csv
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM
from io import BytesIO

import math
import numpy as np
import torch
import torchvision.transforms as T
#from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-2B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             use_flash_attn=False,  
                                             trust_remote_code=True, cache_dir = HF_CACHE).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()
model.config.llm_config._attn_implementation = "eager"
# single-image input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = model.get_text_tokenizer()
model.config.vocab_size = tokenizer.vocab_size
# ─── 3) Helper to open images ─────────────────────────────────────────
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

# ─── 4) Load ScienceQA test split ───────────────────────────────────
dataset = load_dataset("derek-thomas/ScienceQA", split="test")

# ─── 5) Prepare CSV for logging ─────────────────────────────────────
csv_path = "scienceqa_ovis_predictionsff.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["question_id", "prediction", "joint_token_prob", "correct_choice_number", "correct_choice_text"]
    )
    writer.writeheader()

# ─── 6) Inference Loop with Ovis2 ────────────────────────────────────
for ex in tqdm(dataset, desc="Evaluating ScienceQA with Ovis2"):
    qid = ex.get("id") or ex.get("question_id")
    question = ex.get("question", "")
    choices = ex.get("choices", [])
    answer_idx = int(ex.get("answer", 0))

    # format prompt
    choices_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(choices))
    prompt_text = (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n"
        "Please respond **only** with the correct choice in this exact format: "
        "<answer></answer>, where n is the choice number. No other text or explanation."
    )

    # load image if available
    img = open_image(ex.get("image"))
    if img is not None:
        query = f"<image>\n{prompt_text}"
        prompt, input_ids, pixel_values = model.preprocess_inputs(
                query, [img], max_partition=9
            )
    else:
        images=[]
        query = prompt_text
        prompt, input_ids, pixel_values = model.preprocess_inputs(
                query, images, max_partition=None)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    input_len = input_ids.shape[1]
    print(input_len)
    # generate
    with torch.inference_mode():
        gen_out = model.generate(
            input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True, # Collect the full GenerateOutput object
            output_scores=True,
        )
        full_seqs = gen_out.sequences   
        true_vocab_size = gen_out.scores[0].shape[-1]
        model.config.vocab_size = true_vocab_size        # The full sequence (input + generated 
        # decode answer
        pred = text_tokenizer.decode(full_seqs[0], skip_special_tokens=True).strip()
        gen_ids    = gen_out.sequences[:, input_len:]
        #raw_output = processor.decode(gen_ids[0], skip_special_tokens=True).strip()

        # compute average token probability (geometric mean)
        trans_scores = model.compute_transition_scores(
            sequences=gen_out.sequences,
            scores=gen_out.scores,
            normalize_logits=True
        )
        token_probs = torch.exp(trans_scores[0]).cpu().tolist()
        lm_prob = float(np.prod(token_probs) ** (1 / len(token_probs))) if token_probs else 0.0
    
        # Calculate the geometric mean of token probabilities for the joint probability
    
        print(lm_prob)

    # log results
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "prediction", "joint_token_prob","correct_choice_number", "correct_choice_text"]
        )
        writer.writerow({
            "question_id": qid,
            "prediction": pred,
            "joint_token_prob":lm_prob,
            "correct_choice_number": answer_idx + 1,
            "correct_choice_text": choices[answer_idx]
        })
