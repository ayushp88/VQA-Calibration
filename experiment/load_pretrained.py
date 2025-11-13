import torch, logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
    GenerationConfig, # Added as it was imported in the initial code block
)
#from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    LlavaOnevisionForConditionalGeneration
)

from peft import PeftModel
from typing import Any, Tuple, Optional
from transformers import BartForSequenceClassification, BartTokenizer
from huggingface_hub import login # Added for load_causal_lm (vLLM path)
from vllm import LLM, SamplingParams # Added for load_causal_lm (vLLM path)
import numpy as np # Added as it was imported in the initial code block

print(__name__, "load pretrained")


# 0) Optional monkey-patch to satisfy any stray HF calls
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.get_default_device = get_default_device

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device: {device}")

DEFAULT_CACHE_DIR = "/DATA1/Imagenet2012/collaborative-calibration/data/hf_cache"
DEFAULT_NLI_CLS = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def load_entailment_classifier(nli_model: str = DEFAULT_NLI_CLS, cache_dir: str = DEFAULT_CACHE_DIR):
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model, use_fast=True, cache_dir=cache_dir)
    nli_classifier = AutoModelForSequenceClassification.from_pretrained(nli_model, cache_dir=cache_dir)
    logging.debug("entailment_classifier loaded")
    return nli_tokenizer, nli_classifier


def load_causal_lm(
    model_id: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    access_token: Optional[str] = None,
    use_vllm: bool = False,
    adapter_checkpoint: Optional[str] = None
) -> Tuple[Optional[Any], Any]:
    """
    Unified loader for multiple causal / multimodal LMs with optional PEFT adapter.

    Returns:
        (processor_or_tokenizer, model)
        If use_vllm=True, returns (None, vllm_model).
    """
    #logging.info(f"Attempting to load causal LM: {model_id} (use_vllm={use_vllm})")
    lower_id = model_id.lower()

    # ---- Access token check for gated families (example: llama) ----
    if "llama" in lower_id and not access_token and not use_vllm:
        raise ValueError("HF access token required for Llama models.")

    # ---- vLLM branch ----
    if use_vllm:
        if LLM is None:
            raise ImportError("vLLM not installed. Install vllm or set use_vllm=False.")
        if access_token:
            login(token=access_token)
        #logging.debug(f"GenerationConfig: {GenerationConfig.from_pretrained(model_id)}")
        vllm_model = LLM(model=model_id, trust_remote_code=True, download_dir=cache_dir)
        #logging.info(f"{model_id} loaded with vLLM.")
        return None, vllm_model

    # =================================================================
    # Branch 1: Qwen2.5-VL (vision-language)
    # =================================================================
    if "qwen" in lower_id and "vl" in lower_id:
        from transformers import Qwen2_5_VLForConditionalGeneration  # ensure installed
        #logging.info("Detected Qwen2.5-VL model.")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=DEFAULT_CACHE_DIR
        )
    
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=DEFAULT_CACHE_DIR)
        base_model.eval()
        #logging.info("Loaded Qwen2.5-VL successfully.")
        return processor, base_model

    # =================================================================
    # Branch 2: Phi-4 multimodal (e.g., microsoft/Phi-4-multimodal-instruct)
    # =================================================================
    if "phi-4" in lower_id or "phi-4-multimodal" in lower_id:
        #logging.info("Detected Phi-4 multimodal model.")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=DEFAULT_CACHE_DIR
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
            _attn_implementation="sdpa",
        ).eval()

        #logging.info("Loaded Phi-4 multimodal successfully.")
        return processor, model

    # =================================================================
    # Branch 3: IBM Granite Vision models (vision-to-seq)
    # =================================================================
    if "granite" in lower_id and "vision" in lower_id:
        from transformers import AutoModelForVision2Seq  # ensure version supports
        #logging.info("Detected IBM Granite Vision model.")
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=DEFAULT_CACHE_DIR,
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            cache_dir=DEFAULT_CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

        #logging.info("Loaded Granite Vision model successfully.")
        return processor, model

    # =================================================================
    # Branch 4: Gemma3 (vision-language variant) – example logic
    # =================================================================
    if "gemma-3" in lower_id:
        # If you have a pure Gemma Vision model class, import it; else fallback to AutoModelForCausalLM
        from transformers import Gemma3ForConditionalGeneration  # ensure available
        logging.info("Detected Gemma3 model.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=DEFAULT_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=DEFAULT_CACHE_DIR,
            trust_remote_code=True
        )
        FINETUNED_DIR = "gemma-pathvqak-focal-yesno"
        model = PeftModel.from_pretrained(model, FINETUNED_DIR)
        model.eval()
        logging.info("Loaded Gemma3 model successfully.")
        return processor, model

    if "llava" in lower_id:
        # If you have a pure Gemma Vision model class, import it; else fallback to AutoModelForCausalLM
        from transformers import Gemma3ForConditionalGeneration  # ensure available
        tokenizer = AutoTokenizer.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            cache_dir=DEFAULT_CACHE_DIR,
            use_fast=True
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            cache_dir=DEFAULT_CACHE_DIR,
            tokenizer=tokenizer
        )
        processor.tokenizer.padding_side = "right"

        # ─── 3) Load & 4-bit Quantize Model ────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            cache_dir=DEFAULT_CACHE_DIR,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        logging.info("Loaded Gemma3 model successfully.")
        return processor, model


import logging
import torch
import numpy as np
from transformers import GenerationConfig

def causal_lm_generate(
    #model_id: str,
    processor,
    model,
    prompt: str,
    image=None,                         # PIL.Image or None
    choices=None,                      # Optional multiple-choice list -> enumerated into the prompt
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    return_joint_prob: bool = True,
    force_family: str = None,          # "gemma", "qwen", "granite", "phi"
    add_generation_prompt: bool = True,
    **gen_kwargs
):
    """
    Unified inline generation for:
      - Phi-4 multimodal (manual special-token prompt)
      - Gemma, Qwen2.5-VL, Granite Vision, generic chat-template models

    Returns:
        (lm_prob, text)
    """
    
    model_id = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
        
    lower_id = model_id
    if force_family is not None:
        family = force_family.lower()
    else:
        if "phi-4" in lower_id or "phi-4-multimodal" in lower_id:
            family = "phi"
        elif "gemma-3" in lower_id:
            family = "gemma"
        elif "llava" in lower_id:
            family = "gemma"
        elif "qwen" in lower_id and "vl" in lower_id:
            family = "qwen"
        elif "granite" in lower_id and "vision" in lower_id:
            family = "granite"
        else:
            family = "generic"

    # --- 1) Build (possibly MC) prompt text (used by non-Phi families) ---
    if choices:
        enumerated = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        prompt_text = (
            f"Question: {prompt}\n"
            f"Choices:\n{enumerated}\n"
            "Please answer with the correct choice number and the full answer text. No explanation."
        )
    else:
        prompt_text = prompt

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------
    # PHI FAMILY (manual special token construction)
    # ------------------------------------------------------------------
    if family == "phi":
        # 0) Special tokens
        user_prompt      = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix    = "<|end|>"
        image_token      = "<|image_1|>"

        # 1) Raw prompt string
        text_prompt = (
            f"{user_prompt} "
            f"{(image_token + ' ') if image is not None else ''}"
            f"{prompt_text} "
            f"{prompt_suffix} "
            f"{assistant_prompt}"
        )

        # 2) Tokenize
        if image is not None:
            inputs = processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(
                text=text_prompt,
                return_tensors="pt",
                padding=True
            )
        inputs = inputs.to(device)

        # 3) (Optional) GenerationConfig
        try:
            gen_config = GenerationConfig.from_pretrained(model_id)
        except Exception:
            # Fallback to a neutral config if the model_id doesn't have one
            gen_config = GenerationConfig()

        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = temperature
        gen_config.top_p = top_p
        gen_config.do_sample = do_sample

        # 4) Generate
        input_len = inputs["input_ids"].shape[1]
        gen_out = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,          # also explicit; harmless duplication
            return_dict_in_generate=True,
            output_scores=return_joint_prob,
            num_logits_to_keep=1,
            **gen_kwargs
        )

        # 5) Decode
        seqs = gen_out.sequences
        gen_ids = seqs[:, input_len:]
        # For Phi multimodal processors, batch_decode usually present
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        else:
            # fallback to tokenizer if exposed
            text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        logging.info(f"[Phi] Model output: {text}")

        # 6) Joint probability
        lm_prob = None
        if return_joint_prob:
            try:
                trans_scores = model.compute_transition_scores(
                    sequences=seqs,
                    scores=gen_out.scores,
                    normalize_logits=True,
                )
                # Only generated portion
                gen_log_probs = trans_scores[0][input_len:]
                if gen_log_probs.numel() == 0:
                    lm_prob = 0.0
                else:
                    # geometric mean
                    lm_prob = float(torch.exp(gen_log_probs.mean()).item())
                logging.info(f"[Phi] Joint probability: {lm_prob:.4g}")
            except Exception as e:
                logging.warning(f"[Phi] Could not compute joint probability: {e}")
                lm_prob = None

        return lm_prob, text

    # ------------------------------------------------------------------
    # GEMMA / QWEN / GRANITE / GENERIC (chat template path)
    # ------------------------------------------------------------------
    system_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    if image is not None:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text}
        ]
    else:
        user_content = [
            {"type": "text",  "text": prompt_text}
        ]
    messages = [system_msg, {"role": "user", "content": user_content}]

    try:
        inputs = processor.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
        )
    except AttributeError as e:
        raise ValueError(
            "Processor must support .apply_chat_template(...) for this family."
        ) from e

    # Move to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    input_len = inputs["input_ids"].shape[1]

    # GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_id)
    except Exception:
        gen_config = GenerationConfig()

    gen_config.max_new_tokens = max_new_tokens
    gen_config.temperature = temperature
    gen_config.top_p = top_p
    gen_config.do_sample = do_sample

    gen_out = model.generate(
        **inputs,
        generation_config=gen_config,
        return_dict_in_generate=True,
        output_scores=return_joint_prob,
        **gen_kwargs
    )

    sequences = gen_out.sequences
    gen_ids = sequences[:, input_len:]

    if hasattr(processor, "decode"):
        text = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
    elif hasattr(processor, "batch_decode"):
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    elif hasattr(processor, "tokenizer"):
        text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    else:
        raise ValueError("No decode method available in processor/tokenizer.")

    lm_prob = None
    if return_joint_prob:
        try:
            trans_scores = model.compute_transition_scores(
                sequences=gen_out.sequences,
                scores=gen_out.scores,
                normalize_logits=True
            )
            token_probs = torch.exp(trans_scores[0]).cpu().tolist()
            lm_prob = float(np.prod(token_probs) ** (1 / len(token_probs))) if token_probs else 0.0

        except Exception as e:
            logging.warning(f"Could not compute joint probability: {e}")
            lm_prob = None

    return lm_prob, text
