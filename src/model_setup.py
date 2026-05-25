import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
    BitsAndBytesConfig
)

from sentence_transformers import SentenceTransformer
from config import CONFIG


def setup_models():

    models = {}

    # =========================
    # Qwen 4-bit Quantized
    # =========================

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    models["qwen_tokenizer"] = AutoTokenizer.from_pretrained(
        CONFIG['models']['qwen'],
        trust_remote_code=True
    )

    models["qwen_model"] = AutoModelForCausalLM.from_pretrained(
        CONFIG['models']['qwen'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    models["qwen_model"].eval()

    # =========================
    # T5
    # =========================

    models["t5_tokenizer"] = T5Tokenizer.from_pretrained(
        CONFIG['models']['t5']
    )

    models["t5_model"] = T5ForConditionalGeneration.from_pretrained(
        CONFIG['models']['t5']
    ).to(CONFIG['device'])

    # =========================
    # Sentiment Pipeline
    # =========================

    models["sentiment_pipeline"] = pipeline(
        "sentiment-analysis",
        model=CONFIG['models']['sentiment'],
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )

    # =========================
    # Sentence Transformer
    # =========================

    models["sentence_model"] = SentenceTransformer(
        CONFIG['models']['sentence'],
        device=CONFIG['device']
    )

    return models