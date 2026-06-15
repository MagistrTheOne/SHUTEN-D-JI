"""Test tokenizer compatibility with vLLM."""
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("/workspace/models/qwen3-235b-gptq")
print("has all_special_tokens_extended:", hasattr(t, "all_special_tokens_extended"))
print("tokenizer type:", type(t).__name__)
print("TOKENIZER OK")
