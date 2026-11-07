"""Carrega modelo Hugging Face (base ou base+LoRA) e devolve `generate(prompts) -> textos`.
Imports pesados são preguiçosos; este módulo não é coberto por testes unitários (precisa de GPU)."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def load_generate(
    model_id: str, adapter: str | None = None, max_new_tokens: int = 256, load_in_4bit: bool = True
) -> tuple[object, Callable[[Sequence[str]], list[str]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if cc >= 8 else torch.float16  # T4 (7.5) usa fp16
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        if load_in_4bit
        else None
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map="auto", dtype=dtype
    )
    if adapter:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, adapter)
    m.eval()

    def generate(prompts: Sequence[str]) -> list[str]:
        enc = tok(list(prompts), return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    return tok, generate
