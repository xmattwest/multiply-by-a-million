#!/usr/bin/env python3
"""model_io.py (POD-SIDE): load Gemma 4 text-only, re-check the manifest, map layers.

Runs INSIDE the pinned vLLM 0.30.0 image (transformers >= 5.10.4, torch 2.13,
bf16) on one RTX PRO 6000. Not a laptop script (imports torch/transformers).

Gemma 4 12B (google/gemma-4-12B-it) is a unified, encoder-free multimodal model:
config.json has architectures=['Gemma4UnifiedForConditionalGeneration'],
model_type='gemma4_unified', with text_config/vision_config/audio_config.
Weights are nested as model.language_model.layers.N.*. We load the checkpoint
with the class the config names and run only the text path (no pixel/audio
inputs), hooking the decoder layers. Everything here is exploratory.
"""
import hashlib
import os

import torch                                   # pod-only
import transformers                            # pod-only
from transformers import AutoConfig, AutoTokenizer   # pod-only

EXPECTED_MODEL_TYPES = ("gemma4_unified", "gemma4")   # 12B (unified, encoder-free); 31B (vision tower)
SDPA_BACKENDS = None      # set by run_interp's diagnosis when a specific SDPA kernel is needed


def attn_ctx():
    """Context for every model call: the SDPA kernel the diagnosis chose, or nothing."""
    import contextlib
    if SDPA_BACKENDS is None:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel
    return sdpa_kernel(SDPA_BACKENDS)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recheck_manifest(model_dir, manifest_tsv, only=("config.json", "tokenizer_config.json",
                                                    "chat_template.jinja", "tokenizer.json")):
    """Re-verify a few small files against the manifest (the boot script checked all)."""
    want = {}
    with open(manifest_tsv, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] and not parts[0].startswith("#"):
                want[parts[0]] = (int(parts[1]), parts[2])
    checked = []
    for rel in only:
        assert rel in want, f"manifest has no entry for {rel}"
        size, sha = want[rel]
        full = os.path.join(model_dir, rel)
        assert os.path.getsize(full) == size, f"{rel}: size mismatch"
        got = sha256_file(full)
        assert got == sha, f"{rel}: sha256 mismatch"
        checked.append((rel, got))
    return checked


def load_config(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    assert cfg.model_type in EXPECTED_MODEL_TYPES, f"model_type {cfg.model_type!r}, expected one of {EXPECTED_MODEL_TYPES}"
    text_cfg = getattr(cfg, "text_config", None)
    assert text_cfg is not None, "config has no text_config"
    return cfg, text_cfg


def _model_class(cfg):
    """The class the checkpoint was saved from (config.architectures), so the weight names match
    by construction: Gemma4UnifiedForConditionalGeneration (12B), Gemma4ForConditionalGeneration (31B).
    AutoModelForCausalLM is the fallback (it loaded 12B on the pod, transformers 5.17.0, 2026-10-05)."""
    for arch in (getattr(cfg, "architectures", None) or []):
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls, arch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM, "AutoModelForCausalLM"


def _from_pretrained(cls, model_dir, dtype):
    """Returns (model, how, loading_info or None). SI_INTERP_DEVICE=cpu is for the local rehearsal only."""
    kw = {"dtype": dtype}
    device = os.environ.get("SI_INTERP_DEVICE", "cuda")
    try:
        model, info = cls.from_pretrained(model_dir, device_map=device, output_loading_info=True, **kw)
        return model, f"device_map={device}", info
    except TypeError:                         # output_loading_info not accepted
        return cls.from_pretrained(model_dir, device_map=device, **kw), f"device_map={device}", None
    except (ImportError, ValueError) as e:    # accelerate missing (present on the pod 2026-10-05); v5 raises ValueError
        if "accelerate" not in str(e):
            raise
        model, info = cls.from_pretrained(model_dir, output_loading_info=True, **kw)
        return model.to(device), f"no accelerate: loaded, then .to({device})", info


def check_loading_info(info):
    """No weight may be missing or unexpected (tied lm_head excepted): else layers would be random."""
    if info is None:
        return "not reported"
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    unexpected = list(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    assert not missing and not unexpected and not mismatched, (
        f"weights do not match the model: missing {missing[:5]} ({len(missing)}), "
        f"unexpected {unexpected[:5]} ({len(unexpected)}), mismatched {mismatched[:3]}")
    return "all weights matched"


def load_text_only(model_dir, dtype=torch.bfloat16):
    """Load the checkpoint on the GPU; return (model, tokenizer, info)."""
    cfg, text_cfg = load_config(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cls, cls_name = _model_class(cfg)
    model, how, loading = _from_pretrained(cls, model_dir, dtype)
    weights_check = check_loading_info(loading)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = int(text_cfg.num_hidden_layers)
    layers = decoder_layers(model, n_layers)
    info = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_class": cls_name,
        "load": how,
        "model_type": cfg.model_type,
        "text_model_type": getattr(text_cfg, "model_type", None),
        "num_hidden_layers": n_layers,
        "hidden_size": int(text_cfg.hidden_size),
        "vocab_size": int(text_cfg.vocab_size),
        "num_layers_found": len(layers),
        "final_logit_softcapping": getattr(text_cfg, "final_logit_softcapping", None),
        "dtype": str(dtype),
        "device": str(next(model.parameters()).device),
        "weights_check": weights_check,
        "gpu": gpu_memory(),
    }
    return model, tok, info


def gpu_memory():
    """GiB used / total on this GPU, and the peak our process allocated (None without CUDA)."""
    if not torch.cuda.is_available():
        return {"used_gib": None, "total_gib": None, "peak_alloc_gib": None}
    free, total = torch.cuda.mem_get_info()
    return {"used_gib": round((total - free) / 2**30, 1), "total_gib": round(total / 2**30, 1),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}


def text_decoder(model, n_layers=None):
    """The text decoder module (the one owning .layers)."""
    m = getattr(model, "model", model)
    lm = getattr(m, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm
    # Fallback: the ModuleList named 'layers' with the configured length.
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) \
                and (n_layers is None or len(mod) == n_layers) and "vision" not in name and "audio" not in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            return parent
    raise AttributeError("could not locate the text decoder")


def decoder_layers(model, n_layers=None):
    layers = text_decoder(model, n_layers).layers
    assert n_layers is None or len(layers) == n_layers, f"found {len(layers)} layers, config says {n_layers}"
    return layers


def render_messages(tok, messages):
    """Chat template with thinking off -> (prompt_text, input_ids [1, T] LongTensor).

    Renders text first and tokenizes it separately (the template already holds
    <bos>, so add_special_tokens=False). This avoids apply_chat_template's
    return type, which changed across transformers versions.
    """
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return text, ids
