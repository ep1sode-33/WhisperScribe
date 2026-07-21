#!/usr/bin/env python3
"""Dump per-stage golden tensors from the mlx-vlm DeepSeek-OCR-2 reference.

Run:  .venv/bin/python scripts/gen_fixtures.py
Needs: Fixtures/model_dir.txt (Task 2 Step 1), Fixtures/images/ (Step 2).

NOTE on deviations from the original brief draft (see task-2-report.md for
the full rationale): the brief's script was written against a *guessed*
public interface and is deliberately marked as needing verification against
the real source. Verified against the installed mlx-vlm==0.6.6 (identical to
the local mlx-vlm clone), the following differ from the guess:

  * processor(...) returns a dict with key "images" = [patches, global] (a
    2-element list), NOT separate "pixel_values"/"images_crop" keys.
  * model.sam_model is a TOP-LEVEL attribute (Model.sam_model), not
    model.vision_model.sam_model.
  * Model.get_input_embeddings(input_ids, pixel_values, images_spatial_crop,
    images_seq_mask, **kwargs) takes the whole [patches, global] list as
    `pixel_values` (positional), not separate pixel_values/images_crop args.
  * LanguageModel.__call__(inputs, inputs_embeds=None, mask=None, cache=None)
    requires `inputs` positionally (used only for the embed_tokens fallback
    path when inputs_embeds is None) -- `model.language_model(inputs_embeds=x)`
    alone raises TypeError.
  * greedy_tokens: rather than calling the high-level `generate()` and
    re-encoding its returned text with `tokenizer.encode(text)` (lossy: a
    decode->encode round trip is not guaranteed to reproduce the original
    token id sequence for a BPE/byte-level tokenizer with special/added
    tokens), we drive `mlx_vlm.generate.generate_step` directly with the
    exact same input_ids/pixel_values/images_spatial_crop/images_seq_mask
    used for the prefill_logits computation, and record the raw sampled
    token ids from each step. temperature=0.0 selects the built-in argmax
    sampler (sampler_is_greedy = sampler is None and temperature == 0).
  * pixels_global / pixels_patches are saved exactly as the processor
    returns them: channel-first (N, 3, H, W), bf16 (cast to f32 on save),
    NOT (1,1024,1024,3) as the brief's Interfaces section suggested. The
    model transposes to channel-last internally (right before SAM) with
    `.transpose(0, 2, 3, 1)`; this fixture intentionally captures the
    processor's own output layout since that is what a Swift preprocessing
    port must reproduce bit-for-bit. See report for full shape/dtype table.
  * sam_out / encoder_out / projector_out are computed for the GLOBAL view
    only (1024x1024 -> 256 tokens), matching the brief script's own logic
    (it only forwarded pix_g through these stages, never iterated patches).
    The complementary 768x768 -> 144-token path is covered by the standalone
    mask_144 fixture and by patches captured in pixels_patches.safetensors.
  * Loading: `mlx_vlm.load(MODEL_DIR)` fails outright -- its processor
    loader routes through `transformers.AutoProcessor.from_pretrained(...,
    trust_remote_code=True)`, which resolves the HF repo's own custom
    `modeling_deepseekocr2.py`/`modeling_deepseekv2.py` (needed only to
    determine the *tokenizer* class) and that file imports torch/torchvision
    plus APIs (e.g. `LlamaFlashAttention2`) removed from current
    `transformers` -- a dependency rabbit hole unrelated to the mlx
    inference path, which never touches those files. We instead load the
    mlx model via `mlx_vlm.utils.load_model` (unaffected -- it only reads
    `config.json`/`model.safetensors` through the local `deepseekocr_2`
    package) and construct `DeepseekOCR2Processor` directly from a
    `PreTrainedTokenizerFast.from_pretrained(...)` plus `processor_config.json`,
    bypassing `AutoConfig`/`trust_remote_code` entirely.
  * Tokenizer class: `DeepseekOCR2Processor.tokenizer_class` declares
    `("LlamaTokenizer", "LlamaTokenizerFast")`, but this checkpoint's
    `tokenizer.json` is a GPT2-style byte-level BPE vocab (model.type
    "BPE", decoder type "ByteLevel", tokens like "Hello", "Ġworld", "Ċ")
    -- NOT a SentencePiece Llama vocab. Loading it as `LlamaTokenizerFast`
    causes transformers' Llama-specific fast-tokenizer glue to *overwrite*
    the tokenizer.json's own (correct) ByteLevel decoder with a
    SentencePiece Metaspace-style decoder ("▁"->" ") that does not
    recognize "Ġ"/"Ċ" byte-level markers at all -- so `.decode()` silently
    leaves every space/newline as a literal "Ġ"/"Ċ" character in the
    output text (confirmed by round-tripping "Hello world\nSecond line"
    through both loaders). Loading via the generic
    `PreTrainedTokenizerFast.from_pretrained(...)` instead respects
    tokenizer.json's own ByteLevel decoder verbatim and round-trips
    correctly; `meta.json["text"]` is built from that tokenizer.
"""
import json
import pathlib

import mlx.core as mx
from mlx_vlm.generate import generate_step
from mlx_vlm.models.deepseekocr_2.processing_deepseekocr import DeepseekOCR2Processor
from mlx_vlm.models.deepseekocr_2.vision import Qwen2Decoder2Encoder
from mlx_vlm.utils import get_model_path, load_model
from PIL import Image
from transformers import PreTrainedTokenizerFast

ROOT = pathlib.Path(__file__).parent.parent
FIX = ROOT / "Fixtures"
MODEL_DIR = (FIX / "model_dir.txt").read_text().strip()
PROMPT = "<image>\nFree OCR. "
GREEDY_STEPS = 100

print(f"loading model from {MODEL_DIR} ...")
model_path = get_model_path(MODEL_DIR)
model = load_model(model_path, lazy=False)

tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
proc_cfg = json.loads((model_path / "processor_config.json").read_text())
processor = DeepseekOCR2Processor(
    tokenizer=tokenizer,
    candidate_resolutions=[tuple(r) for r in proc_cfg["candidate_resolutions"]],
    patch_size=proc_cfg["patch_size"],
    downsample_ratio=proc_cfg["downsample_ratio"],
    image_mean=tuple(proc_cfg.get("image_mean", (0.5, 0.5, 0.5))),
    image_std=tuple(proc_cfg.get("image_std", (0.5, 0.5, 0.5))),
    normalize=proc_cfg.get("normalize", True),
    image_token=proc_cfg.get("image_token", "<image>"),
    pad_token=proc_cfg.get("pad_token", "<｜▁pad▁｜>"),
    add_special_token=proc_cfg.get("add_special_token", False),
    sft_format=proc_cfg.get("sft_format", "deepseek"),
    mask_prompt=proc_cfg.get("mask_prompt", True),
    ignore_id=proc_cfg.get("ignore_id", -100),
)


def save(d, name, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(d / f"{name}.safetensors"),
        {
            k: (v.astype(mx.float32) if v.dtype != mx.int32 else v)
            for k, v in arrays.items()
        },
    )


# --- masks standalone (P1 单测夹具) ---
# Reproduces the mixed attention mask built inside
# Qwen2Decoder2Encoder.__call__: image tokens attend to all image tokens
# (bidirectional), query tokens attend to all image tokens + causally to
# earlier query tokens, image tokens never see query tokens.
enc = model.vision_model.qwen2_encoder
assert isinstance(enc, Qwen2Decoder2Encoder), type(enc)
for n in (144, 256):
    causal = mx.triu(mx.full((n, n), -1e9, dtype=mx.float32), k=1)
    top = mx.concatenate([mx.zeros((n, n)), mx.full((n, n), -1e9)], axis=1)
    bot = mx.concatenate([mx.zeros((n, n)), causal], axis=1)
    m = mx.concatenate([top, bot], axis=0)
    save(FIX / "masks", f"mask_{n}", mask=m)
print("masks done")

# --- per-image fixtures ---
for img_path in sorted((FIX / "images").glob("*.png")):
    name = img_path.stem
    out = FIX / name
    print(f"=== {name} ===")

    # 预处理(参考实现自身的 processor). process_one()/tokenize_with_images()
    # take List[PIL.Image.Image], not file paths -- pass an opened image.
    pil_image = Image.open(img_path).convert("RGB")
    inputs = processor(text=PROMPT, images=[pil_image], return_tensors="mlx")
    print(name, "input keys:", list(inputs.keys()))

    ids = mx.array(inputs["input_ids"])
    pix_p, pix_g = inputs["images"]  # [patches, global] -- see processing_deepseekocr.py
    pix_p = mx.array(pix_p)
    pix_g = mx.array(pix_g)
    crop = inputs.get("images_spatial_crop")
    if crop is not None:
        crop = mx.array(crop)
    seq_mask = mx.array(inputs["images_seq_mask"])

    save(out, "pixels_global", x=pix_g)
    has_patches = bool(mx.sum(pix_p).item() != 0)
    n_patch = int(pix_p.shape[0]) if has_patches else 0
    if has_patches:
        save(out, "pixels_patches", x=pix_p)

    # 逐级前向(仅 global view: 1024x1024 -> SAM 16x16x896 -> Qwen2Enc 256x896 -> Proj 256x1280)
    global_hwc = pix_g.transpose(0, 2, 3, 1)  # (1,3,1024,1024) CHW -> (1,1024,1024,3) HWC
    sam = model.sam_model(global_hwc)
    save(out, "sam_out", x=sam)
    enc_out = model.vision_model.qwen2_encoder(sam)
    save(out, "encoder_out", x=enc_out)
    proj_out = model.projector(enc_out)
    save(out, "projector_out", x=proj_out)

    # input embeds: pixel_values arg is the whole [patches, global] list,
    # exactly what the processor returned as inputs["images"].
    embeds_feat = model.get_input_embeddings(
        ids,
        [pix_p, pix_g],
        images_spatial_crop=crop,
        images_seq_mask=seq_mask,
    )
    embeds = embeds_feat.inputs_embeds
    save(out, "input_embeds", x=embeds)

    # prefill logits: full teacher-forced forward over the prompt, last
    # position. `inputs` (ids) must be passed positionally even though
    # inputs_embeds is provided -- it is only used as a fallback for
    # embed_tokens() when inputs_embeds is None, but the parameter itself
    # has no default in LanguageModel.__call__.
    logits = model.language_model(ids, inputs_embeds=embeds).logits
    save(out, "prefill_logits", x=logits[:, -1, :])

    # 贪心 100 步 -- drive generate_step directly with the exact same
    # input_ids/pixel_values/spatial_crop/seq_mask used above, and record
    # the raw sampled token ids (no text decode/re-encode round trip).
    gen = generate_step(
        ids,
        model,
        [pix_p, pix_g],
        None,
        images_spatial_crop=crop,
        images_seq_mask=seq_mask,
        max_tokens=GREEDY_STEPS,
        temperature=0.0,
    )
    greedy_ids = []
    for tok, _logprobs in gen:
        greedy_ids.append(int(tok.item()) if hasattr(tok, "item") else int(tok))
    save(out, "greedy_tokens", ids=mx.array(greedy_ids, dtype=mx.int32))
    text = processor.tokenizer.decode(greedy_ids, skip_special_tokens=True)

    meta = {
        "prompt_token_ids": [int(t) for t in ids.flatten().tolist()],
        "images_spatial_crop": (crop.tolist() if crop is not None else None),
        "num_patches": n_patch,
        "text": text,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(name, "prefill_logits", logits[:, -1, :].shape, "greedy[:12]", greedy_ids[:12])

print("fixtures done")
