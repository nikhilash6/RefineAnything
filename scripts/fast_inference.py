"""
RefineAnything – Fast Inference Script
======================================

Minimal inputs:
    --input   Source image to refine
    --mask    Binary mask (white = region to refine)
    --prompt  What to refine, e.g. "Refine the LOGO."
    --ref     (optional) Reference image for style/content guidance

Usage:
    python scripts/fast_inference.py \
        --input  output_image_edit_plus_gui_input.png \
        --mask   output_image_edit_plus_gui_mask.png \
        --ref    output_image_edit_plus_gui_ref.png \
        --prompt "Refine the LOGO."
"""

import argparse
import io
import math
import os

import numpy as np
import torch
from PIL import Image, ImageCms, ImageFilter


def normalize_to_srgb(img: Image.Image) -> Image.Image:
    """Convert a PIL image to sRGB, applying the embedded ICC profile when present.

    Keeps the image in the same color space the model was trained on and
    prevents off-gamut shifts from images tagged with e.g. Display P3.
    """
    if img is None:
        return img
    icc = img.info.get("icc_profile") if hasattr(img, "info") else None
    if icc:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst_profile = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(
                img,
                src_profile,
                dst_profile,
                outputMode="RGB",
            )
        except Exception:
            img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════

def bbox_from_mask(mask_l: Image.Image) -> tuple[int, int, int, int]:
    """Return tight (x1, y1, x2, y2) bounding box of non-zero pixels."""
    arr = np.array(mask_l, dtype=np.uint8)
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        raise ValueError("Mask is empty — nothing to refine.")
    w, h = mask_l.size
    return (max(0, int(xs.min())),
            max(0, int(ys.min())),
            min(w, int(xs.max()) + 1),
            min(h, int(ys.max()) + 1))


def focus_crop(
    image: Image.Image,
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int],
    margin: int = 64,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """
    Crop around *bbox* so the diffusion model works on a ~1024² region.

    Returns (cropped_image, cropped_mask, crop_box).
    """
    iw, ih = image.size
    s = math.sqrt(1024 * 1024 / float(iw * ih))
    x1, y1, x2, y2 = bbox

    cx1 = max(0, int(math.floor(max(0.0, x1 * s - margin) / s)))
    cy1 = max(0, int(math.floor(max(0.0, y1 * s - margin) / s)))
    cx2 = min(iw, int(math.ceil(min(iw * s, x2 * s + margin) / s)))
    cy2 = min(ih, int(math.ceil(min(ih * s, y2 * s + margin) / s)))

    crop_box = (cx1, cy1, cx2, cy2)
    return image.crop(crop_box), mask_l.crop(crop_box), crop_box


def binarise_mask_to_rgb(mask_l: Image.Image) -> Image.Image:
    """Convert an L-mode mask to a clean binary RGB image (spatial condition for the model)."""
    arr = np.where(np.array(mask_l, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

_HF_LORA_REPO = "limuloo1999/RefineAnything"
_HF_LORA_FILENAME = "Qwen-Image-Edit-2511-RefineAny.safetensors"
_HF_LORA_ADAPTER = "refine_anything"

_HF_LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
_HF_LIGHTNING_FILENAME = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
_HF_LIGHTNING_ADAPTER = "lightning"


def build_pipeline(model_dir: str, device: str = "cuda", load_lightning_lora: bool = True):
    """Load QwenImageEditPlusPipeline + RefineAnything LoRA, ready for inference.

    When *load_lightning_lora* is True (default), also loads the Lightning
    distillation LoRA so inference works well at ~8 steps.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from huggingface_hub import hf_hub_download

    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    })

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, scheduler=scheduler,
    )
    pipe.to(device)

    lora_path = hf_hub_download(repo_id=_HF_LORA_REPO, filename=_HF_LORA_FILENAME)
    pipe.load_lora_weights(
        os.path.dirname(lora_path),
        weight_name=os.path.basename(lora_path),
        adapter_name=_HF_LORA_ADAPTER,
    )

    adapter_names = [_HF_LORA_ADAPTER]
    adapter_weights = [1.0]

    if load_lightning_lora:
        lightning_path = hf_hub_download(
            repo_id=_HF_LIGHTNING_REPO, filename=_HF_LIGHTNING_FILENAME,
        )
        pipe.load_lora_weights(
            os.path.dirname(lightning_path),
            weight_name=os.path.basename(lightning_path),
            adapter_name=_HF_LIGHTNING_ADAPTER,
        )
        adapter_names.append(_HF_LIGHTNING_ADAPTER)
        adapter_weights.append(1.0)

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    return pipe


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(
    pipe,
    input_image: Image.Image,
    mask_l: Image.Image,
    prompt: str,
    ref_image: Image.Image | None = None,
    *,
    seed: int = 0,
    num_steps: int = 8,
    true_cfg_scale: float = 1.0,
    guidance_scale: float = 1.0,
    negative_prompt: str = " ",
    device: str = "cuda",
) -> Image.Image:
    """
    Single inference call.

    Assembles the image list the model expects:
        [input_image, (ref_image), spatial_mask_rgb]
    then runs the diffusion pipeline.
    """
    images: list[Image.Image] = [input_image]
    if ref_image is not None:
        images.append(ref_image)
    images.append(binarise_mask_to_rgb(mask_l))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    with torch.inference_mode():
        output = pipe(
            image=images,
            prompt=prompt,
            generator=gen,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )
    return output.images[0]


# ═══════════════════════════════════════════════════════════════════════════
# Paste-back
# ═══════════════════════════════════════════════════════════════════════════

def paste_back(
    original: Image.Image,
    generated: Image.Image,
    mask_l: Image.Image,
    crop_box: tuple[int, int, int, int] | None = None,
    mask_grow: int = 3,
    blend_blur: int = 5,
) -> Image.Image:
    """Blend *generated* back into *original* through a smoothed mask."""
    m = mask_l.convert("L")
    if mask_grow > 0:
        m = m.filter(ImageFilter.MaxFilter(size=2 * mask_grow + 1))
    if blend_blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(blend_blur)))

    target = original.crop(crop_box) if crop_box else original
    dst = np.asarray(target.convert("RGB")).astype(np.float32)
    src = np.asarray(
        generated.convert("RGB").resize(target.size, Image.BICUBIC)
    ).astype(np.float32)
    alpha = np.asarray(m.resize(target.size, Image.BILINEAR)).astype(np.float32) / 255.0

    blended = src * alpha[:, :, None] + dst * (1.0 - alpha[:, :, None])
    composited = Image.fromarray(
        np.clip(blended + 0.5, 0, 255).astype(np.uint8), mode="RGB"
    )

    if crop_box:
        result = original.copy()
        result.paste(composited, (crop_box[0], crop_box[1]))
        return result
    return composited


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end
# ═══════════════════════════════════════════════════════════════════════════

def refine(
    input_path: str,
    mask_path: str,
    prompt: str,
    model_dir: str,
    ref_path: str | None = None,
    output_path: str = "output_refine_result.png",
    device: str = "cuda",
    seed: int = 0,
    num_steps: int = 8,
    true_cfg_scale: float | None = None,
    guidance_scale: float = 1.0,
    negative_prompt: str = " ",
    do_focus_crop: bool = True,
    focus_crop_margin: int = 64,
    mask_grow: int = 3,
    blend_blur: int = 5,
    load_lightning_lora: bool = True,
) -> Image.Image:
    """
    Full RefineAnything pipeline:
        1. Load inputs  →  2. Focus crop  →  3. Inference  →  4. Paste back  →  5. Save

    ``true_cfg_scale`` defaults to 1.0 when the Lightning distillation LoRA
    is loaded (Lightning bakes CFG into the weights, so running CFG>1 on top
    of it stacks the guidance twice and destroys quality). Pass an explicit
    value to override.
    """
    if true_cfg_scale is None:
        true_cfg_scale = 1.0 if load_lightning_lora else 4.0
        print(
            f"[fast_inference] auto true_cfg_scale={true_cfg_scale} "
            f"(lightning_lora={'on' if load_lightning_lora else 'off'})"
        )

    # 1. Load inputs
    input_image = normalize_to_srgb(Image.open(input_path))
    mask_l = Image.open(mask_path).convert("L")
    if mask_l.size != input_image.size:
        mask_l = mask_l.resize(input_image.size, Image.NEAREST)
    bbox = bbox_from_mask(mask_l)

    ref_image = None
    if ref_path and os.path.isfile(ref_path):
        ref_image = normalize_to_srgb(Image.open(ref_path))

    # 2. Focus crop
    crop_box = None
    model_image, model_mask = input_image, mask_l
    if do_focus_crop:
        model_image, model_mask, crop_box = focus_crop(
            input_image, mask_l, bbox, margin=focus_crop_margin,
        )

    # 3. Inference
    pipe = build_pipeline(model_dir, device=device, load_lightning_lora=load_lightning_lora)
    generated = run_inference(
        pipe, model_image, model_mask, prompt, ref_image,
        seed=seed, num_steps=num_steps, true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt,
        device=device,
    )

    # 4. Paste back
    result = paste_back(
        input_image, generated, model_mask,
        crop_box=crop_box, mask_grow=mask_grow, blend_blur=blend_blur,
    )

    # 5. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    result.save(output_path)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RefineAnything – fast inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Source image path")
    p.add_argument("--mask", required=True, help="Mask image path (white = edit region)")
    p.add_argument("--prompt", required=True, help="Refine prompt")
    p.add_argument("--ref", default=None, help="Reference image path (optional)")
    p.add_argument("--output", default="output_refine_result.png")
    p.add_argument("--model_dir", default="Qwen/Qwen-Image-Edit-2511")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument(
        "--true_cfg_scale",
        type=float,
        default=None,
        help="True CFG scale. Defaults to 1.0 when Lightning LoRA is enabled "
             "(required by Lightning distillation) and 4.0 otherwise.",
    )
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--no_focus_crop", action="store_true")
    p.add_argument("--focus_crop_margin", type=int, default=64)
    p.add_argument("--mask_grow", type=int, default=3)
    p.add_argument("--blend_blur", type=int, default=5)
    p.add_argument(
        "--no_lightning_lora",
        action="store_true",
        help="Disable Lightning LoRA (enabled by default for ~8-step inference)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    refine(
        input_path=args.input,
        mask_path=args.mask,
        prompt=args.prompt,
        model_dir=args.model_dir,
        ref_path=args.ref,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
        num_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        do_focus_crop=not args.no_focus_crop,
        focus_crop_margin=args.focus_crop_margin,
        mask_grow=args.mask_grow,
        blend_blur=args.blend_blur,
        load_lightning_lora=not args.no_lightning_lora,
    )


if __name__ == "__main__":
    main()
