import math
import os
import threading
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download

def calculate_dimensions(target_area: int, ratio: float):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height), None


def vit_resize_dims(src_w: int, src_h: int, vit_resize_size: int = 384) -> tuple[int, int]:
    ratio = float(src_w) / float(src_h) if src_h else 1.0
    new_w, new_h, _ = calculate_dimensions(vit_resize_size * vit_resize_size, ratio)
    return new_w, new_h


def scale_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    sx = float(dst_w) / float(src_w) if src_w else 1.0
    sy = float(dst_h) / float(src_h) if src_h else 1.0
    x1, y1, x2, y2 = bbox_xyxy
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def format_bbox_xyxy(bbox_xyxy: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    return f"[{x1}, {y1}, {x2}, {y2}]"


def draw_bbox_on_image(image, bbox_xyxy: tuple[int, int, int, int]):
    from PIL import ImageDraw

    x1, y1, x2, y2 = bbox_xyxy
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    w = max(2, int(round(min(vis.size) * 0.006)))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=w)
    return vis


def draw_points_on_image(image, points: list[tuple[int, int]], *, connect: bool = False):
    from PIL import ImageDraw

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    w, h = vis.size
    r = max(2, int(round(min(w, h) * 0.004)))
    if connect and len(points) >= 2:
        draw.line(points + [points[0]], fill=(255, 64, 64), width=max(1, r // 2))
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(64, 255, 64), outline=(0, 0, 0))
    return vis


_HF_LORA_REPO = "limuloo1999/RefineAnything"
_HF_LORA_FILENAME = "Qwen-Image-Edit-2511-RefineAny.safetensors"
_HF_LORA_ADAPTER = "refine_anything"

_LIGHTNING_LOADED = False
_PIPELINE_LOCK = threading.Lock()


def _build_pipeline(model_dir: str):
    """Build the pipeline at module level. ZeroGPU intercepts .to('cuda')
    and keeps the model on CPU until a @spaces.GPU function runs."""
    scheduler_config = {
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
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=None)

    local_path = hf_hub_download(
        repo_id=_HF_LORA_REPO,
        filename=_HF_LORA_FILENAME,
    )
    lora_dir = os.path.dirname(local_path)
    weight_name = os.path.basename(local_path)
    pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=_HF_LORA_ADAPTER)

    pipe.to("cuda")
    return pipe


_DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen/Qwen-Image-Edit-2511")
print(f"[startup] Loading pipeline from {_DEFAULT_MODEL_DIR} ...")
_PIPELINE = _build_pipeline(_DEFAULT_MODEL_DIR)
print("[startup] Pipeline ready.")


def _get_pipeline(load_lightning_lora: bool):
    global _LIGHTNING_LOADED

    with _PIPELINE_LOCK:
        if load_lightning_lora and not _LIGHTNING_LOADED:
            lightning_path = hf_hub_download(
                repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
                filename="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
            )
            lightning_dir = os.path.dirname(lightning_path)
            lightning_weight = os.path.basename(lightning_path)
            _PIPELINE.load_lora_weights(lightning_dir, weight_name=lightning_weight, adapter_name="lightning")
            _LIGHTNING_LOADED = True

        adapter_names: list[str] = [_HF_LORA_ADAPTER]
        adapter_weights: list[float] = [1.0]
        if _LIGHTNING_LOADED:
            adapter_names.append("lightning")
            adapter_weights.append(1.0 if load_lightning_lora else 0.0)

        if hasattr(_PIPELINE, "set_adapters"):
            try:
                _PIPELINE.set_adapters(adapter_names, adapter_weights=adapter_weights)
            except TypeError:
                _PIPELINE.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))

        return _PIPELINE


def build_app():
    import base64
    import gradio as gr
    import inspect
    import io
    import numpy as np
    import random
    import re
    from PIL import Image

    def _to_float01_rgb(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        return arr

    def _to_float01_mask(mask_img: Image.Image) -> np.ndarray:
        arr = np.asarray(mask_img.convert("L")).astype(np.float32) / 255.0
        return arr

    def composite_masked(
        *,
        destination: Image.Image,
        source: Image.Image,
        mask: Image.Image,
        resize_source: bool = True,
    ) -> Image.Image:
        dst = destination.convert("RGB")
        if resize_source and getattr(source, "size", None) != dst.size:
            src = source.convert("RGB").resize(dst.size, resample=Image.BICUBIC)
        else:
            src = source.convert("RGB")

        m = mask.convert("L")
        if getattr(m, "size", None) != dst.size:
            m = m.resize(dst.size, resample=Image.BILINEAR)

        dst_f = _to_float01_rgb(dst)
        src_f = _to_float01_rgb(src)
        m_f = _to_float01_mask(m)[:, :, None]
        out = src_f * m_f + dst_f * (1.0 - m_f)
        out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def prepare_paste_mask(
        mask_l: Image.Image,
        *,
        mask_grow: int = 0,
        blend_kernel: int = 0,
    ) -> Image.Image:
        from PIL import ImageFilter

        m = mask_l.convert("L")
        if mask_grow and int(mask_grow) > 0:
            k = 2 * int(mask_grow) + 1
            m = m.filter(ImageFilter.MaxFilter(size=k))
        if blend_kernel and int(blend_kernel) > 0:
            m = m.filter(ImageFilter.GaussianBlur(radius=float(blend_kernel)))
        return m

    def make_bbox_mask(
        *,
        size: tuple[int, int],
        bbox_xyxy: tuple[int, int, int, int],
        mask_grow: int = 0,
        blend_kernel: int = 0,
    ) -> Image.Image:
        from PIL import ImageDraw, ImageFilter

        w, h = size
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(1, min(w, int(x2)))
        y2 = max(1, min(h, int(y2)))

        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), fill=255)

        if mask_grow and int(mask_grow) > 0:
            k = 2 * int(mask_grow) + 1
            m = m.filter(ImageFilter.MaxFilter(size=k))

        if blend_kernel and int(blend_kernel) > 0:
            m = m.filter(ImageFilter.GaussianBlur(radius=float(blend_kernel)))

        return m

    def compute_crop_box_xyxy(
        *,
        image_size: tuple[int, int],
        bbox_xyxy: tuple[int, int, int, int],
        margin: int,
    ) -> tuple[int, int, int, int]:
        w, h = image_size
        x1, y1, x2, y2 = bbox_xyxy
        m = max(0, int(margin))
        cx1 = max(0, min(w - 1, int(x1) - m))
        cy1 = max(0, min(h - 1, int(y1) - m))
        cx2 = max(1, min(w, int(x2) + m))
        cy2 = max(1, min(h, int(y2) + m))
        if cx2 <= cx1:
            cx2 = min(w, cx1 + 1)
        if cy2 <= cy1:
            cy2 = min(h, cy1 + 1)
        return (cx1, cy1, cx2, cy2)

    def crop_box_from_1024_area_margin(
        *,
        image_size: tuple[int, int],
        bbox_xyxy: tuple[int, int, int, int],
        margin: int,
    ) -> tuple[int, int, int, int]:
        iw, ih = image_size
        if iw <= 0 or ih <= 0:
            return compute_crop_box_xyxy(image_size=image_size, bbox_xyxy=bbox_xyxy, margin=margin)
        s = math.sqrt(1024 * 1024 / float(iw * ih))
        vw, vh = float(iw) * s, float(ih) * s
        x1, y1, x2, y2 = bbox_xyxy
        vx1 = max(0.0, min(vw - 1.0, float(x1) * s - float(margin)))
        vy1 = max(0.0, min(vh - 1.0, float(y1) * s - float(margin)))
        vx2 = max(1.0, min(vw, float(x2) * s + float(margin)))
        vy2 = max(1.0, min(vh, float(y2) * s + float(margin)))
        if vx2 <= vx1:
            vx2 = min(vw, vx1 + 1.0)
        if vy2 <= vy1:
            vy2 = min(vh, vy1 + 1.0)
        cx1 = max(0, min(iw - 1, int(math.floor(vx1 / s))))
        cy1 = max(0, min(ih - 1, int(math.floor(vy1 / s))))
        cx2 = max(1, min(iw, int(math.ceil(vx2 / s))))
        cy2 = max(1, min(ih, int(math.ceil(vy2 / s))))
        if cx2 <= cx1:
            cx2 = min(iw, cx1 + 1)
        if cy2 <= cy1:
            cy2 = min(ih, cy1 + 1)
        return (cx1, cy1, cx2, cy2)

    def offset_bbox_xyxy(bbox_xyxy: tuple[int, int, int, int], dx: int, dy: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        return (int(x1) - int(dx), int(y1) - int(dy), int(x2) - int(dx), int(y2) - int(dy))

    def _decode_data_url(x):
        if not isinstance(x, str):
            return None
        s = x
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1]
        try:
            data = base64.b64decode(s)
        except Exception:
            return None
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    def _to_rgb_pil(x, *, label: str):
        if x is None:
            return None
        if isinstance(x, str):
            x2 = _decode_data_url(x)
            if x2 is None:
                raise gr.Error(f"{label} 数据格式不支持")
            x = x2
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x.astype(np.uint8))
        if not hasattr(x, "convert"):
            raise gr.Error(f"{label} 数据格式不支持")
        try:
            return x.convert("RGB")
        except Exception as e:
            raise gr.Error(f"{label} 转换 RGB 失败: {type(e).__name__}: {e}")

    def mask_to_points_sample_list(mask_img: Image.Image, *, num_points: int = 64, seed: int = 0) -> tuple[str, list[tuple[int, int]]]:
        arr = np.array(mask_img.convert("L"), dtype=np.uint8)
        if arr.max() <= 1:
            mask = arr.astype(bool)
        else:
            mask = arr > 0
        ys, xs = np.where(mask)
        if xs.size == 0:
            raise gr.Error("mask 为空，无法从中采样点")
        rng = random.Random(int(seed))
        idxs = list(range(int(xs.size)))
        rng.shuffle(idxs)
        idxs = idxs[: int(num_points)]
        pts = [(int(xs[i]), int(ys[i])) for i in idxs]
        s = "[" + ", ".join(f"({int(x)},{int(y)})" for (x, y) in pts) + "]"
        return s, pts

    def strip_special_region(prompt: str) -> str:
        p = (prompt or "").replace("<SPECIAL_REGION>", " ")
        p = p.replace("\n", " ")
        p = re.sub(r"\s{2,}", " ", p).strip()
        return p

    def strip_location_text(prompt: str) -> str:
        p = strip_special_region(prompt)
        p = re.sub(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", "", p)
        p = re.sub(r"\s{2,}", " ", p).strip()
        return p

    def mask_has_foreground(mask_l: Image.Image) -> bool:
        arr = np.array(mask_l.convert("L"), dtype=np.uint8)
        return bool(arr.max() > 0)

    def mask_bbox_xyxy(mask_img_l: Image.Image) -> tuple[int, int, int, int] | None:
        arr = np.array(mask_img_l.convert("L"), dtype=np.uint8)
        ys, xs = np.where(arr > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        w, h = mask_img_l.size
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def render_spatial_prompt(mask_img_l: Image.Image, *, source: str, bbox_margin: int = 0) -> Image.Image | None:
        src = (source or "mask").strip().lower()
        if src == "bbox":
            bbox = mask_bbox_xyxy(mask_img_l)
            if bbox is None:
                return None
            w, h = mask_img_l.size
            out = Image.new("L", (w, h), 0)
            x1, y1, x2, y2 = bbox
            m = max(0, int(bbox_margin))
            x1 = max(0, x1 - m)
            y1 = max(0, y1 - m)
            x2 = min(w, x2 + m)
            y2 = min(h, y2 + m)
            from PIL import ImageDraw

            draw = ImageDraw.Draw(out)
            draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), fill=255)
            return out
        arr = np.array(mask_img_l.convert("L"), dtype=np.uint8)
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def overlay_mask_on_image(image_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        base = image_rgb.convert("RGB")
        m = mask_l.convert("L")
        if getattr(m, "size", None) != base.size:
            m = m.resize(base.size, resample=Image.NEAREST)
        base_f = np.asarray(base).astype(np.float32)
        mf = (np.asarray(m).astype(np.float32) > 0)[:, :, None].astype(np.float32)
        color = np.array([64.0, 255.0, 64.0], dtype=np.float32)[None, None, :]
        alpha = 0.35
        out = base_f * (1.0 - alpha * mf) + color * (alpha * mf)
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    def extract_bbox_from_image1(image1_value):
        if image1_value is None:
            raise gr.Error("image1 必须上传")
        if not isinstance(image1_value, dict):
            raise gr.Error("image1 数据格式不支持")

        if "image" in image1_value and "mask" in image1_value:
            img = image1_value["image"]
            mask = image1_value["mask"]
            if img is None:
                raise gr.Error("image1 必须上传")
        elif "background" in image1_value and "layers" in image1_value:
            img = image1_value.get("background") or image1_value.get("composite")
            layers = image1_value.get("layers") or []
            if img is None:
                raise gr.Error("image1 数据缺少 background/composite")
            mask = layers if layers else None
        else:
            raise gr.Error("请在 image1 上涂抹选择区域")

        if isinstance(img, str):
            img2 = _decode_data_url(img)
            if img2 is None:
                raise gr.Error("image1 数据格式不支持（image）")
            img = img2
        if isinstance(mask, str):
            mask2 = _decode_data_url(mask)
            if mask2 is None:
                raise gr.Error("image1 数据格式不支持（mask）")
            mask = mask2

        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8))
        else:
            img_pil = img

        if hasattr(img_pil, "convert"):
            img_pil = img_pil.convert("RGB")

        iw, ih = img_pil.size
        vit_w, vit_h = vit_resize_dims(iw, ih, vit_resize_size=384)

        if mask is None:
            return img_pil, None, None, None, (vit_w, vit_h)

        if isinstance(mask, list):
            mask_arr = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8)
            for layer in mask:
                if isinstance(layer, str):
                    layer2 = _decode_data_url(layer)
                    if layer2 is None:
                        continue
                    layer = layer2
                if isinstance(layer, np.ndarray):
                    layer_pil = Image.fromarray(layer.astype(np.uint8))
                else:
                    layer_pil = layer
                if layer_pil is None:
                    continue
                if getattr(layer_pil, "size", None) != img_pil.size:
                    layer_pil = layer_pil.resize(img_pil.size)
                layer_arr = np.array(layer_pil, dtype=np.uint8)
                if layer_arr.ndim == 3 and layer_arr.shape[2] >= 4:
                    layer_mask = layer_arr[:, :, 3]
                elif layer_arr.ndim == 3:
                    layer_mask = layer_arr.max(axis=2)
                else:
                    layer_mask = layer_arr
                mask_arr = np.maximum(mask_arr, layer_mask.astype(np.uint8))
        elif isinstance(mask, np.ndarray):
            mask_arr = mask.astype(np.uint8)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.max(axis=2)
            mask_pil_l = Image.fromarray(mask_arr, mode="L")
            if getattr(mask_pil_l, "size", None) != img_pil.size:
                mask_pil_l = mask_pil_l.resize(img_pil.size, resample=Image.NEAREST)
                mask_arr = np.array(mask_pil_l, dtype=np.uint8)
        else:
            mask_pil_l = mask.convert("L")
            if getattr(mask_pil_l, "size", None) != img_pil.size:
                mask_pil_l = mask_pil_l.resize(img_pil.size, resample=Image.NEAREST)
            mask_arr = np.array(mask_pil_l, dtype=np.uint8)
        if isinstance(mask, list):
            mask_pil_l = Image.fromarray(mask_arr, mode="L")

        ys, xs = np.where(mask_arr > 0)
        if xs.size == 0 or ys.size == 0:
            return img_pil, None, None, None, (vit_w, vit_h)

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1

        x1 = max(0, min(iw - 1, x1))
        y1 = max(0, min(ih - 1, y1))
        x2 = max(1, min(iw, x2))
        y2 = max(1, min(ih, y2))

        bbox_raw = (x1, y1, x2, y2)
        bbox_vit = scale_bbox_xyxy(bbox_raw, iw, ih, vit_w, vit_h)
        return img_pil, mask_pil_l, bbox_raw, bbox_vit, (vit_w, vit_h)

    def extract_ref_from_image2(image2_value):
        """Return (ref_pil_rgb | None, crop_info_str | None).

        If the user painted on image2, crop to the brush bounding-box and
        return only that region.  Otherwise return the full image.
        """
        if image2_value is None:
            return None, None

        if not isinstance(image2_value, dict):
            return _to_rgb_pil(image2_value, label="image2"), None

        if "image" in image2_value and "mask" in image2_value:
            img = image2_value["image"]
            mask = image2_value["mask"]
        elif "background" in image2_value and "layers" in image2_value:
            img = image2_value.get("background") or image2_value.get("composite")
            layers = image2_value.get("layers") or []
            mask = layers if layers else None
        else:
            img = image2_value
            mask = None

        if img is None:
            return None, None

        if isinstance(img, str):
            img2 = _decode_data_url(img)
            if img2 is None:
                return None, None
            img = img2
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8))
        else:
            img_pil = img
        img_pil = img_pil.convert("RGB")

        if mask is None:
            return img_pil, None

        if isinstance(mask, list):
            mask_arr = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8)
            for layer in mask:
                if isinstance(layer, str):
                    layer2 = _decode_data_url(layer)
                    if layer2 is None:
                        continue
                    layer = layer2
                if isinstance(layer, np.ndarray):
                    layer_pil = Image.fromarray(layer.astype(np.uint8))
                else:
                    layer_pil = layer
                if layer_pil is None:
                    continue
                if getattr(layer_pil, "size", None) != img_pil.size:
                    layer_pil = layer_pil.resize(img_pil.size)
                layer_arr = np.array(layer_pil, dtype=np.uint8)
                if layer_arr.ndim == 3 and layer_arr.shape[2] >= 4:
                    layer_mask = layer_arr[:, :, 3]
                elif layer_arr.ndim == 3:
                    layer_mask = layer_arr.max(axis=2)
                else:
                    layer_mask = layer_arr
                mask_arr = np.maximum(mask_arr, layer_mask.astype(np.uint8))
        elif isinstance(mask, np.ndarray):
            mask_arr = mask.astype(np.uint8)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.max(axis=2)
            tmp = Image.fromarray(mask_arr, mode="L")
            if getattr(tmp, "size", None) != img_pil.size:
                tmp = tmp.resize(img_pil.size, resample=Image.NEAREST)
                mask_arr = np.array(tmp, dtype=np.uint8)
        else:
            tmp = mask.convert("L")
            if getattr(tmp, "size", None) != img_pil.size:
                tmp = tmp.resize(img_pil.size, resample=Image.NEAREST)
            mask_arr = np.array(tmp, dtype=np.uint8)

        ys, xs = np.where(mask_arr > 0)
        if xs.size == 0 or ys.size == 0:
            return img_pil, None

        iw, ih = img_pil.size
        x1 = max(0, min(iw - 1, int(xs.min())))
        y1 = max(0, min(ih - 1, int(ys.min())))
        x2 = max(1, min(iw, int(xs.max()) + 1))
        y2 = max(1, min(ih, int(ys.max()) + 1))

        cropped = img_pil.crop((x1, y1, x2, y2))
        crop_info = f"ref_crop=[{x1},{y1},{x2},{y2}] ({x2 - x1}x{y2 - y1})"
        return cropped, crop_info

    def _predict_impl(
        image1_value,
        image2,
        prompt,
        mode,
        spatial_source,
        seed,
        steps,
        true_cfg_scale,
        guidance_scale,
        load_lightning_lora,
        paste_back_bbox,
        paste_back_mode,
        focus_crop_for_bbox,
        focus_crop_margin,
        paste_mask_grow,
        paste_blend_kernel,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("prompt 为空")

        img_pil, mask_pil_l, bbox_raw, bbox_vit, (vit_w, vit_h) = extract_bbox_from_image1(image1_value)
        img_pil = _to_rgb_pil(img_pil, label="image1")
        image2, ref_crop_info = extract_ref_from_image2(image2)

        has_mask = (mask_pil_l is not None) and mask_has_foreground(mask_pil_l)
        has_bbox = bbox_raw is not None

        use_focus_crop = bool(paste_back_bbox) and bool(focus_crop_for_bbox) and has_bbox
        crop_xyxy = None
        bbox_for_model_raw = bbox_raw
        img_for_model = img_pil
        image2_for_model = image2
        mask_for_model_l = mask_pil_l if has_mask else None
        vit_wh_for_prompt = (vit_w, vit_h)

        if use_focus_crop:
            iw, ih = img_pil.size
            margin = int(focus_crop_margin) if focus_crop_margin is not None and str(focus_crop_margin).strip() else 0
            crop_xyxy = crop_box_from_1024_area_margin(image_size=(iw, ih), bbox_xyxy=bbox_raw, margin=margin)
            cx1, cy1, cx2, cy2 = crop_xyxy
            img_for_model = img_pil.crop((cx1, cy1, cx2, cy2))
            bbox_for_model_raw = offset_bbox_xyxy(bbox_raw, cx1, cy1)
            if has_mask and mask_pil_l is not None:
                mask_for_model_l = mask_pil_l.crop((cx1, cy1, cx2, cy2))
            vit_w2, vit_h2 = vit_resize_dims(img_for_model.size[0], img_for_model.size[1], vit_resize_size=384)
            vit_wh_for_prompt = (vit_w2, vit_h2)
            bbox_vit = scale_bbox_xyxy(bbox_for_model_raw, img_for_model.size[0], img_for_model.size[1], vit_w2, vit_h2)

        prompt_for_model = strip_location_text(prompt)

        spatial_source = (spatial_source or "mask").strip().lower()
        spatial_mask_l = None
        if mask_for_model_l is not None and mask_has_foreground(mask_for_model_l):
            spatial_mask_l = render_spatial_prompt(mask_for_model_l, source=spatial_source, bbox_margin=0)

        info = ""
        if has_bbox:
            info = f"BBox(raw)={format_bbox_xyxy(bbox_raw)}"
        else:
            info = "未检测到涂抹区域"
        if has_bbox:
            info += f" -> QwenVit(384-area)={format_bbox_xyxy(bbox_vit)} vit_wh=({vit_wh_for_prompt[0]},{vit_wh_for_prompt[1]})"
        if ref_crop_info:
            info += f" {ref_crop_info}"
        if spatial_mask_l is not None:
            info += f" spatial={spatial_source}"
        if crop_xyxy is not None:
            info += f" crop={format_bbox_xyxy(crop_xyxy)} bbox_in_crop={format_bbox_xyxy(bbox_for_model_raw)}"

        vis_base = img_for_model.resize(vit_wh_for_prompt, resample=Image.BICUBIC)
        if spatial_mask_l is not None:
            spatial_vis = spatial_mask_l.resize(vit_wh_for_prompt, resample=Image.NEAREST)
            vis = overlay_mask_on_image(vis_base, spatial_vis)
        elif has_bbox:
            vis = draw_bbox_on_image(vis_base, bbox_vit)
        else:
            vis = vis_base

        if mode == "Preview only":
            return (img_pil, img_pil), prompt_for_model, vis, "Done"

        seed = int(seed) if seed is not None and str(seed).strip() else 0
        steps = int(steps) if steps is not None and str(steps).strip() else 8
        true_cfg_scale = float(true_cfg_scale) if true_cfg_scale is not None and str(true_cfg_scale).strip() else 4.0
        guidance_scale = float(guidance_scale) if guidance_scale is not None and str(guidance_scale).strip() else 1.0

        pipe = _get_pipeline(load_lightning_lora=bool(load_lightning_lora))

        img = img_for_model if image2_for_model is None else [img_for_model, image2_for_model]
        if spatial_mask_l is not None:
            spatial_rgb = spatial_mask_l.convert("RGB")
            if isinstance(img, list):
                img = img + [spatial_rgb]
            else:
                img = [img, spatial_rgb]
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        t0 = time.time()
        with torch.inference_mode():
            try:
                out = pipe(
                    image=img,
                    prompt=prompt_for_model,
                    generator=gen,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                )
            except Exception as e:
                raise gr.Error(f"Inference failed: {type(e).__name__}: {e}")
        dt = time.time() - t0
        out_img = out.images[0]

        if paste_back_bbox:
            paste_back_mode = (paste_back_mode or "bbox").strip().lower()
            mg = int(paste_mask_grow) if paste_mask_grow is not None and str(paste_mask_grow).strip() else 0
            bk = int(paste_blend_kernel) if paste_blend_kernel is not None and str(paste_blend_kernel).strip() else 0
            paste_mask = None
            if paste_back_mode.startswith("mask") and mask_for_model_l is not None and mask_has_foreground(mask_for_model_l):
                paste_mask = prepare_paste_mask(mask_for_model_l, mask_grow=mg, blend_kernel=bk)
            elif bbox_for_model_raw is not None:
                paste_mask = make_bbox_mask(size=img_for_model.size, bbox_xyxy=bbox_for_model_raw, mask_grow=mg, blend_kernel=bk)

            if paste_mask is not None:
                out_img_crop = composite_masked(destination=img_for_model, source=out_img, mask=paste_mask, resize_source=True)
                if crop_xyxy is not None:
                    cx1, cy1, cx2, cy2 = crop_xyxy
                    out_full = img_pil.copy()
                    out_full.paste(out_img_crop, (cx1, cy1))
                    out_img = out_full
                else:
                    out_img = out_img_crop

        status = f"Done ({dt:.2f}s)"
        return (img_pil, out_img), prompt_for_model, vis, status

    predict = _predict_impl

    _DESCRIPTION_EN = """\
**RefineAnything** refines local regions of an image guided by a text prompt. \
Upload a source image, **brush over the area** you want to edit, and describe the desired change. \
Optionally upload a reference image for style/content guidance — \
leave it as-is to reference the whole image, or **brush on it** to specify exactly which region to reference.\
"""

    _DESCRIPTION_CN = """\
**RefineAnything** 根据文字提示精修图片的局部区域。\
上传一张源图，**用画笔涂抹**需要编辑的区域，再输入想要的修改描述即可。\
可选上传第二张参考图来引导风格/内容——不涂抹则参考整张图，**涂抹则精确指定参考区域**。\
"""

    _NOTE_EN = (
        "For refinement tasks, prompts starting with **refine** usually work better. "
        "The model also shows some grounding edit ability (for example **add**, **remove**, **modify**) "
        "even without dedicated grounding training."
    )
    _NOTE_CN = (
        "做 refine 任务时，prompt 以 **refine** 开头通常效果更好。"
        "此外模型也具备一定 grounding edit 能力（如 **add**、**remove**、**modify**），"
        "虽然我们没有使用专门的 grounding 数据训练。"
    )

    def _randomize_seed():
        return random.randint(0, 2**31 - 1)

    with gr.Blocks(title="RefineAnything", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RefineAnything")
        gr.Markdown(_DESCRIPTION_EN)
        gr.Markdown(_DESCRIPTION_CN)
        gr.Markdown(_NOTE_EN)
        gr.Markdown(_NOTE_CN)

        with gr.Row():
            with gr.Column():
                if hasattr(gr, "ImageMask"):
                    image1 = gr.ImageMask(label="Source image (brush to select region)", type="pil")
                else:
                    image1 = gr.Image(label="Source image", type="pil")
            with gr.Column():
                if hasattr(gr, "ImageMask"):
                    image2 = gr.ImageMask(label="Reference image (optional, brush to crop)", type="pil")
                else:
                    image2 = gr.Image(label="Reference image (optional)", type="pil")

        prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Describe the edit you want...")

        with gr.Row():
            mode = gr.Radio(["Run inference", "Preview only"], value="Run inference", label="Mode", scale=2)
            seed = gr.Number(label="Seed", value=0, precision=0, scale=1)
            seed_dice = gr.Button("🎲 Random", scale=0, min_width=110)
            steps = gr.Number(label="Steps", value=8, precision=0, scale=1)

        with gr.Row():
            spatial_source = gr.Radio(["mask", "bbox"], value="mask", label="Spatial prompt source", scale=2)
            load_lightning_lora = gr.Checkbox(label="Lightning LoRA (faster)", value=True, scale=1)
        with gr.Row():
            paste_back_mode = gr.Radio(["bbox", "mask"], value="bbox", label="Paste-back mode", scale=1)
            with gr.Column(scale=1):
                focus_crop_margin = gr.Number(label="Crop margin (px)", value=64, precision=0)
                gr.Markdown(
                    "Note: Increasing this value usually improves harmony between the refined region and surrounding areas; decreasing it usually improves fine-detail recovery."
                )

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                true_cfg_scale = gr.Number(label="True CFG scale", value=4.0)
                guidance_scale = gr.Number(label="Guidance scale", value=1.0)
            with gr.Row():
                paste_back_bbox = gr.Checkbox(label="Composite paste-back", value=True)
                focus_crop_for_bbox = gr.Checkbox(label="Focus-crop edit region", value=True)
            with gr.Row():
                paste_mask_grow = gr.Number(label="Mask grow", value=3, precision=0)
                paste_blend_kernel = gr.Number(label="Blend kernel", value=5, precision=0)

        run_btn = gr.Button("Run", variant="primary", size="lg")

        gr.Markdown("### Output")
        out_image = gr.ImageSlider(label="Before / After")
        with gr.Row():
            replaced_prompt = gr.Textbox(label="Actual prompt sent", lines=2)
            status = gr.Textbox(label="Status", lines=1)
        image1_vis = gr.Image(label="Input preview (ViT 384) + region overlay", type="pil")

        run_btn.click(
            fn=predict,
            inputs=[
                image1, image2, prompt, mode, spatial_source,
                seed, steps, true_cfg_scale, guidance_scale,
                load_lightning_lora,
                paste_back_bbox, paste_back_mode,
                focus_crop_for_bbox, focus_crop_margin,
                paste_mask_grow, paste_blend_kernel,
            ],
            outputs=[out_image, replaced_prompt, image1_vis, status],
        )
        seed_dice.click(fn=_randomize_seed, inputs=None, outputs=seed)

    return demo


demo = build_app()

if __name__ == "__main__":
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    demo.launch(server_name="0.0.0.0", show_error=True, ssr_mode=False)
