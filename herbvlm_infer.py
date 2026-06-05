import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

import clip_herbvlm as clip


ROOT = Path(__file__).resolve().parent
DEFAULT_META_JSON = ROOT / "datasets" / "jsons" / "chinese_medicine_163_text_meta.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_PROMPTS = [
    "a photo of {}, a Chinese medicinal herb.",
    "a photo of dried {}, a traditional Chinese medicine herb.",
    "一张中药材{}的照片。",
]


@dataclass
class InferenceConfig:
    fuse_type: int = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run HerbVLM inference on one or more herb images.")
    parser.add_argument("--image", nargs="+", default=[], help="Input image path(s).")
    parser.add_argument("--image-dir", default="", help="Directory containing images.")
    parser.add_argument("--recursive", action="store_true", help="Search --image-dir recursively.")
    parser.add_argument("--checkpoint", default="", help="Trained HerbVLM checkpoint path, e.g. best.pth.")
    parser.add_argument("--backbone", default="ViT-B/16", help="CLIP backbone name or local CLIP checkpoint.")
    parser.add_argument("--model-cache-dir", default="./model/clip", help="Directory for CLIP pretrained weights.")
    parser.add_argument("--meta-json", default=str(DEFAULT_META_JSON), help="Class metadata JSON.")
    parser.add_argument("--prompt-mode", default="multi", choices=["baseline", "base", "multi"], help="Prompt template mode.")
    parser.add_argument("--tkpm-mode", default="auto", choices=["auto", "mean", "weighted", "herb", "similarity"],
                        help="How to aggregate multiple text prompts.")
    parser.add_argument("--tkpm-temp", type=float, default=8.0, help="Temperature for weighted prompt aggregation.")
    parser.add_argument("--text-cache", default="", help="Optional path for cached text features.")
    parser.add_argument("--branch", default="tot", choices=["clip", "mlp", "ada", "tot", "total"],
                        help="Logit branch used for prediction.")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to print per image.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda, cpu, or cuda:N.")
    parser.add_argument("--fuse-type", type=int, default=2, choices=[1, 2, 3], help="HIA feature fusion type.")
    parser.add_argument("--output", default="", help="Optional CSV output path.")
    return parser.parse_args()


def load_metadata(meta_json):
    meta_path = Path(meta_json)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Class metadata JSON not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"Class metadata JSON must be a non-empty object: {meta_path}")
    return metadata


def to_pinyin(text):
    try:
        from pypinyin import lazy_pinyin
        return " ".join(lazy_pinyin(text))
    except Exception:
        return text


def infer_morphology(classname, meta):
    morph_en = meta.get("morphology_en", "") if isinstance(meta, dict) else ""
    morph_zh = meta.get("morphology_zh", "") if isinstance(meta, dict) else ""
    if morph_en and morph_zh:
        return morph_en, morph_zh

    rules = [
        ("切片", "slices", "切片"),
        ("片", "slices", "片状"),
        ("子", "seeds", "种子"),
        ("仁", "kernels", "仁"),
        ("花", "flowers", "花朵"),
        ("叶", "leaves", "叶片"),
        ("根", "roots", "根部"),
        ("皮", "barks/peels", "皮类"),
        ("果", "fruits", "果实"),
        ("茎", "stems", "茎部"),
        ("藤", "vine stems", "藤类"),
        ("草", "whole herbs", "全草"),
        ("壳", "shells", "壳类"),
        ("块", "chunks", "块状"),
        ("条", "strips", "条状"),
        ("虫", "insect-derived materials", "虫类"),
        ("角", "horn-like materials", "角类"),
    ]
    for token, en_desc, zh_desc in rules:
        if token in classname:
            return en_desc, zh_desc
    return "medicinal materials", "干燥药材"


def build_prompts(classnames, metadata, prompt_mode):
    if prompt_mode in {"baseline", "base"}:
        return BASE_PROMPTS

    prompts = {}
    for classname in classnames:
        meta = metadata.get(classname, {}) if isinstance(metadata, dict) else {}
        pinyin = meta.get("pinyin") if isinstance(meta, dict) else None
        english_alias = meta.get("english_alias") if isinstance(meta, dict) else None
        morph_en, morph_zh = infer_morphology(classname, meta if isinstance(meta, dict) else {})

        pinyin = pinyin or to_pinyin(classname)
        english_alias = english_alias or pinyin.title()
        prompts[classname] = [
            f"a photo of {classname}, a Chinese medicinal herb.",
            f"a photo of dried {classname}, a traditional Chinese medicine herb.",
            f"一张中药材{classname}的照片。",
            f"a photo of {pinyin}, a Chinese medicinal herb name in pinyin.",
            f"a photo of {english_alias}, a Chinese medicinal herb.",
            f"a close-up photo of dried {morph_en} of {classname}.",
            f"中药材{classname}，其形态为{morph_zh}。",
        ]
    return prompts


def build_text_features(cache_path, classnames, prompts, model, device, tkpm_mode="auto", tkpm_temperature=8.0):
    if cache_path:
        cache = Path(cache_path)
        if cache.is_file():
            text_features = torch.load(cache, map_location="cpu")
            if text_features.ndim != 2 or text_features.shape[1] != len(classnames):
                raise ValueError(
                    f"Text cache shape {tuple(text_features.shape)} does not match {len(classnames)} classes."
                )
            return text_features.to(device)

    weights = []
    with torch.no_grad():
        for classname in classnames:
            if isinstance(prompts, list):
                texts = [template.format(classname) for template in prompts]
            else:
                texts = prompts[classname]

            tokens = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(tokens)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            use_weighted = False
            if class_embeddings.shape[0] > 1:
                if tkpm_mode in {"weighted", "herb", "similarity"}:
                    use_weighted = True
                elif tkpm_mode == "auto":
                    use_weighted = isinstance(prompts, dict)

            if use_weighted:
                anchor = class_embeddings.mean(dim=0)
                anchor = anchor / anchor.norm(dim=0, keepdim=False)
                prompt_scores = class_embeddings @ anchor
                prompt_weights = torch.softmax(prompt_scores * tkpm_temperature, dim=0)
                class_embedding = (prompt_weights.unsqueeze(1) * class_embeddings).sum(dim=0)
            else:
                class_embedding = class_embeddings.mean(dim=0)

            class_embedding = class_embedding / class_embedding.norm()
            weights.append(class_embedding)

    text_features = torch.stack(weights, dim=1)
    if cache_path:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(text_features.detach().cpu(), cache)
    return text_features.to(device)


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return None

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "adapter_state_dict" in payload:
        state_dict = payload["adapter_state_dict"]
    elif isinstance(payload, dict) and "full_model_state_dict" in payload:
        state_dict = payload["full_model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"loaded": len(state_dict) - len(unexpected), "missing": len(missing), "unexpected": len(unexpected)}


def collect_images(image_paths, image_dir, recursive=False):
    paths = [Path(path) for path in image_paths]
    if image_dir:
        root = Path(image_dir)
        pattern = "**/*" if recursive else "*"
        paths.extend(path for path in root.glob(pattern) if path.suffix.lower() in IMAGE_EXTS)

    paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image file(s) not found: {missing}")
    if not paths:
        raise ValueError("No input images found. Use --image or --image-dir.")
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def predict_one(model, preprocess, text_features, image_path, classnames, branch, topk, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_logits, mlp_logits, ada_logits, total_logits, aux = model.my_forward(
            image_tensor, text_features, return_aux=True
        )
        branch_logits = {
            "clip": clip_logits,
            "mlp": mlp_logits,
            "ada": ada_logits,
            "tot": total_logits,
            "total": total_logits,
        }[branch]
        probs = torch.softmax(branch_logits.float(), dim=1)[0]
        k = min(max(1, topk), len(classnames))
        scores, indices = probs.topk(k)

    predictions = [
        {
            "rank": rank,
            "label": classnames[idx],
            "score": float(score),
        }
        for rank, (score, idx) in enumerate(zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1)
    ]
    alpha = None
    if isinstance(aux, dict) and "alpha" in aux:
        alpha = float(aux["alpha"].detach().float().cpu().view(-1)[0])
    return {"image": str(image_path), "alpha": alpha, "predictions": predictions}


def write_csv(records, output_path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "rank", "label", "score", "alpha"])
        writer.writeheader()
        for record in records:
            for pred in record["predictions"]:
                writer.writerow({
                    "image": record["image"],
                    "rank": pred["rank"],
                    "label": pred["label"],
                    "score": f"{pred['score']:.8f}",
                    "alpha": "" if record["alpha"] is None else f"{record['alpha']:.8f}",
                })


def main():
    args = parse_args()
    if not args.checkpoint and args.branch != "clip":
        raise SystemExit("A HerbVLM checkpoint is required for non-CLIP branches. Add --checkpoint or use --branch clip.")

    device = torch.device(args.device)
    metadata = load_metadata(args.meta_json)
    classnames = list(metadata.keys())
    prompts = build_prompts(classnames, metadata, args.prompt_mode)
    image_paths = collect_images(args.image, args.image_dir, args.recursive)

    config = InferenceConfig(fuse_type=args.fuse_type)
    model, preprocess = clip.load(
        args.backbone,
        device=device,
        download_root=args.model_cache_dir,
        num_classes=len(classnames),
        config=config,
    )
    model.eval()
    ckpt_info = load_checkpoint(model, args.checkpoint) if args.checkpoint else None
    text_features = build_text_features(
        args.text_cache,
        classnames,
        prompts,
        model,
        device,
        tkpm_mode=args.tkpm_mode,
        tkpm_temperature=args.tkpm_temp,
    )

    if ckpt_info is not None:
        print(
            f"Loaded checkpoint: {args.checkpoint} "
            f"(loaded={ckpt_info['loaded']}, unexpected={ckpt_info['unexpected']})"
        )
    print(f"Loaded {len(classnames)} classes. Running inference on {len(image_paths)} image(s).")

    records = [
        predict_one(model, preprocess, text_features, path, classnames, args.branch, args.topk, device)
        for path in image_paths
    ]

    for record in records:
        print(f"\n{record['image']}")
        if record["alpha"] is not None and args.branch in {"tot", "total"}:
            print(f"alpha={record['alpha']:.4f}")
        for pred in record["predictions"]:
            print(f"{pred['rank']:>2}. {pred['label']}\t{pred['score']:.4f}")

    if args.output:
        write_csv(records, args.output)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
