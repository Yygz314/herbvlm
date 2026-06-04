import os
import random
import re
import json
from collections import defaultdict

from .utils import Datum, DatasetBase


template = [
    "a photo of {}, a Chinese medicinal herb.",
    "a photo of dried {}, a traditional Chinese medicine herb.",
    "一张中药材{}的照片。",
]


class ChineseMedicine163(DatasetBase):

    dataset_dir = "Chinese-Medicine-163"
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root, num_shots, val_ratio=0.2, seed=2024):
        self.dataset_dir = root
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.prompt_mode = os.getenv("HERBVLM_TEXT_PROMPT_MODE", "baseline").lower()
        self.meta_json = os.getenv("HERBVLM_TEXT_META_JSON", "").strip()
        self.template = template

        lab2meta = self._build_label_map(self.train_dir, self.test_dir)
        classnames = [v["classname"] for _, v in sorted(lab2meta.items(), key=lambda x: x[1]["label"])]
        self.template = self._build_template(classnames, self.prompt_mode, self.meta_json)
        train_full = self._read_split(self.train_dir, lab2meta)
        test = self._read_split(self.test_dir, lab2meta)

        train, val = self._split_train_val(train_full, val_ratio=val_ratio, seed=seed)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def _load_meta_json(meta_path):
        if not meta_path:
            return {}
        if not os.path.isfile(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _to_pinyin(text):
        try:
            from pypinyin import lazy_pinyin  # optional dependency
            return " ".join(lazy_pinyin(text))
        except Exception:
            return text

    @staticmethod
    def _infer_morphology(classname, meta):
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

    @classmethod
    def _build_template(cls, classnames, prompt_mode, meta_json):
        if prompt_mode in {"baseline", "base"}:
            return template

        metadata = cls._load_meta_json(meta_json)
        prompt_dict = {}
        for classname in classnames:
            meta = metadata.get(classname, {}) if isinstance(metadata, dict) else {}
            pinyin = meta.get("pinyin") if isinstance(meta, dict) else None
            if not pinyin:
                pinyin = cls._to_pinyin(classname)
            english_alias = meta.get("english_alias") if isinstance(meta, dict) else None
            if not english_alias:
                english_alias = pinyin.title()
            morph_en, morph_zh = cls._infer_morphology(classname, meta if isinstance(meta, dict) else {})

            prompt_dict[classname] = [
                f"a photo of {classname}, a Chinese medicinal herb.",
                f"a photo of dried {classname}, a traditional Chinese medicine herb.",
                f"一张中药材{classname}的照片。",
                f"a photo of {pinyin}, a Chinese medicinal herb name in pinyin.",
                f"a photo of {english_alias}, a Chinese medicinal herb.",
                f"a close-up photo of dried {morph_en} of {classname}.",
                f"中药材{classname}，其形态为{morph_zh}。",
            ]
        return prompt_dict

    @classmethod
    def _is_image_file(cls, name):
        return name.lower().endswith(cls.image_exts)

    @staticmethod
    def _decode_class_name(raw_name):
        # Dataset folder names are stored as "#U4e09#U4e03"-style Unicode tokens.
        tokens = re.findall(r"#U([0-9a-fA-F]{4,6})", raw_name)
        if tokens:
            try:
                return "".join(chr(int(t, 16)) for t in tokens)
            except ValueError:
                pass
        return raw_name.replace("_", " ")

    def _build_label_map(self, train_dir, test_dir):
        train_classes = sorted(
            d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
        )
        test_classes = sorted(
            d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
        )
        if train_classes != test_classes:
            raise ValueError(
                f"Train/Test class mismatch: train={len(train_classes)}, test={len(test_classes)}"
            )

        label_map = {}
        for label, raw_name in enumerate(train_classes):
            label_map[raw_name] = {
                "label": label,
                "classname": self._decode_class_name(raw_name),
            }
        return label_map

    def _read_split(self, split_dir, label_map):
        items = []
        for raw_name, meta in label_map.items():
            cls_dir = os.path.join(split_dir, raw_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")

            for file_name in sorted(os.listdir(cls_dir)):
                if not self._is_image_file(file_name):
                    continue
                impath = os.path.join(cls_dir, file_name)
                items.append(
                    Datum(
                        impath=impath,
                        label=meta["label"],
                        classname=meta["classname"],
                    )
                )
        return items

    @staticmethod
    def _split_train_val(train_items, val_ratio=0.2, seed=2024):
        rng = random.Random(seed)
        grouped = defaultdict(list)
        for item in train_items:
            grouped[item.label].append(item)

        train, val = [], []
        for _, items in grouped.items():
            rng.shuffle(items)
            if len(items) <= 1:
                train.extend(items)
                continue

            n_val = max(1, int(round(len(items) * val_ratio)))
            n_val = min(n_val, len(items) - 1)
            val.extend(items[:n_val])
            train.extend(items[n_val:])

        return train, val
