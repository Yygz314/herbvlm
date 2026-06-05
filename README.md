# HerbVLM

一种图文知识增强的中药材鉴别模型。

> 本仓库仅发布论文对应的推理代码、模型结构和类别提示信息，不包含训练入口、训练损失、优化器、数据划分或训练日志生成代码。

## 模型简介

HerbVLM 以 CLIP 为基础，在中药材识别任务中引入面向中药文本提示与图像特征融合的改进模块，用于提升细粒度中药材类别识别的准确率与鲁棒性。

- 视觉分支（HIA）：提取并融合多层图像特征。
- 文本分支（TKPM）：使用中药类别名称、拼音、英文别名和形态信息构建文本提示。
- 动态融合：结合 CLIP 相似度、适配器分支和样本级融合权重输出最终预测。

## 模型结构图

### 1) HerbVLM 总体结构

![HerbVLM Overall](./assets/HerbVLM.jpg)

### 2) TKPM 文本层知识提示模块

![TKPM Module](./assets/TKPM.jpg)

### 3) HIA 图像层融合机制

![HIA Module](./assets/HIA.jpg)

## 仓库结构

```text
HerbVLM/
├── herbvlm_infer.py                  # 推理入口
├── clip_herbvlm/                     # HerbVLM/CLIP 模型结构与 tokenizer
├── datasets/jsons/
│   ├── chinese_medicine_163_text_meta.json
│   └── chinese_medicine_163_text_meta.example.json
├── assets/                           # README 图示
├── model/clip/README.md              # CLIP 预训练权重说明
├── requirements.txt
└── .gitignore
```

## 环境安装

```bash
cd HerbVLM
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd HerbVLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 权重准备

推理需要两类权重：

1. CLIP 预训练权重：首次运行会自动下载到 `--model-cache-dir`，也可以手动放到 `model/clip/`。
2. HerbVLM 训练后 checkpoint：请将论文发布的 `best.pth` 或等价 checkpoint 放在本地路径，并通过 `--checkpoint` 指定。

`.gitignore` 默认忽略 `*.pt`、`*.pth` 等权重文件，避免大文件误提交。

## 单张图片推理

```bash
python herbvlm_infer.py \
  --image path/to/herb.jpg \
  --checkpoint path/to/best.pth \
  --backbone ViT-B/16 \
  --model-cache-dir ./model/clip \
  --meta-json datasets/jsons/chinese_medicine_163_text_meta.json \
  --branch tot \
  --topk 5
```

如果只想运行 CLIP zero-shot 分支进行代码连通性检查，可以不提供 HerbVLM checkpoint：

```bash
python herbvlm_infer.py --image path/to/herb.jpg --branch clip --topk 5
```

## 文件夹批量推理

```bash
python herbvlm_infer.py \
  --image-dir path/to/images \
  --recursive \
  --checkpoint path/to/best.pth \
  --output result/predictions.csv
```

常用参数：

- `--branch`: 预测分支，可选 `clip`、`mlp`、`ada`、`tot`，默认 `tot`。
- `--prompt-mode`: 文本提示模式，可选 `baseline` 或 `multi`，默认 `multi`。
- `--tkpm-mode`: 多提示聚合方式，默认 `auto`。
- `--text-cache`: 可选文本特征缓存路径，用于加速重复推理。
- `--device`: 推理设备，默认自动选择 CUDA 或 CPU。

## 类别与提示信息

`datasets/jsons/chinese_medicine_163_text_meta.json` 保存 163 个类别的类别顺序和 TKPM 元信息。checkpoint 的输出维度必须与该 JSON 中的类别数量和顺序一致。

