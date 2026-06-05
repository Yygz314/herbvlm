# CLIP 预训练权重说明

本目录用于存放 CLIP 预训练权重（例如 `ViT-B-16.pt`、`RN50.pt`）。

- 方式 1：首次运行推理时通过 `--model-cache-dir ./model/clip` 自动下载。
- 方式 2：手动下载后放到本目录，再在推理命令中指定 `--model-cache-dir ./model/clip`。
