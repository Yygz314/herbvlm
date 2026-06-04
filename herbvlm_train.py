import os
import json
import time
import csv
import torch
import random
import numpy as np
from tqdm import tqdm
import clip_herbvlm as clip
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.utils import DatasetWrapper
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from alisuretool.Tools import Tools  # type: ignore
except Exception:
    class Tools:
        @staticmethod
        def new_dir(path):
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            return path

        @staticmethod
        def print(msg, txt_path=None):
            text = str(msg)
            print(text)
            if txt_path:
                Tools.new_dir(txt_path)
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(text + "\n")


MODEL_CACHE_DIR = os.getenv("HERBVLM_MODEL_CACHE_DIR", "./model/clip")
DATA_ROOT = os.getenv("HERBVLM_DATA_ROOT", "./your/data/path")
LOG_ROOT = os.getenv("HERBVLM_LOG_ROOT", "./result/log")
CKPT_ROOT = os.getenv("HERBVLM_CKPT_ROOT", "./result/checkpoints")
VIS_ROOT = os.getenv("HERBVLM_VIS_ROOT", "./result/vis")


class MyTransform(object):

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    @staticmethod
    def transform_train(size, scale=(0.8, 1.0)):
        funcs = [
            T.RandomResizedCrop(size=size, scale=scale, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    @staticmethod
    def transform_test(size):
        funcs = [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    pass


class Config10Dataset(object):

    def __init__(self, dataset_name, seed=2024, shots=16, backbone="RN50", lr=0.001, batch_size=64, train_epoch=50,
                 loss_lambda=[1.0, 1.0, 1.0, 1.0, 1.0], fuse_type=2):
        self.setup_seed(seed)

        self.seed = seed
        self.shots = shots
        self.lr = lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.backbone = backbone  # RN50 RN101 ViT-B/32 ViT-B/16

        self.loss_lambda = loss_lambda
        self.fuse_type = fuse_type

        _dataset_info = self.dataset_info()
        self.dataset_name = dataset_name
        assert self.dataset_name in _dataset_info.keys()
        self.data_path = os.path.join(DATA_ROOT, _dataset_info[self.dataset_name][2])
        self.dataset = _dataset_info[self.dataset_name][0](self.data_path, self.shots)
        self.num_classes = _dataset_info[self.dataset_name][1]

        self.cache_dir = MODEL_CACHE_DIR
        pass

    def get_detail(self):
        detail_str = (f"dataset_name={self.dataset_name}, shots={self.shots}, lr={self.lr}, seed={self.seed}, "
                      f"train_epoch={self.train_epoch}, batch_size={self.batch_size}, backbone={self.backbone}, "
                      f"num_classes={self.num_classes}, loss_lambda={self.loss_lambda}, fuse_type={self.fuse_type}")
        return detail_str

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def dataset_info():
        from datasets.chinese_medicine_163 import ChineseMedicine163

        return {"chinese_medicine_163": [ChineseMedicine163, 163, "Chinese-Medicine-163"]}

    pass


class Eval(object):

    def __init__(self, batch_size, clip_model, val_loader, text_feats, model_variant="herbvlm"):
        self.clip_model = clip_model
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.model_variant = model_variant.lower()
        pass

    def eval(self, best_beta=None):
        self.clip_model.eval()
        all_labels, all_logits = [], []
        all_alpha = []
        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    if self.model_variant == "herbvlm":
                        clip_logits, mlp_logits, ada_logits, tot_logits, aux = self.clip_model.my_forward(
                            images.cuda(), self.text_feats, return_aux=True
                        )
                        if isinstance(aux, dict) and "alpha" in aux:
                            all_alpha.append(aux["alpha"].detach().view(-1))
                    else:
                        clip_logits, mlp_logits, ada_logits, tot_logits = self.clip_model.my_forward(
                            images.cuda(), self.text_feats
                        )
                    all_logits.append([clip_logits, mlp_logits, ada_logits, tot_logits])
                    all_labels.append(labels)
                    pass
                pass
            pass
        all_labels = torch.cat(all_labels, dim=0)
        clip_all = torch.cat([one[0] for one in all_logits], dim=0)
        mlp_all = torch.cat([one[1] for one in all_logits], dim=0)
        ada_all = torch.cat([one[2] for one in all_logits], dim=0)
        tot_all = torch.cat([one[3] for one in all_logits], dim=0)

        result_acc = {}
        branch_map = {
            "clip": ("clip_logits", clip_all),
            "mlp": ("mlp_logits", mlp_all),
            "ada": ("ada_logits", ada_all),
            "tot": ("tot_logits", tot_all),
        }
        for branch_prefix, (acc_key, logits_branch) in branch_map.items():
            m = self.cal_macro_metrics(logits_branch, all_labels)
            result_acc[acc_key] = m["acc"]
            result_acc[f"{branch_prefix}_precision_macro"] = m["precision_macro"]
            result_acc[f"{branch_prefix}_recall_macro"] = m["recall_macro"]
            result_acc[f"{branch_prefix}_f1_macro"] = m["f1_macro"]
            Tools.print(
                f"test {acc_key} acc={m['acc']:.2f}% "
                f"P={m['precision_macro']:.2f}% R={m['recall_macro']:.2f}% F1={m['f1_macro']:.2f}%"
            )

        if len(all_alpha) > 0:
            alpha_tensor = torch.cat(all_alpha, dim=0).float()
            result_acc["alpha_mean"] = float(alpha_tensor.mean().item())
            result_acc["alpha_std"] = float(alpha_tensor.std(unbiased=False).item())
            Tools.print(
                f"test alpha mean={result_acc['alpha_mean']:.4f}, std={result_acc['alpha_std']:.4f}"
            )

        if self.model_variant == "herbvlm":
            result_acc["acc"] = result_acc["tot_logits"]
            result_acc["precision_macro"] = result_acc["tot_precision_macro"]
            result_acc["recall_macro"] = result_acc["tot_recall_macro"]
            result_acc["f1_macro"] = result_acc["tot_f1_macro"]
            Tools.print(
                f"test HerbVLM main acc(tot_logits)={result_acc['acc']:.2f}% "
                f"P={result_acc['precision_macro']:.2f}% "
                f"R={result_acc['recall_macro']:.2f}% "
                f"F1={result_acc['f1_macro']:.2f}%"
            )
            return best_beta, result_acc

        if best_beta is None:
            best_beta, last_acc, best_acc = self.search_hp(mlp_all, ada_all, all_labels)
            logits = self.fuse_logits(mlp_all, ada_all, beta=best_beta)
            main_m = self.cal_macro_metrics(logits, all_labels)
            result_acc["acc"] = main_m["acc"]
            result_acc["precision_macro"] = main_m["precision_macro"]
            result_acc["recall_macro"] = main_m["recall_macro"]
            result_acc["f1_macro"] = main_m["f1_macro"]
            Tools.print(f"val best beta = {best_beta:.4f} => last_acc={last_acc:.2f}% [best_acc={best_acc}]")
            return best_beta, result_acc
        else:
            logits = self.fuse_logits(mlp_all, ada_all, beta=best_beta)
            main_m = self.cal_macro_metrics(logits, all_labels)
            result_acc["acc"] = main_m["acc"]
            result_acc["precision_macro"] = main_m["precision_macro"]
            result_acc["recall_macro"] = main_m["recall_macro"]
            result_acc["f1_macro"] = main_m["f1_macro"]
            Tools.print(
                f"test main acc={main_m['acc']:.2f}% "
                f"P={main_m['precision_macro']:.2f}% R={main_m['recall_macro']:.2f}% F1={main_m['f1_macro']:.2f}%"
            )
            return best_beta, result_acc
        # return best_beta, acc

    @staticmethod
    def fuse_logits(mlp_logits, clip_logits, beta=1.0):
        return beta * mlp_logits + (1 - beta) * clip_logits

    @staticmethod
    def cal_acc(logits, labels):
        pred = torch.argmax(logits, -1)
        labels = labels.to(pred.device)
        acc_num = (pred == labels).sum().item()
        return 1.0 * acc_num / len(labels)

    @staticmethod
    def cal_prf_macro(logits, labels, eps=1e-12):
        # Macro-averaged precision/recall/F1 over all classes in logits.
        pred = torch.argmax(logits, -1).detach().to(torch.int64).cpu()
        gold = labels.detach().to(torch.int64).cpu()
        num_classes = int(logits.shape[1])

        idx = gold * num_classes + pred
        cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        valid = (tp + fn) > 0
        if valid.any():
            return (
                float(precision[valid].mean().item()),
                float(recall[valid].mean().item()),
                float(f1[valid].mean().item()),
            )
        return 0.0, 0.0, 0.0

    @classmethod
    def cal_macro_metrics(cls, logits, labels):
        acc = cls.cal_acc(logits, labels)
        precision, recall, f1 = cls.cal_prf_macro(logits, labels)
        return {
            "acc": acc * 100.0,
            "precision_macro": precision * 100.0,
            "recall_macro": recall * 100.0,
            "f1_macro": f1 * 100.0,
        }

    def search_hp(self, mlp_logits, clip_logits, all_labels, start=0, end=1, step=50):
        beta_list = [i * (end - start) / step + start for i in range(step + 1)]
        accs, best_beta, best_acc = [], start, 0.
        for beta in beta_list:
            logits = self.fuse_logits(mlp_logits, clip_logits, beta=beta)
            acc = self.cal_acc(logits, all_labels) * 100.
            accs.append((beta, acc))
            if acc > best_acc:
                best_acc = acc
                best_beta = beta
        return best_beta, accs[-1][-1], best_acc

    pass


class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
        pass

    def step(self, logits, labels):
        pred = torch.argmax(logits, -1)
        acc_num = (pred == labels.cuda()).sum().item()
        total = len(labels)
        self.acc_num += acc_num
        self.total += total
        pass

    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total

    pass


class Runner(object):

    def __init__(self, config):
        self.config = config

        Tools.print(f"Preparing {self.config.backbone} model.")
        self.clip_model, self.preprocess = clip.load(self.config.backbone, download_root=self.config.cache_dir,
                                                     num_classes=self.config.num_classes, config=self.config)
        self.clip_model.eval()
        self.model_variant = os.getenv("HERBVLM_MODEL_VARIANT", "herbvlm").lower()
        self.tkpm_mode = os.getenv("HERBVLM_TKPM_MODE", "auto").lower()
        self.tkpm_temperature = float(os.getenv("HERBVLM_TKPM_TEMP", "8.0"))
        self.conf_margin = float(os.getenv("HERBVLM_CONF_MARGIN", "0.2"))
        self.conf_topk = max(1, int(os.getenv("HERBVLM_CONF_TOPK", "3")))
        self.lambda_conf = float(
            os.getenv("HERBVLM_LAMBDA_CONF", "0.2" if self.model_variant == "herbvlm" else "0.0")
        )

        Tools.print("Getting cached textual weights W ...")
        prompt_mode = os.getenv("HERBVLM_TEXT_PROMPT_MODE", "baseline").lower()
        meta_json = os.getenv("HERBVLM_TEXT_META_JSON", "").strip()
        meta_tag = ""
        if meta_json:
            meta_base = os.path.splitext(os.path.basename(meta_json))[0]
            meta_tag = f"_meta-{meta_base}"
        backbone_tag = self.config.backbone.replace("/", "-")
        text_feat_path = os.path.join(
            self.config.cache_dir,
            f"{self.config.dataset_name}_{backbone_tag}_textfeats_{prompt_mode}{meta_tag}_tkpm-{self.tkpm_mode}.pt"
        )
        self.text_feats = self.clip_classifier(
            text_feat_path,
            self.config.dataset.classnames,
            self.config.dataset.template,
            self.clip_model,
            tkpm_mode=self.tkpm_mode,
            tkpm_temperature=self.tkpm_temperature,
        )

        # Preparation for training
        for param in self.clip_model.parameters():
            param.requires_grad = False
            pass
        for name, param in self.clip_model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            pass

        Tools.print(f"Preparing {self.config.dataset_name} dataset.")
        self.train_loader = DataLoader(
            DatasetWrapper(self.config.dataset.train_x, input_size=224, transform=MyTransform.transform_train(224), is_train=True),
            batch_size=self.config.batch_size, num_workers=8, shuffle=True, drop_last=False, pin_memory=(torch.cuda.is_available()))
        self.val_loader = DataLoader(
            DatasetWrapper(self.config.dataset.val, input_size=224, transform=self.preprocess, is_train=False),
            batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
        self.test_loader = DataLoader(
            DatasetWrapper(self.config.dataset.test, input_size=224, transform=self.preprocess, is_train=False),
            batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
        self.test_loader_list = [self.test_loader]

        self.optimizer = torch.optim.AdamW(self.clip_model.parameters(), lr=self.config.lr / 10, weight_decay=1e-4, eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.train_epoch * len(self.train_loader))

        self.eval = Eval(
            self.config.batch_size, self.clip_model, self.test_loader, self.text_feats, model_variant=self.model_variant
        )
        self.eval_interval = max(1, int(os.getenv("HERBVLM_EVAL_INTERVAL", "1")))
        self.report_to_file = os.getenv("HERBVLM_REPORT_TO_FILE", "1") == "1"
        self.save_ckpt = os.getenv("HERBVLM_SAVE_CKPT", "1") == "1"
        self.save_every_epoch = os.getenv("HERBVLM_SAVE_EVERY_EPOCH", "0") == "1"
        self.save_full_model = os.getenv("HERBVLM_SAVE_FULL_MODEL", "0") == "1"
        self.auto_plot = os.getenv("HERBVLM_AUTO_PLOT", "1") == "1"
        self.save_all_to_vis = os.getenv("HERBVLM_SAVE_ALL_TO_VIS", "0") == "1"
        self.vis_exp_name = os.getenv("HERBVLM_VIS_EXP_NAME", "").strip()
        self.ckpt_root = os.getenv("HERBVLM_CKPT_ROOT", CKPT_ROOT)
        self.vis_root = os.getenv("HERBVLM_VIS_ROOT", VIS_ROOT)
        safe_backbone = self.config.backbone.replace("/", "-")
        report_prompt_tag = f"{prompt_mode}_tkpm-{self.tkpm_mode}_{self.model_variant}"
        if meta_tag:
            report_prompt_tag = f"{report_prompt_tag}{meta_tag}"
        report_base = f"progress_{self.config.dataset_name}_{safe_backbone}_{report_prompt_tag}_seed{self.config.seed}"
        self.run_name = report_base

        os.makedirs(self.vis_root, exist_ok=True)
        os.makedirs(LOG_ROOT, exist_ok=True)
        if self.save_all_to_vis:
            ts_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            default_exp_name = f"{report_base}_{ts_tag}"
            exp_name = self.vis_exp_name if self.vis_exp_name else default_exp_name
            self.exp_dir = os.path.join(self.vis_root, exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)
            self.report_jsonl_path = os.path.join(self.exp_dir, f"{report_base}.jsonl")
            self.report_txt_path = os.path.join(self.exp_dir, f"{report_base}.txt")
            self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
            self.vis_output_dir = self.exp_dir
        else:
            self.exp_dir = None
            self.report_jsonl_path = os.path.join(LOG_ROOT, f"{report_base}.jsonl")
            self.report_txt_path = os.path.join(LOG_ROOT, f"{report_base}.txt")
            self.ckpt_dir = os.path.join(self.ckpt_root, self.run_name)
            self.vis_output_dir = self.vis_root

        if self.save_ckpt:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.report_jsonl_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_txt_path), exist_ok=True)
        os.makedirs(self.vis_output_dir, exist_ok=True)
        self.epoch_history = []
        self.best_main_acc = -1.0
        self._append_report({
            "event": "start",
            "dataset": self.config.dataset_name,
            "backbone": self.config.backbone,
            "seed": self.config.seed,
            "shots": self.config.shots,
            "epochs": self.config.train_epoch,
            "batch_size": self.config.batch_size,
            "eval_interval": self.eval_interval,
            "model_variant": self.model_variant,
            "tkpm_mode": self.tkpm_mode,
            "tkpm_temperature": self.tkpm_temperature,
            "lambda_conf": self.lambda_conf,
            "conf_margin": self.conf_margin,
            "conf_topk": self.conf_topk,
            "save_ckpt": self.save_ckpt,
            "auto_plot": self.auto_plot,
            "ckpt_dir": self.ckpt_dir if self.save_ckpt else None,
            "vis_root": self.vis_root,
            "save_all_to_vis": self.save_all_to_vis,
            "exp_dir": self.exp_dir,
            "vis_output_dir": self.vis_output_dir,
        })
        pass

    def train_epoch(self, epoch):
        self.clip_model.adapter.train()
        self.clip_model.visual.adapter.train()

        train_acc_mlp, train_acc_total, train_loss = AvgACC(), AvgACC(), 0.0
        loss_list = [0, 0, 0, 0, 0, 0]
        alpha_sum, alpha_sq_sum, alpha_count = 0.0, 0.0, 0
        with tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Train {epoch + 1}/{self.config.train_epoch}",
                ncols=120) as tqdm_train:
            for step, (images, labels) in tqdm_train:
                images, labels = images.cuda(), labels.cuda()
                if self.model_variant == "herbvlm":
                    clip_logits, mlp_logits, ada_logits, total_logits, aux = self.clip_model.my_forward(
                        images, self.text_feats, return_aux=True
                    )
                    alpha = aux.get("alpha", None) if isinstance(aux, dict) else None
                else:
                    clip_logits, mlp_logits, ada_logits, total_logits = self.clip_model.my_forward(images, self.text_feats)
                    alpha = None
                loss, losses = self.get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits,
                                             lambda_value=self.config.loss_lambda,
                                             lambda_conf=self.lambda_conf if self.model_variant == "herbvlm" else 0.0,
                                             conf_margin=self.conf_margin,
                                             conf_topk=self.conf_topk)
                train_loss += loss.item()
                train_acc_mlp.step(mlp_logits, labels)
                train_acc_total.step(total_logits, labels)

                for i, l in enumerate(losses):
                    loss_list[i] += l.item()

                if alpha is not None:
                    alpha_flat = alpha.detach().view(-1).float()
                    alpha_sum += alpha_flat.sum().item()
                    alpha_sq_sum += (alpha_flat * alpha_flat).sum().item()
                    alpha_count += alpha_flat.numel()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                avg_loss = train_loss / (step + 1)
                tqdm_train.set_postfix(
                    cur_loss=f"{loss.item():.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    mlp_acc=f"{train_acc_mlp.cal() * 100:.2f}%",
                    total_acc=f"{train_acc_total.cal() * 100:.2f}%",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

            train_acc_mlp_result = train_acc_mlp.cal()
            train_acc_total_result = train_acc_total.cal()
            train_loss = train_loss / len(self.train_loader)
            pass

        avg_losses = [one / len(self.train_loader) for one in loss_list]
        alpha_mean = alpha_sum / alpha_count if alpha_count > 0 else 0.0
        alpha_var = alpha_sq_sum / alpha_count - alpha_mean * alpha_mean if alpha_count > 0 else 0.0
        alpha_std = float(max(0.0, alpha_var) ** 0.5)
        metrics = {
            "train_loss": train_loss,
            "train_mlp_acc": train_acc_mlp_result * 100.0,
            "train_total_acc": train_acc_total_result * 100.0,
            "train_main_acc": (train_acc_total_result if self.model_variant == "herbvlm" else train_acc_mlp_result) * 100.0,
            "l1_loss_mlp_clip": avg_losses[0],
            "l1_loss_ada_clip": avg_losses[1],
            "ce_loss_mlp": avg_losses[2],
            "ce_loss_ada": avg_losses[3],
            "ce_loss_total": avg_losses[4],
            "conf_loss": avg_losses[5],
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
        }
        return metrics

    def train(self):
        last_test_acc_list = None
        best_ckpt_path = None
        last_ckpt_path = None
        for epoch in range(self.config.train_epoch):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)

            need_eval = ((epoch + 1) % self.eval_interval == 0) or (epoch + 1 == self.config.train_epoch)
            eval_summary = {}
            is_best_epoch = False
            if need_eval:
                last_test_acc_list = self.test()
                eval_summary = self._summarize_eval_result(last_test_acc_list)
                if "main_acc" in eval_summary:
                    cur_main_acc = float(eval_summary["main_acc"])
                    if cur_main_acc > self.best_main_acc:
                        self.best_main_acc = cur_main_acc
                        is_best_epoch = True

            epoch_seconds = time.time() - epoch_start
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            report = {
                "event": "epoch_end",
                "epoch": epoch + 1,
                "epochs": self.config.train_epoch,
                "epoch_seconds": round(epoch_seconds, 3),
                "lr": lr,
                **train_metrics,
                **eval_summary,
                "best_main_acc": self.best_main_acc if self.best_main_acc >= 0 else None,
            }
            self._append_report(report)
            self.epoch_history.append(dict(report))

            if self.save_ckpt:
                last_ckpt_path = self._save_checkpoint("last.pth", report, is_best=False)
                if self.save_every_epoch:
                    self._save_checkpoint(f"epoch_{epoch + 1:03d}.pth", report, is_best=False)
                if is_best_epoch:
                    best_ckpt_path = self._save_checkpoint("best.pth", report, is_best=True)

            short_msg = (
                f"Epoch [{epoch + 1}/{self.config.train_epoch}] "
                f"loss={train_metrics['train_loss']:.4f} "
                f"main_acc={train_metrics['train_main_acc']:.2f}% "
                f"lr={lr:.8f} "
                f"time={epoch_seconds:.1f}s"
            )
            if "main_acc" in eval_summary:
                short_msg += f" eval_main_acc={eval_summary['main_acc']:.2f}%"
            if "main_precision_macro" in eval_summary and "main_recall_macro" in eval_summary and "main_f1_macro" in eval_summary:
                short_msg += (
                    f" P={eval_summary['main_precision_macro']:.2f}% "
                    f"R={eval_summary['main_recall_macro']:.2f}% "
                    f"F1={eval_summary['main_f1_macro']:.2f}%"
                )
            Tools.print(short_msg, self.report_txt_path if self.report_to_file else None)

        if last_test_acc_list is None:
            last_test_acc_list = self.test()
            self._append_report({
                "event": "final_eval",
                **self._summarize_eval_result(last_test_acc_list)
            })

        if self.save_ckpt:
            if best_ckpt_path:
                Tools.print(f"Saved best checkpoint: {best_ckpt_path}", self.report_txt_path if self.report_to_file else None)
            if last_ckpt_path:
                Tools.print(f"Saved last checkpoint: {last_ckpt_path}", self.report_txt_path if self.report_to_file else None)

        if self.auto_plot:
            vis_paths = self._plot_training_artifacts()
            if len(vis_paths) > 0:
                Tools.print("Saved training artifacts:", self.report_txt_path if self.report_to_file else None)
                for p in vis_paths:
                    Tools.print(f"  - {p}", self.report_txt_path if self.report_to_file else None)
        return last_test_acc_list

    def test(self):
        self.eval.clip_model = self.clip_model
        val_best_beta = None
        if self.val_loader:
            self.eval.val_loader = self.val_loader
            val_best_beta, val_result_acc = self.eval.eval()
            pass
        test_acc_list = []
        for test_loader in self.test_loader_list:
            self.eval.val_loader = test_loader
            val_best_beta, test_result_acc = self.eval.eval(best_beta=val_best_beta)
            test_acc_list.append(test_result_acc)
            pass
        return test_acc_list

    def _summarize_eval_result(self, test_acc_list):
        # For non-ImageNet datasets test_acc_list usually has one element.
        summary = {}
        if isinstance(test_acc_list, list) and len(test_acc_list) > 0 and isinstance(test_acc_list[0], dict):
            main = test_acc_list[0]
            for k in [
                "clip_logits", "mlp_logits", "ada_logits", "tot_logits", "acc", "alpha_mean", "alpha_std",
                "clip_precision_macro", "clip_recall_macro", "clip_f1_macro",
                "mlp_precision_macro", "mlp_recall_macro", "mlp_f1_macro",
                "ada_precision_macro", "ada_recall_macro", "ada_f1_macro",
                "tot_precision_macro", "tot_recall_macro", "tot_f1_macro",
                "precision_macro", "recall_macro", "f1_macro",
            ]:
                if k in main:
                    summary[f"eval_{k}"] = float(main[k])
            if "acc" in main:
                summary["main_acc"] = float(main["acc"])
            if "precision_macro" in main:
                summary["main_precision_macro"] = float(main["precision_macro"])
            if "recall_macro" in main:
                summary["main_recall_macro"] = float(main["recall_macro"])
            if "f1_macro" in main:
                summary["main_f1_macro"] = float(main["f1_macro"])
        return summary

    def _append_report(self, payload):
        if not self.report_to_file:
            return
        record = dict(payload)
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.report_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _to_float_or_none(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _dedup_epoch_rows(epoch_rows):
        merged = {}
        for row in epoch_rows:
            if "epoch" in row:
                merged[int(row["epoch"])] = row
        return [merged[k] for k in sorted(merged.keys())]

    def _get_trainable_state(self):
        # This project only trains adapter-related params by default.
        state = {}
        for name, tensor in self.clip_model.state_dict().items():
            if "adapter" in name:
                state[name] = tensor.detach().cpu()
        return state

    def _save_checkpoint(self, filename, epoch_record, is_best=False):
        if not self.save_ckpt:
            return None
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        payload = {
            "meta": {
                "model_name": "HerbVLM",
                "model_variant": self.model_variant,
                "dataset": self.config.dataset_name,
                "backbone": self.config.backbone,
                "seed": self.config.seed,
                "shots": self.config.shots,
                "tkpm_mode": self.tkpm_mode,
                "tkpm_temperature": self.tkpm_temperature,
                "lambda_conf": self.lambda_conf,
                "conf_margin": self.conf_margin,
                "conf_topk": self.conf_topk,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            },
            "is_best": bool(is_best),
            "epoch_record": dict(epoch_record),
            "adapter_state_dict": self._get_trainable_state(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        if self.save_full_model:
            payload["full_model_state_dict"] = {
                k: v.detach().cpu() for k, v in self.clip_model.state_dict().items()
            }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def _plot_training_artifacts(self):
        epoch_rows = self._dedup_epoch_rows([r for r in self.epoch_history if r.get("event") == "epoch_end"])
        if not epoch_rows:
            Tools.print("Skip plotting: no epoch records found.", self.report_txt_path if self.report_to_file else None)
            return []

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            Tools.print(f"Skip plotting: matplotlib unavailable ({e}).", self.report_txt_path if self.report_to_file else None)
            return []

        out_prefix = os.path.join(self.vis_output_dir, f"{self.run_name}_paper")
        paths = []

        # CSV export for paper tables.
        csv_path = f"{out_prefix}_epoch_metrics.csv"
        fields = [
            "epoch", "epochs", "epoch_seconds", "lr",
            "train_loss", "train_mlp_acc", "train_total_acc", "train_main_acc",
            "l1_loss_mlp_clip", "l1_loss_ada_clip", "ce_loss_mlp", "ce_loss_ada", "ce_loss_total", "conf_loss",
            "alpha_mean", "alpha_std",
            "eval_clip_logits", "eval_mlp_logits", "eval_ada_logits", "eval_tot_logits", "eval_acc", "main_acc",
            "eval_clip_precision_macro", "eval_clip_recall_macro", "eval_clip_f1_macro",
            "eval_mlp_precision_macro", "eval_mlp_recall_macro", "eval_mlp_f1_macro",
            "eval_ada_precision_macro", "eval_ada_recall_macro", "eval_ada_f1_macro",
            "eval_tot_precision_macro", "eval_tot_recall_macro", "eval_tot_f1_macro",
            "eval_precision_macro", "eval_recall_macro", "eval_f1_macro",
            "main_precision_macro", "main_recall_macro", "main_f1_macro",
            "best_main_acc", "ts",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in epoch_rows:
                writer.writerow({k: row.get(k) for k in fields})
        paths.append(csv_path)

        epochs = [int(r["epoch"]) for r in epoch_rows]
        train_loss = [self._to_float_or_none(r.get("train_loss")) for r in epoch_rows]
        train_main_acc = [self._to_float_or_none(r.get("train_main_acc")) for r in epoch_rows]
        eval_main_acc = [self._to_float_or_none(r.get("main_acc")) for r in epoch_rows]

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
        })

        # Figure 1: main curves.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=220)
        ax = axes[0]
        ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False, loc="upper right")

        ax = axes[1]
        ax.plot(epochs, train_main_acc, color="#2ca02c", linewidth=2.0, marker="o", markersize=3, label="Train Main Acc")
        ax.plot(epochs, eval_main_acc, color="#d62728", linewidth=2.0, marker="s", markersize=3, label="Test Main Acc")
        valid_eval = [(i, v) for i, v in enumerate(eval_main_acc) if v is not None]
        if valid_eval:
            best_idx = max(valid_eval, key=lambda x: x[1])[0]
            ax.scatter([epochs[best_idx]], [eval_main_acc[best_idx]], color="#d62728", s=42, zorder=5)
            ax.annotate(
                f"Best: {eval_main_acc[best_idx]:.2f}% (E{epochs[best_idx]})",
                xy=(epochs[best_idx], eval_main_acc[best_idx]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                color="#d62728",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Main Accuracy Curves")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        p1_pdf = f"{out_prefix}_main_curves.pdf"
        p1_png = f"{out_prefix}_main_curves.png"
        fig.savefig(p1_pdf, bbox_inches="tight")
        fig.savefig(p1_png, bbox_inches="tight", dpi=420)
        plt.close(fig)
        paths.extend([p1_pdf, p1_png])

        # Figure 2: branch accuracies.
        keys = [
            ("eval_clip_logits", "CLIP Logits", "#9467bd"),
            ("eval_mlp_logits", "MLP Logits", "#1f77b4"),
            ("eval_ada_logits", "Ada Logits", "#ff7f0e"),
            ("eval_tot_logits", "Total Logits", "#2ca02c"),
            ("eval_acc", "Main Acc", "#d62728"),
        ]
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=240)
        for key, label, color in keys:
            y = [self._to_float_or_none(r.get(key)) for r in epoch_rows]
            if all(v is None for v in y):
                continue
            ax.plot(epochs, y, linewidth=2.0, marker="o", markersize=3, label=label, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Test Accuracy by Logit Branch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        p2_pdf = f"{out_prefix}_branch_curves.pdf"
        p2_png = f"{out_prefix}_branch_curves.png"
        fig.savefig(p2_pdf, bbox_inches="tight")
        fig.savefig(p2_png, bbox_inches="tight", dpi=420)
        plt.close(fig)
        paths.extend([p2_pdf, p2_png])

        # Figure 3: objective decomposition.
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=240)
        loss_keys = [
            ("ce_loss_mlp", "CE-MLP", "#1f77b4"),
            ("ce_loss_ada", "CE-ADA", "#ff7f0e"),
            ("ce_loss_total", "CE-Total", "#2ca02c"),
            ("l1_loss_mlp_clip", "L1(MLP,CLIP)", "#9467bd"),
            ("l1_loss_ada_clip", "L1(ADA,CLIP)", "#8c564b"),
            ("conf_loss", "Confusion Loss", "#d62728"),
        ]
        for key, label, color in loss_keys:
            y = [self._to_float_or_none(r.get(key)) for r in epoch_rows]
            if all(v is None for v in y):
                continue
            ax.plot(epochs, y, linewidth=2.0, marker="o", markersize=3, label=label, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Decomposition")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        p3_pdf = f"{out_prefix}_loss_curves.pdf"
        p3_png = f"{out_prefix}_loss_curves.png"
        fig.savefig(p3_pdf, bbox_inches="tight")
        fig.savefig(p3_png, bbox_inches="tight", dpi=420)
        plt.close(fig)
        paths.extend([p3_pdf, p3_png])

        # Figure 4: macro Precision/Recall/F1.
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=240)
        prf_keys = [
            ("main_precision_macro", "Macro Precision", "#1f77b4"),
            ("main_recall_macro", "Macro Recall", "#ff7f0e"),
            ("main_f1_macro", "Macro F1", "#2ca02c"),
        ]
        has_any = False
        for key, label, color in prf_keys:
            y = [self._to_float_or_none(r.get(key)) for r in epoch_rows]
            if all(v is None for v in y):
                continue
            has_any = True
            ax.plot(epochs, y, linewidth=2.0, marker="o", markersize=3, label=label, color=color)
        if has_any:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score (%)")
            ax.set_title("Macro Precision/Recall/F1")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.legend(frameon=False, loc="lower right")
            fig.tight_layout()
            p4_pdf = f"{out_prefix}_macro_prf_curves.pdf"
            p4_png = f"{out_prefix}_macro_prf_curves.png"
            fig.savefig(p4_pdf, bbox_inches="tight")
            fig.savefig(p4_png, bbox_inches="tight", dpi=420)
            paths.extend([p4_pdf, p4_png])
        plt.close(fig)

        return paths

    @staticmethod
    def clip_classifier(feat_path, classnames, template, clip_model, tkpm_mode="auto", tkpm_temperature=8.0):
        if os.path.exists(feat_path):
            Tools.print(f"Loading texture features from {feat_path}")
            text_feats = torch.load(feat_path, map_location='cpu')
            return text_feats.cuda()

        Tools.print(f"Building textual weights with TKPM mode={tkpm_mode}, temp={tkpm_temperature}")
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                classname = classname.replace('_', ' ')
                if isinstance(template, list):
                    texts = [t.format(classname) for t in template]
                elif isinstance(template, dict):
                    texts = template[classname]

                texts = clip.tokenize(texts).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                use_weighted = False
                if class_embeddings.shape[0] > 1:
                    if tkpm_mode in {"weighted", "herb", "similarity"}:
                        use_weighted = True
                    elif tkpm_mode == "auto":
                        use_weighted = isinstance(template, dict)
                if use_weighted:
                    anchor = class_embeddings.mean(dim=0)
                    anchor = anchor / anchor.norm(dim=0, keepdim=False)
                    prompt_scores = class_embeddings @ anchor
                    prompt_weights = torch.softmax(prompt_scores * tkpm_temperature, dim=0)
                    class_embedding = (prompt_weights.unsqueeze(1) * class_embeddings).sum(dim=0)
                else:
                    class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)
                pass

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
            torch.save(clip_weights, Tools.new_dir(feat_path))

        return clip_weights

    @staticmethod
    def get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits,
                 lambda_value=[1.0, 1.0, 1.0, 1.0, 1.0], lambda_conf=0.0, conf_margin=0.2, conf_topk=3):
        ce_loss = F.cross_entropy(mlp_logits, labels) * lambda_value[0]
        ce_loss2 = F.cross_entropy(ada_logits, labels) * lambda_value[1]
        ce_loss3 = F.cross_entropy(total_logits, labels) * lambda_value[2]

        l1_loss1 = F.l1_loss(mlp_logits, clip_logits) * lambda_value[3]
        l1_loss2 = F.l1_loss(ada_logits, clip_logits) * lambda_value[4]
        conf_loss = total_logits.new_tensor(0.0)
        if lambda_conf > 0:
            conf_loss = Runner.confusion_aware_loss(
                student_logits=total_logits, teacher_logits=clip_logits, labels=labels,
                margin=conf_margin, topk=conf_topk
            ) * lambda_conf

        loss = l1_loss1 + l1_loss2 + ce_loss + ce_loss2 + ce_loss3 + conf_loss
        return loss, [l1_loss1, l1_loss2, ce_loss, ce_loss2, ce_loss3, conf_loss]

    @staticmethod
    def confusion_aware_loss(student_logits, teacher_logits, labels, margin=0.2, topk=3):
        if topk <= 0 or student_logits.shape[1] <= 1:
            return student_logits.new_tensor(0.0)

        with torch.no_grad():
            conf_probs = F.softmax(teacher_logits, dim=1)
            conf_probs.scatter_(1, labels.view(-1, 1), -1.0)
            k = min(topk, conf_probs.shape[1] - 1)
            hard_neg_idx = conf_probs.topk(k=k, dim=1).indices
            neg_teacher_logits = teacher_logits.gather(1, hard_neg_idx)
            neg_weights = F.softmax(neg_teacher_logits, dim=1)

        pos_logits = student_logits.gather(1, labels.view(-1, 1))
        neg_logits = student_logits.gather(1, hard_neg_idx)
        pair_margin = margin - (pos_logits - neg_logits)
        margin_loss = F.relu(pair_margin)
        return (margin_loss * neg_weights).sum(dim=1).mean()

    pass


def run_single_dataset():
    dataset_name = os.getenv("HERBVLM_TARGET_DATASET", "chinese_medicine_163")
    shots = int(os.getenv("HERBVLM_SHOTS", "-1"))
    backbone = os.getenv("HERBVLM_BACKBONE", "ViT-B/16")
    train_epoch = int(os.getenv("HERBVLM_EPOCHS", "20"))
    batch_size = int(os.getenv("HERBVLM_BATCH_SIZE", "64"))
    seed = int(os.getenv("HERBVLM_SEED", "2024"))
    lr = float(os.getenv("HERBVLM_LR", "0.001"))
    text_prompt_mode = os.getenv("HERBVLM_TEXT_PROMPT_MODE", "baseline")
    text_meta_json = os.getenv("HERBVLM_TEXT_META_JSON", "")

    Tools.print(
        f"Initializing config: dataset={dataset_name}, backbone={backbone}, shots={shots}, "
        f"epochs={train_epoch}, batch_size={batch_size}, text_prompt_mode={text_prompt_mode}, "
        f"text_meta_json={text_meta_json if text_meta_json else 'None'}. "
        f"Building dataset index may take a while..."
    )
    config = Config10Dataset(
        dataset_name=dataset_name,
        seed=seed,
        shots=shots,
        backbone=backbone,
        lr=lr,
        batch_size=batch_size,
        train_epoch=train_epoch,
    )
    Tools.print({"mode": "single_dataset", "detail": config.get_detail()})
    Tools.print(
        f"Dataset ready: train={len(config.dataset.train_x)}, val={len(config.dataset.val)}, "
        f"test={len(config.dataset.test)}, classes={config.num_classes}"
    )
    runner = Runner(config=config)
    Tools.print(
        f"Progress report files: txt={runner.report_txt_path}, jsonl={runner.report_jsonl_path}"
    )
    if runner.save_all_to_vis and runner.exp_dir:
        Tools.print(f"Experiment dir (all outputs): {runner.exp_dir}")
    if runner.save_ckpt:
        Tools.print(f"Checkpoint dir: {runner.ckpt_dir}")
    if runner.auto_plot:
        Tools.print(f"Visualization dir: {runner.vis_output_dir}")
    acc_list = runner.train()
    Tools.print({"name": dataset_name, "acc": acc_list, "detail": config.get_detail()})


if __name__ == '__main__':
    run_single_dataset()
