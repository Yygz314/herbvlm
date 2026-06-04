import random
import os.path as osp
from collections import defaultdict
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class Datum:
    """Single data sample descriptor."""

    def __init__(self, impath: str = "", label: int = 0, domain: int = -1, classname: str = ""):
        self._impath = impath
        self._label = int(label)
        self._domain = int(domain)
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """Minimal dataset base used by HerbVLM."""

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x or []
        self._train_u = train_u or []
        self._val = val or []
        self._test = test or []

        self._num_classes = self.get_num_classes(self._train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(self._train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classnames(self):
        return self._classnames

    @staticmethod
    def get_num_classes(data_source: List[Datum]):
        if not data_source:
            return 0
        labels = {item.label for item in data_source}
        return max(labels) + 1

    @staticmethod
    def get_lab2cname(data_source: List[Datum]) -> Tuple[dict, list]:
        container = {}
        for item in data_source:
            container[item.label] = item.classname
        labels = sorted(container.keys())
        classnames = [container[label] for label in labels]
        return container, classnames

    @staticmethod
    def _split_dataset_by_label(data_source: List[Datum]):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        if num_shots < 1:
            return data_sources[0] if len(data_sources) == 1 else data_sources

        output = []
        for data_source in data_sources:
            dataset = []
            tracker = self._split_dataset_by_label(data_source)
            for _, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = random.choices(items, k=num_shots) if repeat else items
                dataset.extend(sampled_items)
            output.append(dataset)

        return output[0] if len(output) == 1 else output


def read_image(path):
    if not osp.exists(path):
        raise FileNotFoundError(f"No file exists at {path}")
    return Image.open(path).convert("RGB")


class DatasetWrapper(TorchDataset):
    """Wrap Datum list into a torch Dataset."""

    def __init__(self, data_source, input_size=224, transform=None, is_train=False):
        self.data_source = data_source
        self.input_size = input_size
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = read_image(item.impath)
        if self.transform is not None:
            image = self.transform(image)
        return image, item.label
