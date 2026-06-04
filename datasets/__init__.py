from .chinese_medicine_163 import ChineseMedicine163


dataset_list = {
    "chinese_medicine_163": ChineseMedicine163,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
