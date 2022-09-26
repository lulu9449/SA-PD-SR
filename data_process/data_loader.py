import os

import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from data_process.process import info_log


img_suffix = ["jpg", "png", "PNG", "JPEG", "bmp", "JPG"]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_imgs_path(dirpath):
    imgs_name = os.listdir(dirpath)  # [:200]
    names = []
    for img_name in imgs_name:
        path_abs = os.path.join(dirpath, img_name)
        if not os.path.exists(path_abs):
            continue
        if img_name.strip().split(".")[-1] in img_suffix:
            names.append(os.path.join(dirpath, img_name))
    names.sort()
    return names


class TrainSet(Dataset):
    def __init__(self, hr_imgs_path, transform):
        imgs_res = get_imgs_path(hr_imgs_path)  #[:800]
        imgs_res.sort()
        self.items = imgs_res  #[:308]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path = self.items[index]
        hr_img = Image.open(img_path)
        hr_img = np.array(hr_img)
        hr_img = self.transform(hr_img)
        return hr_img


class TestSet(Dataset):
    def __init__(self, hr_lr_path, transform):
        hr_path, lr_path = hr_lr_path
        hr_imgs = get_imgs_path(hr_path)
        lr_imgs = get_imgs_path(lr_path)
        if len(hr_imgs) != len(lr_imgs):
            raise ValueError("img nums not equal in {} and {}!".format(hr_path, lr_path))
        self.items = list(zip(hr_imgs, lr_imgs))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def read_img(self, path):
        img = Image.open(path)
        img = np.array(img)
        return self.transform(img)

    def __getitem__(self, index):
        hr_path, lr_path = self.items[index]
        hr_img, lr_img = [self.read_img(path) for path in [hr_path, lr_path]]
        return hr_img, lr_img, lr_path


def build_dataloader(dataset_dirname, batch_size, log_file, is_train=True, num_workers=2):
    if is_train:
        dataset = TrainSet(dataset_dirname, transform)
    else:
        dataset = TestSet(dataset_dirname, transform)
    data_len = dataset.__len__()
    step_num = data_len // batch_size + 1 if data_len % batch_size != 0 else data_len // batch_size
    info_log(log_file, "INFO building dataloader, length of dataset is {}, step num is {}!\n".format(data_len, step_num))
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers), step_num




