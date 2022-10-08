import os

import PIL.Image as Image
import numpy as np
import torch
from torchvision import transforms
import random


from data_process.image_sampler import Bicubic
from data_process.ori_bicubic import bicubic
from data_process.process import save_tensor2imgs

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


sampler = Bicubic(22, normal_bicubic=False).cuda()
# sampler_ori = bicubic()

path = "/data/users/luluzhang/datasets/DIV2K/DIV2K_train_HR"
imgs_name = os.listdir(path)
imgs_name.sort()
imgs_name = [os.path.join(path, name) for name in imgs_name[9:10]]
for img_name in imgs_name:
    img = Image.open(img_name)
    img = np.array(img)
    img_hr = torch.unsqueeze(transform(img), dim=0).cuda()
    _, _, h, w = img_hr.shape
    scaleH, scaleW = random.uniform(0.19, 0.4), random.uniform(0.19, 0.4)
    scaleH, scaleW = 0.21, 0.21
    print(img_name, scaleH, scaleW)
    lowH, lowW = int(h * scaleH), int(w * scaleW)
    img_lr, kernels = sampler(img_hr, [lowH, lowW])
    # img_lr_bicubic, _ = sampler_ori(img_hr, [lowH, lowW])
    # print(img_hr.shape, torch.max(img_hr), torch.min(img_hr))
    # print(img_lr.shape, torch.max(img_lr), torch.min(img_lr))

    single_name = os.path.basename(img_name).split(".")[0]
    save_tensor2imgs(img_lr, "tmp_degrade", flag="{}_lr".format(single_name))
    # save_tensor2imgs(img_lr_bicubic, "tmp_degrade", flag="{}_lr_bicubic".format(single_name))
    save_tensor2imgs(img_hr, "tmp_degrade", flag="{}_hr".format(single_name))

