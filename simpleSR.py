import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
# from torchvision import transforms
from torch.utils.data import DataLoader
from data_process.ori_bicubic import build_sampler as bicubic_ori_function
from data_process.image_sampler import build_sampler
from data_process.process import save_tensor2imgs
from Network.TAUS import TAUS
import random

from data_process.data_loader import TrainSet, transform

kernel_width = 22
bicubic_ori = bicubic_ori_function(kernel_width)
bicubic_ori = DataParallel(bicubic_ori)
down_sampler = build_sampler(kernel_width)
down_sampler = DataParallel(down_sampler)

def down_sample(hr_imgs):

    bs, cn, highH, high_W = hr_imgs.shape
    scaleH, scaleW = random.uniform(0.19, 0.7), random.uniform(0.19, 0.7)
    # scaleH, scaleW = 0.5, 0.5
    lowH, lowW = int(highH * scaleH), int(high_W * scaleW)
    lr_imgs, kernels = down_sampler(hr_imgs, [lowH, lowW])
    # lr_imgs = torch.nn.functional.interpolate(hr_imgs, [lowH, lowW], mode='bicubic', align_corners=True)
    # lr_imgs, _ = bicubic_ori(hr_imgs, [lowH, lowW])
    lr_imgs = torch.clip(lr_imgs, -1.0, 1.0)
    # kernels = kernels.permute(0, 4, 2, 3, 1)[..., 0]
    return lr_imgs


class SRnet(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=16, scale_factor=2):
        super(SRnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(mid_channel + in_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(mid_channel + in_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv3_1 = nn.Conv2d(mid_channel, 1, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.convGlob3_2 = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(mid_channel * 3 + in_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.scale_factor = scale_factor
        self.upscale = TAUS(mid_channel, out_channel, 10)
    def forward(self, lr_imgs, dst_size):
        bs, cn, hi, wi = lr_imgs.shape
        texture = self.conv1(lr_imgs)
        texture = self.conv2(torch.cat([lr_imgs, texture], dim=1))
        texture = self.conv3(torch.cat([lr_imgs, texture], dim=1))
        texture1 = self.conv3_1(texture)
        texture2 = self.convGlob3_2(texture)
        texture = torch.cat([texture, texture * texture1, texture * texture2], dim=1)
        texture = self.conv4(torch.cat([lr_imgs, texture], dim=1))
        texture = self.conv5(texture)
        # sr_imgs = torch.reshape(texture, [bs, cn, self.scale_factor, self.scale_factor, hi, wi])
        # sr_imgs = sr_imgs.permute(0, 1, 4, 2, 5, 3)
        # sr_imgs = torch.reshape(sr_imgs, [bs, cn, hi * self.scale_factor, wi * self.scale_factor])
        sr_imgs = self.upscale(texture, None, dst_size)
        return sr_imgs

trainset_dirname = "/data/users/luluzhang/datasets/DIV2K/DIV2K_train_HR_p"
trainset = TrainSet(trainset_dirname, transform)
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

scale_factor = 2
sr_model = SRnet(3, 3, mid_channel=32, scale_factor=scale_factor).cuda()
sr_model = DataParallel(sr_model)

optimizer_sr = torch.optim.Adam(sr_model.parameters(), lr=0.0005)
scheduler_sr = lr_scheduler.ExponentialLR(optimizer_sr, gamma=0.999)

# loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()

for ep in range(0, 10):
    for batch_id, hr_imgs in enumerate(train_dataloader):
        optimizer_sr.zero_grad()

        hr_imgs = hr_imgs.cuda()
        _, _, hr_h, hr_w = hr_imgs.shape
        lr_imgs = down_sample(hr_imgs)
        sr_imgs = sr_model(lr_imgs, [hr_h, hr_w])

        _, _, ho, wo = hr_imgs.shape
        # print(lr_imgs.shape, hr_imgs.shape, sr_imgs.shape)
        cur_loss = loss(sr_imgs, hr_imgs)

        cur_loss.backward()
        optimizer_sr.step()
        scheduler_sr.step()

        if batch_id % 10 == 0:
            lr_imgs_up =  torch.nn.functional.interpolate(lr_imgs, [ho, wo], mode='bicubic', align_corners=True)
            save_tensor2imgs(torch.cat([lr_imgs_up, sr_imgs, hr_imgs], dim=3), "simpleSR_tmp_res", "lr_sr_hr")
            # save_tensor2imgs(lr_imgs, "simpleSR_tmp_res", "lr")
            bicubic_loss = loss(lr_imgs_up, hr_imgs).detach().cpu().numpy()
            print(ep, batch_id, cur_loss.detach().cpu().numpy(), bicubic_loss)







