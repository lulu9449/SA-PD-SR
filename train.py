import datetime
import os
import random

import torch
from torch.nn import DataParallel
from torch.optim import lr_scheduler

from Network.mainNet import build_model
from data_process.data_loader import build_dataloader
from data_process.image_sampler import build_sampler
from data_process.ori_bicubic import build_sampler as bicubic_ori
from data_process.process import save_tensor2imgs, info_log
from loss.loss_function import build_loss


class Trainer(object):

    def __init__(self,
                 trainset_dirname,
                 batch_size,
                 is_train,
                 num_workers,
                 kernel_width,
                 mid_channels_sr,
                 kpc_dims,
                 implicit_dims,
                 img_channels,
                 block_num,
                 interact_num,
                 pre_trained=None
                 ):
        time_now = datetime.datetime.now()
        date, time = time_now.date(), time_now.time()
        time_log = [date.year, date.month, date.day, time.hour, time.minute, time.second]
        time_log = [str(time_log[0]), ] + [str(item).zfill(2) for item in time_log[1:]]
        time_log = "_".join(time_log)

        self.save_path = time_log
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        os.system("cp {} {}".format("/data/users/luluzhang/practice0808/train.py",
                                    os.path.join(self.save_path, "train_file.py")))

        self.log_file = open(os.path.join(self.save_path, "{}.txt".format(time_log)), "w")

        info_log(self.log_file, "INFO building trainer!\n")
        self.train_data_loader, self.step_num = build_dataloader(trainset_dirname, batch_size, self.log_file, is_train, num_workers)
        self.down_sampler = build_sampler(kernel_width)
        self.sr_model, self.en_decoder = build_model(kernel_width, mid_channels_sr, kpc_dims, implicit_dims, img_channels, block_num, interact_num, self.log_file)
        # print(type(self.sr_model.parameters()), type(self.en_decoder.parameters()))
        if pre_trained is not None:
            dirname, ep, step, mode = pre_trained
            self.load_pth(dirname, ep, step, mode)
        info_log(self.log_file, "INFO sr model has {} parameters!\n".format(sum(param.numel() for param in self.sr_model.parameters())))
        info_log(self.log_file, "INFO kpc model has {} parameters!\n".format(sum(param.numel() for param in self.en_decoder.parameters())))
        self.sr_model = DataParallel(self.sr_model)
        self.en_decoder = DataParallel(self.en_decoder)
        self.down_sampler = DataParallel(self.down_sampler)
        info_log(self.log_file, "INFO model parallelled!\n")
        self.loss_function = build_loss()
        self.optimizer_sr = torch.optim.Adam(self.sr_model.parameters(), lr=0.00015)
        # optimazer for kernel re-construction
        self.optimizer_krc = torch.optim.Adam(self.en_decoder.parameters(), lr=0.00015)
        self.optimizer_all = torch.optim.Adam([
            {'params': self.sr_model.parameters()},
            {'params': self.en_decoder.parameters()}
        ], lr=0.00015)

        self.bicubic_ori = bicubic_ori(kernel_width)
        self.bicubic_ori = DataParallel(self.bicubic_ori)

        info_log(self.log_file, "INFO trainer build!\n")

    def save_pth(self, ep, step):
        sr_path = os.path.join(self.save_path, "sr_{}_{}.pth".format(ep, step))
        kpc_path = os.path.join(self.save_path, "kpc_{}_{}.pth".format(ep, step))
        torch.save(self.sr_model, sr_path)
        info_log(self.log_file, "INFO sr model saved to {}!\n".format(sr_path))
        torch.save(self.en_decoder, kpc_path)
        info_log(self.log_file, "INFO kpc model saved to {}!\n".format(kpc_path))

    def load_pth(self, path, ep, step, load_mode=3):
        if load_mode in [1, 3]:
            kpc_path = os.path.join(path, "kpc_{}_{}.pth".format(ep, step))
            self.en_decoder.load_state_dict(torch.load(kpc_path).module.state_dict())
            info_log(self.log_file, "INFO kpc model loaded from {}!\n".format(kpc_path))
        if load_mode in [2, 3]:
            sr_path = os.path.join(path, "sr_{}_{}.pth".format(ep, step))
            self.sr_model.load_state_dict(torch.load(sr_path).module.state_dict())
            info_log(self.log_file, "INFO sr model loaded from {}!\n".format(sr_path))

    def down_sample(self, hr_imgs):
        bs, cn, highH, high_W = hr_imgs.shape
        scaleH, scaleW = random.uniform(0.19, 0.7), random.uniform(0.19, 0.7)
        # scaleH, scaleW = 0.5, 0.5
        lowH, lowW = int(highH * scaleH), int(high_W * scaleW)
        lr_imgs, kernels = self.down_sampler(hr_imgs, [lowH, lowW])
        # lr_imgs = torch.nn.functional.interpolate(hr_imgs, [lowH, lowW], mode='bicubic', align_corners=True)
        # lr_imgs, _ = self.bicubic_ori(hr_imgs, [lowH, lowW])
        lr_imgs = torch.clip(lr_imgs, -1.0, 1.0)
        kernels = kernels.permute(0, 4, 2, 3, 1)[..., 0]
        return lr_imgs, kernels

    def train(self, start_ep, num_ep, mode="2"):
        if mode not in ["1", "2", "3"]:
            info_log(self.log_file, "ERROR mode must in [1, 2, 3]")
            return

        optimizer_cur = self.optimizer_krc if mode in ["1"] else self.optimizer_all
        optimizer_cur = self.optimizer_sr if mode in ["2"] else optimizer_cur

        scheduler_cur = lr_scheduler.StepLR(optimizer_cur, step_size=self.step_num // 22, gamma=0.9)

        info_log(self.log_file, "INFO begin training!\n")
        for epoch in range(start_ep, start_ep + num_ep):
            info_log(self.log_file, "\nINFO current epoch {} !\n".format(epoch))
            for batch_id, hr_imgs in enumerate(self.train_data_loader):
                optimizer_cur.zero_grad()

                hr_imgs = hr_imgs.cuda()
                lr_imgs, kernels = self.down_sample(hr_imgs)

                kernel_pc, kernel_rc = self.en_decoder(kernels)
                kernel_pc = kernel_pc.detach()

                sr_imgs, kernel_estimated = self.sr_model.forward(lr_imgs, [hr_imgs.shape[-2], hr_imgs.shape[-1]])

                loss_sr = self.loss_function(sr_imgs, hr_imgs)
                loss_ke = self.loss_function(kernel_estimated, kernel_pc)
                loss_krc = self.loss_function(kernel_rc, kernels)

                loss_sr_ke = loss_sr + loss_ke
                loss_all = loss_sr_ke + loss_krc

                # 1 for kpc en-decoder; 2 for sr and kernel estimation; 3 for the whole model
                loss_cur = loss_krc if mode in ["1"] else loss_all
                loss_cur = loss_sr_ke if mode in ["2"] else loss_cur

                loss_cur.backward()
                optimizer_cur.step()
                scheduler_cur.step()
                if (batch_id + 1) % 5 == 0:
                    loss_list = [loss_sr, loss_ke, loss_krc, loss_sr_ke, loss_all]
                    loss_list = [item.detach() if item != 0 else 0. for item in loss_list]
                    # loss_names = ["loss_sr", "loss_ke", "loss_krc", "loss_sr_ke", "loss_all"]
                    # loss_dict = dict(zip(loss_names, loss_list))
                    info_log(self.log_file, "INFO ep{} step{}: loss sr:{:.4},ke:{:.4},krc:{:.4},sr&ke:{:.4},sr&ke&krc:{:.4}!".format(epoch, batch_id, *loss_list))
                    # print(torch.max(sr_imgs).detach().cpu().numpy(), torch.min(sr_imgs).detach().cpu().numpy())
                    print(scheduler_cur.get_last_lr())
                    bs, cn, ho, wo = hr_imgs.shape
                    lr_imgs_up = torch.nn.functional.interpolate(lr_imgs, [ho, wo], mode='bicubic', align_corners=True)
                    bicubic_loss = self.loss_function(lr_imgs_up, hr_imgs).detach().cpu().numpy()
                    print(epoch, batch_id, loss_list[0], bicubic_loss)
                    if (batch_id + 1) % 40 == 0:
                        # save_tensor2imgs(lr_imgs, os.path.join(self.save_path, "train_imgs"), "lr")
                        save_tensor2imgs(torch.cat([lr_imgs_up, sr_imgs, hr_imgs], dim=3), os.path.join(self.save_path, "train_imgs"), "lr_sr_hr")
                if (batch_id + 1) % (self.step_num // 2) == 0:
                    self.save_pth(epoch, batch_id)

            # break

            self.save_pth(epoch, batch_id)

    def close_log(self):
        self.log_file.close()


if __name__ == "__main__":

    trainset_dirname = "/data/users/luluzhang/datasets/DIV2K/DIV2K_train_HR_p"
    batch_size = 28
    is_train = True
    num_workers = 4
    kernel_width = 22
    mid_channels_sr = 32
    kpc_dims = 10
    implicit_dims = 24
    img_channels = 3
    block_num = 4
    interact_num = 4
    pre_trained = None
    # pre_trained = ["2022_09_02_12_33_31", 1, 5214, 3]
    # pre_trained = ["2022_09_02_16_14_30", 11, 5214, 1]  # 周末训练结果
    # pre_trained = ["2022_09_05_14_19_24", 0, 10428, 2]
    # pre_trained = ["2022_09_05_16_49_10", 10, 10428, 1]
    pre_trained = ["2022_09_21_02_55_08", 1, 4344, 3] #


    trainer = Trainer(trainset_dirname,
                      batch_size,
                      is_train,
                      num_workers,
                      kernel_width,
                      mid_channels_sr,
                      kpc_dims,
                      implicit_dims,
                      img_channels,
                      block_num,
                      interact_num,
                      pre_trained
                      )

    # trainer.train(0, 1, mode="1")
    trainer.train(1, 10, mode="2")
    trainer.close_log()

