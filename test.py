import datetime
import os

import numpy as np
import torch
from torch.nn import DataParallel

from Network.mainNet import build_model
from data_process.data_loader import build_dataloader
from data_process.process import calculate_psnr_ssim
from data_process.process import info_log


class Tester(object):

    def __init__(self,
                 hrset_dirname,
                 lrset_dirnames,
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
        self.hr_dirname = hrset_dirname
        self.lr_dirnames = lrset_dirnames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train
        self.save_path = time_log if pre_trained is None else pre_trained[0]
        if not os.path.exists(self.save_path):
            info_log(self.log_file, "ERROR there is no directory named {}!".format(self.save_path))
        os.system("cp {} {}".format("/data/users/luluzhang/practice0808/test.py",
                                    os.path.join(self.save_path, "test_file.py")))

        self.log_file = open(os.path.join(self.save_path, "{}_test.txt".format(time_log)), "w")

        info_log(self.log_file, "INFO building tester!\n")
        self.sr_model, self.en_decoder = build_model(kernel_width, mid_channels_sr, kpc_dims, implicit_dims, img_channels, block_num, interact_num, self.log_file)
        # print(type(self.sr_model.parameters()), type(self.en_decoder.parameters()))
        if pre_trained is not None:
            self.load_pth(*pre_trained)

        self.sr_model.eval()
        self.sr_model = DataParallel(self.sr_model)
        self.en_decoder.eval()
        self.en_decoder = DataParallel(self.en_decoder)
        info_log(self.log_file, "INFO model parallelled!\n")
        info_log(self.log_file, "INFO tester build!\n")

    def load_pth(self, path, ep, step, load_mode=3):
        if load_mode in [1, 3]:
            kpc_path = os.path.join(path, "kpc_{}_{}.pth".format(ep, step))
            self.en_decoder.load_state_dict(torch.load(kpc_path).module.state_dict())
            info_log(self.log_file, "INFO kpc model loaded from {}!\n".format(kpc_path))
        if load_mode in [2, 3]:
            sr_path = os.path.join(path, "sr_{}_{}.pth".format(ep, step))
            self.sr_model.load_state_dict(torch.load(sr_path).module.state_dict())
            info_log(self.log_file, "INFO sr model loaded from {}!\n".format(sr_path))

    def test(self):
        info_log(self.log_file, "INFO begin testing!\n")
        for lr_dirname in self.lr_dirnames:
            test_data_loader, _ = build_dataloader([self.hr_dirname, lr_dirname],
                                                   self.batch_size,
                                                   self.log_file,
                                                   self.is_train,
                                                   self.num_workers)
            ours_psnr_list, ours_ssim_list, cubic_psnr_list, cubic_ssim_list = [], [], [], []
            for batch_id, hr_lr_imgs in enumerate(test_data_loader):
                hr_imgs, lr_imgs = hr_lr_imgs[0].cuda(), hr_lr_imgs[1].cuda()
                lr_path = hr_lr_imgs[2]
                ori_shape = [lr_imgs.shape[2], lr_imgs.shape[3], lr_imgs.shape[1]]
                dst_shape = [hr_imgs.shape[2], hr_imgs.shape[3], hr_imgs.shape[1]]
                shape_list = ori_shape + dst_shape
                info_log(self.log_file, "INFO processing image {} from {}*{}*{} to {}*{}*{}!\n".format(lr_path, *shape_list))
                with torch.no_grad():
                    sr_imgs, estimated_kpc = self.sr_model.forward(lr_imgs, [dst_shape[0], dst_shape[1]])
                    # print(sr_imgs.shape, hr_imgs.shape)
                    print(estimated_kpc.shape)
                    print(estimated_kpc[0, 0, :10, :10])
                    from data_process.process import save_tensor2imgs
                    save_tensor2imgs(torch.unsqueeze(torch.squeeze(estimated_kpc), dim=1), dir_path="tmp_ekpc", flag="ekpc", auto_scale=True)
                    lr_imgs_up = torch.nn.functional.interpolate(lr_imgs, [dst_shape[0], dst_shape[1]], mode='bicubic', align_corners=True)
                    # print(hr_imgs.shape, hr_imgs.dtype, torch.min(hr_imgs), torch.max(hr_imgs), type(hr_imgs))

                    # lr_sr_hr = torch.cat([lr_imgs_up, sr_imgs, hr_imgs], dim=-1)
                    # save_tensor2imgs(lr_sr_hr, dir_path=os.path.join(self.save_path, "test_res"), flag="{}_{}_lr_sr_hr".format(os.path.basename(lr_dirname), batch_id))
                    # save_tensor2imgs(lr_imgs, dir_path=os.path.join(self.save_path, "test_res"), flag="lr_{}_{}".format(os.path.basename(lr_dirname), batch_id))

                    hr_imgs = np.array(np.clip((hr_imgs[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5, 0.0, 255.0), dtype=np.uint8)
                    sr_imgs = np.array(np.clip((sr_imgs[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5, 0.0, 255.0), dtype=np.uint8)
                    lr_imgs_up = np.array(np.clip((lr_imgs_up[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5, 0.0, 255.0), dtype=np.uint8)
                    ours_psnr, ours_ssim = calculate_psnr_ssim(sr_imgs, hr_imgs)
                    cubic_psnr, cubic_ssim = calculate_psnr_ssim(lr_imgs_up, hr_imgs)
                    cubic_psnr_list.append(cubic_psnr)
                    cubic_ssim_list.append(cubic_ssim)
                    ours_psnr_list.append(ours_psnr)
                    ours_ssim_list.append(ours_ssim)
                    info_log(self.log_file, "[CUBIC] psnr:{}, ssim:{}\n".format(cubic_psnr, cubic_ssim))
                    info_log(self.log_file, "[OURS] psnr:{}, ssim:{}\n".format(ours_psnr, ours_ssim))
            info_log(self.log_file, "{} test done!\n".format(lr_dirname))
            info_log(self.log_file, "[CUBIC] average psnr:{}, average ssim:{}\n".format(sum(cubic_psnr_list) / len(cubic_psnr_list),
                                                                    sum(cubic_ssim_list) / len(cubic_ssim_list)))
            info_log(self.log_file, "[OURS] average psnr:{}, average ssim:{}\n".format(sum(ours_psnr_list) / len(ours_psnr_list),
                                                                   sum(ours_ssim_list) / len(ours_ssim_list)))
            info_log(self.log_file, "\n")


    def close_log(self):
        self.log_file.close()


if __name__ == "__main__":

    # hrset_dirname = "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_HR"
    # lrset_dirname = ["/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_bicubic/X4",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_bicubic/X3",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_bicubic/X2",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_difficult",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_mild",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_unknown/X2",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_unknown/X3",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_unknown/X4",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_wild",
    #                  "/data/users/luluzhang/datasets/DIV2K/DIV2K_valid_LR_x8",
    #                  ]
    hrset_dirname = "/data/users/luluzhang/datasets/DIV2K/img_test/hr"
    lrset_dirname = ["/data/users/luluzhang/datasets/DIV2K/img_test/lr"]
    batch_size = 1
    is_train = False
    num_workers = 4
    kernel_width = 22
    mid_channels_sr = 32
    kpc_dims = 10
    implicit_dims = 24
    img_channels = 3
    block_num = 4
    interact_num = 4
    pre_trained = None
    pre_trained = ["2022_09_30_01_19_28", 1, 4344, 3]


    tester = Tester(hrset_dirname,
                    lrset_dirname,
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

    tester.test()
    tester.close_log()

