import torch.nn as nn
from Network.TAUS import TAUS
from Network.interActor import initSR, interActor
from Network.KpcCoder import Encoder, Decoder
from data_process.process import info_log

class SRNetWork(nn.Module):
    def __init__(self, mid_channels, in_channels, out_channels, kpc_channels, implicit_channles, block_num, interact_num=4):
        super(SRNetWork, self).__init__()
        self.interAct_num = interact_num
        self.initSRer = initSR(in_channels, implicit_channles, mid_channels, block_num)
        self.SRer = interActor(in_channels, kpc_channels, implicit_channles, mid_channels, block_num)
        self.estimater = interActor(in_channels, implicit_channles, kpc_channels, mid_channels, block_num, res=False)
        self.taus_tail = TAUS(implicit_channles, out_channels, kpc_channels)

    def forward(self, lr_imgs, dst_size):
        self.taus_tail.eval()
        sr_texture = self.initSRer(lr_imgs)
        for _ in range(self.interAct_num):
        # for _ in range(1):
            ke_texture = self.estimater(lr_imgs, sr_texture)
            sr_texture = self.SRer(lr_imgs, ke_texture)
        sr_imgs = self.taus_tail(sr_texture, ke_texture, dst_size)

        return sr_imgs, ke_texture


# class SRNetWork_plain(nn.Module):
#     def __init__(self, mid_channels, in_channels, out_channels, kpc_channels, implicit_channles, block_num, interact_num=4):
#         super(SRNetWork_plain, self).__init__()
#         # self.sr_head = nn.Sequential(
#         #     nn.Conv2d(in_channels, mid_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         #     nn.LeakyReLU(),
#         #     nn.Conv2d(mid_channels // 2, mid_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         #     nn.LeakyReLU(),
#         #     nn.Conv2d(mid_channels // 2, implicit_channles, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         # )
#         # self.sr_head1 = nn.Sequential(
#         #     nn.Conv2d(in_channels, implicit_channles, kernel_size=1, stride=1),
#         #     nn.LeakyReLU(),
#         # )
#         # self.sr_body = nn.Sequential(
#         #     nn.Conv2d(implicit_channles, mid_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         #     nn.LeakyReLU(),
#         #     nn.Conv2d(mid_channels // 2, mid_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         #     nn.LeakyReLU(),
#         #     nn.Conv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
#         # )
#         # self.sr_body1 = nn.Sequential(
#         #     nn.Conv2d(implicit_channles, mid_channels // 4, kernel_size=1, stride=1),
#         #     nn.LeakyReLU(),
#         # )
#         # self.sr_tail = nn.Conv2d(mid_channels // 4, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
#         self.SRer_init = initSR(in_channels, implicit_channles, mid_channels, block_num)
#         self.estimater = interActor(in_channels, implicit_channles, kpc_channels, mid_channels, block_num, res=False)
#         self.upscaler = TAUS(implicit_channles, out_channels, kpc_channels)
#
#     def forward(self, lr_imgs, dst_size):
#         # sr_texture = self.sr_head(lr_imgs) + self.sr_head1(lr_imgs)
#         # ke_texture = self.estimater(lr_imgs, sr_texture)
#         # sr_texture = torch.nn.functional.interpolate(sr_texture, dst_size, mode='bicubic', align_corners=True)
#         # sr_texture = self.sr_body(sr_texture) + self.sr_body1(sr_texture)
#         # sr_imgs = self.sr_tail(sr_texture)
#         # print(sr_texture.shape, ke_texture.shape)
#         sr_texture = self.SRer_init(lr_imgs)
#         ke_texture = self.estimater(lr_imgs, sr_texture)
#         sr_imgs = self.upscaler(sr_texture, ke_texture, dst_size)
#
#         return sr_imgs, ke_texture


# lr_imgs = torch.randn([96, 3, 70, 85], dtype=torch.float32).cuda()
# model = SRNetWork(64, 3, 3, 10, 10, 4).cuda()
# sr_res, ke = model(lr_imgs, [192, 192])
# print(sr_res.shape, ke.shape)

class KPCNetWork(nn.Module):
    def __init__(self, kernel_width, kpc_channels):
        super(KPCNetWork, self).__init__()
        kernel_width = kernel_width + 2
        self.encoder = Encoder(kernel_width, kpc_channels)
        self.decoder = Decoder(kernel_width, kpc_channels)

    def forward(self, kernels):
        kpc = self.encoder(kernels)
        krc = self.decoder(kpc)
        return kpc, krc


def build_model(kernel_width, mid_channels, kpc_dims, implicit_dims, img_channels, block_num, interact_num, log_file):
    info_log(log_file, "INFO building SR interActor and en-decoder!\n")
    return SRNetWork(mid_channels, img_channels, img_channels, kpc_dims, implicit_dims, block_num, interact_num).cuda(), KPCNetWork(kernel_width, kpc_dims).cuda()

