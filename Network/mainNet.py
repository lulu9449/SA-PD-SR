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

