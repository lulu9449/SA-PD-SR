import torch
import torch.nn as nn
from data_process.process import rgb2yuv


class SRloss(nn.Module):
    def __init__(self, mode='l2'):
        super(SRloss, self).__init__()
        self.loss_function = None
        if mode == 'l2':
            self.loss_function = torch.nn.MSELoss()
        elif mode == 'l1':
            self.loss_function = torch.nn.L1Loss()
    def forward(self, SR, HR, yuv=False):
        if yuv:
            SR = rgb2yuv(SR)
            HR = rgb2yuv(HR)
        return self.loss_function(SR, HR)

def build_loss():
    return SRloss(mode='l1').cuda()


