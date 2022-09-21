import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio


class bicubic(nn.Module):

    def __init__(self, kernel_width):
        super(bicubic, self).__init__()
        self.kernel_width = kernel_width

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale_hw):
        kernel_width = self.kernel_width
        # if scale < 1:
        #     kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale_hw[0] + 0.5 * (1 - 1 / scale_hw[0])
        u1 = x1 / scale_hw[1] + 0.5 * (1 - 1 / scale_hw[1])

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale_hw[0] < 1 or scale_hw[1] < 1:
            mid0 = mid0 * scale_hw[0]
            mid1 = mid1 * scale_hw[1]
            # print(mid0.shape, mid1.shape)
            weight0 = scale_hw[0] * self.cubic(mid0)
            weight1 = scale_hw[1] * self.cubic(mid1)

        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, output_size):
        [b, c, h, w] = input.shape
        # output_size = [b, c, int(h * scale), int(w * scale)]
        scaleh = float(output_size[0]) / float(h)
        scalew = float(output_size[1]) / float(w)

        weight0, weight1, indice0, indice1 = self.contribute([h, w], output_size, [scaleh, scalew])

        weight0 = np.asarray(weight0[0], dtype=np.float32)
        weight0 = torch.from_numpy(weight0).cuda()

        indice0 = np.asarray(indice0[0], dtype=np.float32)
        indice0 = torch.from_numpy(indice0).cuda().long()
        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        weight1 = np.asarray(weight1[0], dtype=np.float32)
        weight1 = torch.from_numpy(weight1).cuda()

        indice1 = np.asarray(indice1[0], dtype=np.float32)
        indice1 = torch.from_numpy(indice1).cuda().long()
        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.round(255 * torch.sum(out, dim=3).permute(0, 1, 3, 2)) / 255

        return out, [weight0, weight1]


def build_sampler(kernel_width):
    return bicubic(kernel_width).cuda()
