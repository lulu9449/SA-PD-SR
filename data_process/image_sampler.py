import torch
import torch.nn as nn
import numpy as np
from data_process.process import save_kernel2imgs


def get_indice(in_size, out_size, kernel_width):

    x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
    x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

    scale0 = float(out_size[0]) / float(in_size[0])
    scale1 = float(out_size[1]) / float(in_size[1])

    u0 = x0 / scale0 + 0.5 * (1 - 1 / scale0)
    u1 = x1 / scale1 + 0.5 * (1 - 1 / scale1)

    left0 = torch.floor(u0 - kernel_width / 2)
    left1 = torch.floor(u1 - kernel_width / 2)

    P = np.ceil(kernel_width) + 2

    indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
    indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

    offset0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
    offset1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

    indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
    indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

    kernel_h = indice0.shape[2]
    kernel_w = indice1.shape[2]

    indice_h, indice_w = torch.meshgrid(torch.reshape(indice0, [-1]), torch.reshape(indice1, [-1]))
    indice = torch.cat([torch.unsqueeze(indice_h, -1), torch.unsqueeze(indice_w, -1)], dim=-1)
    indice = torch.reshape(indice, [-1, out_size[0], kernel_h, out_size[1], kernel_w, 2])
    indice = indice.permute(0, 1, 3, 2, 4, 5)
    return indice[0:1, ...], [offset0, offset1]


def cubic(offset):
    absx = torch.abs(offset)
    # print(torch.max(absx), torch.min(absx))
    absx2 = offset * offset
    absx3 = absx * absx2

    condition1 = (absx <= 1).to(torch.float32)
    condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)
    condition3 = (absx > 2).to(torch.float32)

    f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2 + (0.0) * condition3
    # print(torch.max(f), torch.min(f))
    return f


def get_sin_clip(step, params_h, params_w, limit_low=None, limit_high=None):
    batch_size = params_h.shape[0]
    step_h, step_w = step
    resh = torch.sin((step_h * params_h[:, 1:2] + params_h[:, 2:3]) * 2 * 3.1415926) * params_h[:, 0:1] + params_h[:, 3:] + (params_h[:, 0:1] * 0.5)
    resw = torch.sin((step_w * params_w[:, 1:2] + params_w[:, 2:3]) * 2 * 3.1415926) * params_w[:, 0:1] + params_w[:, 3:] + (params_w[:, 0:1] * 0.5)
    resHW_list = [torch.meshgrid(resh[i], resw[i]) for i in range(batch_size)]
    resHW_list = [torch.unsqueeze(torch.cat([torch.unsqueeze(resH, dim=-1), torch.unsqueeze(resW, dim=-1)], dim=-1), dim=0) for resH, resW in resHW_list]
    res = torch.cat(resHW_list, dim=0)
    res = torch.mean(res, dim=-1, keepdim=True)
    # print(res.shape)
    if limit_low is not None:
        res = torch.clip(res, min=limit_low)
    if limit_high is not None:
        res = torch.clip(res, max=limit_high)
    res = torch.reshape(res, [batch_size, 1, step_h.shape[0], step_w.shape[0], 1, 1, 1])
    return res


def get_rand_scale_theta(batch_size, out_size):
    step_h = torch.linspace(0., 1., steps=out_size[0])
    step_w = torch.linspace(0., 1., steps=out_size[1])

    bias_scale = torch.Tensor([[[0, 0.1, 0, 0.1]]])
    factor_scale = torch.Tensor([[[1.0, 5.0, 1.0, 1.0]]])

    scaleh_param = torch.rand([2, batch_size, 4]) * factor_scale + bias_scale
    scalew_param = torch.rand([2, batch_size, 4]) * factor_scale + bias_scale

    scaleh = get_sin_clip([step_h, step_w], scaleh_param[0, ...], scalew_param[0, ...], 0.1, 1.6)
    scalew = get_sin_clip([step_h, step_w], scaleh_param[1, ...], scalew_param[1, ...], 0.1, 1.6)

    bias_theta = torch.Tensor([[[0, 0.1, 0, 0.1]]])
    factor_theta = torch.Tensor([[[3.1416 / 2., 2.0, 1.0, 3.1416 / 2]]])
    theta_paramH = torch.rand([1, batch_size, 4]) * factor_theta + bias_theta
    theta_paramW = torch.rand([1, batch_size, 4]) * factor_theta + bias_theta
    thetaHW = get_sin_clip([step_h, step_w], theta_paramH[0], theta_paramW[0], limit_low=0, limit_high=3.1415926)
    # print(scaleh.shape, scalew.shape, thetaHW.shape)

    return scaleh.cuda(), scalew.cuda(), thetaHW.cuda()



def scale_rotate_offset(h, w, batch_size, out_size, scale):
    if scale is not None:
        scaleh = torch.ones([batch_size, 1, out_size[0], out_size[1], 1, 1, 1]).cuda()*scale[0]
        scalew = torch.ones([batch_size, 1, out_size[0], out_size[1], 1, 1, 1]).cuda()*scale[1]
        theta_rotate = torch.zeros([batch_size, 1, out_size[0], out_size[1], 1, 1, 1]).cuda()
    else:
        scaleh, scalew, theta_rotate = get_rand_scale_theta(batch_size, out_size)
    h_new = h * scaleh
    w_new = w * scalew
    kernel_h, kernel_w = h_new.shape[4], h_new.shape[5]
    offset_hw = torch.unsqueeze(torch.cat([h_new, w_new], dim=-1), dim=-1)
    sin_theta = torch.sin(theta_rotate)
    cos_theta = torch.cos(theta_rotate)
    rotate_mat = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
    rotate_mat = torch.reshape(rotate_mat, [-1, 1, out_size[0], out_size[1], 1, 1, 2, 2])
    offset_hw = torch.reshape(torch.matmul(rotate_mat, offset_hw), [-1, 1, out_size[0], out_size[1], kernel_h, kernel_w, 2])

    return offset_hw[..., 0], offset_hw[..., 1]


def get_weight(out_size, batch_size, offset, sample_wise=False, scale=None):

    offset0, offset1 = offset
    kernel_h = offset0.shape[2]
    kernel_w = offset1.shape[2]

    offset_h, offset_w = torch.meshgrid(torch.reshape(offset0, [-1]), torch.reshape(offset1, [-1]))
    offset_hw = torch.cat([torch.unsqueeze(offset_h, -1), torch.unsqueeze(offset_w, -1)], dim=-1)
    offset_hw = torch.reshape(offset_hw, [-1, out_size[0], kernel_h, out_size[1], kernel_w, 2])
    offset_hw = offset_hw.permute(0, 1, 3, 2, 4, 5).cuda()

    mini_batch = batch_size if sample_wise else 1

    offset_h, offset_w = scale_rotate_offset(offset_hw[..., 0:1], offset_hw[..., 1:], mini_batch, out_size, scale=scale)

    # print(offset_h.shape, offset_w.shape)
    weight_h, weight_w = cubic(offset_h), cubic(offset_w)
    weight_mat = weight_h * weight_w

    weight_sum = torch.sum(weight_mat, dim=[-1, -2], keepdim=True)
    weight_mat = weight_mat / weight_sum

    if not sample_wise:
        weight_mat = weight_mat.repeat(batch_size, 1, 1, 1, 1, 1, 1)
    # print(weight_h.shape, weight_w.shape, weight_mat.shape)
    # print(torch.sum(weight_mat, dim=[-1, -2]))

    return weight_mat


def slice_mul(gpu_tensor, offset, scale):
    b, c, o_h, o_w, k_s_h, k_s_w = gpu_tensor.shape
    kernel_size = k_s_h * k_s_w
    batch_size = 16
    batch_num = int(b // batch_size) + 1 if b % batch_size > 0 else int(b // batch_size)
    batch_res_list = list()
    batch_weight_list = list()
    for id in range(batch_num):
        cur_gpu_tensor = gpu_tensor[id * batch_size: (id + 1) * batch_size]
        cur_gpu_tensor = torch.reshape(cur_gpu_tensor, [-1, c, o_h, o_w, kernel_size])

        cur_weight_mat = get_weight([o_h, o_w], cur_gpu_tensor.shape[0], offset, scale=scale)
        cur_weight_mat = torch.reshape(cur_weight_mat, [-1, 1, o_h, o_w, kernel_size])
        batch_weight_list.append(cur_weight_mat)
        # print(torch.max(cur_weight_mat), torch.min(cur_weight_mat))

        cur_res = torch.mul(cur_gpu_tensor, cur_weight_mat)
        cur_res = torch.sum(cur_res, dim=-1, keepdim=False)
        batch_res_list.append(cur_res)
    res = torch.cat(batch_res_list, dim=0)
    kernel = torch.cat(batch_weight_list, dim=0)

    return res, kernel



class Bicubic(nn.Module):

    def __init__(self, kernel_width, normal_bicubic=False):
        super(Bicubic, self).__init__()
        self.kernel_width = kernel_width
        self.normal_bicubic = normal_bicubic

    def forward(self, input, output_hw):
        [_, _, h, w] = input.shape

        indice, offset = get_indice([h, w], output_hw, self.kernel_width)
        scale_h, scale_w = float(output_hw[0]) / float(h), float(output_hw[1]) / float(w)
        scale = [scale_h, scale_w] if self.normal_bicubic else None

        indice = np.asarray(indice[0] - 1, dtype=np.float32)
        indice = torch.from_numpy(indice).cuda().long()
        selected_pixels = input[:, :, indice[..., 0], indice[..., 1]]

        res, kernel = slice_mul(selected_pixels, offset, scale)

        return res, kernel


def build_sampler(kernel_width):
    return Bicubic(kernel_width, normal_bicubic=True).cuda()

