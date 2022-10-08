import torch
from Network.TAUS import get_offset_scale
from image_sampler import scale_rotate_offset, gaussian

# test TAUS
# def show_xy(x, y, offset_scale, indice, center_hw):
#     offset_scale = offset_scale[:, x, y, :, :]
#     indice = indice[x, y, :, :]
#     center_hw = center_hw[x, y, :, :]
#     print(torch.squeeze(offset_scale).detach().cpu().numpy())
#     print(torch.squeeze(indice).detach().cpu().numpy())
#     print(torch.squeeze(center_hw).detach().cpu().numpy())
#     center_hw = torch.floor(center_hw).long()
#     print(torch.squeeze(center_hw).detach().cpu().numpy())
#     print()
#
# in_size = [10, 17]
# out_size = [20, 34]
# kernel_size = 4
# offset_scale, indice, center_hw = get_offset_scale(in_size, out_size, kernel_size)
# print(offset_scale.shape, indice.shape, center_hw.shape)
# print(offset_scale.dtype, indice.dtype, center_hw.dtype)
# print()
#
#
# x, y = 0, 0
# show_xy(x, y, offset_scale, indice, center_hw)
# x, y = 0, 1
# show_xy(x, y, offset_scale, indice, center_hw)
# x, y = 0, 2
# show_xy(x, y, offset_scale, indice, center_hw)


# test image sampler

from data_process.image_sampler import Bicubic
from data_process.process import save_kernel2imgs




sampler = Bicubic(22, normal_bicubic=False).cuda()
lr = torch.zeros([1, 3, 55, 55]).cuda()
hr, kernel = sampler(lr, [27, 29])
# print(lr.shape, hr.shape, kernel.shape)
# kernel = kernel[:,:,0:1,0:1,:]
save_kernel2imgs(kernel / torch.max(kernel) * 1.0, "tmp_kernels", "kernel")
# print(kernel[:,:,0:1,0:1,:].detach().cpu().numpy())


kernel_size = 24
indice = torch.range(0, kernel_size - 1)
# print(indice.shape)
offset = indice - (torch.max(indice) + torch.min(indice)) / 2
# print(offset)

offset_hw_h, offset_hw_w = torch.meshgrid(torch.reshape(offset, [-1]), torch.reshape(offset, [-1]))
# print(offset_hw_h.shape, offset_hw_w.shape)
offset_hw = torch.cat([torch.unsqueeze(offset_hw_h, -1), torch.unsqueeze(offset_hw_w, -1)], dim=-1)
offset_hw = torch.reshape(offset_hw, [-1, 1, kernel_size, 1, kernel_size, 2])
offset_hw = offset_hw.permute(0, 1, 3, 2, 4, 5).cuda()
# print(offset_hw.shape)


def scale_rotate_offset(h, w, batch_size, out_size, scale, theta):
    scaleh = 0.2
    scalew = 0.6
    theta_rotate = torch.Tensor([theta])
    h_new = torch.unsqueeze(h, dim=0)
    w_new = torch.unsqueeze(w, dim=0)
    kernel_h, kernel_w = h_new.shape[4], h_new.shape[5]
    # print(h_new.shape, w_new.shape)
    offset_hw = torch.unsqueeze(torch.cat([h_new, w_new], dim=-1), dim=-1)
    sin_theta = torch.sin(theta_rotate).cuda()
    cos_theta = torch.cos(theta_rotate).cuda()
    rotate_mat = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
    rotate_mat = torch.reshape(rotate_mat, [-1, 1, out_size[0], out_size[1], 1, 1, 2, 2])
    # print(scaleh.shape, scalew.shape, theta_rotate.shape, rotate_mat.shape, offset_hw.shape)
    # print(rotate_mat)
    # print(offset_hw[0,0,0,0,0,0])
    # print(scaleh[0, 0, 0, 0, 0, 0])
    # print(scalew[0, 0, 0, 0, 0, 0])
    # print(rotate_mat.shape, offset_hw.shape)
    # print(offset_hw[..., 9, 12, :, 0])

    offset_hw = torch.matmul(rotate_mat, offset_hw)
    # print(offset_hw.shape)
    offset_hw = torch.reshape(offset_hw, [-1, 1, out_size[0], out_size[1], kernel_h, kernel_w, 2])
    # print(offset_hw.shape)
    # print(offset_hw[..., 9, 12, :])

    # print(offset_hw[0,0,0,0,:,:,0])
    # print(offset_hw[0, 0, 0, 0, :, :, 1])

    return offset_hw[..., 0] * scaleh, offset_hw[..., 1] * scalew

offset_h, offset_w = scale_rotate_offset(offset_hw[..., 0:1], offset_hw[..., 1:], batch_size=1, out_size=[1, 1], scale=None, theta=3.1415926 * 0.75)
# print(offset_h.shape, offset_w.shape)
weight_h, weight_w = gaussian(offset_h), gaussian(offset_w)
weight_mat = weight_h * weight_w

weight_sum = torch.sum(weight_mat, dim=[-1, -2], keepdim=True)
weight_mat = weight_mat / weight_sum
weight_mat = torch.reshape(weight_mat, [1, 1, 1, 1, -1])
# print(weight_mat.shape)
# save_kernel2imgs(weight_mat / torch.max(weight_mat) * 1.0, "tmp_kernels", "kernel")


