import torch
from Network.TAUS import get_offset_scale


def show_xy(x, y, offset_scale, indice, center_hw):
    offset_scale = offset_scale[:, x, y, :, :]
    indice = indice[x, y, :, :]
    center_hw = center_hw[x, y, :, :]
    print(torch.squeeze(offset_scale).detach().cpu().numpy())
    print(torch.squeeze(indice).detach().cpu().numpy())
    print(torch.squeeze(center_hw).detach().cpu().numpy())
    center_hw = torch.floor(center_hw).long()
    print(torch.squeeze(center_hw).detach().cpu().numpy())
    print()

in_size = [10, 17]
out_size = [20, 34]
kernel_size = 4
offset_scale, indice, center_hw = get_offset_scale(in_size, out_size, kernel_size)
print(offset_scale.shape, indice.shape, center_hw.shape)
print(offset_scale.dtype, indice.dtype, center_hw.dtype)
print()


x, y = 0, 0
show_xy(x, y, offset_scale, indice, center_hw)
x, y = 0, 1
show_xy(x, y, offset_scale, indice, center_hw)
x, y = 0, 2
show_xy(x, y, offset_scale, indice, center_hw)
