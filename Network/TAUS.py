import torch
import torch.nn as nn
from data_process.process import save_tensor2imgs


def get_offset_scale(ori_size, dst_size, kernel_size):
    h_in, w_in = ori_size
    h_out, w_out = dst_size

    indice_h = torch.arange(start=1, end=h_out + 1).to(torch.float32)
    indice_w = torch.arange(start=1, end=w_out + 1).to(torch.float32)

    scaleh = float(h_out) / float(h_in)
    scalew = float(w_out) / float(w_in)

    center_h = indice_h / scaleh + 0.5 * (1 - 1 / scaleh) - 1.0
    center_w = indice_w / scalew + 0.5 * (1 - 1 / scalew) - 1.0
    indice_h, indice_w = torch.floor(center_h), torch.floor(center_w)
    start_ind = -((kernel_size - 1) // 2)

    indice_h = torch.reshape(indice_h, [-1, 1]) + torch.arange(start=start_ind, end=kernel_size + start_ind)
    indice_w = torch.reshape(indice_w, [-1, 1]) + torch.arange(start=start_ind, end=kernel_size + start_ind)

    ind_left_h, ind_left_w = torch.reshape(torch.floor(indice_h), [-1]), torch.reshape(torch.floor(indice_w), [-1])

    indice = torch.meshgrid(ind_left_h, ind_left_w)
    indice = torch.cat([torch.unsqueeze(item, dim=-1) for item in indice], dim=-1)

    indice = torch.reshape(indice, [h_out, kernel_size, w_out, kernel_size, -1])
    indice = indice.permute(0, 2, 1, 3, 4)

    center_hw = torch.meshgrid(center_h, center_w)
    center_hw = torch.cat([torch.unsqueeze(item, dim=-1) for item in center_hw], dim=-1)
    center_hw = torch.reshape(center_hw, [h_out, w_out, 1, 1, 2])
    offset = indice[:, :, -start_ind:-start_ind + 1, -start_ind:-start_ind + 1] - center_hw

    scaleh = torch.ones([h_out, w_out, 1, 1, 1]) * scaleh
    scalew = torch.ones([h_out, w_out, 1, 1, 1]) * scalew

    offset_scale = torch.cat([offset, scaleh, scalew], dim=-1)
    offset_scale = offset_scale.permute(4, 0, 1, 2, 3)
    # print(indice.shape, center_hw.shape, offset_scale.shape)

    return offset_scale.cuda(), indice.cuda(), center_hw.cuda()


def pad_select(tensor, indice, pad_width):
    tensor = nn.functional.pad(tensor, (pad_width, pad_width, pad_width, pad_width), mode='reflect')
    tensor = tensor[:, :, indice[..., 0] + pad_width, indice[..., 1] + pad_width]
    return tensor


def slice_mul(selected_pixels, weight):
    # res = torch.mul(selected_pixels, weight)
    n, h, w, cn, channel_num = weight.shape
    batch_size = 2
    batch_num = channel_num // batch_size if channel_num % batch_size == 0 else channel_num // batch_size + 1
    res = torch.zeros([n, h, w, cn]).cuda()
    for i in range(batch_num):
        cur_selected = selected_pixels[..., i * batch_size:(i + 1) * batch_size]
        cur_weight = weight[..., i * batch_size:(i + 1) * batch_size]
        res = res + torch.sum(cur_selected * cur_weight, dim=-1, keepdim=False)
    # print(res.shape)
    return res



class TAUS(nn.Module):
    def __init__(self, in_channel, out_channel, kpc_channels=10, kernel_size=4):
        super(TAUS, self).__init__()
        kpc_trans_dim = kpc_channels // 2
        ft_trans_dim = in_channel // 2
        self.linear_dim_input = 4 + ft_trans_dim
        self.out_channel = out_channel
        self.linear_dim_output = in_channel * out_channel * (kernel_size ** 2)
        self.linear1 = nn.Sequential(nn.Linear(self.linear_dim_input, self.linear_dim_output // 8),
                                    nn.Tanh(),
                                    nn.Linear(self.linear_dim_output // 8, self.linear_dim_output // 8)
                                    )
        self.linear2 = nn.Sequential(nn.Linear(self.linear_dim_input, self.linear_dim_output // 8),
                                    nn.Tanh(),
                                    nn.Linear(self.linear_dim_output // 8, self.linear_dim_output // 8)
                                    )
        self.linear3 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.linear_dim_output// 4, self.linear_dim_output // 2),
            nn.LeakyReLU(),
            nn.Linear(self.linear_dim_output // 2, self.linear_dim_output)
        )
        self.kpc_trans = nn.Conv2d(kpc_channels, kpc_trans_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.feature_trans = nn.Conv2d(in_channel, ft_trans_dim, kernel_size=4, stride=1, padding=0, padding_mode='reflect')
        self.kernel_size = kernel_size
        self.tanh = nn.Tanh()
        # self.upscale_weight = nn.Parameter(torch.randn(1, 192, 192, in_channel*out_channel*(kernel_size**2)) * 0.07 + 0.007)
        # self.conv_tail = nn.ModuleList([
        #     nn.Conv2d(ft_trans_dim, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        #     nn.Tanh(),
        #     nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        #     nn.Tanh(),
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        # ])

    def forward_bkup(self, lr_features, kernel_pc, dst_size):
        # kpc_trans = self.kpc_trans(kernel_pc)
        feature_trans = self.feature_trans(lr_features)
        # feature = torch.cat([kpc_trans, feature_trans], dim=1)
        feature_us = nn.functional.interpolate(feature_trans, dst_size, mode='bicubic')
        for layer in self.conv_tail:
            feature_us = layer(feature_us)
        return self.tanh(feature_us)

    def offscale2weight(self, features):
        features1 = self.linear1(features)
        features2 = self.linear2(features)
        features = torch.cat([features1 * features2, features1], dim=-1)
        features = self.linear3(features)
        return features

    def forward(self, lr_features, kernel_pc, dst_size):
        # print(lr_features.shape, feature_trans.shape, nn.functional.pad(lr_features, (1, 2, 1, 2), mode='reflect').shape)
        bs, cn, h_in, w_in = lr_features.shape
        h_out, w_out = dst_size
        offset_scale, indice, center_ind = get_offset_scale([h_in, w_in], [h_out, w_out], self.kernel_size)

        indice, center_ind = indice.long(), torch.floor(center_ind).long()
        min_ind_lu = torch.min(indice)
        min_ind_rl = max([torch.max(indice[..., 0] - h_in + 1), torch.max(indice[..., 1] - w_in + 1)])
        pad_width = max(-min_ind_lu, min_ind_rl)

        # pixels to be weighted
        selected_pixels = pad_select(lr_features, indice, pad_width)
        selected_pixels = torch.reshape(selected_pixels.permute(0, 2, 3, 1, 4, 5), [bs, h_out, w_out, 1, -1])

        # get kpc padded and translated
        # kpc_trans = self.kpc_trans(kernel_pc)
        # kpc_trans = pad_select(kpc_trans, center_ind, pad_width)

        # get sr features padded and translated
        feature_trans = self.feature_trans(nn.functional.pad(lr_features, (1, 2, 1, 2), mode='reflect'))
        sr_features = pad_select(feature_trans, center_ind, pad_width)

        # concat offset, scale and img textures
        offset_scale = torch.unsqueeze(offset_scale, dim=0).repeat(bs, 1, 1, 1, 1, 1)
        offset_scale_features =  torch.cat([sr_features, offset_scale], dim=1)
        offset_scale_features = offset_scale_features.permute(0, 2, 3, 1, 4, 5)[..., 0, 0]

        try:
            weight = self.offscale2weight(offset_scale_features)
            weight = torch.reshape(weight, [bs, h_out, w_out, self.out_channel, -1])
            sr_res = slice_mul(selected_pixels, weight).permute(0, 3, 1, 2)
        except:
            sr_res = torch.zeros([bs, self.out_channel, h_out, w_out], dtype=torch.float32).cuda()
            step = 4
            step_num = h_out // step if h_out % step == 0 else h_out // step + 1
            for i in range(step_num):
                # print(i)
                cur_weight = self.offscale2weight(offset_scale_features[:, i * step: (i + 1) * step, :, :])
                cur_weight = torch.reshape(cur_weight, [bs, step, w_out, self.out_channel, -1])
                cur_res = slice_mul(selected_pixels[:, i * step: (i + 1) * step], cur_weight).permute(0, 3, 1, 2)
                sr_res[:, :, i * step: (i + 1) * step, :] = cur_res
            # print(offset_scale_features.shape, selected_pixels.shape, sr_res.shape)


        sr_res = self.tanh(sr_res)

        # save_tensor2imgs(selected_tmp, "feature_tmp", "feature")
        # save_tensor2imgs(lr_features[:, 0:3, :, :], "feature_tmp", "feature_lr")
        # print(selected_pixels.shape, weight.shape)
        # print(sr_res.shape)
        # print(torch.max(sr_res), torch.min(sr_res))
        # print(sr_res[0, :, 80, 80])

        return sr_res
