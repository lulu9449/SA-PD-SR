import torch
import torch.nn as nn


class attentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, res=True):
        super(attentionModule, self).__init__()
        self.conv_head = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                   padding_mode='reflect')
        self.conv_glob = nn.AdaptiveAvgPool2d(1)
        self.attention_channel = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.tail = nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_res = res

    def forward(self, tensor):
        tensor_input = tensor
        tensor = self.conv_head(tensor)
        global_tensor = self.conv_glob(tensor)
        channel_tensor = self.attention_channel(tensor)
        global_tensor = tensor * global_tensor
        channel_tensor = tensor * channel_tensor
        tensor = torch.cat([tensor, global_tensor, channel_tensor], dim=1)
        tensor = self.lrelu(self.tail(tensor))
        # print(tensor.shape, global_tensor.shape, channel_tensor.shape)
        # print(tensor.shape, tensor_input.shape)
        if self.is_res:
            return tensor + tensor_input
        return tensor



class initSR(nn.Module):
    def __init__(self, in_channel, out_channels, mid_channels=16, block_num=4):
        super(initSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(mid_channels + in_channel, mid_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.attention_blocks = nn.ModuleList([attentionModule(mid_channels + in_channel, mid_channels, False) for _ in range(block_num)])

        self.conv5 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, lr_imgs):
        texture = self.conv1(lr_imgs)
        texture = self.conv2(torch.cat([lr_imgs, texture], dim=1))
        for layer in self.attention_blocks:
            texture = torch.cat([texture, lr_imgs], dim=1)
            texture = layer(texture)
        texture = self.conv5(texture)
        return texture


class interActor(nn.Module):
    def __init__(self, feature1_dim, feature2_dim, out_channels, mid_channels, block_num, res=True, skip=True):
        super(interActor, self).__init__()

        self.skip = skip

        self.f1_head = nn.Conv2d(feature1_dim, mid_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.f1_head_ = nn.Conv2d(mid_channels // 2 + feature1_dim, mid_channels // 2, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect')
        self.f2_head = nn.Conv2d(feature2_dim, mid_channels // 2, kernel_size=3, stride=1, padding=1,
                                 padding_mode='reflect')

        self.attention_blocks = nn.ModuleList(
            [attentionModule(mid_channels + feature1_dim, mid_channels, False) for _ in range(block_num)])

        self.tail = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        # self.f1_head = nn.Conv2d(feature1_dim, mid_channels // 8, kernel_size=3, stride=1, padding=1,
        #                            padding_mode='reflect')
        # self.f2_head = nn.Conv2d(feature2_dim, mid_channels // 8, kernel_size=3, stride=1, padding=1,
        #                            padding_mode='reflect')
        # attention_channels = mid_channels // 4
        # self.body = nn.ModuleList([attentionModule(attention_channels, attention_channels, res) for _ in range(block_num)])
        # self.tail = nn.Conv2d(attention_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, tensor1, tensor2):
        texture1 = self.f1_head_(torch.cat([self.f1_head(tensor1), tensor1], dim=1))
        texture = torch.cat([texture1, self.f2_head(tensor2)], dim=1)
        for layer in self.attention_blocks:
            texture = torch.cat([texture, tensor1], dim=1) if self.skip else texture
            texture = layer(texture)
        texture = self.tail(texture)
        return texture


# model = initSR(3, 10, 48, 6).cuda()
# input1 = torch.randn(12, 3, 8, 9).cuda()
# # input2 = torch.randn(12, 10, 8, 9).cuda()
# res = model(input1)
# print(res.shape)
#
#
# model = interActor(10, 10, 10, 48, 6).cuda()
# input1 = torch.randn(12, 10, 8, 9).cuda()
# input2 = torch.randn(12, 10, 8, 9).cuda()
# res = model(input1, input2)
# print(res.shape)

