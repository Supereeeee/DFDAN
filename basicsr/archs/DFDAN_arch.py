from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
# from fvcore.nn import FlopCountAnalysis, parameter_count_table    # 计算参数量和运算量
# from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis


class BSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class DBSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 dilation, stride=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise_dilated
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class DLSA(nn.Module):
    '''
    Dynamic lightweight spatial attention (DLSA)
    '''
    def __init__(self, channels):
        super(DLSA, self).__init__()

        self.DBSConv1 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=1, padding=1)
        self.DBSConv2 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=2, padding=2)
        self.DBSConv3 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=3, padding=3)
        self.DBSConv3_stride = DBSConv(channels // 4, channels // 4, kernel_size=3, stride=2, dilation=3, padding=3)
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(in_channels=channels // 4, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x0):
        h0 = x0.size(2)
        w0 = x0.size(3)

        x1 = self.conv1_1(x0)
        x2 = self.DBSConv3_stride(x1)
        h = x2.size(2)
        w = x2.size(3)
        down_size2 = (h // 2, w // 2)
        down_size4 = (h // 4, w // 4)
        down_size8 = (h // 8, w // 8)

        s2 = F.adaptive_max_pool2d(x2, down_size2)
        h2 = s2.size(2)
        w2 = s2.size(3)
        s4 = F.adaptive_max_pool2d(x2, down_size4)
        h4 = s4.size(2)
        w4 = s4.size(3)
        s8 = F.adaptive_max_pool2d(x2, down_size8)

        s1 = self.act(self.conv1_2(x2))
        s1_down = F.adaptive_max_pool2d(s1, down_size2)

        s2 = s2 + s1_down
        s2 = self.act(self.DBSConv3(s2))
        s2_down = F.adaptive_max_pool2d(s2, down_size4)

        s4 = s4 + s2_down
        s4 = self.act(self.DBSConv2(s4))
        s4_down = F.adaptive_max_pool2d(s4, down_size8)

        s8 = s8 + s4_down
        s8 = self.act(self.DBSConv1(s8))
        s8_up = F.interpolate(s8, size=(h4, w4), mode='bilinear', align_corners=False)
        s8 = F.interpolate(s8, size=(h, w), mode='bilinear', align_corners=False)

        s4 = s4 + s8_up
        s4 = self.act(self.DBSConv2(s4))
        s4_up = F.interpolate(s4, size=(h2, w2), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(h, w), mode='bilinear', align_corners=False)

        s2 = s2 + s4_up
        s2 = self.act(self.DBSConv3(s2))
        s2_up = F.interpolate(s2, size=(h, w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=(h, w), mode='bilinear', align_corners=False)

        s1 = s1 + s2_up
        s1 = self.act(self.conv1_2(s1))
        out = s1 + s2 + s4 + s8

        out = F.interpolate(out, size=(h0, w0), mode='bilinear', align_corners=False)

        out = out + x1
        out = self.conv1_3(out)
        out = self.sigmoid(out)
        out = out * x0
        return out

class DLCA(nn.Module):
    '''
    Dynamic lightweight channel attention (DLSA)
    '''

    def __init__(self, channels):
        super(DLCA, self).__init__()
        self.split_channels = int(channels // 4)
        self.DBSConv1 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=1, padding=1)
        self.DBSConv2 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=2, padding=2)
        self.DBSConv3 = DBSConv(channels // 4, channels // 4, kernel_size=3, dilation=3, padding=3)
        self.DBSConv3_stride = DBSConv(channels, channels, kernel_size=3, stride=2, dilation=3, padding=3)
        self.conv1_mid = nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1_up = nn.Conv2d(in_channels=channels // 4, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x0):
        h = x0.size(2)
        w = x0.size(3)
        x1 = self.DBSConv3_stride(x0)
        split_c1, split_c2, split_c3, split_c4 = torch.split(x1, (self.split_channels, self.split_channels, self.split_channels, self.split_channels), dim=1)
        split_c1 = self.act(self.DBSConv1(split_c1))
        split_c2 = self.act(self.DBSConv2(split_c2))
        split_c3 = self.act(self.DBSConv3(split_c3))
        split_c4 = self.act(self.conv1_mid(split_c4))
        out = split_c1 + split_c2 + split_c3 + split_c4
        out = self.conv1_up(out)
        out = out + x1
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = self.sigmoid(out)
        out = x0 * out
        return out

class DFDB(nn.Module):
    '''
    Dilated Feature Distillation Block (DFDB)
    '''
    def __init__(self, channels):
        super(DFDB, self).__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = BSConv(channels, channels, kernel_size=3, padding=1)
        self.c1_r_d = DBSConv(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = BSConv(channels, channels, kernel_size=3, padding=1)
        self.c2_r_d = DBSConv(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = BSConv(channels, channels, kernel_size=3, padding=1)
        self.c3_r_d = DBSConv(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.c4 = BSConv(channels, channels // 2, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(channels // 2 * 4, channels, 1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.act(self.c1_r_d(input))
        r_c1 = self.act(self.c1_r(r_c1))
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c2))
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c3))
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        return out


class DFDAB(nn.Module):
    '''
    Dilated Feature Distillation Attention Block (DFDAB)
    '''
    def __init__(self, channels):
        super(DFDAB, self).__init__()
        self.DFDB = DFDB(channels)
        self.DLSA = DLSA(channels)
        self.DLCA = DLCA(channels)

    def forward(self, x):
        out = self.DFDB(x)
        out = self.DLSA(out)
        out = self.DLCA(out)
        return out + x


@ARCH_REGISTRY.register()
class DFDAN(nn.Module):
    def __init__(self, in_channels, channels, num_block, out_channels, upscale):
        super(DFDAN, self).__init__()
        self.fea_conv = DBSConv(in_channels * 4, channels, kernel_size=3, padding=3, dilation=3)
        self.B1 = DFDAB(channels)
        self.B2 = DFDAB(channels)
        self.B3 = DFDAB(channels)
        self.B4 = DFDAB(channels)
        self.B5 = DFDAB(channels)
        self.B6 = DFDAB(channels)
        self.B7 = DFDAB(channels)
        self.B8 = DFDAB(channels)
        self.c1 = nn.Conv2d(channels * num_block, channels, 1)
        self.GELU = nn.GELU()
        self.c2 = BSConv(channels, channels, kernel_size=3, padding=1)
        self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=channels, num_out_ch=out_channels)

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

# net = DFDAN(in_channels=3, channels=56, num_block=8, out_channels=3, upscale=4)  # 定义好的网络模型,实例化
# print(net)
# input = torch.randn(1, 3, 320, 180)  # 1280*720-640, 360-427, 240-320, 180
# flops, params = profile(net, (input,))
# print('flops[G]: ', flops/1e9, 'params[K]: ', params/1e3)

# flops[G]:  21.69967968 params[K]:  407.0   (56channels, 8blocks, x4)
# flops[G]:  37.52432208 params[K]:  396.395   (56channels, 8blocks, x3)
# flops[G]:  82.61875328 params[K]:  388.82   (56channels, 8blocks, x2)