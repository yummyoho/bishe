import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from torchvision.models import resnet34 as resnet

'''
Second Stage   
QAnet = Quality-aware Region Selection Subnet
'''


class RCInet(nn.Module):
    def __init__(self):
        super(RCInet, self).__init__()
        self.resnet_D = resnet()
        self.resnet_T = resnet()
        self.resnet_V = resnet()

        # 全局平均池化15个
        self.avg_pool_V0 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_V1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_V2 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_V3 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_V4 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_T0 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_T1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_T2 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_T3 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_T4 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_D0 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_D1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_D2 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_D3 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_D4 = nn.AdaptiveAvgPool2d(1)

        self.w0_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64 * 3, kernel_size=1)
        self.ReLU = nn.ReLU(inplace=True)  # 不产生额外张量
        self.sigmoid = nn.Sigmoid()

        ##################################################################################################################

        self.resconvV1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvD1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvT1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, rgb, t, d):
        # D Branch
        # Encoder
        x_u_D = self.resconvD1(d)
        x_u_D = self.resnet_D.bn1(x_u_D)
        x_u_D = self.resnet_D.relu(x_u_D)

        x0_D = self.resnet_D.maxpool(x_u_D)
        x1_D = self.resnet_D.layer1(x0_D)
        x2_D = self.resnet_D.layer2(x1_D)
        x3_D = self.resnet_D.layer3(x2_D)
        x4_D = self.resnet_D.layer4(x3_D)
        # T Branch
        # Encoder
        x_u_T = self.resconvT1(t)
        x_u_T = self.resnet_T.bn1(x_u_T)
        x_u_T = self.resnet_T.relu(x_u_T)

        x0_T = self.resnet_T.maxpool(x_u_T)
        x1_T = self.resnet_T.layer1(x0_T)
        x2_T = self.resnet_T.layer2(x1_T)
        x3_T = self.resnet_T.layer3(x2_T)
        x4_T = self.resnet_T.layer4(x3_T)
        # V Branch
        # Encoder
        x_u_V = self.resconvV1(rgb)
        x_u_V = self.resnet_V.bn1(x_u_V)
        x_u_V = self.resnet_V.relu(x_u_V)

        x0_V = self.resnet_V.maxpool(x_u_V)
        x1_V = self.resnet_V.layer1(x0_V)
        x2_V = self.resnet_V.layer2(x1_V)
        x3_V = self.resnet_V.layer3(x2_V)
        x4_V = self.resnet_V.layer4(x3_V)

        # F0
        x0_D_gap = self.avg_pool_D0(x0_D)
        x0_T_gap = self.avg_pool_T0(x0_T)
        x0_V_gap = self.avg_pool_V0(x0_V)

        x0_VDT = torch.cat((x0_D_gap, x0_T_gap, x0_V_gap), dim=1)  # VDT按照通道维度拼接
        w0_VDT = self.w0_conv1x1(x0_VDT)  # 生成三模态的对应权重图
        w0_D, w0_T, w0_V = torch.chunk(w0_VDT, 3, dim=1)  # 通道拆分

        x_u_D_weighted = x_u_D * w0_D  # (batch_size, 64, H, W)
        x_u_T_weighted = x_u_T * w0_T  # (batch_size, 64, H, W)
        x_u_V_weighted = x_u_V * w0_V  # (batch_size, 64, H, W)
        x_u_VDT = torch.cat((x_u_D_weighted, x_u_T_weighted, x_u_V_weighted), dim=1)

    def load_pretrained_model(self):
        self.resnet_D.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        self.resnet_T.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        self.resnet_V.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        print('loading pretrained model success!')


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
