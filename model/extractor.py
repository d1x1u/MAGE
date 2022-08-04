import math
from multiprocessing.sharedctypes import Value
import torch
from torch import nn
from torch.nn import functional as F
from utils.toolbox import initialize


# def vanilla(args):
#     """
#     vanilla version 1.0
#     """
#     in_channel = args.pca_component if args.use_pca else args.bands_num
#     b0 = nn.Sequential(
#         nn.Conv2d(in_channel, args.hsi_channels[0], kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(args.hsi_channels[0]),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )
#     b1 = nn.Sequential(
#         nn.Conv2d(args.hsi_channels[0], args.hsi_channels[1], kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(args.hsi_channels[1]),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )
#     b2 = nn.Sequential(
#         nn.Conv2d(args.hsi_channels[1], args.hsi_channels[2], kernel_size=1, stride=1),
#         nn.BatchNorm2d(args.hsi_channels[2]),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )
#     net = nn.Sequential(
#         b0, b1, b2,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten()
#     )
#     return net

def vanilla(args):
    """
    vanilla version 2.0
    """
    in_channel = args.pca_component if args.use_pca else args.bands_num
    b0 = nn.Sequential(
        nn.Conv2d(in_channel, args.hsi_channels[0], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(args.hsi_channels[0]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b1 = nn.Sequential(
        nn.Conv2d(args.hsi_channels[0], args.hsi_channels[1], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(args.hsi_channels[1]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(
        nn.Conv2d(args.hsi_channels[1], args.hsi_channels[2], kernel_size=1, stride=1),
        nn.BatchNorm2d(args.hsi_channels[2]),
        nn.ReLU(inplace=True),
    )
    # TODO: 加了ECA之后，NMI有接近一个点的提升，ACC有三个点的提升，两点启示：
    # 1. 在这个vanilla基础上逐渐加入残差连接、空间注意力、探索注意力模块加的位置
    # 2. 匈牙利算法可能导致ACC不能很好的反映聚类的好坏，看看NMI是否是一个更好的评估指标？
    net = nn.Sequential(
        b0, b1, b2,
            ECA(args.hsi_channels[-1]),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return net

"""
Attention Mecanism, attention mask is the output.
"""
class ECA(nn.Module):
    """
    Spectral attention module from Efficient Channel Attention.
    Reference: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1 ,-2)).transpose(-1, -2).unsqueeze(-1)
        mask = self.sigmoid(y)

        return mask

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

class CA(nn.Module):
    """
    Spatial attention module from Coordinate Attention.
    Reference: https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_channels = max(8, in_channels // reduction)

        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        _, _, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv(y)
        y = self.bn(y)
        y = self.act(y)

        mask_h, mask_w = torch.split(y, [H, W], dim=2)
        mask_w = mask_w.permute(0, 1, 3, 2)
        
        mask_h = self.conv_h(mask_h).sigmoid()
        mask_w = self.conv_w(mask_w).sigmoid()

        mask = mask_h * mask_w
        return mask

class CUSTOM_SPEC(nn.Module):
    """
    Spectral attention module combining the ECA and CA methods.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k, padding=(k-1)//2, bias=False)
        self.conv_w = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k, padding=(k-1)//2, bias=False)
        
    def forward(self, x):
        h = self.pool_h(x)
        h = self.conv_h(h.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        h = h.sigmoid()

        w = self.pool_w(x)
        w = self.conv_w(w.squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2)
        w = w.sigmoid()

        mask = h * w
        return mask

class CUSTOM_SPAT(nn.Module):
    """
    Spatial attention module by custom design.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        assert H == W, 'Please check the LiDAR patch size. H should equal W.'
        center_pixel_index = (H * W) // 2
        
        y = x.permute(0,2,3,1).reshape(N, -1, C)
        center = y[:, center_pixel_index, :].unsqueeze(1)
        mask = torch.exp(-3.*torch.cdist(center, y)).reshape(N, H, W).unsqueeze(1)

        return mask

class Conv_1x1_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, use_swish):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class Conv_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, use_swish):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class Residual_Unit(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, att_module, use_swish):
        super().__init__()
        self.conv1 = Conv_Unit(in_channels, hidden_channels, use_swish)

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.att = att_module

        self.conv1_2 = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out) * out

        residual = self.conv1_2(x)

        out += residual
        out = self.act(out)
        return out

class Extractor(nn.Module):
    def __init__(self, raw_channel, channels, att_module_after_first, att_module, use_swish):
        super().__init__()
        self.conv_unit = Conv_Unit(raw_channel, channels[0], use_swish)
        self.att_module_after_first = att_module_after_first
        self.residual_unit = Residual_Unit(channels[0], channels[1], channels[2], att_module, use_swish)
        self.conv1x1 = Conv_1x1_Unit(channels[2], channels[2], use_swish)
    
    def forward(self, x):
        out = self.conv_unit(x)
        out = self.att_module_after_first(out) * out
        out = self.residual_unit(out)
        out = self.conv1x1(out)
        return out

class SSARN(nn.Module):
    """"
    猜想搜参的时候,应该不用ACC评价聚类效果。而是使用NMI、F、ARI等指标
    可能0.55和0.6差别不大,是因为二者的NMI差不多。

    现在准备不使用聚类任务作为第一阶段了。考虑使用fine tune的classification accuracy作为评价指标。
    """
    def __init__(self, args):
        super().__init__()
        self.raw_hsi_channels = args.pca_component if args.use_pca else args.hsi_bands_num
        self.raw_lidar_channels = args.lidar_bands_num

        """Define the attention module after the first conv unit."""
        if args.att_after_first_mode == 0:
            self.spec_att_after_first = nn.Identity()
            self.spat_att_after_first = nn.Identity()
        elif args.att_after_first_mode == 1:
            self.spec_att_after_first = ECA(channels=args.hsi_channels[0])
            self.spat_att_after_first = ECA(channels=args.lidar_channels[0])
        elif args.att_after_first_mode == 2:
            self.spec_att_after_first = CA(in_channels=args.hsi_channels[0], out_channels=args.hsi_channels[0])
            self.spat_att_after_first = CA(in_channels=args.lidar_channels[0], out_channels=args.lidar_channels[0])
        elif args.att_after_first_mode == 3:
            self.spec_att_after_first = CUSTOM_SPEC(channels=args.patch_size)
            self.spat_att_after_first = CUSTOM_SPEC(channels=args.patch_size)
        else:
            raise ValueError

        """Define the spectral attention module within the residual unit."""
        if args.spec_att_mode == 0:
            self.spec_att = nn.Identity()
        elif args.spec_att_mode == 1:
            self.spec_att = ECA(channels=args.hsi_channels[-1])
        elif args.spec_att_mode == 2:
            self.spec_att = CA(in_channels=args.hsi_channels[-1], out_channels=args.hsi_channels[-1])
        elif args.spec_att_mode == 3:
            self.spec_att = CUSTOM_SPEC(channels=args.patch_size)
        else:
            raise ValueError
        
        self.hsi_extractor = Extractor(self.raw_hsi_channels, args.hsi_channels, self.spec_att_after_first, self.spec_att, args.use_swish)
        
        """Define the spatial attention module."""
        if args.spat_att_mode == 0:
            self.spat_att = nn.Identity()
        elif args.spat_att_mode == 1:
            self.lidar_extractor = Extractor(self.raw_lidar_channels, args.lidar_channels, self.spat_att_after_first, nn.Identity(), args.use_swish)
            self.spat_att = CA(in_channels=args.lidar_channels[-1], out_channels=1)
        elif args.spat_att_mode == 2:
            self.spat_att = CUSTOM_SPAT()
        else:
            raise ValueError
        self.spat_att_mode = args.spat_att_mode

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        hsi_raw = x[:, :-self.raw_lidar_channels, ...]
        lidar_raw = x[:, -self.raw_lidar_channels:, ...]
        
        if self.spat_att_mode == 0:
            hsi_feat = self.hsi_extractor(hsi_raw)
        elif self.spat_att_mode == 1:
            """Borrowing the idea from CBAM, channel attention first, then spatial attention."""
            hsi_feat = self.hsi_extractor(hsi_raw)
            lidar_feat = self.lidar_extractor(lidar_raw)
            lidar_mask = self.spat_att(lidar_feat)
            hsi_feat = lidar_mask * hsi_feat
        elif self.spat_att_mode == 2:
            """Modulate the raw hsi first."""
            lidar_mask = self.spat_att(lidar_raw)
            hsi_raw = lidar_mask * hsi_raw
            hsi_feat = self.hsi_extractor(hsi_raw)
        else:
            raise ValueError

        feat = self.avg_pool(hsi_feat)
        feat = self.flatten(feat)
        return feat

if __name__ == "__main__":
    dataset_name = "MUUFL"
    args, device, labels_text, ignored_label = initialize(dataset_name)
    net = SSARN(args).to(device)
    HSI = torch.randn((512, 32, 9, 9)).to(device)
    LiDAR = torch.randn((512, 2, 9, 9)).to(device)
    X = torch.cat((HSI, LiDAR), dim=1)
    print(net(X).shape)
