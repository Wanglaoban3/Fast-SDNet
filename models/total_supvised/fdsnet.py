"""FDSNet: An Accurate Real-Time Surface Defect Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['FDSNet']


class FDSNet(nn.Module):
    def __init__(self, num_classes):
        super(FDSNet, self).__init__()
        self.encoder = Encoder(32, 48, 64, (64, 96, 128), 128)
        self.feature_fusion = FeatureFusionModule(64, 64, 128)
        self.classifier = Classifer(128, num_classes)

    def forward(self, x):
        size = x.size()[2:]
        x8, x16,x32, x_en = self.encoder(x)
        x, x8_aux = self.feature_fusion(x8, x_en, x16)
        x = self.classifier(x,size)
        return x


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _GroupConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, groupss=8, **kwargs):
        super(_GroupConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=groupss),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class MBlock(nn.Module):
    """LinearBottleneck used in MobileNetV2 and SElayer added in MobileNetV3"""
    def __init__(self, in_channels, out_channels, seLayer=None, t=6, stride=2, **kwargs):
        super(MBlock, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_se = seLayer
        assert self.use_se in ['selayer','spatial',None]
        if self.use_se =='selayer':
            self.atten = SELayer(out_channels)
        elif self.use_se == 'spatial':
            self.atten = SpatialGate()

    def forward(self, x):
        out = self.block(x)
        if self.use_se:
            out = self.atten(out)
        if self.use_shortcut:
            out = x + out
        return out


class GCU(nn.Module):
    """Global Context Upsampling module"""
    def __init__(self, in_ch1=128, in_ch2=128, in_ch3=128):
        super(GCU, self).__init__()
        self.gcblock1 = ContextBlock2d(in_ch1, in_ch1,pool='avg')
        self.group_conv1 = _GroupConv(in_ch1, in_ch1, 1, 2)
        self.group_conv2 = _GroupConv(in_ch2, in_ch2, 1, 2)
        self.gcblock3 = ContextBlock2d(in_ch3, in_ch3)

    def forward(self, x32, x16, x8):
        x32 = self.gcblock1(x32)
        x16_32 = F.interpolate(x32, x16.size()[2:], mode='bilinear', align_corners=True)
        x16_32 = self.group_conv1(x16_32)
        x16_fusion = x16 + x16_32

        x8_16 = F.interpolate(x16_fusion, x8.size()[2:], mode='bilinear', align_corners=True)
        x8_fusion = torch.mul(x8 , x8_16)
        x8gp = self.group_conv2(x8_fusion)
        x8gc = self.gcblock3(x8gp)
        return x8gc

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class Encoder(nn.Module):
    """Global feature extractor module"""

    def __init__(self, dw_channels1=32, dw_channels2=48,in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(2, 2, 2), **kwargs):
        super(Encoder, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, in_channels, 2)

        self.mblock1 = self._make_layer(MBlock, in_channels, block_channels[0], num_blocks[0], t, 2,[None,None])
        self.mblock2 = self._make_layer(MBlock, block_channels[0], block_channels[1], num_blocks[1], t, 2,[None,'selayer'])
        self.mblock3 = self._make_layer(MBlock, block_channels[1], block_channels[2], num_blocks[2], t, 1,[None,None])

        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1,atten=None):
        layers = []
        layers.append(block(inplanes, planes,atten[0], t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes,atten[i], t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x --> 1/8x
        x = self.conv(x)
        x = self.dsconv1(x)
        x8 = self.dsconv2(x)
        # stack the MoileNetV3 blocks
        x16 = self.mblock1(x8)
        x32 = self.mblock2(x16)
        x32_2 = self.mblock3(x32)
        out = self.ppm(x32_2)
        return x8,x16, x32_2, out


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, x8_in_ch=64, x16_in_ch=64, out_channels=128,  **kwargs):
        super(FeatureFusionModule, self).__init__()

        self.conv_8 = nn.Sequential(
            nn.Conv2d(x8_in_ch, out_channels, 1),
            nn.BatchNorm2d(out_channels),)

        self.conv_16 = nn.Sequential(
            nn.Conv2d(x16_in_ch, out_channels, 1),
            nn.BatchNorm2d(out_channels))

        self.gcu = GCU(out_channels, out_channels, out_channels)

    def forward(self, x8, x32, x16):
        # 1*1 conv
        x8 = self.conv_8(x8)
        x16 = self.conv_16(x16)

        out = self.gcu(x32, x16, x8)

        return out,x8


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        # self.group_conv1 = _GroupConv(dw_channels, dw_channels, stride, 2)
        self.conv1 = nn.Conv2d(dw_channels, dw_channels, stride, )

        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x,size):
        x = self.dsconv1(x)
        # x = self.group_conv1(x)
        x = self.conv1(x)
        x = self.conv(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.max(x, 1)[0].unsqueeze(1)
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3 # 7这里卷积和的大小
        # self.compress = ChannelPool()
        # self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = nn.Sequential(
            nn.Conv2d(1,1, kernel_size,stride=1,padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(1)
        )
    def forward(self, x):
        # x_compress = self.compress(x)
        x_compress = torch.mean(x,1).unsqueeze(1)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        torch.nn.init.constant(m[-1], val=0)
        m[-1].inited = True
    else:
        torch.nn.init.constant(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add']): #pool='att', fusions=['channel_add'], ratio=8
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            torch.nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in')
            self.conv_mask.inited = True

        # if self.channel_add_conv is not None:
        #     last_zero_init(self.channel_add_conv)
        # if self.channel_mul_conv is not None:
        #     last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

# 这个是有ratio的计算
        # if 'channel_add' in fusions:
        #     self.channel_add_conv = nn.Sequential(
        #         nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
        #         nn.LayerNorm([self.planes // ratio, 1, 1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
        #     )
        # else:
        #     self.channel_add_conv = None
        # if 'channel_mul' in fusions:
        #     self.channel_mul_conv = nn.Sequential(
        #         nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
        #         nn.LayerNorm([self.planes // ratio, 1, 1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
        #     )


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if(shape_x[-1] < self.output_size[-1]):
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1]))
            paddzero = paddzero.to('cuda:0')
            x = torch.cat((x, paddzero), axis=-1)

        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


if __name__ == '__main__':
    mod = FDSNet(2)
    t = torch.rand(2, 3, 224, 224)
    p = mod(t)
    print(p)