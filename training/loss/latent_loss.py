import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import os
import math
from face_model.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class TeacherEncoder(nn.Module):

    def __init__(self,size = 512, narrow = 1, channel_multiplier = 2, pretrained_weight = None, device='cpu'):

        super(TeacherEncoder,self).__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel

        if pretrained_weight is not None:

            for name, weight in pretrained_weight.items():

                ecd = getattr(self, name)
                ecd.load_state_dict(weight)

    def forward(self,x):

        latent = []
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            x = ecd(x)
        
        latent.append(x)

        return latent

class LatentLoss(nn.Module):

    def __init__(self, args, weight, device="cpu"):

        super().__init__()

        self.teacher_net = TeacherEncoder(size=args.size, narrow=args.narrow, channel_multiplier=args.channel_multiplier, pretrained_weight=weight, device=device)

        for params in self.teacher_net.parameters():
            params.requires_grad = False

    def forward(self, x, student_ans):
        
        teacher_soln = self.teacher_net(x)
        batch = x.shape[0]
        loss = 0.0 
        for ts, sa in zip(teacher_soln, student_ans):
            for i in range(batch):
                loss = loss + F.mse_loss(GramMatrix(ts[i]), GramMatrix(sa[i]))

        return loss
 
def GramMatrix(x):

    c,h,w = x.shape
    x = x.view(c,-1)
    GM_result = torch.mm(x, x.permute(1,0).contiguous()) / (c * h * w)

    return GM_result

