# Adapted from
# https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py
from typing import NamedTuple

import torch.nn as nn

from superres.nn.model import common

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    return EDSR(args)


class Args(NamedTuple):
    # Number of super-resolution blocks
    n_resblocks: int = 16
    # Number of feature maps
    n_feats: int = 64
    # Max value of color channels
    max_color_level: int = 255
    # Number of color channels
    n_colors: int = 3
    # Residual scaling
    res_scale: float = 0.1


class EDSR(nn.Module):
    def __init__(self, args: Args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        if args.n_colors == 3:
            self.sub_mean = common.MeanShiftRgb(args.max_color_level)
            self.add_mean = common.MeanShiftRgb(args.max_color_level, sign=1)
        elif args.n_colors == 1:
            self.sub_mean = common.MeanShiftGrayscale(args.max_color_level)
            self.add_mean = common.MeanShiftGrayscale(args.max_color_level,
                                                      sign=1)
        else:
            raise RuntimeError("Only grayscale and RGB is supported")

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                                .format(name, own_state[name].size(),
                                        param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
