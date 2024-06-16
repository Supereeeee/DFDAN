from thop import profile
import torch
from .DFDAN_arch import DFDAN

net = DFDAN(in_channels=3, channels=56, num_block=8, out_channels=3, upscale=4)
input = torch.randn(1, 3, 320, 180)
flops, params = profile(net, (input,))
print('flops[G]: ', flops/1e9, 'params[K]: ', params/1e3)

# flops[G]:  21.69967968 params[K]:  407.0   (56channels, 8blocks, x4)
# flops[G]:  37.52432208 params[K]:  396.395   (56channels, 8blocks, x3)
# flops[G]:  82.61875328 params[K]:  388.82   (56channels, 8blocks, x2)
