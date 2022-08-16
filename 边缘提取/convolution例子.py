import numpy
import torch
import torch.nn as nn

m1 = nn.Conv2d(16,33,3,stride=2) # input_channel, output_channel, kernal_size,stride
# 定义输入
input = torch.randn(20,16,50,100) # batch, channel, height, width

output=m1(input)
print(output.detach().numpy().shape)

"""
    特征图尺寸：
        (H/W - kernel_size + padding*2) / stride
"""